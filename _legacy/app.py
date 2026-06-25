# -*- coding: utf-8 -*-
import os
import logging
import tempfile
from pathlib import Path
from itertools import product
import numpy as np
import skrf as rf
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
import werkzeug.utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==================== 元件库 ====================
class ComponentLibrary:
    def __init__(self):
        self.components = {}

    def _parse_nominal(self, part_no, comp_type):
        part = part_no.upper().replace('-', '')
        import re
        if comp_type == 'L':
            m = re.search(r'(\d+)([NR])(\d+)', part)
            if m:
                a, unit, b = int(m.group(1)), m.group(2), int(m.group(3))
                return float(a) + b / 10.0 if a > 0 else b / 10.0
            m = re.search(r'(\d+)N', part)
            if m: return float(m.group(1))
        elif comp_type == 'C':
            m = re.search(r'(\d+)[R](\d+)', part)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                return float(a) + b / 10.0
        return None

    def load_from_folders(self, l_folder="", c_folder=""):
        self.components.clear()
        def scan_folder(folder, comp_type):
            if not folder or not os.path.exists(folder):
                logger.warning(f"Folder not found: {folder}")
                return 0
            folder_path = Path(folder)
            s2p_files = list(folder_path.rglob("*.s2p")) + list(folder_path.rglob("*.S2P"))
            count = 0
            for file in s2p_files:
                try:
                    ntwk = rf.Network(str(file))
                    if ntwk.nports != 2: continue
                    part_no = file.stem
                    unique_name = f"{comp_type}_{part_no}"
                    self.components[unique_name] = {
                        'network': ntwk,
                        'type': comp_type,
                        'filename': str(file),
                        'original_name': part_no,
                        'nominal': self._parse_nominal(part_no, comp_type),
                        'unit': 'nH' if comp_type == 'L' else 'pF'
                    }
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to load {file}: {e}")
            return count

        l_cnt = scan_folder(l_folder, 'L')
        c_cnt = scan_folder(c_folder, 'C')
        logger.info(f"Loaded L: {l_cnt}, C: {c_cnt}, total: {len(self.components)}")
        return len(self.components)

    def get_component_names_by_type(self, comp_type=None):
        if comp_type:
            return [name for name, info in self.components.items() if info['type'] == comp_type]
        return list(self.components.keys())


# ==================== 拓扑生成器（相邻不同） ====================
class TopologyGenerator:
    @classmethod
    def get_topologies(cls, min_elements=1, max_elements=4):
        topologies = []
        for n in range(min_elements, max_elements + 1):
            for combo in product(['S', 'P'], repeat=n):
                valid = True
                for i in range(n - 1):
                    if combo[i] == combo[i+1]:
                        valid = False
                        break
                if valid:
                    topologies.append(combo)
        return topologies


# ==================== 匹配优化器 ====================
class MatchingOptimizer:
    def __init__(self, component_lib, antenna_net, bands_mhz, num_points=5,
                 l_min_nh=0.1, l_max_nh=20.0, c_min_pf=0.1, c_max_pf=20.0,
                 use_ga=False, ga_population=80, ga_generations=50,
                 use_ideal=False, ideal_l_step=0.05, ideal_c_step=0.05):
        self.lib = component_lib
        self.antenna = antenna_net
        self.bands = bands_mhz
        self.num_points = num_points
        self.l_min, self.l_max = l_min_nh, l_max_nh
        self.c_min, self.c_max = c_min_pf, c_max_pf
        self.use_ga = use_ga
        self.ga_pop = ga_population
        self.ga_gen = ga_generations
        self.use_ideal = use_ideal
        self.ideal_l_step = ideal_l_step
        self.ideal_c_step = ideal_c_step

        self.eval_freqs_hz = np.unique(np.concatenate([np.linspace(s*1e6, e*1e6, num_points) for s, e in bands_mhz]))
        self.full_freqs_hz = self.antenna.frequency.f
        self.antenna_interp = self.antenna.interpolate(self.eval_freqs_hz)
        self.antenna_full = self.antenna

        self.short_1port = rf.Network(
            frequency=self.full_freqs_hz,
            s=np.array([[[-1.+0j, 0.+0j], [0.+0j, -1.+0j]]] * len(self.full_freqs_hz), dtype=complex),
            z0=50
        )
        self._cache = {}

    def _get_key(self, topology, component_or_specs):
        return (tuple(topology), tuple(component_or_specs))

    def _build_ideal_matching_network(self, topology, comp_specs, freqs_hz):
        Z0 = 50
        omega = 2 * np.pi * freqs_hz
        s_thru = np.array([[[0.+0j, 1.+0j], [1.+0j, 0.+0j]]] * len(freqs_hz), dtype=complex)
        current = rf.Network(frequency=freqs_hz, s=s_thru, z0=Z0)

        for (comp_type, value), topo_type in zip(comp_specs, topology):
            if comp_type == 'L':
                Z = 1j * omega * (value * 1e-9)
            else:  # 'C'
                Z = 1.0 / (1j * omega * (value * 1e-12))

            if topo_type == 'S':
                s = np.zeros((len(freqs_hz), 2, 2), dtype=complex)
                s[:,0,0] = Z / (2*Z0 + Z)
                s[:,1,1] = s[:,0,0]
                s[:,0,1] = (2*Z0) / (2*Z0 + Z)
                s[:,1,0] = s[:,0,1]
                ntwk = rf.Network(frequency=freqs_hz, s=s, z0=Z0)
            else:  # 'P'
                Y = 1.0 / Z
                Y0 = 1.0 / Z0
                s = np.zeros((len(freqs_hz), 2, 2), dtype=complex)
                s[:,0,0] = -Y / (2*Y0 + Y)
                s[:,1,1] = s[:,0,0]
                s[:,0,1] = (2*Y0) / (2*Y0 + Y)
                s[:,1,0] = s[:,0,1]
                ntwk = rf.Network(frequency=freqs_hz, s=s, z0=Z0)

            current = current ** ntwk
        return current

    def _build_matching_network(self, topology, networks, freqs_hz):
        s_thru = np.array([[[0.+0j, 1.+0j], [1.+0j, 0.+0j]]] * len(freqs_hz), dtype=complex)
        current = rf.Network(frequency=freqs_hz, s=s_thru, z0=50)

        for idx, (topo_type, ntwk) in enumerate(zip(topology, networks)):
            ntwk = ntwk.interpolate(current.frequency)
            try:
                if topo_type == 'S':
                    current = current ** ntwk
                elif topo_type == 'P':
                    short_interp = self.short_1port.interpolate(ntwk.frequency)
                    shunt_1port = ntwk ** short_interp
                    Y_shunt = shunt_1port.y[:, 0, 0]
                    Y_shunt_2port = np.zeros(current.y.shape, dtype=complex)
                    Y_shunt_2port[:, 0, 0] = Y_shunt
                    y_new = current.y + Y_shunt_2port
                    current = rf.Network(frequency=current.frequency, y=y_new, z0=50)
            except Exception as e:
                logger.error(f"构建失败 (位置 {idx}, {topo_type}): {e}")
                return None
        return current

    def evaluate_network(self, topology, comp_spec_or_names):
        if self.use_ideal:
            comp_specs = comp_spec_or_names
            if len(topology) != len(comp_specs):
                logger.error(f"拓扑长度({len(topology)})与元件数({len(comp_specs)})不符")
                return None
            key = (tuple(topology), tuple(comp_specs))
            if key in self._cache:
                return self._cache[key]
            match_net_opt = self._build_ideal_matching_network(topology, comp_specs, self.eval_freqs_hz)
            match_net_full = self._build_ideal_matching_network(topology, comp_specs, self.full_freqs_hz)
            components_display = [f"{t}{v:.2f}" for t, v in comp_specs]
        else:
            component_names = comp_spec_or_names
            if len(topology) != len(component_names):
                logger.error(f"拓扑长度({len(topology)})与元件数({len(component_names)})不符")
                return None
            key = (tuple(topology), tuple(component_names))
            if key in self._cache:
                return self._cache[key]
            try:
                networks = [self.lib.components[name]['network'] for name in component_names]
            except KeyError:
                return None
            match_net_opt = self._build_matching_network(topology, networks, self.eval_freqs_hz)
            match_net_full = self._build_matching_network(topology, networks, self.full_freqs_hz)
            components_display = component_names

        if match_net_opt is None or match_net_full is None:
            return None

        cascaded_opt = match_net_opt ** self.antenna_interp
        gamma_opt = cascaded_opt.s[:, 0, 0]
        eff = 1.0 - np.abs(gamma_opt)**2
        avg_eff = float(np.mean(eff))

        cascaded_full = match_net_full ** self.antenna_full
        gamma_full = cascaded_full.s[:, 0, 0]
        full_s11_db = 20 * np.log10(np.abs(gamma_full) + 1e-12)

        original_gamma = self.antenna_full.s[:, 0, 0]

        result = {
            'avg_efficiency': avg_eff,
            'efficiency_points': [float(e) for e in eff],
            's11_match_db': [float(v) for v in 20*np.log10(np.abs(gamma_opt) + 1e-12)],
            'frequencies_ghz': (self.eval_freqs_hz / 1e9).tolist(),
            'full_frequencies_ghz': (self.full_freqs_hz / 1e9).tolist(),
            'full_s11_match_db': [float(v) for v in full_s11_db],
            'topology': list(topology),
            'components': components_display,
            'smith_gamma_real': np.real(gamma_full).tolist(),
            'smith_gamma_imag': np.imag(gamma_full).tolist(),
            'original_gamma_real': np.real(original_gamma).tolist(),
            'original_gamma_imag': np.imag(original_gamma).tolist()
        }

        self._cache[key] = result
        return result

    def optimize(self, topologies, max_components, limit_per_type=50):
        if self.use_ideal:
            logger.info("使用理想LCR模式，启动遗传算法...")
            results = self._ga_ideal_optimize(topologies, max_components)
        else:
            l_names = [name for name in self.lib.get_component_names_by_type('L')
                       if (info := self.lib.components[name]).get('nominal') is None or
                       self.l_min <= info.get('nominal', 0) <= self.l_max][:limit_per_type]
            c_names = [name for name in self.lib.get_component_names_by_type('C')
                       if (info := self.lib.components[name]).get('nominal') is None or
                       self.c_min <= info.get('nominal', 0) <= self.c_max][:limit_per_type]
            all_names = l_names + c_names
            logger.info(f"优化启动 | 可用 L={len(l_names)}, C={len(c_names)}")

            if self.use_ga or len(all_names) > 60:
                logger.info("使用遗传算法模式...")
                results = self._ga_optimize(topologies, all_names, max_components)
            else:
                logger.info("使用穷举模式...")
                results = self._exhaustive_optimize(topologies, all_names, max_components)

        unique_results = self._remove_duplicates(results)
        unique_results.sort(key=lambda x: x['avg_efficiency'], reverse=True)
        logger.info(f"优化完成 | 唯一有效结果: {len(unique_results)} 个")
        return unique_results[:200]

    def _exhaustive_optimize(self, topologies, all_names, max_components):
        results = []
        total = 0
        seen = set()
        for topo in topologies:
            n = len(topo)
            if n > max_components: continue
            for indices in product(range(len(all_names)), repeat=n):
                comp_choice = [all_names[i] for i in indices]
                key = self._get_key(topo, comp_choice)
                if key in seen: continue
                seen.add(key)
                total += 1
                if total % 10000 == 0:
                    logger.info(f"已评估 {total:,} 个候选...")
                res = self.evaluate_network(topo, comp_choice)
                if res and res['avg_efficiency'] > 0.12:
                    results.append(res)
        return results

    def _ga_optimize(self, topologies, all_names, max_components):
        population_size = self.ga_pop
        generations = self.ga_gen
        results = []
        seen = set()

        def create_individual():
            n = random.randint(1, max_components)
            valid_topos = [t for t in topologies if len(t) == n]
            if not valid_topos:
                topo = random.choice(topologies)
                n = len(topo)
            else:
                topo = random.choice(valid_topos)
            comps = [random.choice(all_names) for _ in range(n)]
            return (topo, comps)

        def fitness(ind):
            topo, comps = ind
            res = self.evaluate_network(topo, comps)
            return res['avg_efficiency'] if res else 0.0

        population = [create_individual() for _ in range(population_size)]

        for gen in range(generations):
            scored = [(ind, fitness(ind)) for ind in population]
            scored.sort(key=lambda x: x[1], reverse=True)

            for ind, score in scored[:12]:
                key = self._get_key(ind[0], ind[1])
                if key not in seen and score > 0.12:
                    res = self.evaluate_network(ind[0], ind[1])
                    if res:
                        results.append(res)
                        seen.add(key)

            elite = [ind for ind, _ in scored[:population_size//3]]
            new_pop = elite[:]
            while len(new_pop) < population_size:
                p1, p2 = random.sample(elite, 2)
                child_topo = p1[0] if random.random() < 0.75 else p2[0]
                child_comps = p1[1][:]
                if random.random() < 0.6 and child_comps:
                    idx = random.randrange(len(child_comps))
                    child_comps[idx] = random.choice(all_names)
                new_pop.append((child_topo, child_comps))
            population = new_pop[:population_size]

            if gen % 8 == 0:
                logger.info(f"GA 第 {gen} 代 | 最佳效率: {scored[0][1]*100:.2f}% | 已发现唯一结果: {len(seen)}")
        logger.info(f"GA 完成 | 发现唯一结果: {len(seen)} 个")
        return results

    def _ga_ideal_optimize(self, topologies, max_components):
        population_size = self.ga_pop
        generations = self.ga_gen
        results = []
        seen = set()

        def create_individual():
            n = random.randint(1, max_components)
            valid_topos = [t for t in topologies if len(t) == n]
            if not valid_topos:
                topo = random.choice(topologies)
                n = len(topo)
            else:
                topo = random.choice(valid_topos)
            specs = []
            for _ in range(n):
                if random.random() < 0.5:
                    typ = 'L'
                    raw_val = random.uniform(self.l_min, self.l_max)
                    val = round(raw_val / self.ideal_l_step) * self.ideal_l_step
                    val = max(self.l_min, min(self.l_max, val))
                else:
                    typ = 'C'
                    raw_val = random.uniform(self.c_min, self.c_max)
                    val = round(raw_val / self.ideal_c_step) * self.ideal_c_step
                    val = max(self.c_min, min(self.c_max, val))
                specs.append((typ, val))
            return (topo, specs)

        def fitness(ind):
            topo, specs = ind
            res = self.evaluate_network(topo, specs)
            return res['avg_efficiency'] if res else 0.0

        population = [create_individual() for _ in range(population_size)]

        for gen in range(generations):
            scored = [(ind, fitness(ind)) for ind in population]
            scored.sort(key=lambda x: x[1], reverse=True)

            for ind, score in scored[:12]:
                key = (tuple(ind[0]), tuple(ind[1]))
                if key not in seen and score > 0.12:
                    res = self.evaluate_network(ind[0], ind[1])
                    if res:
                        results.append(res)
                        seen.add(key)

            elite = [ind for ind, _ in scored[:population_size//3]]
            new_pop = elite[:]
            while len(new_pop) < population_size:
                p1, p2 = random.sample(elite, 2)
                # 交叉拓扑：选择长度一致的父本或调整
                if random.random() < 0.75:
                    child_topo = p1[0]
                    child_specs = p1[1][:]
                else:
                    child_topo = p2[0]
                    child_specs = p2[1][:]
                # 确保拓扑长度与规格长度一致
                if len(child_topo) != len(child_specs):
                    # 如果不一致，从父本2拷贝一份规格（但拓扑是p1的，需重新生成）
                    child_topo = p1[0]
                    child_specs = p1[1][:]
                # 变异
                if random.random() < 0.6 and child_specs:
                    idx = random.randrange(len(child_specs))
                    if random.random() < 0.5:
                        typ = 'L'
                        raw_val = random.uniform(self.l_min, self.l_max)
                        val = round(raw_val / self.ideal_l_step) * self.ideal_l_step
                        val = max(self.l_min, min(self.l_max, val))
                    else:
                        typ = 'C'
                        raw_val = random.uniform(self.c_min, self.c_max)
                        val = round(raw_val / self.ideal_c_step) * self.ideal_c_step
                        val = max(self.c_min, min(self.c_max, val))
                    child_specs[idx] = (typ, val)
                new_pop.append((child_topo, child_specs))
            population = new_pop[:population_size]

            if gen % 8 == 0:
                logger.info(f"GA 第 {gen} 代 | 最佳效率: {scored[0][1]*100:.2f}%")
        logger.info(f"GA 完成 | 发现唯一结果: {len(seen)} 个")
        return results

    def _remove_duplicates(self, results):
        seen = {}
        for res in results:
            if self.use_ideal:
                key = (tuple(res['topology']), tuple(res['components']))
            else:
                key = (tuple(res['topology']), tuple(res['components']))
            if key not in seen or res['avg_efficiency'] > seen[key]['avg_efficiency']:
                seen[key] = res
        return list(seen.values())


# ==================== Flask 路由 ====================
lib = ComponentLibrary()
antenna_net = None

@app.route('/')
def index():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>index.html 文件未找到，请放在同一目录</h1>"

@app.route('/api/load_library', methods=['POST'])
def load_library():
    data = request.get_json()
    count = lib.load_from_folders(data.get('l_folder', ''), data.get('c_folder', ''))
    return jsonify({'count': count})

@app.route('/api/load_antenna', methods=['POST'])
def load_antenna():
    global antenna_net
    file = request.files['file']
    filename = werkzeug.utils.secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    antenna_net = rf.Network(filepath)
    freqs = antenna_net.frequency.f / 1e9
    s11 = 20 * np.log10(np.abs(antenna_net.s[:, 0, 0]) + 1e-12)
    return jsonify({'frequencies': freqs.tolist(), 's11_db': s11.tolist()})

@app.route('/api/optimize', methods=['POST'])
def optimize():
    global antenna_net, lib
    if not antenna_net:
        return jsonify({'error': '请先加载天线文件'}), 400

    data = request.get_json()
    bands = data.get('bands', [[2400, 2500]])
    num_points = int(data.get('num_points', 7))
    min_elem = int(data.get('min_elements', 1))
    max_elem = int(data.get('max_elements', 3))
    limit_per_type = int(data.get('limit_per_type', 40))
    l_min = float(data.get('l_min_nh', 0.1))
    l_max = float(data.get('l_max_nh', 20.0))
    c_min = float(data.get('c_min_pf', 0.1))
    c_max = float(data.get('c_max_pf', 20.0))
    use_ga = data.get('use_ga', True)
    ga_pop = int(data.get('ga_population', 100))
    ga_gen = int(data.get('ga_generations', 80))

    use_ideal = data.get('use_ideal', False)
    ideal_l_step = float(data.get('ideal_l_step', 0.05))
    ideal_c_step = float(data.get('ideal_c_step', 0.05))

    if not use_ideal and not lib.components:
        return jsonify({'error': '请先加载元件库，或勾选"使用理想LCR元件"'}), 400

    topologies = TopologyGenerator.get_topologies(min_elem, max_elem)

    optimizer = MatchingOptimizer(lib, antenna_net, bands, num_points,
                                  l_min, l_max, c_min, c_max,
                                  use_ga, ga_pop, ga_gen,
                                  use_ideal, ideal_l_step, ideal_c_step)

    results = optimizer.optimize(topologies, max_components=max_elem, limit_per_type=limit_per_type)
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)