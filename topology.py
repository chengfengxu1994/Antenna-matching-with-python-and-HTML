# engine/topology.py — 独立拓扑引擎
# ============================================
# 完全独立的 RF 匹配网络拓扑生成器。
# 不依赖 Flask/前端，仅依赖 numpy + rf_utils。
#
# 输入: SNP 文件 + port配置 + 元件数量范围
# 输出: 拓扑定义、拓扑图片(SVG)、文本图、日志
#
# 独立运行:
#   python engine/topology.py --snp antenna.s2p --config port_config.json
#
# 作为库调用:
#   from engine.topology import TopologyEngine
#   engine = TopologyEngine()
#   engine.load_snp("antenna.s2p")
#   engine.load_port_configs([{...}, {...}])
#   result = engine.generate()
#   print(result.summary)
#   result.save_svg("output/")
#
# 拓扑结构定义:
#   1port:  load-1el(S/P)-2el(S/P)-...-Nel(S/P)-port1-|SNP|
#
#   2port:  (load/shunt/open)-1el(S/P)-...-Nel(S/P)-port1
#           -|SNP|-port2-1el(S/P)-...-Nel(S/P)-(load/shunt/open)
#
#   3port:  (load/shunt/open)-1el(S/P)-...-Nel(S/P)-port1
#           -|SNP|-port2-1el(S/P)-...-Nel(S/P)-(load/shunt/open)
#                  |
#           port3-1el(S/P)-...-Nel(S/P)-(load/shunt/open)
#
#   Nport:  (load/shunt/open)-1el(S/P)-...-Nel(S/P)-port1
#           -|SNP|-port2-1el(S/P)-...-Nel(S/P)-(load/shunt/open)
#                  |-port3-1el(S/P)-...-Nel(S/P)-(load/shunt/open)
#                  |-portN-1el(S/P)-...-Nel(S/P)-(load/shunt/open)

import os
import re
import sys
import json
import math
import logging
import textwrap
from dataclasses import dataclass, field, asdict
from itertools import product as iproduct
from typing import Optional, Union

import numpy as np

# ── 日志 ──────────────────────────────────────────────────────────
logger = logging.getLogger('topology')
_log_initialized = False


def _setup_logger(level=logging.INFO):
    """Init logger: console (encoding-safe) + file."""
    global _log_initialized
    if _log_initialized:
        return
    _log_initialized = True

    logger.setLevel(level)

    # Console handler with GBK-safe encoding on Windows
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(
        '[%(levelname)s] %(message)s'))

    # Windows GBK console: encode/decode to prevent UnicodeEncodeError
    import codecs
    if sys.platform == 'win32' and sys.stdout.encoding and \
       sys.stdout.encoding.upper() not in ('UTF-8', 'UTF8'):
        try:
            ch.setStream(codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace'))
        except Exception:
            pass
    logger.addHandler(ch)

    # File handler
    try:
        fh = logging.FileHandler('topology_engine.log', mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s'))
        logger.addHandler(fh)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# 1. 数据类 — 拓扑定义
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PortConfig:
    """单个端口的配置.

    Attributes:
        port: 0-based port索引
        port_type: 'load'(匹配到50Ω) | 'ground'(短/开/负载地) | 'short'(直短) | 'open'(直开)
        elem_min: 最小匹配元件数 (1-6)
        elem_max: 最大匹配元件数 (1-6)
        l_min_nh: 电感最小值 (nH)
        l_max_nh: 电感最大值 (nH)
        c_min_pf: 电容最小值 (pF)
        c_max_pf: 电容最大值 (pF)
        termination: 地端口终结类型: 'short'(默认) | 'open' | 'load'
        bands_mhz: [[start, end], ...] 频段列表
    """
    port: int
    port_type: str = 'load'
    elem_min: int = 2
    elem_max: int = 4
    l_min_nh: float = 0.1
    l_max_nh: float = 60.0
    c_min_pf: float = 0.1
    c_max_pf: float = 30.0
    termination: Optional[str] = None
    bands_mhz: list = field(default_factory=list)

    def __post_init__(self):
        # 规范化
        self.elem_min = max(1, min(6, self.elem_min))
        self.elem_max = max(self.elem_min, min(6, self.elem_max))
        if self.port_type in ('short', 'open', 'ground'):
            self.termination = self.termination or (
                'short' if self.port_type != 'open' else 'open')
        if self.port_type == 'load' or self.port_type not in ('short', 'open'):
            self.termination = self.termination or 'load'

    @classmethod
    def from_dict(cls, d: dict) -> 'PortConfig':
        """从字典解析 (兼容前端 JSON)."""
        return cls(
            port=int(d.get('port', 0)),
            port_type=d.get('type', d.get('port_type', 'load')),
            elem_min=int(d.get('elem_min', d.get('elem_count', d.get('element_count', 2)))),
            elem_max=int(d.get('elem_max', d.get('elem_count', d.get('element_count', 4)))),
            l_min_nh=float(d.get('l_min', d.get('l_min_nh', 0.1))),
            l_max_nh=float(d.get('l_max', d.get('l_max_nh', 60.0))),
            c_min_pf=float(d.get('c_min', d.get('c_min_pf', 0.1))),
            c_max_pf=float(d.get('c_max', d.get('c_max_pf', 30.0))),
            termination=d.get('termination'),
            bands_mhz=d.get('bands', d.get('bands_mhz', [])),
        )

    def to_dict(self) -> dict:
        return {
            'port': self.port,
            'type': self.port_type,
            'elem_min': self.elem_min,
            'elem_max': self.elem_max,
            'l_min_nh': self.l_min_nh,
            'l_max_nh': self.l_max_nh,
            'c_min_pf': self.c_min_pf,
            'c_max_pf': self.c_max_pf,
            'termination': self.termination,
            'bands_mhz': self.bands_mhz,
        }

    def __repr__(self):
        term = f", term={self.termination}" if self.termination else ""
        return (f"PortConfig(P{self.port+1} {self.port_type} "
                f"el=[{self.elem_min}-{self.elem_max}]{term})")


@dataclass
class ElementNode:
    """单个匹配元件节点.

    Attributes:
        position: 0-based 在分支中的位置
        topology_type: 'S'(串联) | 'P'(并联/分路)
        component_type: 'L'(电感) | 'C'(电容) | None(未赋值)
    """
    position: int
    topology_type: str
    component_type: Optional[str] = None

    def __repr__(self):
        ct = self.component_type or '?'
        return f"Elem[{self.position}] {ct}({self.topology_type})"


@dataclass
class PortBranch:
    """一个天线端口的完整匹配分支.

    描述从端口到终结的匹配元件链.

    Attributes:
        port: 天线Port index (0-based)
        port_type: 'load' | 'ground' | 'short' | 'open'
        element_pattern: tuple of 'S'/'P' (纯模式，无元件类型)
        element_count: 元件数量
        termination: 'short' | 'open' | 'load'
        comp_types: Optional tuple of 'L'/'C' (元件类型赋值)
    """
    port: int
    port_type: str = 'load'
    element_pattern: tuple = ()
    element_count: int = 0
    termination: str = 'load'
    comp_types: Optional[tuple] = None

    def __post_init__(self):
        if not self.element_pattern and self.element_count > 0:
            logger.warning(f"PortBranch P{self.port+1}: element_pattern empty but element_count={self.element_count}")
        if self.element_pattern:
            self.element_count = len(self.element_pattern)

    @property
    def pattern_str(self) -> str:
        """返回 'SPS' 格式的拓扑模式字符串."""
        return ''.join(self.element_pattern)

    @pattern_str.setter
    def pattern_str(self, s: str):
        self.element_pattern = tuple(s)
        self.element_count = len(s)

    @property
    def type_str(self) -> str:
        """返回 'LCL' 格式的元件类型字符串."""
        if self.comp_types:
            return ''.join(self.comp_types)
        return ''

    @property
    def is_terminated(self) -> bool:
        """是否需要终结 (load端口需要匹配到50Ω，ground端口需要短/开/负载)."""
        return self.port_type in ('load', 'ground', 'short', 'open')

    def get_termination_label(self) -> str:
        """返回终结类型的显示标签."""
        labels = {
            'load': '50Ω',
            'short': 'GND',
            'open': 'OPEN',
            'ground': 'GND',
        }
        return labels.get(self.termination, self.termination.upper())

    def get_termination_color(self) -> str:
        """返回终结描点的颜色."""
        colors = {
            'load': '#0d904f',
            'short': '#d93025',
            'open': '#f9ab00',
        }
        return colors.get(self.termination, '#666')

    def __repr__(self):
        return (f"PortBranch(P{self.port+1} {self.port_type} "
                f"{self.pattern_str}/{self.type_str} → {self.termination})")


@dataclass
class TopologyVariant:
    """一个完整的拓扑变体 (拓扑模式 + 元件类型赋值).

    代表一种具体的匹配网络结构: 所有端口的 S/P 组合 + L/C 分配.

    Attributes:
        nports: 天线端口数
        branches: list of PortBranch (每个端口一个分支)
        variant_id: 唯一标识符
    """
    nports: int
    branches: list  # list of PortBranch
    variant_id: str = ''

    def __post_init__(self):
        if not self.variant_id:
            self._make_id()

    def _make_id(self):
        """生成变体ID: 例如 'P1_SP-S_LC|P2_SPS-LCL'."""
        parts = []
        for b in self.branches:
            p_str = b.pattern_str
            t_str = b.type_str or '?' * b.element_count
            parts.append(f"P{b.port+1}_{p_str}-{t_str}")
        self.variant_id = '|'.join(parts)

    @property
    def total_elements(self) -> int:
        return sum(b.element_count for b in self.branches)

    @property
    def branch_count(self) -> int:
        return len(self.branches)

    def get_branch(self, port: int) -> Optional[PortBranch]:
        for b in self.branches:
            if b.port == port:
                return b
        return None

    def __repr__(self):
        return f"TopologyVariant({self.variant_id} {self.total_elements}el)"


# ═══════════════════════════════════════════════════════════════════
# 2. 模式生成 — 核心拓扑生成算法
# ═══════════════════════════════════════════════════════════════════

def generate_element_patterns(elem_min: int, elem_max: Optional[int] = None) -> list:
    """生成所有有效的 S/P 拓扑模式.

    生成 [elem_min, elem_max] 范围内所有有效的 S(串联)/P(并联) 排列组合.
    排除全相同模式的无效组合 (如 SSSS, PPPP) — 因为 N>=2 时这些是冗余的.

    Args:
        elem_min: 最小元件数
        elem_max: 最大元件数 (默认 = elem_min)

    Returns:
        list of tuple: 例如 [('S','P'), ('P','S'), ('S','P','S'), ...]
    """
    if elem_max is None:
        elem_max = elem_min
    topos = []
    for n in range(elem_min, elem_max + 1):
        if n < 1:
            topos.append(())
            continue
        for combo in iproduct(['S', 'P'], repeat=n):
            topos.append(combo)
    return topos


def generate_type_variants(pattern: tuple, filter_invalid: bool = True) -> list:
    """为拓扑模式生成所有有效的 L/C 元件类型赋值.

    对每个位置分配 L(电感) 或 C(电容), 生成 2^N 种可能.
    过滤: 并联位置 (P) 不能用电感 (L), 否则构成直流短路.

    Args:
        pattern: ('S','P','S') 格式的拓扑模式
        filter_invalid: 是否过滤无效组合 (默认是, 过滤 P+L)

    Returns:
        list of tuple: 例如 [('L','C','L'), ('C','C','L'), ...]
    """
    variants = [tuple(combo) for combo in iproduct(['L', 'C'], repeat=len(pattern))]
    if filter_invalid:
        variants = [v for v in variants
                    if not any(tt == 'P' and ct == 'L' for tt, ct in zip(pattern, v))]
    return variants


def default_component_types(pattern: tuple) -> tuple:
    """返回默认元件类型: 串联→L, 并联→C."""
    return tuple('L' if tt == 'S' else 'C' for tt in pattern)


# ═══════════════════════════════════════════════════════════════════
# 3. 拓扑引擎 — 主控类
# ═══════════════════════════════════════════════════════════════════

class TopologyEngine:
    """拓扑引擎主类.

    加载 SNP 文件 + port配置 → 生成所有拓扑变体 → 渲染图形 + 日志.

    用法:
        engine = TopologyEngine()
        engine.load_snp("antenna.s2p")
        engine.load_port_configs([{...}, {...}])
        result = engine.generate()
    """

    def __init__(self):
        self._snp_path = None
        self._snp_network = None       # rf_utils Network 对象
        self._snp_nports = 0
        self._snp_freqs_hz = None
        self._port_configs: list[PortConfig] = []
        self._bands_mhz: list = []
        self._result = None

    # ── SNP 文件加载 ─────────────────────────────────────────

    def load_snp(self, path: str) -> dict:
        """加载 Touchstone SNP 文件.

        Args:
            path: .s1p / .s2p / .sNp 文件路径

        Returns:
            dict: {nports, frequencies, filename, ...}
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"SNP file not found: {path}")

        self._snp_path = path

        try:
            from rf_utils.network import Network
            net = Network(path)
        except Exception as e:
            # 尝试使用 snp_utils
            try:
                import snp_utils as rf
                net = rf.Network(path)
            except Exception as e2:
                raise RuntimeError(
                    f"Cannot load SNP file: {e}\nFallback load also failed: {e2}")

        self._snp_network = net
        self._snp_nports = net.nports
        self._snp_freqs_hz = net.frequency.f
        nf = len(self._snp_freqs_hz)

        info = {
            'file': os.path.basename(path),
            'path': path,
            'nports': self._snp_nports,
            'freqs': nf,
            'freq_min_ghz': round(self._snp_freqs_hz[0] / 1e9, 6),
            'freq_max_ghz': round(self._snp_freqs_hz[-1] / 1e9, 6),
        }

        logger.info(f"Loaded SNP: {info['file']} | "
                     f"{info['nports']}-port | "
                     f"{nf} freqs | "
                     f"[{info['freq_min_ghz']:.4f}-{info['freq_max_ghz']:.4f}] GHz")

        return info

    # ── port配置加载 ─────────────────────────────────────────

    def load_port_configs(self, configs: list, bands_mhz: Optional[list] = None):
        """加载并验证端口配置.

        Args:
            configs: list of dict, 每项为端口配置
            bands_mhz: 全局频段 [[start_mhz, end_mhz], ...]
        """
        parsed = [PortConfig.from_dict(c) for c in configs]
        self._port_configs = parsed

        if bands_mhz:
            self._bands_mhz = bands_mhz
        else:
            # 从各端口收集频段
            for pc in parsed:
                if pc.bands_mhz:
                    self._bands_mhz.extend(pc.bands_mhz)

        # 验证端口索引
        if self._snp_nports > 0:
            for pc in parsed:
                if pc.port >= self._snp_nports:
                    raise ValueError(
                        f"Port index {pc.port} exceeds SNP port count {self._snp_nports}")

        logger.info("Port configs (%d ports):", len(parsed))
        for pc in parsed:
            logger.info(f"  {pc}")

        if self._bands_mhz:
            bands_str = ', '.join(f"[{s:.0f}-{e:.0f}]MHz" for s, e in self._bands_mhz)
            logger.info(f"Bands: {bands_str}")

    # ── 核心生成 ─────────────────────────────────────────────

    def generate(self) -> 'TopologyResult':
        """执行拓扑生成.

        Returns:
            TopologyResult 包含所有生成的拓扑信息.
        """
        t0 = __import__('time').time()

        nports = self._snp_nports or self._get_nports_from_configs()
        configs = self._port_configs

        # 如果未配置端口，自动生成默认配置
        if not configs:
            configs = self._auto_configure_ports(nports)
            self._port_configs = configs

        # 生成每个端口的拓扑模式
        per_port_patterns = []
        per_port_configs = []

        # 按端口类型分类: load端口和ground端口
        load_configs = [pc for pc in configs if pc.port_type == 'load']
        ground_configs = [pc for pc in configs if pc.port_type != 'load']

        all_branches_config = []  # 所有要生成拓扑的分支

        if nports == 1:
            # 1-port: 只用 load port配置
            for pc in load_configs:
                patterns = generate_element_patterns(pc.elem_min, pc.elem_max)
                per_port_patterns.append(patterns)
                all_branches_config.append(pc)
            if not all_branches_config:
                pc = PortConfig(port=0)
                patterns = generate_element_patterns(pc.elem_min, pc.elem_max)
                per_port_patterns.append(patterns)
                all_branches_config.append(pc)
        else:
            # N-port: 按端口类型分组，先load后ground
            ordered_configs = load_configs + ground_configs
            if not ordered_configs:
                # 自动配置: port0=load, 其余=ground short
                for p in range(nports):
                    if p == 0:
                        pc = PortConfig(port=p, port_type='load')
                    else:
                        pc = PortConfig(port=p, port_type='ground',
                                        termination='short', elem_min=1, elem_max=1)
                    ordered_configs.append(pc)
                self._port_configs = ordered_configs

            for pc in ordered_configs:
                patterns = generate_element_patterns(pc.elem_min, pc.elem_max)
                per_port_patterns.append(patterns)
                all_branches_config.append(pc)

        # 构建 PortBranch 列表 (纯模式，无类型)
        branch_templates = []
        for pc in all_branches_config:
            branch_templates.append(PortBranch(
                port=pc.port,
                port_type=pc.port_type,
                element_pattern=(),  # 暂空
                element_count=0,
                termination=self._resolve_termination(pc),
            ))

        # ── 生成所有组合 ──
        # 如果某端口无模式 (elem=0), 设为 [()]
        pattern_lists = [pp if pp else [()] for pp in per_port_patterns]
        total_combos = 0
        total_variants = 0

        all_variants = []
        all_patterns_seen = set()

        for combo in iproduct(*pattern_lists):
            combo_str = '|'.join(''.join(c) for c in combo)
            if combo_str in all_patterns_seen:
                continue
            all_patterns_seen.add(combo_str)

            total_combos += 1

            # 对该组合生成 type variants (Cartesian product of per-port type variants)
            per_port_type_variants = []
            for i, pattern in enumerate(combo):
                tv = generate_type_variants(pattern)
                per_port_type_variants.append(tv)

            # Cartesian product 跨端口
            for type_combo in iproduct(*per_port_type_variants):
                total_variants += 1

                branches = []
                for i, (pc, pattern, types) in enumerate(
                        zip(all_branches_config, combo, type_combo)):
                    branch = PortBranch(
                        port=pc.port,
                        port_type=pc.port_type,
                        element_pattern=pattern,
                        element_count=len(pattern),
                        termination=self._resolve_termination(pc),
                        comp_types=types,
                    )
                    branches.append(branch)

                variant = TopologyVariant(
                    nports=nports,
                    branches=branches,
                )
                all_variants.append(variant)

        elapsed = __import__('time').time() - t0

        # ── 统计 ──
        per_port_stats = []
        for pc, pl in zip(all_branches_config, pattern_lists):
            per_port_stats.append({
                'port': pc.port,
                'type': pc.port_type,
                'elem_range': [pc.elem_min, pc.elem_max],
                'patterns': len(pl),
            })

        # 构建结果
        self._result = TopologyResult(
            nports=nports,
            snp_info=self._get_snp_info(),
            port_configs=[pc.to_dict() for pc in self._port_configs],
            bands_mhz=self._bands_mhz,
            per_port_stats=per_port_stats,
            topology_patterns=list(all_patterns_seen),
            total_patterns=total_combos,
            total_variants=total_variants,
            variants=all_variants,
            generation_time_sec=round(elapsed, 4),
        )

        logger.info(f"=== Topology Generation Complete ===")
        logger.info(f"  Ports: {nports}")
        logger.info(f"  Port branches: {len(all_branches_config)}")
        for pps in per_port_stats:
            logger.info(f"    P{pps['port']+1} ({pps['type']}): "
                         f"el=[{pps['elem_range'][0]}-{pps['elem_range'][1]}] "
                         f"{pps['patterns']} patterns")
        logger.info(f"  Total pattern combos: {total_combos}")
        logger.info(f"  Total variants (w/ LC types): {total_variants}")
        logger.info(f"  Gen time: {elapsed:.4f}s")

        return self._result

    def _resolve_termination(self, pc: PortConfig) -> str:
        """根据端口配置解析终结类型."""
        if pc.port_type == 'load':
            return 'load'
        if pc.termination:
            return pc.termination
        if pc.port_type == 'short':
            return 'short'
        if pc.port_type == 'open':
            return 'open'
        return 'short'

    def _get_nports_from_configs(self) -> int:
        """从端口配置推断端口数."""
        if not self._port_configs:
            return 1
        max_port = max(pc.port for pc in self._port_configs)
        return max(max_port + 1, 1)

    def _auto_configure_ports(self, nports: int) -> list:
        """未配置端口时自动生成默认配置."""
        configs = []
        if nports == 1:
            configs.append(PortConfig(port=0, port_type='load'))
        else:
            configs.append(PortConfig(port=0, port_type='load'))
            for p in range(1, nports):
                configs.append(PortConfig(
                    port=p, port_type='ground',
                    termination='short', elem_min=1, elem_max=1))
        logger.info(f"自动生成 {len(configs)} port configs ({nports}-port)")
        return configs

    def _get_snp_info(self) -> dict:
        if self._snp_network is None:
            return {'file': '', 'nports': 0, 'freqs': 0}
        return {
            'file': os.path.basename(self._snp_path or ''),
            'path': self._snp_path or '',
            'nports': self._snp_nports,
            'freqs': len(self._snp_freqs_hz) if self._snp_freqs_hz is not None else 0,
            'freq_min_ghz': round(self._snp_freqs_hz[0] / 1e9, 6)
                            if self._snp_freqs_hz is not None and len(self._snp_freqs_hz) > 0 else 0,
            'freq_max_ghz': round(self._snp_freqs_hz[-1] / 1e9, 6)
                            if self._snp_freqs_hz is not None and len(self._snp_freqs_hz) > 0 else 0,
        }

    def get_result(self) -> Optional['TopologyResult']:
        return self._result


# ═══════════════════════════════════════════════════════════════════
# 4. 结果类 — 包含所有生成结果和渲染方法
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TopologyResult:
    """拓扑生成结果.

    包含所有生成的拓扑变体、统计信息、以及渲染方法.
    """
    nports: int = 0
    snp_info: dict = field(default_factory=dict)
    port_configs: list = field(default_factory=list)
    bands_mhz: list = field(default_factory=list)
    per_port_stats: list = field(default_factory=list)
    topology_patterns: list = field(default_factory=list)
    total_patterns: int = 0
    total_variants: int = 0
    variants: list = field(default_factory=list)  # list of TopologyVariant
    generation_time_sec: float = 0.0

    # 渲染缓存
    _svg_cache: dict = field(default_factory=dict, repr=False)
    _text_cache: list = field(default_factory=list, repr=False)

    @property
    def summary(self) -> str:
        """ASCII-safe summary text."""
        lines = [
            "=" * 70,
            "RF Topology Engine -- Generation Report",
            "=" * 70,
            f"  SNP: {self.snp_info.get('file', 'N/A')}  "
            f"({self.snp_info.get('nports', '?')}-port, "
            f"{self.snp_info.get('freqs', '?')} freqs)",
            f"  Ports: {self.nports}",
            f"  Bands: {self._bands_str()}",
            "-" * 70,
            "Port branches:",
        ]
        for ps in self.per_port_stats:
            lines.append(
                f"  P{ps['port']+1:2d} ({ps['type']:6s}): "
                f"el=[{ps['elem_range'][0]}-{ps['elem_range'][1]}]  "
                f"{ps['patterns']} S/P patterns"
            )
        lines += [
            "-" * 70,
            f"  Total pattern combos: {self.total_patterns}",
            f"  Total variants (w/ LC types): {self.total_variants}",
            f"  Gen time: {self.generation_time_sec*1000:.1f}ms",
            "-" * 70,
        ]
        return '\n'.join(lines)

    def _bands_str(self) -> str:
        if not self.bands_mhz:
            return '未指定'
        return ', '.join(f"[{s:.0f}-{e:.0f}]MHz" for s, e in self.bands_mhz)

    def render_all(self, render_svg: bool = True, render_text: bool = True):
        """渲染所有拓扑变体.

        Args:
            render_svg: 是否生成 SVG 图形
            render_text: 是否生成 ASCII 文本图
        """
        self._svg_cache.clear()
        self._text_cache.clear()

        renderer = TopologyRenderer()

        for variant in self.variants:
            vid = variant.variant_id
            if render_svg:
                self._svg_cache[vid] = renderer.render_svg(variant, self.nports)
            if render_text:
                self._text_cache.append(renderer.render_text(variant, self.nports))

        logger.info(f"已渲染 {len(self.variants)} 个变体: "
                     f"{'SVG' if render_svg else ''} "
                     f"{'Text' if render_text else ''}")

    def get_svg(self, variant_id: str) -> Optional[str]:
        """获取指定变体的 SVG."""
        return self._svg_cache.get(variant_id)

    def get_text(self, index: int) -> Optional[str]:
        """获取指定索引的文本图."""
        if 0 <= index < len(self._text_cache):
            return self._text_cache[index]
        return None

    def save_svg(self, output_dir: str = 'topology_output', prefix: str = 'topo'):
        """将所有 SVG 保存到文件.

        Args:
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        os.makedirs(output_dir, exist_ok=True)
        saved = 0
        for vid, svg in self._svg_cache.items():
            # 文件名安全化
            safe_name = re.sub(r'[<>:"/\\|?*]', '_', vid)[:100]
            fpath = os.path.join(output_dir, f"{prefix}_{safe_name}.svg")
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(svg)
            saved += 1
        logger.info(f"已保存 {saved} 个 SVG 文件到 {output_dir}/")

    def print_text(self, max_variants: int = 20):
        """打印文本拓扑图到控制台.

        Args:
            max_variants: 最多打印数量 (0=全部)
        """
        n_show = len(self._text_cache) if max_variants == 0 else min(max_variants, len(self._text_cache))
        logger.info(f"打印前 {n_show}/{len(self._text_cache)} 个拓扑文本图:")
        print()
        for i in range(n_show):
            print(self._text_cache[i])
            print()

    def to_dict(self) -> dict:
        """序列化为字典 (用于 JSON 输出)."""
        return {
            'nports': self.nports,
            'snp_info': self.snp_info,
            'port_configs': self.port_configs,
            'bands_mhz': self.bands_mhz,
            'per_port_stats': self.per_port_stats,
            'total_patterns': self.total_patterns,
            'total_variants': self.total_variants,
            'variants_count': len(self.variants),
            'generation_time_sec': self.generation_time_sec,
            'variants': [v.variant_id for v in self.variants],
        }


# ═══════════════════════════════════════════════════════════════════
# 5. SVG 渲染器 — 生成拓扑图片
# ═══════════════════════════════════════════════════════════════════

class TopologyRenderer:
    """拓扑 SVG/文本渲染器.

    根据端口数量选择不同布局:
      - 1-port: 水平单线, 左终结→元件→SNP
      - 2-port: 水平, 左终结→元件→SNP→元件→右终结
      - 3-port: 左终结→元件→SNP→元件→右终结 + 向下分支
      - N-port: 同上, 右侧垂直堆叠多个端口分支
    """

    # ── 布局常量 (所有单位: px) ──
    ELEM_W = 36       # 元件框宽度
    ELEM_H = 24       # 元件框高度
    ELEM_GAP = 20     # 元件间距
    PORT_R = 14       # port/终结圆半径
    SNP_W = 80        # SNP 框宽度
    SNP_H = 50        # SNP 框高度
    MARGIN_X = 30     # 左右边距
    MARGIN_Y = 30     # 上下边距
    BRANCH_GAP_X = 20 # 分支间距
    BRANCH_GAP_Y = 60 # 垂直方向分支间距
    LINE_Y_OFFSET = 40  # 主传输线 Y 偏移 (从顶部)

    # ── 颜色 ──
    COLOR_SERIES = '#27ae60'
    COLOR_SHUNT = '#f39c12'
    COLOR_SERIES_BORDER = '#1e8449'
    COLOR_SHUNT_BORDER = '#d68910'
    COLOR_PORT_LOAD = '#0d904f'
    COLOR_PORT_GND = '#d93025'
    COLOR_PORT_OPEN = '#f9ab00'
    COLOR_PORT = '#1a73e8'
    COLOR_SNP_BG = '#e8f0fe'
    COLOR_SNP_BORDER = '#1a73e8'
    COLOR_LINE = '#333'
    COLOR_LINE_SHUNT = '#d68910'
    COLOR_BG = '#f8f9fa'
    COLOR_BG_BORDER = '#dee2e6'
    COLOR_TEXT = '#495057'
    COLOR_GND_SYMBOL = '#333'

    ELEM_TEXT_COLOR = 'white'
    PORT_TEXT_COLOR = 'white'

    def render_svg(self, variant: TopologyVariant, nports: int) -> str:
        """渲染一个拓扑变体的 SVG 图形.

        Args:
            variant: 拓扑变体
            nports: port数

        Returns:
            str: SVG 标记文本
        """
        if nports == 1:
            return self._render_1port(variant)
        elif nports == 2:
            return self._render_2port(variant)
        elif nports == 3:
            return self._render_3port(variant)
        else:
            return self._render_nport(variant, nports)

    def render_text(self, variant: TopologyVariant, nports: int) -> str:
        """渲染一个拓扑变体的 ASCII 文本图.

        Args:
            variant: 拓扑变体
            nports: port数

        Returns:
            str: 文本图
        """
        if nports == 1:
            return self._render_text_1port(variant)
        elif nports == 2:
            return self._render_text_2port(variant)
        elif nports == 3:
            return self._render_text_3port(variant)
        else:
            return self._render_text_nport(variant, nports)

    # ══════════════════════════════════════════════════════════
    # 1-Port SVG
    # ══════════════════════════════════════════════════════════

    def _render_1port(self, variant: TopologyVariant) -> str:
        branch = variant.branches[0]
        n_el = branch.element_count

        # 计算尺寸
        total_w = (self.MARGIN_X * 2 + self.PORT_R * 2 +
                   n_el * (self.ELEM_W + self.ELEM_GAP) +
                   self.SNP_W + 40)
        total_h = 120
        w = max(total_w, 280)
        h = total_h
        cy = h // 2  # 中心 Y
        x = self.MARGIN_X  # 当前 X

        parts = []
        self._svg_header(parts, w, h)

        # 背景
        self._svg_background(parts, w, h)

        # 顶部标签
        label = f"1-Port Matching: {branch.pattern_str} / {branch.type_str}"
        self._svg_label(parts, w, label)

        # 传输线
        line_start = x + self.PORT_R + 5
        line_end = x + self.SNP_W + self.PORT_R * 2 + n_el * (self.ELEM_W + self.ELEM_GAP) + 20
        self._svg_line(parts, line_start, cy, line_end, cy)

        # 终结端 (左侧, 50Ω)
        self._svg_termination(parts, x, cy, branch.termination,
                              branch.get_termination_label())

        x += self.PORT_R + 15

        # 元件
        for i, (tt, ct) in enumerate(zip(branch.element_pattern, branch.comp_types or [])):
            if tt == 'S':
                # 串联: 在线上
                self._svg_series_element(parts, x, cy, tt, ct)
                x += self.ELEM_W + self.ELEM_GAP
            else:
                # 并联: 向下分支
                self._svg_shunt_element(parts, x, cy, tt, ct)
                x += self.ELEM_W + self.ELEM_GAP

        # SNP 框 (右侧)
        snp_x = x + 10
        self._svg_snp_box(parts, snp_x, cy - self.SNP_H // 2,
                          self.SNP_W, self.SNP_H,
                          f"{self._snp_label(variant.nports)}")

        parts.append('</svg>')
        return ''.join(parts)

    # ══════════════════════════════════════════════════════════
    # 2-Port SVG
    # ══════════════════════════════════════════════════════════

    def _render_2port(self, variant: TopologyVariant) -> str:
        b_left = variant.get_branch(0) or variant.branches[0]
        b_right = variant.get_branch(1) if variant.nports >= 2 else \
                  (variant.branches[1] if len(variant.branches) > 1 else
                   PortBranch(port=1, port_type='ground', termination='short'))

        n_left = b_left.element_count
        n_right = b_right.element_count

        # 计算尺寸
        left_width = self.PORT_R + 15 + n_left * (self.ELEM_W + self.ELEM_GAP)
        right_width = self.PORT_R + 15 + n_right * (self.ELEM_W + self.ELEM_GAP)
        snp_center_x = self.MARGIN_X + left_width + self.SNP_W // 2 + 20

        total_w = (self.MARGIN_X + left_width + self.SNP_W + right_width + self.MARGIN_X + 40)
        total_h = 140
        w = max(total_w, 320)
        h = total_h
        cy = h // 2 - 10

        parts = []
        self._svg_header(parts, w, h)
        self._svg_background(parts, w, h)

        # 标签
        label = (f"2-Port: "
                 f"P1({b_left.pattern_str}/{b_left.type_str}) → "
                 f"SNP → P2({b_right.pattern_str}/{b_right.type_str})")
        self._svg_label(parts, w, label)

        # 左分支
        x = self.MARGIN_X + 10

        # 左传输线 (从终结到SNP)
        line_l_start = x + self.PORT_R + 5
        line_l_end = snp_center_x - self.SNP_W // 2 - 10
        self._svg_line(parts, line_l_start, cy, line_l_end, cy)

        # 左终结
        self._svg_termination(parts, x, cy, b_left.termination,
                              b_left.get_termination_label())

        x += self.PORT_R + 15

        # 左元件
        for i, (tt, ct) in enumerate(zip(b_left.element_pattern, b_left.comp_types or [])):
            if tt == 'S':
                self._svg_series_element(parts, x, cy, tt, ct)
            else:
                self._svg_shunt_element(parts, x, cy, tt, ct)
            x += self.ELEM_W + self.ELEM_GAP

        # SNP 框 (中间)
        snp_x = snp_center_x - self.SNP_W // 2
        snp_y = cy - self.SNP_H // 2
        self._svg_snp_box(parts, snp_x, snp_y, self.SNP_W, self.SNP_H,
                          f"{self._snp_label(variant.nports)}")

        # 右分支
        rx = snp_x + self.SNP_W + 20

        # 右传输线 (从SNP到终结)
        line_r_start = snp_x + self.SNP_W + 10
        line_r_end = rx + self.PORT_R + 5 + n_right * (self.ELEM_W + self.ELEM_GAP)
        self._svg_line(parts, line_r_start, cy, line_r_end, cy)

        # 右元件
        for i, (tt, ct) in enumerate(zip(b_right.element_pattern, b_right.comp_types or [])):
            if tt == 'S':
                self._svg_series_element(parts, rx, cy, tt, ct)
            else:
                self._svg_shunt_element(parts, rx, cy, tt, ct)
            rx += self.ELEM_W + self.ELEM_GAP

        # 右终结
        self._svg_termination(parts, rx, cy, b_right.termination,
                              b_right.get_termination_label())

        # SNP ↔ 元件 连线圆点
        self._svg_junction(parts, snp_x - 5, cy)
        self._svg_junction(parts, snp_x + self.SNP_W + 5, cy)

        parts.append('</svg>')
        return ''.join(parts)

    # ══════════════════════════════════════════════════════════
    # 3-Port SVG
    # ══════════════════════════════════════════════════════════

    def _render_3port(self, variant: TopologyVariant) -> str:
        b_left = variant.get_branch(0) or variant.branches[0]
        b_right = variant.get_branch(1) or (variant.branches[1] if len(variant.branches) > 1 else
                  PortBranch(port=1, port_type='ground', termination='short'))
        b_bottom = variant.get_branch(2) or (variant.branches[2] if len(variant.branches) > 2 else
                   PortBranch(port=2, port_type='ground', termination='short'))

        n_left = b_left.element_count
        n_right = b_right.element_count
        n_bottom = b_bottom.element_count

        # 计算尺寸
        left_width = self.PORT_R + 15 + n_left * (self.ELEM_W + self.ELEM_GAP)
        right_width = self.PORT_R + 15 + n_right * (self.ELEM_W + self.ELEM_GAP)
        bottom_width = self.PORT_R + 15 + n_bottom * (self.ELEM_W + self.ELEM_GAP)

        main_width = left_width + self.SNP_W + max(right_width, bottom_width + 40) + 40
        total_w = max(main_width + self.MARGIN_X * 2, 400)
        total_h = 220
        w = total_w
        h = total_h

        # 布局坐标
        center_x = self.MARGIN_X + left_width + self.SNP_W // 2 + 30
        cy_main = 65  # 主传输线 Y (左/右)
        cy_bottom = 155  # 底部传输线 Y

        parts = []
        self._svg_header(parts, w, h)
        self._svg_background(parts, w, h)

        # 标签
        label = (f"3-Port: "
                 f"P1({b_left.pattern_str}/{b_left.type_str}) → "
                 f"SNP → P2({b_right.pattern_str}/{b_right.type_str}) "
                 f"↕ P3({b_bottom.pattern_str}/{b_bottom.type_str})")
        self._svg_label(parts, w, label)

        snp_x = center_x - self.SNP_W // 2
        snp_y = cy_main - self.SNP_H // 2

        # ── 左分支 ──
        x = self.MARGIN_X + 10
        line_l_end = snp_x - 10
        self._svg_line(parts, x + self.PORT_R + 5, cy_main, line_l_end, cy_main)
        self._svg_termination(parts, x, cy_main, b_left.termination,
                              b_left.get_termination_label())
        x += self.PORT_R + 15
        for tt, ct in zip(b_left.element_pattern, b_left.comp_types or []):
            if tt == 'S':
                self._svg_series_element(parts, x, cy_main, tt, ct)
            else:
                self._svg_shunt_element(parts, x, cy_main, tt, ct)
            x += self.ELEM_W + self.ELEM_GAP

        # SNP 框
        self._svg_snp_box(parts, snp_x, snp_y, self.SNP_W, self.SNP_H,
                          f"{self._snp_label(variant.nports)}")

        # ── 右分支 ──
        rx = snp_x + self.SNP_W + 20
        line_r_start = snp_x + self.SNP_W + 10
        line_r_end = rx + self.PORT_R + 5 + n_right * (self.ELEM_W + self.ELEM_GAP)
        self._svg_line(parts, line_r_start, cy_main, line_r_end, cy_main)
        for tt, ct in zip(b_right.element_pattern, b_right.comp_types or []):
            if tt == 'S':
                self._svg_series_element(parts, rx, cy_main, tt, ct)
            else:
                self._svg_shunt_element(parts, rx, cy_main, tt, ct)
            rx += self.ELEM_W + self.ELEM_GAP
        self._svg_termination(parts, rx, cy_main, b_right.termination,
                              b_right.get_termination_label())

        # ── 底部分支 (Port 3) ──
        # 从 SNP 中心向下
        branch_x = center_x
        self._svg_line(parts, branch_x, cy_main + self.SNP_H // 2,
                       branch_x, cy_bottom - self.PORT_R - 5)

        bx = branch_x - (n_bottom * (self.ELEM_W + self.ELEM_GAP)) // 2
        # 底部水平线
        line_b_start = bx - 5
        line_b_end = bx + self.PORT_R + 15 + n_bottom * (self.ELEM_W + self.ELEM_GAP)
        self._svg_line(parts, line_b_start, cy_bottom, line_b_end, cy_bottom)

        # 底部终止
        self._svg_termination(parts, bx - 10, cy_bottom, b_bottom.termination,
                              b_bottom.get_termination_label())
        bx += self.PORT_R + 5

        # 左→右排列 (从终止开始向右)
        for tt, ct in zip(b_bottom.element_pattern, b_bottom.comp_types or []):
            if tt == 'S':
                self._svg_series_element(parts, bx, cy_bottom, tt, ct)
            else:
                self._svg_shunt_element(parts, bx, cy_bottom, tt, ct)
            bx += self.ELEM_W + self.ELEM_GAP

        # SNP 连接点
        self._svg_junction(parts, snp_x - 5, cy_main)
        self._svg_junction(parts, snp_x + self.SNP_W + 5, cy_main)
        self._svg_junction(parts, branch_x, cy_main + self.SNP_H // 2)

        parts.append('</svg>')
        return ''.join(parts)

    # ══════════════════════════════════════════════════════════
    # N-Port SVG
    # ══════════════════════════════════════════════════════════

    def _render_nport(self, variant: TopologyVariant, nports: int) -> str:
        # 分支: port0=左, port1=右上, port2..N-1=右侧垂直堆叠
        branches = variant.branches
        b_left = branches[0] if branches else PortBranch(port=0)
        # 剩余分支按端口号排列 (port2+ 右侧堆叠)
        right_branches = list(branches[1:]) if len(branches) > 1 else []
        n_right = len(right_branches)

        n_left = b_left.element_count
        right_ns = [b.element_count for b in right_branches]
        max_right_el = max(right_ns) if right_ns else 0

        # 计算尺寸
        left_width = self.PORT_R + 15 + n_left * (self.ELEM_W + self.ELEM_GAP)
        right_width = (self.PORT_R + 15 + max_right_el * (self.ELEM_W + self.ELEM_GAP)
                       if max_right_el > 0 else self.PORT_R * 2 + 20)

        main_width = left_width + self.SNP_W + right_width + 60
        port_h = 50  # 每个端口行的高度
        total_right_h = max(1, n_right) * (port_h + 10) + 20

        total_w = max(main_width + self.MARGIN_X * 2, 400)
        total_h = max(160, total_right_h + self.MARGIN_Y * 2 + 10)
        w = total_w
        h = total_h

        center_x = self.MARGIN_X + left_width + self.SNP_W // 2 + 30
        snp_x = center_x - self.SNP_W // 2

        parts = []
        self._svg_header(parts, w, h)
        self._svg_background(parts, w, h)

        # 标签
        label = f"{nports}-Port Matching Network"
        self._svg_label(parts, w, label)

        # ── 左分支 (Port 1) ──
        # 居中放置
        right_area_top = self.MARGIN_Y + 20
        left_cy = right_area_top + (total_right_h // 2) if n_right > 0 else 65
        if n_right == 0:
            left_cy = h // 2 - 10

        x = self.MARGIN_X + 10
        line_l_end = snp_x - 10
        self._svg_line(parts, x + self.PORT_R + 5, left_cy, line_l_end, left_cy)
        self._svg_termination(parts, x, left_cy, b_left.termination,
                              b_left.get_termination_label())
        x += self.PORT_R + 15
        for tt, ct in zip(b_left.element_pattern, b_left.comp_types or []):
            if tt == 'S':
                self._svg_series_element(parts, x, left_cy, tt, ct)
            else:
                self._svg_shunt_element(parts, x, left_cy, tt, ct)
            x += self.ELEM_W + self.ELEM_GAP

        # SNP 框高度取决于右侧分支数量
        snp_h = max(self.SNP_H, n_right * port_h + 10 + 10)
        snp_y = left_cy - snp_h // 2
        self._svg_snp_box(parts, snp_x, snp_y, self.SNP_W, snp_h,
                          f"{self._snp_label(variant.nports)}")

        # ── 右侧分支 (Port 2..N) ──
        for idx, rb in enumerate(right_branches):
            rcy = snp_y + 15 + idx * port_h
            rx = snp_x + self.SNP_W + 20

            # 从 SNP 到元件的连线
            line_r_start = snp_x + self.SNP_W + 10
            line_r_end = rx + self.PORT_R + 5 + rb.element_count * (self.ELEM_W + self.ELEM_GAP)
            self._svg_line(parts, line_r_start, rcy, line_r_end, rcy)

            # 连接点
            self._svg_junction(parts, snp_x + self.SNP_W + 5, rcy)

            # 元件
            for tt, ct in zip(rb.element_pattern, rb.comp_types or []):
                if tt == 'S':
                    self._svg_series_element(parts, rx, rcy, tt, ct)
                else:
                    self._svg_shunt_element(parts, rx, rcy, tt, ct)
                rx += self.ELEM_W + self.ELEM_GAP

            # 终结
            self._svg_termination(parts, rx, rcy, rb.termination,
                                  rb.get_termination_label() + f" P{rb.port+1}")

        # SNP 左连接点
        self._svg_junction(parts, snp_x - 5, left_cy)

        parts.append('</svg>')
        return ''.join(parts)

    # ══════════════════════════════════════════════════════════
    # SVG 图形元素
    # ══════════════════════════════════════════════════════════

    def _svg_header(self, parts: list, w: int, h: int):
        parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">\n'
        )

    def _svg_background(self, parts: list, w: int, h: int):
        parts.append(
            f'<rect x="5" y="5" width="{w-10}" height="{h-10}" '
            f'rx="12" ry="12" fill="{self.COLOR_BG}" '
            f'stroke="{self.COLOR_BG_BORDER}" stroke-width="1.5"/>\n'
        )

    def _svg_label(self, parts: list, w: int, text: str):
        parts.append(
            f'<text x="{w/2}" y="20" text-anchor="middle" '
            f'font-size="11" font-weight="600" fill="{self.COLOR_TEXT}">'
            f'{self._escape_xml(text)}</text>\n'
        )

    def _svg_line(self, parts: list, x1: int, y1: int, x2: int, y2: int):
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{self.COLOR_LINE}" stroke-width="2"/>\n'
        )

    def _svg_series_element(self, parts: list, x: int, cy: int, tt: str, ct: str):
        """串联元件: 在传输线上的方框."""
        color = self.COLOR_SERIES
        border = self.COLOR_SERIES_BORDER
        parts.append(
            f'<rect x="{x}" y="{cy-self.ELEM_H//2}" '
            f'width="{self.ELEM_W}" height="{self.ELEM_H}" '
            f'rx="4" ry="4" fill="{color}" '
            f'stroke="{border}" stroke-width="1" opacity="0.85"/>\n'
        )
        parts.append(
            f'<text x="{x+self.ELEM_W//2}" y="{cy+4}" text-anchor="middle" '
            f'fill="{self.ELEM_TEXT_COLOR}" font-size="10" font-weight="600">'
            f'{tt}</text>\n'
        )

    def _svg_shunt_element(self, parts: list, x: int, cy: int, tt: str, ct: str):
        """并联元件: 在传输线下方的方框 + 分支线."""
        color = self.COLOR_SHUNT
        border = self.COLOR_SHUNT_BORDER
        branch_y = cy + 45

        # 元件框 (在传输线下方)
        parts.append(
            f'<rect x="{x}" y="{branch_y-self.ELEM_H//2}" '
            f'width="{self.ELEM_W}" height="{self.ELEM_H}" '
            f'rx="4" ry="4" fill="{color}" '
            f'stroke="{border}" stroke-width="1" opacity="0.85"/>\n'
        )
        parts.append(
            f'<text x="{x+self.ELEM_W//2}" y="{branch_y+4}" text-anchor="middle" '
            f'fill="{self.ELEM_TEXT_COLOR}" font-size="10" font-weight="600">'
            f'P</text>\n'
        )
        # 分支连接线
        parts.append(
            f'<line x1="{x+self.ELEM_W//2}" y1="{cy}" '
            f'x2="{x+self.ELEM_W//2}" y2="{branch_y-self.ELEM_H//2}" '
            f'stroke="{border}" stroke-width="1.5"/>\n'
        )
        # 地符号
        parts.append(
            f'<line x1="{x+self.ELEM_W//2-6}" y1="{branch_y+self.ELEM_H//2}" '
            f'x2="{x+self.ELEM_W//2+6}" y2="{branch_y+self.ELEM_H//2}" '
            f'stroke="{self.COLOR_GND_SYMBOL}" stroke-width="1.2"/>\n'
        )

    def _svg_termination(self, parts: list, x: int, y: int,
                         term_type: str, label: str):
        """终结符号: 圆 + 标签.

        根据终结类型显示不同颜色:
          - load (50Ω): 绿色
          - short (GND): 红色 + 地符号
          - open: 黄色
        """
        color = {
            'load': '#0d904f',
            'short': '#d93025',
            'open': '#f9ab00',
            'ground': '#d93025',
        }.get(term_type, '#666')

        # 圆
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="{self.PORT_R}" '
            f'fill="{color}" stroke="#333" stroke-width="1.2"/>\n'
        )
        # 标签文字
        parts.append(
            f'<text x="{x}" y="{y+4}" text-anchor="middle" '
            f'fill="white" font-size="7" font-weight="600">'
            f'{self._escape_xml(label)}</text>\n'
        )

        # 地端口额外添加地符号
        if term_type in ('short', 'ground'):
            gs_y = y + self.PORT_R + 3
            gs_x = x
            parts.append(
                f'<line x1="{gs_x-6}" y1="{gs_y}" x2="{gs_x+6}" y2="{gs_y}" '
                f'stroke="{self.COLOR_GND_SYMBOL}" stroke-width="1.5"/>\n'
            )
            parts.append(
                f'<line x1="{gs_x-3}" y1="{gs_y+4}" x2="{gs_x+3}" y2="{gs_y+4}" '
                f'stroke="{self.COLOR_GND_SYMBOL}" stroke-width="1.0"/>\n'
            )

    def _svg_snp_box(self, parts: list, x: int, y: int, w: int, h: int, label: str):
        """SNP 框: 蓝色边框圆角矩形."""
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="8" ry="8" fill="{self.COLOR_SNP_BG}" '
            f'stroke="{self.COLOR_SNP_BORDER}" stroke-width="2"/>\n'
        )
        parts.append(
            f'<text x="{x+w//2}" y="{y+h//2+4}" text-anchor="middle" '
            f'fill="{self.COLOR_SNP_BORDER}" font-size="12" font-weight="700">'
            f'{self._escape_xml(label)}</text>\n'
        )

    def _svg_junction(self, parts: list, x: int, y: int):
        """连接点: 小圆点."""
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="4" '
            f'fill="{self.COLOR_LINE}" stroke="none"/>\n'
        )

    def _snp_label(self, nports: int) -> str:
        return f"|SNP|"

    @staticmethod
    def _escape_xml(s: str) -> str:
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    # ══════════════════════════════════════════════════════════
    # ASCII 文本渲染
    # ══════════════════════════════════════════════════════════

    def _render_text_1port(self, variant: TopologyVariant) -> str:
        branch = variant.branches[0]
        n_el = branch.element_count

        lines = [
            f"── 1-Port Topology: {variant.variant_id} ──",
            f"   Port 1 ({branch.port_type} → {branch.termination})",
        ]

        # 拓扑行
        topo_str = branch.pattern_str
        type_str = branch.type_str or '?' * n_el
        term_label = branch.get_termination_label()

        # 构建 ASCII 图
        # [50Ω] --- S --- P --- S --- [|SNP|]
        ascii_parts = [f"[{term_label}]"]
        for tt, ct in zip(branch.element_pattern, branch.comp_types or []):
            ascii_parts.append(f"─── {ct}({tt}) ───")
        ascii_parts.append("[|SNP|]")

        line = ''.join(ascii_parts)
        lines.append(f"   {line}")
        lines.append(f"   模式: {topo_str}  类型: {type_str}")
        return '\n'.join(lines)

    def _render_text_2port(self, variant: TopologyVariant) -> str:
        b_left = variant.get_branch(0) or variant.branches[0]
        b_right = variant.get_branch(1) if variant.nports >= 2 else \
                  (variant.branches[1] if len(variant.branches) > 1 else
                   PortBranch(port=1, port_type='ground', termination='short'))

        lines = [
            f"── 2-Port Topology: {variant.variant_id} ──",
        ]

        # Port 1 (左)
        line_l = f"[{b_left.get_termination_label()}]"
        for tt, ct in zip(b_left.element_pattern, b_left.comp_types or []):
            line_l += f"─── {ct}({tt}) ───"

        # SNP
        mid = "─── [|SNP|] ───"

        # Port 2 (右)
        line_r = ""
        for tt, ct in zip(b_right.element_pattern, b_right.comp_types or []):
            line_r += f"─── {ct}({tt}) ───"
        line_r += f"[{b_right.get_termination_label()}]"

        lines.append(f"   P1 {line_l}{mid}{line_r}")
        lines.append(f"   P1模式: {b_left.pattern_str}  类型: {b_left.type_str}")
        lines.append(f"   P2模式: {b_right.pattern_str}  类型: {b_right.type_str}  (终结: {b_right.termination})")
        return '\n'.join(lines)

    def _render_text_3port(self, variant: TopologyVariant) -> str:
        b_left = variant.get_branch(0) or variant.branches[0]
        b_right = variant.get_branch(1) or (variant.branches[1] if len(variant.branches) > 1 else
                  PortBranch(port=1))
        b_bottom = variant.get_branch(2) or (variant.branches[2] if len(variant.branches) > 2 else
                   PortBranch(port=2))

        lines = [
            f"── 3-Port Topology: {variant.variant_id} ──",
        ]

        # Port 1 (左行)
        left_str = f"[{b_left.get_termination_label()}]"
        for tt, ct in zip(b_left.element_pattern, b_left.comp_types or []):
            left_str += f"── {ct}({tt}) ──"

        # Port 2 (右行)
        right_str = ""
        for tt, ct in zip(b_right.element_pattern, b_right.comp_types or []):
            right_str += f"── {ct}({tt}) ──"
        right_str += f"[{b_right.get_termination_label()}]"

        # Port 3 (下行)
        bottom_str = f"[{b_bottom.get_termination_label()}]"
        for tt, ct in zip(b_bottom.element_pattern, b_bottom.comp_types or []):
            bottom_str += f"── {ct}({tt}) ──"

        lines.append(f"   P1 {left_str}── |SNP| ──{right_str}")
        lines.append(f"                │")
        lines.append(f"                {bottom_str}")
        lines.append(f"   P1: {b_left.pattern_str}/{b_left.type_str}  "
                     f"P2: {b_right.pattern_str}/{b_right.type_str}  "
                     f"P3: {b_bottom.pattern_str}/{b_bottom.type_str}")
        return '\n'.join(lines)

    def _render_text_nport(self, variant: TopologyVariant, nports: int) -> str:
        branches = variant.branches
        b_left = branches[0] if branches else PortBranch(port=0)
        right_branches = list(branches[1:]) if len(branches) > 1 else []

        lines = [
            f"── {nports}-Port Topology: {variant.variant_id} ──",
        ]

        # Port 1 (左)
        left_str = f"[{b_left.get_termination_label()}]"
        for tt, ct in zip(b_left.element_pattern, b_left.comp_types or []):
            left_str += f"── {ct}({tt}) ──"

        # 右侧分支 (每个一行)
        right_lines = []
        for rb in right_branches:
            rstr = ""
            for tt, ct in zip(rb.element_pattern, rb.comp_types or []):
                rstr += f"── {ct}({tt}) ──"
            rstr += f"[{rb.get_termination_label()}]"
            right_lines.append(rstr)

        # 构建 N-port 布局
        lines.append(f"   P1 {left_str}── |{'SNP':^12}| ── P2{right_lines[0] if right_lines else ''}")
        for i, rl in enumerate(right_lines[1:], start=3):
            lines.append(f"                    ├── P{i}{rl}")
        lines.append(f"   P1: {b_left.pattern_str}/{b_left.type_str}")
        for i, rb in enumerate(right_branches, start=2):
            lines.append(f"   P{i}: {rb.pattern_str}/{rb.type_str}  (终结: {rb.termination})")
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════
# 6. 辅助函数 — 外部调用接口
# ═══════════════════════════════════════════════════════════════════

def create_topology_engine(snp_path: str = None,
                           port_configs: list = None,
                           bands_mhz: list = None) -> TopologyEngine:
    """便捷工厂函数: 创建并配置拓扑引擎.

    Args:
        snp_path: SNP 文件路径 (可选)
        port_configs: list of dict Port configs (可选)
        bands_mhz: [[start, end], ...] 频段 (可选)

    Returns:
        TopologyEngine 实例
    """
    engine = TopologyEngine()
    if snp_path:
        engine.load_snp(snp_path)
    if port_configs:
        engine.load_port_configs(port_configs, bands_mhz)
    return engine


def generate_topologies(snp_path: str,
                        port_configs: list,
                        bands_mhz: list = None,
                        output_dir: str = 'topology_output',
                        render_svg: bool = True,
                        render_text: bool = True,
                        save_svg: bool = True,
                        print_summary: bool = True,
                        print_text: bool = True,
                        max_print: int = 20) -> TopologyResult:
    """一键式拓扑生成函数.

    完整的拓扑生成 + 渲染 + 输出流程.

    Args:
        snp_path: SNP 文件路径
        port_configs: list of dict port配置
        bands_mhz: [[start, end], ...] 频段
        output_dir: SVG 输出目录
        render_svg: 是否生成 SVG
        render_text: 是否生成文本图
        save_svg: 是否保存 SVG 到文件
        print_summary: 是否打印摘要
        print_text: 是否打印文本图到控制台
        max_print: 文本图最大打印数量 (0=全部)

    Returns:
        TopologyResult 对象
    """
    _setup_logger()

    engine = TopologyEngine()
    logger.info("=" * 60)
    logger.info("RF Topology Engine starting")
    logger.info("=" * 60)

    # 1. 加载 SNP
    snp_info = engine.load_snp(snp_path)
    logger.info(f"  SNP: {snp_info['file']}  ({snp_info['nports']}-port, "
                f"{snp_info['freqs']} freqs)")
    logger.info(f"  频率: [{snp_info['freq_min_ghz']:.4f} - "
                f"{snp_info['freq_max_ghz']:.4f}] GHz")

    # 2. 加载端口配置
    engine.load_port_configs(port_configs, bands_mhz)

    # 3. 生成拓扑
    result = engine.generate()

    # 4. 渲染
    result.render_all(render_svg=render_svg, render_text=render_text)

    # 5. 输出
    if print_summary:
        print()
        print(result.summary)
        print()

    if save_svg and render_svg:
        result.save_svg(output_dir)

    if print_text and render_text:
        result.print_text(max_variants=max_print)

    logger.info("拓扑引擎完成!")
    logger.info(f"  Output dir: {os.path.abspath(output_dir)}/")
    logger.info(f"  Total variants: {result.total_variants}")

    return result


# ═══════════════════════════════════════════════════════════════════
# 7. CLI 入口
# ═══════════════════════════════════════════════════════════════════

def main():
    """命令行入口.

    用法:
        python engine/topology.py --snp antenna.s2p --config config.json
        python engine/topology.py --snp antenna.s2p --nports 2
        python engine/topology.py --snp antenna.s1p --1port
    """
    # Ensure project root is in path for module imports
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    import argparse

    _setup_logger(logging.DEBUG)

    parser = argparse.ArgumentParser(
        description='RF 拓扑引擎 — 独立的匹配网络拓扑生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例:
              # 使用 JSON 配置
              python engine/topology.py --snp ant.s2p --config port_config.json

              # 快速 2-port 测试 (自动配置)
              python engine/topology.py --snp ant.s2p --nports 2

              # 自定义端口
              python engine/topology.py --snp ant.s2p \\
                --port "port=0,type=load,elem_min=2,elem_max=4" \\
                --port "port=1,type=ground,elem_min=1,elem_max=1"

            port配置 JSON 格式:
            [
                {"port": 0, "type": "load", "elem_min": 2, "elem_max": 4},
                {"port": 1, "type": "ground", "elem_min": 1, "elem_max": 2},
                {"port": 2, "type": "ground", "elem_min": 1, "elem_max": 1}
            ]
        """),
    )

    parser.add_argument('--snp', '-s', type=str, required=True,
                        help='SNP 文件路径 (*.s1p, *.s2p, *.sNp)')
    parser.add_argument('--config', '-c', type=str,
                        help='端口配置 JSON 文件路径')
    parser.add_argument('--nports', '-n', type=int,
                        help='端口数 (自动配置, 与 --config 互斥)')
    parser.add_argument('--port', '-p', action='append',
                        help='端口配置 "key=val,key=val" 格式')
    parser.add_argument('--bands', '-b', type=str,
                        help='频段 "start-end,start-end" 格式, 例如 "2400-2500,5150-5850"')
    parser.add_argument('--output', '-o', type=str, default='topology_output',
                        help='SVG 输出目录 (默认: topology_output)')
    parser.add_argument('--max-print', type=int, default=20,
                        help='文本图最大打印数量 (默认: 20, 0=全部)')
    parser.add_argument('--no-svg', action='store_true',
                        help='不生成 SVG 文件')
    parser.add_argument('--no-text', action='store_true',
                        help='不打印文本图')
    parser.add_argument('--summary-only', action='store_true',
                        help='仅打印摘要, 不打印详细拓扑')

    args = parser.parse_args()

    # ── 解析端口配置 ──
    port_configs = None

    if args.config:
        # 从 JSON 文件加载
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                port_configs = json.load(f)
            logger.info(f"Loaded port config: {args.config}")
        except Exception as e:
            logger.error(f"Cannot load config file {args.config}: {e}")
            sys.exit(1)
    elif args.port:
        # 从命令行参数解析
        port_configs = []
        for p_str in args.port:
            kwargs = {}
            for kv in p_str.split(','):
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    kwargs[k.strip()] = v.strip()
            port_configs.append(kwargs)
    elif args.nports:
        # 自动配置
        port_configs = []
        for p in range(args.nports):
            if p == 0:
                port_configs.append({'port': p, 'type': 'load',
                                     'elem_min': 2, 'elem_max': 4})
            else:
                port_configs.append({'port': p, 'type': 'ground',
                                     'termination': 'short',
                                     'elem_min': 1, 'elem_max': 1})
    else:
        parser.print_help()
        print("\n错误: 必须指定 --config, --nports, 或 --port 之一")
        sys.exit(1)

    # ── 解析频段 ──
    bands_mhz = None
    if args.bands:
        bands_mhz = []
        for band_str in args.bands.split(','):
            parts = band_str.strip().split('-')
            if len(parts) == 2:
                bands_mhz.append([float(parts[0]), float(parts[1])])

    # ── 执行拓扑生成 ──
    result = generate_topologies(
        snp_path=args.snp,
        port_configs=port_configs,
        bands_mhz=bands_mhz,
        output_dir=args.output,
        render_svg=not args.no_svg,
        render_text=not args.no_text,
        save_svg=not args.no_svg,
        print_summary=True,
        print_text=not args.summary_only and not args.no_text,
        max_print=args.max_print if not args.summary_only else 0,
    )

    # ── 输出 JSON 摘要 (到 stdout) ──
    if args.summary_only:
        print()
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
