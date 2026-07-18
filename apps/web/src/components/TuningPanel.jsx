import React, { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import TopologySchematic from './TopologySchematic';
import EngineeringSmithChart from './EngineeringSmithChart';

const COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];

/* ── Objective presets (matching backend) ── */
const OBJECTIVE_PRESETS = [
  { name: 'average_efficiency', label: '最高平均效率', desc: '优先提升所有端口和频段的平均总效率' },
  { name: 'worst_case', label: '最佳最差点', desc: '优先改善效率最低的频点与端口' },
  { name: 'balanced', label: '均衡优化', desc: '兼顾平均效率和最差点表现' },
  { name: 'low_coupling', label: '低耦合 / MIMO', desc: '提高端口间耦合损耗的惩罚权重' },
  { name: 'low_cost', label: '低元件数量', desc: '在性能与网络复杂度之间偏向简单 BOM' },
];

/* ── Tuning modes ── */
const TUNING_MODES = [
  { name: 'fixed_lc', label: '固定 LC', desc: '传统固定元件匹配' },
  { name: 'grid_s2p', label: 'S2P 网格', desc: '实测元件窄带网格搜索' },
  { name: 'transmission_line', label: '传输线', desc: '物理线与开短路枝节综合' },
  { name: 'tunable_c', label: '可调电容', desc: '一个或多个可变电容位置' },
  { name: 'switch', label: '开关状态', desc: '多状态开关与分支 LC' },
];

const SEARCH_QUALITY_PRESETS = {
  quick: { label: '快速探索', timeout: 15, beamWidth: 8, bandPoints: 3, desc: '快速生成短名单；大型目录可能只完成部分搜索。' },
  balanced: { label: '工程均衡', timeout: 45, beamWidth: 10, bandPoints: 5, desc: '适用于日常实测元件工程搜索。' },
  thorough: { label: '深度约束', timeout: 120, beamWidth: 10, bandPoints: 10, desc: '执行深度耦合优化；指定端口拓扑时效果最佳。' },
  exhaustive: { label: '自动拓扑深搜', timeout: 150, beamWidth: 50, bandPoints: 3, desc: '面向符合条件的 2–3 端口系统进行离线拓扑发现。' },
};

function inferSearchQuality(timeout) {
  if (timeout < 30) return 'quick';
  if (timeout < 60) return 'balanced';
  if (timeout < 150) return 'thorough';
  return 'exhaustive';
}

/* ── Band presets (matching backend) ── */
const BAND_PRESETS = {
  "GPS L1": [1574, 1576], "GPS L5": [1176, 1177],
  "WiFi 2.4GHz": [2400, 2500], "WiFi 5GHz": [5150, 5850],
  "LTE B1": [1920, 2170], "LTE B3": [1710, 1880],
  "LTE B7": [2500, 2690], "5G n77": [3300, 4200],
  "5G n78": [3300, 3800], "5G n79": [4400, 5000],
  "Bluetooth": [2400, 2480], "NB-IoT": [700, 960],
};

const TOPOLOGY_POLICY_PRESETS = [
  { value: 'auto', label: '自动选择（推荐）', codes: null },
  { value: 'bare', label: '仅评估裸 DUT', codes: ['0'] },
  { value: 'one', label: '任意单元件网络', codes: ['SL', 'PL', 'SC', 'PC'] },
  { value: 'series', label: '仅串联元件', codes: ['SL', 'SC'] },
  { value: 'shunt', label: '仅并联元件', codes: ['PL', 'PC'] },
  { value: 'l_network', label: '任意双元件 L 网络', codes: ['PLSL', 'SLPL', 'PLSC', 'SLPC', 'PCPL', 'SCPL', 'SCSL', 'PCSL'] },
];

function topologyPolicyValue(codes) {
  if (codes == null) return 'auto';
  const signature = [...codes].sort().join(',');
  return TOPOLOGY_POLICY_PRESETS.find(
    preset => preset.codes && [...preset.codes].sort().join(',') === signature
  )?.value || 'custom';
}

/* ── Sub-components ── */

function ModeSelector({ mode, setMode }) {
  return (
    <div className="card mode-card">
      <h3><span>01</span> 匹配方案</h3>
      <div className="mode-grid">
        {TUNING_MODES.map(m => (
          <button key={m.name}
            className={`mode-option ${mode === m.name ? 'active' : ''}`}
            onClick={() => setMode(m.name)}>
            <i>{m.name === 'fixed_lc' ? 'LC' : m.name === 'grid_s2p' ? 'S2P' : m.name === 'transmission_line' ? 'TL' : m.name === 'tunable_c' ? 'VC' : 'SW'}</i>
            <span><strong>{m.label}</strong><small>{m.desc}</small></span>
          </button>
        ))}
      </div>
    </div>
  );
}

function BandEditor({ bands, setBands, bandWeights = [], setBandWeights, setBandConfig }) {
  const normalizedWeights = bands.map((_, index) => Number(bandWeights[index] ?? 1));
  const updateBoth = (nextBands, nextWeights) => {
    if (setBandConfig) setBandConfig(nextBands, nextWeights);
    else {
      setBands(nextBands);
      setBandWeights?.(nextWeights);
    }
  };
  const addBand = () => updateBoth([...bands, [2400, 2500]], [...normalizedWeights, 1]);
  const removeBand = (i) => updateBoth(
    bands.filter((_, j) => j !== i),
    normalizedWeights.filter((_, j) => j !== i),
  );
  const updateBand = (i, idx, val) => {
    const nb = bands.map((b, j) => j === i ? [idx === 0 ? val : b[0], idx === 1 ? val : b[1]] : b);
    setBands(nb);
  };
  const updateWeight = (i, value) => setBandWeights?.(
    normalizedWeights.map((weight, index) => index === i ? value : weight),
  );

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <span style={{ fontSize: 11, fontWeight: 600 }}>工作频段</span>
        <button className="btn btn-xs btn-primary" onClick={addBand}>+ 添加</button>
        {/* Band presets dropdown */}
        <select style={{ fontSize: 10, padding: '1px 4px' }}
          onChange={e => {
            if (e.target.value) {
              const preset = BAND_PRESETS[e.target.value];
              if (preset) updateBoth([...bands, preset], [...normalizedWeights, 1]);
            }
            e.target.value = '';
          }}>
          <option value="">频段预设…</option>
          {Object.keys(BAND_PRESETS).map(k => <option key={k} value={k}>{k}</option>)}
        </select>
      </div>
      {bands.map((b, i) => (
        <div key={i} style={{ display: 'flex', gap: 4, alignItems: 'center', marginBottom: 3 }}>
          <input type="number" value={b[0]} onChange={e => updateBand(i, 0, +e.target.value)}
            style={{ width: 65, fontSize: 11, padding: '2px 4px' }} placeholder="MHz" />
          <span style={{ fontSize: 10 }}>—</span>
          <input type="number" value={b[1]} onChange={e => updateBand(i, 1, +e.target.value)}
            style={{ width: 65, fontSize: 11, padding: '2px 4px' }} placeholder="MHz" />
          <label title="该频段相对于同端口其他频段的目标优先级" style={{ display: 'flex', alignItems: 'center', gap: 2, fontSize: 9, color: 'var(--text-secondary)' }}>
            权重
            <input type="number" min="0" max="100" step="0.1" value={normalizedWeights[i]}
              onChange={e => updateWeight(i, Number(e.target.value))}
              style={{ width: 48, fontSize: 10, padding: '2px 3px' }} />
          </label>
          <button className="btn btn-xs btn-danger" onClick={() => removeBand(i)} style={{ padding: '1px 6px' }}>×</button>
          <span style={{ fontSize: 9, color: 'var(--text-secondary)' }}>
            {((b[0] + b[1]) / 2).toFixed(0)} MHz
          </span>
        </div>
      ))}
    </div>
  );
}

function PortConfigPanel({ portConfigs, setPortConfigs, numPorts, showTopologyConstraints }) {
  const updatePort = (i, key, val) => {
    const nc = portConfigs.map((p, j) => j === i ? { ...p, [key]: val } : p);
    setPortConfigs(nc);
  };
  const updateBand = (pi, bi, idx, val) => {
    const nc = portConfigs.map((p, j) => {
      if (j !== pi) return p;
      const nb = p.bands_mhz.map((b, k) => k === bi ? [idx === 0 ? val : b[0], idx === 1 ? val : b[1]] : b);
      return { ...p, bands_mhz: nb };
    });
    setPortConfigs(nc);
  };
  const addBand = (pi) => {
    const nc = portConfigs.map((p, j) => j === pi ? { ...p, bands_mhz: [...p.bands_mhz, [2400, 2500]] } : p);
    setPortConfigs(nc);
  };
  const removeBand = (pi, bi) => {
    const nc = portConfigs.map((p, j) => j === pi ? { ...p, bands_mhz: p.bands_mhz.filter((_, k) => k !== bi) } : p);
    setPortConfigs(nc);
  };
  const updateMaxComponents = (i, current, value) => {
    const patch = { max_components: value };
    if (value === 0) patch.allowed_topology_codes = ['0'];
    else if (current.max_components === 0) patch.allowed_topology_codes = null;
    const nc = portConfigs.map((port, index) => index === i ? { ...port, ...patch } : port);
    setPortConfigs(nc);
  };

  return (
    <div className="card">
      <h3><span>02</span> 端口与频段</h3>
      {portConfigs.map((pc, i) => (
        <div key={i} style={{
          marginBottom: 8, padding: 8, borderRadius: 6,
          background: pc.enabled ? 'rgba(46,204,113,0.06)' : 'rgba(0,0,0,0.02)',
          border: `1px solid ${pc.enabled ? 'rgba(46,204,113,0.2)' : 'transparent'}`,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <span style={{
              display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
              background: COLORS[i % COLORS.length],
            }} />
            <strong style={{ fontSize: 12 }}>端口 {i + 1}</strong>
            <label style={{ fontSize: 10, marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 4 }}>
              <input type="checkbox" checked={pc.enabled} onChange={e => updatePort(i, 'enabled', e.target.checked)} />
              启用
            </label>
          </div>

          {pc.enabled && (
            <>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontSize: 10 }}>最大元件数</span>
                {[0, 1, 2, 3, 4, 5, 6].map(n => (
                  <button key={n}
                    className={`btn btn-xs ${pc.max_components === n ? 'btn-primary' : ''}`}
                    onClick={() => updateMaxComponents(i, pc, n)}
                  >{n}</button>
                ))}
              </div>

              <label title="端口权重会与每个频段权重相乘；1 表示默认优先级" style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6, fontSize: 10 }}>
                端口优先级
                <input type="number" min="0" max="100" step="0.1"
                  value={Number(pc.port_weight ?? 1)}
                  onChange={event => updatePort(i, 'port_weight', Number(event.target.value))}
                  style={{ width: 58, fontSize: 10, padding: '2px 4px' }} />
                <small style={{ color: 'var(--text-secondary)' }}>有效权重 = 端口 × 频段</small>
              </label>

              {showTopologyConstraints && (
                <div style={{ marginBottom: 7 }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10 }}>
                    <span>拓扑策略</span>
                    <select
                      value={topologyPolicyValue(pc.allowed_topology_codes)}
                      disabled={pc.max_components === 0}
                      onChange={event => {
                        const preset = TOPOLOGY_POLICY_PRESETS.find(item => item.value === event.target.value);
                        if (preset) updatePort(i, 'allowed_topology_codes', preset.codes);
                        else updatePort(i, 'allowed_topology_codes', pc.allowed_topology_codes || ['SL']);
                      }}
                      style={{ flex: 1, minWidth: 0, fontSize: 10, padding: '2px 4px' }}
                      title="Codes are ordered from the DUT outward: S/P = series/shunt, L/C = inductor/capacitor"
                    >
                      {TOPOLOGY_POLICY_PRESETS.map(item => (
                        <option key={item.value} value={item.value}>{item.label}</option>
                      ))}
                      <option value="custom">自定义白名单…</option>
                    </select>
                  </label>
                  {topologyPolicyValue(pc.allowed_topology_codes) === 'custom' && pc.max_components > 0 && (
                    <input
                      value={(pc.allowed_topology_codes || []).join(', ')}
                      onChange={event => updatePort(i, 'allowed_topology_codes', event.target.value
                        .toUpperCase().split(/[\s,]+/).filter(Boolean))}
                      placeholder="例如：SL, PC, PLSC"
                      style={{ width: '100%', marginTop: 4, fontSize: 10, padding: '3px 5px' }}
                    />
                  )}
                  <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 2 }}>
                    编码从 DUT 向外：S/P 为串联/并联，L/C 为电感/电容；“0” 表示不加网络。
                  </div>
                </div>
              )}

              <BandEditor
                bands={pc.bands_mhz}
                setBands={(nb) => updatePort(i, 'bands_mhz', nb)}
                bandWeights={pc.band_weights}
                setBandWeights={(weights) => updatePort(i, 'band_weights', weights)}
                setBandConfig={(bands, weights) => setPortConfigs(portConfigs.map((port, index) =>
                  index === i ? { ...port, bands_mhz: bands, band_weights: weights } : port
                ))}
              />
            </>
          )}
        </div>
      ))}
    </div>
  );
}

function ObjectiveSelector({ objective, setObjective }) {
  return (
    <div className="card objective-card">
      <h3><span>03</span> 优化目标</h3>
      <div className="objective-list">
        {OBJECTIVE_PRESETS.map(o => (
          <label key={o.name} className={`objective-option ${objective === o.name ? 'active' : ''}`}>
            <input type="radio" name="objective" checked={objective === o.name}
              onChange={() => setObjective(o.name)} />
            <div>
              <strong>{o.label}</strong>
              <span>{o.desc}</span>
            </div>
          </label>
        ))}
      </div>
    </div>
  );
}

const LINE_TOPOLOGY_LABELS = {
  through_line: 'Through line', open_stub: 'Open stub', short_stub: 'Short stub',
  connector_line_open_stub_dut: 'Connector → line → open stub → DUT',
  connector_line_short_stub_dut: 'Connector → line → short stub → DUT',
  connector_open_stub_line_dut: 'Connector → open stub → line → DUT',
  connector_short_stub_line_dut: 'Connector → short stub → line → DUT',
};

function TransmissionLineConfigEditor({ config, setConfig, availableFiles = [] }) {
  const update = (key, value) => setConfig({ ...config, [key]: value });
  const microstrip = config.microstrip || {};
  const updateMicrostrip = (key, value) => setConfig({
    ...config,
    ...(key === 'enabled' && value ? { attenuation_db: 0 } : {}),
    microstrip: { ...microstrip, [key]: value },
  });
  const toggleTopology = topology => {
    const active = config.topologies.includes(topology);
    const next = active
      ? config.topologies.filter(item => item !== topology)
      : [...config.topologies, topology];
    if (next.length) update('topologies', next);
  };
  const layoutFiles = availableFiles.filter(file =>
    String(file.filename || file).toLowerCase().endsWith('.s2p')
  );
  const layoutBlocks = config.layout_blocks || [];
  const setLayoutBlocks = blocks => update('layout_blocks', blocks);
  const addLayoutBlock = () => {
    if (!layoutFiles.length) return;
    setLayoutBlocks([...layoutBlocks, {
      filename: layoutFiles[0].filename || layoutFiles[0],
      location: 'connector_side',
      passivity_policy: 'warn',
      reverse_ports: false,
      reference_impedance_mode: 'native',
      left_fixture_filename: null,
      left_fixture_reverse_ports: false,
      right_fixture_filename: null,
      right_fixture_reverse_ports: false,
      maximum_deembedding_condition_number: 1e10,
    }]);
  };
  const updateLayoutBlock = (index, key, value) => setLayoutBlocks(
    layoutBlocks.map((block, itemIndex) => itemIndex === index ? { ...block, [key]: value } : block)
  );
  return (
    <div className="card">
      <h3>Transmission-Line Search</h3>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5 }}>
        {[
          ['characteristic_impedance_min_ohm', 'Minimum Z0 (ohm)'],
          ['characteristic_impedance_max_ohm', 'Maximum Z0 (ohm)'],
          ['electrical_length_min_deg', 'Minimum length (deg)'],
          ['electrical_length_max_deg', 'Maximum length (deg)'],
          ['attenuation_db', 'One-way loss (dB)'],
          ['loss_frequency_exponent', 'Loss frequency exponent'],
        ].map(([key, label]) => (
          <label key={key} style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
            {label}
            <input type="number" min="0" step="0.1" value={config[key]}
              disabled={key === 'attenuation_db' && microstrip.enabled}
              onChange={e => update(key, Number(e.target.value))}
              style={{ width: '100%', boxSizing: 'border-box', marginTop: 2, fontSize: 11 }} />
          </label>
        ))}
      </div>
      <label style={{ display: 'flex', gap: 5, alignItems: 'center', fontSize: 11, fontWeight: 600, marginTop: 9 }}>
        <input type="checkbox" checked={Boolean(microstrip.enabled)}
          onChange={e => updateMicrostrip('enabled', e.target.checked)} />
        Convert and score as manufacturable microstrip
      </label>
      {microstrip.enabled && (
        <div style={{ marginTop: 5, padding: 6, border: '1px solid var(--border)', borderRadius: 5 }}>
          <label style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
            Substrate name
            <input value={microstrip.substrate_name || ''}
              onChange={e => updateMicrostrip('substrate_name', e.target.value)}
              style={{ width: '100%', boxSizing: 'border-box', marginTop: 2, fontSize: 11 }} />
          </label>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5, marginTop: 5 }}>
            {[
              ['relative_permittivity', 'Relative permittivity'],
              ['substrate_height_mm', 'Dielectric height (mm)'],
              ['loss_tangent', 'Loss tangent'],
              ['copper_thickness_um', 'Copper thickness (µm)'],
              ['copper_roughness_rms_um', 'Copper roughness RMS (µm)'],
              ['minimum_width_mm', 'Minimum trace width (mm)'],
              ['maximum_width_mm', 'Maximum trace width (mm)'],
              ['width_tolerance_pct', 'Trace width tolerance (%)'],
              ['length_tolerance_pct', 'Etched length tolerance (%)'],
              ['substrate_height_tolerance_pct', 'Dielectric height tolerance (%)'],
              ['relative_permittivity_tolerance_pct', 'Permittivity tolerance (%)'],
            ].map(([key, label]) => (
              <label key={key} style={{ fontSize: 9, color: 'var(--text-secondary)' }}>
                {label}
                <input type="number" min="0" step="0.01" value={microstrip[key]}
                  onChange={e => updateMicrostrip(key, Number(e.target.value))}
                  style={{ width: '100%', boxSizing: 'border-box', marginTop: 2, fontSize: 10 }} />
              </label>
            ))}
          </div>
          <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 5 }}>
            Hammerstad–Jensen geometry · Kirschning–Jansen dispersion · Wheeler conductor loss. Solder mask, side grounds, bends and discontinuities are not included.
            Manufacturing tolerances are sampled explicitly during yield analysis; width tolerance follows Optenni msWidthToler semantics.
          </div>
        </div>
      )}
      <div style={{ fontSize: 10, fontWeight: 600, margin: '8px 0 3px' }}>Allowed physical topologies</div>
      {Object.entries(LINE_TOPOLOGY_LABELS).map(([topology, label]) => (
        <label key={topology} style={{ display: 'flex', gap: 5, alignItems: 'center', fontSize: 10, marginBottom: 2 }}>
          <input type="checkbox" checked={config.topologies.includes(topology)}
            onChange={() => toggleTopology(topology)} />
          {label}
        </label>
      ))}
      <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginTop: 9 }}>
        <span style={{ fontSize: 10, fontWeight: 600 }}>Fixed EM/VNA S2P layout blocks</span>
        <button className="btn btn-xs btn-primary" onClick={addLayoutBlock}
          disabled={!layoutFiles.length}>+ Add S2P</button>
      </div>
      {!layoutFiles.length && (
        <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 3 }}>
          Place two-port .s2p launch/via/EM files inside the configured SNP directory.
        </div>
      )}
      {layoutBlocks.map((block, index) => (
        <div key={`${block.filename}-${index}`} style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr auto auto', gap: 4, marginTop: 4, alignItems: 'center', padding: 4, border: '1px solid var(--border)', borderRadius: 4 }}>
          <select value={block.filename}
            onChange={e => updateLayoutBlock(index, 'filename', e.target.value)}
            style={{ minWidth: 0, fontSize: 9 }}>
            {layoutFiles.map(file => {
              const filename = file.filename || file;
              return <option key={filename} value={filename}>{filename}</option>;
            })}
          </select>
          <select value={block.location}
            onChange={e => updateLayoutBlock(index, 'location', e.target.value)}
            style={{ minWidth: 0, fontSize: 9 }}>
            <option value="dut_side">DUT side</option>
            <option value="connector_side">Connector side</option>
          </select>
          <select value={block.passivity_policy}
            onChange={e => updateLayoutBlock(index, 'passivity_policy', e.target.value)}
            style={{ minWidth: 0, fontSize: 9 }}>
            <option value="warn">Warn if active</option>
            <option value="reject">Reject if active</option>
          </select>
          <select value={block.reference_impedance_mode || 'native'}
            title="Keep the Touchstone reference impedance or renormalize its power-wave S parameters to the DUT/system impedance"
            onChange={e => updateLayoutBlock(index, 'reference_impedance_mode', e.target.value)}
            style={{ minWidth: 0, fontSize: 9 }}>
            <option value="native">Native Z0</option>
            <option value="system">System Z0</option>
          </select>
          <label title="Swap S11↔S22 and S21↔S12 before cascading" style={{ display: 'flex', gap: 2, alignItems: 'center', fontSize: 9, whiteSpace: 'nowrap' }}>
            <input type="checkbox" checked={Boolean(block.reverse_ports)}
              onChange={e => updateLayoutBlock(index, 'reverse_ports', e.target.checked)} />
            Flip 1↔2
          </label>
          <button className="btn btn-xs btn-danger"
            onClick={() => setLayoutBlocks(layoutBlocks.filter((_, itemIndex) => itemIndex !== index))}>×</button>
          {['left', 'right'].map(side => (
            <React.Fragment key={side}>
              <span style={{ fontSize: 9 }}>{side === 'left' ? 'Left fixture' : 'Right fixture'}</span>
              <select value={block[`${side}_fixture_filename`] || ''}
                onChange={e => updateLayoutBlock(index, `${side}_fixture_filename`, e.target.value || null)}
                style={{ gridColumn: 'span 3', minWidth: 0, fontSize: 9 }}>
                <option value="">None</option>
                {layoutFiles.map(file => {
                  const fixtureName = file.filename || file;
                  return <option key={fixtureName} value={fixtureName}>{fixtureName}</option>;
                })}
              </select>
              <label style={{ display: 'flex', gap: 2, alignItems: 'center', fontSize: 9, whiteSpace: 'nowrap' }}>
                <input type="checkbox" checked={Boolean(block[`${side}_fixture_reverse_ports`])}
                  disabled={!block[`${side}_fixture_filename`]}
                  onChange={e => updateLayoutBlock(index, `${side}_fixture_reverse_ports`, e.target.checked)} />
                Flip 1↔2
              </label>
              <span />
            </React.Fragment>
          ))}
        </div>
      ))}
      <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 5 }}>
        Z0 is shared by the line and stub in two-element networks. Electrical lengths are referenced to the center of all active bands.
      </div>
    </div>
  );
}

function TunableConfigEditor({ mdifPath, setMdifPath, configurations, setConfigurations, fixedComponents, setFixedComponents, autoSynthesize, setAutoSynthesize, hardwareKind = 'tunable', stateOptions = {}, setStateOptions = () => {}, measuredRefine = false, setMeasuredRefine = () => {}, maxInputComponents = 2, setMaxInputComponents = () => {} }) {
  const isSwitch = hardwareKind === 'switch';
  const updateConfiguration = (index, patch) => setConfigurations(
    configurations.map((item, i) => i === index ? { ...item, ...patch } : item)
  );
  const updateBand = (ci, bi, position, value) => {
    const bands = configurations[ci].bands_mhz.map((band, i) => {
      if (i !== bi) return band;
      const next = [...band];
      next[position] = value;
      return next;
    });
    updateConfiguration(ci, { bands_mhz: bands });
  };
  const updateFixed = (index, patch) => setFixedComponents(
    fixedComponents.map((item, i) => i === index ? { ...item, ...patch } : item)
  );
  return (
    <div className="card">
      <h3>{isSwitch ? 'Multi-Throw Switch MDIF' : 'MDIF Multi-State Hardware'}</h3>
      <label style={{ fontSize: 10 }}>MDIF file path
        <input value={mdifPath} onChange={e => setMdifPath(e.target.value)}
          placeholder={isSwitch ? 'C:\\models\\sp2t-or-sp3t.mdif' : 'C:\\models\\variable-capacitor.mdif'} style={{ width: '100%', fontSize: 10 }} />
      </label>
      <div style={{ fontSize: 9, color: 'var(--text-secondary)', margin: '4px 0 8px' }}>
        {isSwitch
          ? 'Each switch throw receives a synthesized series L/C branch. Leave allowed states empty for automatic state mapping.'
          : 'Each configuration can contain multiple simultaneously active bands. The same fixed network is used in every configuration.'}
      </div>
      {configurations.map((configuration, ci) => (
        <div key={ci} style={{ border: '1px solid var(--border-color)', borderRadius: 4, padding: 5, marginBottom: 5 }}>
          <div style={{ display: 'flex', gap: 4 }}>
            <input value={configuration.name} onChange={e => updateConfiguration(ci, { name: e.target.value })}
              style={{ width: 80, fontSize: 10 }} />
            <button className="btn btn-xs btn-primary" onClick={() => updateConfiguration(ci, { bands_mhz: [...configuration.bands_mhz, [1920, 2170]] })}>+ Band</button>
            <button className="btn btn-xs btn-danger" onClick={() => setConfigurations(configurations.filter((_, i) => i !== ci))}>×</button>
          </div>
          {configuration.bands_mhz.map((band, bi) => (
            <div key={bi} style={{ display: 'flex', gap: 3, marginTop: 3, alignItems: 'center' }}>
              <input type="number" value={band[0]} onChange={e => updateBand(ci, bi, 0, +e.target.value)} style={{ width: 62, fontSize: 10 }} />
              <span style={{ fontSize: 9 }}>–</span>
              <input type="number" value={band[1]} onChange={e => updateBand(ci, bi, 1, +e.target.value)} style={{ width: 62, fontSize: 10 }} />
              <span style={{ fontSize: 9 }}>MHz</span>
              <button className="btn btn-xs btn-danger" onClick={() => updateConfiguration(ci, { bands_mhz: configuration.bands_mhz.filter((_, i) => i !== bi) })}>×</button>
            </div>
          ))}
          {isSwitch && (
            <input
              value={(stateOptions[configuration.name] || []).join(', ')}
              onChange={e => {
                const values = e.target.value.split(',').map(value => value.trim()).filter(Boolean);
                setStateOptions({ ...stateOptions, [configuration.name]: values });
              }}
              placeholder="Allowed states, e.g. 100 (optional)"
              style={{ width: '100%', fontSize: 10, marginTop: 4 }}
            />
          )}
        </div>
      ))}
      <button className="btn btn-xs btn-primary" onClick={() => setConfigurations([...configurations, { name: `Set ${configurations.length + 1}`, bands_mhz: [[700, 960]], weight: 1 }])}>+ Configuration</button>
      {isSwitch ? (
        <>
          <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 7 }}>
            Automatically enumerates every branch L/C combination, optimizes continuous values, and selects the best allowed state per configuration. The current input synthesis block is one series inductor.
          </div>
          <label style={{ display: 'flex', gap: 5, alignItems: 'center', marginTop: 7, fontSize: 10 }}>
            <input type="checkbox" checked={measuredRefine} onChange={e => setMeasuredRefine(e.target.checked)} />
            Refine with measured 0402CS / GJM15 S2P parts
          </label>
          <label style={{ display: 'block', marginTop: 6, fontSize: 10 }}>Maximum shared input components
            <select value={maxInputComponents} onChange={e => setMaxInputComponents(+e.target.value)} style={{ marginLeft: 6, fontSize: 10 }}>
              <option value={0}>0</option><option value={1}>1</option><option value={2}>2</option>
            </select>
          </label>
        </>
      ) : <>
      <label style={{ display: 'flex', gap: 5, alignItems: 'center', marginTop: 8, fontSize: 10 }}>
        <input type="checkbox" checked={autoSynthesize} onChange={e => setAutoSynthesize(e.target.checked)} />
        Automatically synthesize fixed topology and measured parts
      </label>
      {autoSynthesize && <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 3 }}>
        Jointly searches topology, vendor S2P parts and the best MDIF state for every configuration.
      </div>}
      <fieldset disabled={autoSynthesize} style={{ border: 0, padding: 0, margin: 0, opacity: autoSynthesize ? 0.45 : 1 }}>
      <div style={{ fontSize: 10, fontWeight: 600, marginTop: 8 }}>Shared fixed network (DUT outward)</div>
      {fixedComponents.map((component, index) => (
        <div key={index} style={{ display: 'flex', gap: 3, marginTop: 3 }}>
          <select value={component.connection} onChange={e => updateFixed(index, { connection: e.target.value })} style={{ fontSize: 10 }}>
            <option value="series">Series</option><option value="shunt">Shunt</option>
          </select>
          <select value={component.kind} onChange={e => updateFixed(index, { kind: e.target.value })} style={{ fontSize: 10 }}>
            <option value="C">C (pF)</option><option value="L">L (nH)</option>
          </select>
          <input type="number" step="0.1" value={component.value} onChange={e => updateFixed(index, { value: +e.target.value })} style={{ width: 58, fontSize: 10 }} />
          <button className="btn btn-xs btn-danger" onClick={() => setFixedComponents(fixedComponents.filter((_, i) => i !== index))}>×</button>
        </div>
      ))}
      <button className="btn btn-xs btn-primary" style={{ marginTop: 4 }} onClick={() => setFixedComponents([...fixedComponents, { connection: 'series', kind: 'C', value: 1 }])}>+ Fixed part</button>
      </fieldset>
      </>}
    </div>
  );
}

function IsolationTargetEditor({ targets, setTargets, enabledPorts }) {
  if (enabledPorts.length < 2) return null;
  const portIndices = enabledPorts.map(port => port.port_index);
  const addTarget = () => setTargets([...targets, {
    source_port: portIndices[0],
    destination_port: portIndices[1],
    band_mhz: enabledPorts[0].bands_mhz?.[0] || [2400, 2500],
    maximum_db: -20,
    weight: 1,
    average_weight: 0,
  }]);
  const updateTarget = (index, patch) => setTargets(
    targets.map((target, i) => i === index ? { ...target, ...patch } : target)
  );
  const updateBand = (index, position, value) => {
    const current = targets[index].band_mhz || [2400, 2500];
    const next = [...current];
    next[position] = value;
    updateTarget(index, { band_mhz: next });
  };

  return (
    <div className="card">
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 6 }}>
        <h3 style={{ margin: 0 }}>4. Isolation Constraints</h3>
        <button className="btn btn-xs btn-primary" style={{ marginLeft: 'auto' }} onClick={addTarget}>+ Add Sij</button>
      </div>
      {targets.length === 0 && (
        <div style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
          Optional hard limits for directed transmission, e.g. S21 ≤ -20 dB.
        </div>
      )}
      {targets.map((target, index) => (
        <div key={index} style={{ padding: 6, marginBottom: 6, border: '1px solid var(--border-color)', borderRadius: 4 }}>
          <div style={{ display: 'flex', gap: 4, alignItems: 'center', marginBottom: 5 }}>
            <span style={{ fontSize: 10 }}>Drive</span>
            <select value={target.source_port} onChange={e => updateTarget(index, { source_port: +e.target.value })} style={{ fontSize: 10 }}>
              {portIndices.map(port => <option key={port} value={port}>P{port + 1}</option>)}
            </select>
            <span style={{ fontSize: 10 }}>→ Receive</span>
            <select value={target.destination_port} onChange={e => updateTarget(index, { destination_port: +e.target.value })} style={{ fontSize: 10 }}>
              {portIndices.map(port => <option key={port} value={port}>P{port + 1}</option>)}
            </select>
            <button className="btn btn-xs btn-danger" style={{ marginLeft: 'auto' }} onClick={() => setTargets(targets.filter((_, i) => i !== index))}>×</button>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 4 }}>
            <label style={{ fontSize: 9 }}>Start MHz
              <input type="number" value={target.band_mhz?.[0] ?? 2400} onChange={e => updateBand(index, 0, +e.target.value)} style={{ width: '100%', fontSize: 10 }} />
            </label>
            <label style={{ fontSize: 9 }}>Stop MHz
              <input type="number" value={target.band_mhz?.[1] ?? 2500} onChange={e => updateBand(index, 1, +e.target.value)} style={{ width: '100%', fontSize: 10 }} />
            </label>
            <label style={{ fontSize: 9 }}>Maximum dB
              <input type="number" value={target.maximum_db ?? -20} onChange={e => updateTarget(index, { maximum_db: +e.target.value })} style={{ width: '100%', fontSize: 10 }} />
            </label>
          </div>
          <div style={{ marginTop: 3, fontSize: 9, color: 'var(--text-secondary)' }}>
            S{Number(target.destination_port) + 1}{Number(target.source_port) + 1} = b{Number(target.destination_port) + 1}/a{Number(target.source_port) + 1}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── Sortable Results Table ── */

function SortHeader({ label, sortKey, currentSort, onSort, unit }) {
  const isActive = currentSort.key === sortKey;
  const arrow = isActive ? (currentSort.dir === 'asc' ? ' ▲' : ' ▼') : '';
  return (
    <th onClick={() => onSort(sortKey)}
      style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap', fontSize: 11 }}
      title={`Sort by ${label}`}>
      {label}{unit && <span style={{ fontSize: 9 }}> ({unit})</span>}{arrow}
    </th>
  );
}

function scoreColor(s) { return s > 0.7 ? 'var(--accent-green)' : s > 0.4 ? 'var(--accent-yellow)' : 'var(--accent-red)'; }
function effColor(e) { return e > 70 ? 'var(--accent-green)' : e > 40 ? 'var(--accent-yellow)' : 'var(--accent-red)'; }
function s11Color(d) { return Math.abs(d) > 15 ? 'var(--accent-green)' : Math.abs(d) > 8 ? 'var(--accent-yellow)' : 'var(--accent-red)'; }
function scoreIsDb(basis) { return String(basis || '').startsWith('rfmatch_core_physical') || String(basis || '').includes('mdif'); }

function solutionComponents(solution) {
  return Object.entries(solution?.per_port || {}).flatMap(([port, metrics]) =>
    (metrics.components || []).map((component, index) => ({
      ...component,
      port: Number(port),
      position: component.position ?? index,
    })),
  );
}

function candidateYield(solution) {
  const analysis = solution?.search_diagnostics?.yield_analysis;
  if (analysis?.yield_fraction == null) return null;
  return {
    fraction: Number(analysis.yield_fraction),
    interval: (analysis.yield_confidence_interval || []).map(Number),
  };
}

function worstBandSummary(solution, portConfigs = []) {
  let worst = null;
  Object.entries(solution?.per_port || {}).forEach(([portKey, metrics]) => {
    const port = Number(portKey);
    const frequencies = metrics.band_freqs_hz || [];
    const efficiencies = metrics.band_total_eff || [];
    if (!frequencies.length || frequencies.length !== efficiencies.length) return;
    const configuredBands = portConfigs.find(item => Number(item.port_index) === port)?.bands_mhz || [];
    const bands = configuredBands.length
      ? configuredBands
      : [[Math.min(...frequencies) / 1e6, Math.max(...frequencies) / 1e6]];
    bands.forEach(([startMhz, stopMhz]) => {
      const values = efficiencies.filter((_, index) => {
        const mhz = Number(frequencies[index]) / 1e6;
        return mhz >= Number(startMhz) - 1e-6 && mhz <= Number(stopMhz) + 1e-6;
      }).map(Number).filter(Number.isFinite);
      if (!values.length) return;
      const minimum = Math.min(...values);
      if (!worst || minimum < worst.minimum) {
        worst = { port, startMhz: Number(startMhz), stopMhz: Number(stopMhz), minimum };
      }
    });
  });
  return worst;
}

function bomDifference(solution, baseline) {
  const signature = item => [
    item.port,
    item.position,
    item.connection_type || item.connection || '',
    item.part_number || item.part || `${item.type || ''}:${item.value || ''}`,
  ].join('|');
  const counts = items => items.reduce((result, item) => {
    const key = signature(item);
    result.set(key, (result.get(key) || 0) + 1);
    return result;
  }, new Map());
  const current = counts(solutionComponents(solution));
  const reference = counts(solutionComponents(baseline));
  let added = 0;
  let removed = 0;
  new Set([...current.keys(), ...reference.keys()]).forEach(key => {
    const delta = (current.get(key) || 0) - (reference.get(key) || 0);
    if (delta > 0) added += delta;
    if (delta < 0) removed -= delta;
  });
  return { added, removed, changed: Math.max(added, removed) };
}

function priorityWeightLabel(solution) {
  const diagnostics = solution?.search_diagnostics || {};
  if (diagnostics.priority_weights_by_port) {
    return Object.entries(diagnostics.priority_weights_by_port).map(([port, values]) =>
      `P${Number(port) + 1}: ${values.effective_band_weights?.join(' / ') || '—'}`
    ).join(' · ');
  }
  const values = diagnostics.priority_weights;
  return values?.effective_band_weights?.length
    ? `P${Number(solution?.port_indices?.[0] ?? 0) + 1}: ${values.effective_band_weights.join(' / ')}`
    : '默认 1.0';
}

function ResultsTable({ solutions, onSelectSolution, selectedIndex, comparisonIndices = [], onToggleComparison }) {
  // Preserve the backend's constraint-aware professional ranking by default.
  // Users can still opt into a metric-only sort by clicking a column header.
  const [sort, setSort] = useState({ key: 'index', dir: 'asc' });

  if (!solutions || solutions.length === 0) return null;

  const rows = solutions.map((s, i) => ({
    index: i,
    score: s.system_score || s.balanced_score || s.efficiency_score || 0,
    avg_eff: (s.avg_total_efficiency != null ? s.avg_total_efficiency :
              s.avg_system_efficiency != null ? s.avg_system_efficiency :
              s.avg_efficiency != null ? s.avg_efficiency : 0) * 100,
    min_eff: (s.min_total_efficiency != null ? s.min_total_efficiency :
              s.min_system_efficiency != null ? s.min_system_efficiency :
              s.min_efficiency != null ? s.min_efficiency : 0) * 100,
    coupling: (s.avg_coupling_loss != null ? s.avg_coupling_loss :
               s.max_coupling_loss != null ? s.max_coupling_loss : null) * 100,
    comp_loss: (s.total_component_loss != null ? s.total_component_loss :
                s.component_loss_total != null ? s.component_loss_total : null) * 100,
    comp_count: s.total_component_count || s.component_count || 0,
    isolation_count: s.isolation_targets?.length || 0,
    isolation_passed: s.isolation_constraints_passed !== false,
    isolation_penalty: s.isolation_penalty_db || 0,
    score_db: scoreIsDb(s.efficiency_basis),
  }));

  const sortedRows = [...rows].sort((a, b) => {
    const va = a[sort.key] ?? 0;
    const vb = b[sort.key] ?? 0;
    return sort.dir === 'desc' ? vb - va : va - vb;
  });

  const handleSort = (key) => {
    setSort(prev => ({ key, dir: prev.key === key && prev.dir === 'desc' ? 'asc' : 'desc' }));
  };

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ minWidth: 500, fontSize: 12 }}>
        <thead>
          <tr>
            <th title="最多选择三个候选方案并排比较">对比</th>
            <th>#</th>
            <SortHeader label="Score" sortKey="score" currentSort={sort} onSort={handleSort} />
            <SortHeader label="Avg η" sortKey="avg_eff" currentSort={sort} onSort={handleSort} unit="%" />
            <SortHeader label="Min η" sortKey="min_eff" currentSort={sort} onSort={handleSort} unit="%" />
            <SortHeader label="Coupling" sortKey="coupling" currentSort={sort} onSort={handleSort} unit="%" />
            <SortHeader label="Comp.Loss" sortKey="comp_loss" currentSort={sort} onSort={handleSort} unit="%" />
            <SortHeader label="#Comps" sortKey="comp_count" currentSort={sort} onSort={handleSort} />
            <SortHeader label="Isolation" sortKey="isolation_penalty" currentSort={sort} onSort={handleSort} unit="dB viol." />
          </tr>
        </thead>
        <tbody>
          {sortedRows.map((r) => (
            <tr key={r.index}
              className={`clickable-row ${selectedIndex === r.index ? 'selected' : ''}`}
              onClick={() => onSelectSolution(r.index)}
              style={{ opacity: r.index < 5 ? 1 : 0.7 }}
            >
              <td onClick={event => event.stopPropagation()}>
                <input type="checkbox" aria-label={`对比方案 ${r.index + 1}`} checked={comparisonIndices.includes(r.index)} onChange={() => onToggleComparison?.(r.index)} />
              </td>
              <td style={{ fontSize: 10, color: 'var(--text-secondary)' }}>{r.index + 1}</td>
              <td style={{ fontWeight: 600, fontFamily: 'monospace', color: r.score_db ? effColor(r.avg_eff) : scoreColor(r.score) }}>
                {r.score_db ? `${r.score.toFixed(2)} dB` : `${(r.score * 100).toFixed(1)}%`}
              </td>
              <td style={{ fontFamily: 'monospace', color: effColor(r.avg_eff) }}>
                {r.avg_eff.toFixed(1)}
              </td>
              <td style={{ fontFamily: 'monospace', color: effColor(r.min_eff) }}>
                {r.min_eff.toFixed(1)}
              </td>
              <td style={{ fontFamily: 'monospace', color: (r.coupling || 0) > 20 ? 'var(--accent-red)' : 'var(--text-secondary)' }}>
                {r.coupling != null ? r.coupling.toFixed(1) : '-'}
              </td>
              <td style={{ fontFamily: 'monospace', color: (r.comp_loss || 0) > 10 ? 'var(--accent-red)' : 'var(--text-secondary)' }}>
                {r.comp_loss != null ? r.comp_loss.toFixed(2) : '-'}
              </td>
              <td style={{ textAlign: 'center' }}>{r.comp_count}</td>
              <td style={{ fontFamily: 'monospace', color: !r.isolation_count ? 'var(--text-secondary)' : r.isolation_passed ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                {!r.isolation_count ? '—' : r.isolation_passed ? 'PASS' : `+${r.isolation_penalty.toFixed(2)}`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SolutionComparison({ solutions, indices, selectedIndex, onSelect, portConfigs = [] }) {
  const candidates = indices.map(index => ({ index, solution: solutions[index] })).filter(item => item.solution);
  if (!candidates.length) return null;
  const baseline = solutions[0] || {};
  const metric = (solution, ...keys) => {
    const key = keys.find(name => solution[name] != null);
    return key ? Number(solution[key]) : 0;
  };
  const score = solution => metric(solution, 'system_score', 'balanced_score', 'efficiency_score');
  const average = solution => 100 * metric(solution, 'avg_total_efficiency', 'avg_system_efficiency', 'avg_efficiency');
  const minimum = solution => 100 * metric(solution, 'min_total_efficiency', 'min_system_efficiency', 'min_efficiency');
  const loss = solution => 100 * metric(solution, 'total_component_loss', 'component_loss_total');
  const bom = solution => solutionComponents(solution);

  return (
    <div className="comparison-panel">
      <div className="comparison-heading"><div><span className="eyebrow">CANDIDATE TRADE-OFF</span><h3>候选方案对比</h3></div><span>最多同时比较 3 个方案</span></div>
      <div className="comparison-grid">
        {candidates.map(({ index, solution }) => {
          const parts = bom(solution);
          const scoreDb = scoreIsDb(solution.efficiency_basis);
          const scoreDelta = score(solution) - score(baseline);
          const yieldSummary = candidateYield(solution);
          const worstBand = worstBandSummary(solution, portConfigs);
          const bomDelta = bomDifference(solution, baseline);
          return <button key={index} className={`comparison-card ${selectedIndex === index ? 'selected' : ''}`} onClick={() => onSelect(index)}>
            <div className="comparison-card-head"><b>方案 #{index + 1}</b><span>{selectedIndex === index ? '当前方案' : '设为当前'}</span></div>
            <div className="comparison-score"><strong>{scoreDb ? `${score(solution).toFixed(2)} dB` : `${(score(solution) * 100).toFixed(1)}%`}</strong><small>{index === 0 ? '排名基准' : `${scoreDelta >= 0 ? '+' : ''}${scoreDb ? scoreDelta.toFixed(2) + ' dB' : (scoreDelta * 100).toFixed(1) + '%'}`}</small></div>
            <div className="comparison-metrics"><span>平均效率<b>{average(solution).toFixed(1)}%</b></span><span>最差点<b>{minimum(solution).toFixed(1)}%</b></span><span>元件损耗<b>{loss(solution).toFixed(2)}%</b></span><span>元件数<b>{parts.length}</b></span></div>
            <div className="comparison-decision-row">
              <span><small>制造良率</small><b className={yieldSummary ? (yieldSummary.fraction >= 0.9 ? 'good' : 'warn') : ''}>{yieldSummary ? `${(yieldSummary.fraction * 100).toFixed(1)}%` : '未计算'}</b></span>
              <span title={worstBand ? `端口 ${worstBand.port + 1}，${worstBand.startMhz}–${worstBand.stopMhz} MHz` : '该候选没有可用的带内曲线'}><small>最差频带</small><b>{worstBand ? `P${worstBand.port + 1} · ${(worstBand.minimum * 100).toFixed(1)}%` : '—'}</b></span>
              <span><small>BOM 差异</small><b>{index === 0 ? '排名基准' : `换 ${bomDelta.changed} · +${bomDelta.added}/−${bomDelta.removed}`}</b></span>
            </div>
            <div className="comparison-topology"><span>{solution.search_diagnostics?.topology_code || solution.topology_code || '自动拓扑'}</span>{parts.map((part, partIndex) => <em key={partIndex}>{part.part_number || part.part || part.value || 'ideal'}</em>)}</div>
          </button>;
        })}
      </div>
    </div>
  );
}

/* ── Power Balance Bar Chart ── */

function PowerBalanceBar({ powerBalance, chartData }) {
  if (!powerBalance && !chartData) return null;

  // Use chart_data if available, else convert power_balance
  const data = chartData || (powerBalance?.per_port
    ? Object.entries(powerBalance.per_port).map(([k, v]) => ({
      port: `Port ${parseInt(k) + 1}`,
      port_index: parseInt(k),
      reflected: (v.reflected || 0) * 100,
      coupled: (v.coupled || 0) * 100,
      component_loss: (v.component_loss || 0) * 100,
      antenna_loss: (v.antenna_loss || 0) * 100,
      radiated: (v.radiated || 0) * 100,
    }))
    : []);

  if (data.length === 0) return null;

  return (
    <div>
      {data.map((d) => (
        <div key={d.port_index} style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 2 }}>{d.port}</div>
          <div style={{ height: 22, display: 'flex', borderRadius: 4, overflow: 'hidden', fontSize: 9 }}>
            <div style={{ width: `${d.reflected}%`, background: '#e74c3c', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', minWidth: d.reflected > 5 ? 0 : undefined }}
              title={`Reflected: ${d.reflected.toFixed(1)}%`}>
              {d.reflected > 8 ? `${d.reflected.toFixed(0)}%` : ''}
            </div>
            <div style={{ width: `${d.coupled}%`, background: '#f39c12', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff' }}
              title={`Coupled: ${d.coupled.toFixed(1)}%`}>
              {d.coupled > 8 ? `${d.coupled.toFixed(0)}%` : ''}
            </div>
            <div style={{ width: `${d.component_loss}%`, background: '#9b59b6', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff' }}
              title={`Comp Loss: ${d.component_loss.toFixed(1)}%`}>
              {d.component_loss > 8 ? `${d.component_loss.toFixed(0)}%` : ''}
            </div>
            <div style={{ width: `${d.antenna_loss}%`, background: '#95a5a6', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff' }}
              title={`Antenna Loss: ${d.antenna_loss.toFixed(1)}%`}>
              {d.antenna_loss > 8 ? `${d.antenna_loss.toFixed(0)}%` : ''}
            </div>
            <div style={{ width: `${d.radiated}%`, background: '#2ecc71', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff' }}
              title={`Radiated: ${d.radiated.toFixed(1)}%`}>
              {d.radiated > 8 ? `${d.radiated.toFixed(0)}%` : ''}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, fontSize: 9, color: 'var(--text-secondary)', marginTop: 2 }}>
            <span>η_total: <strong style={{ color: effColor(d.radiated) }}>{d.radiated.toFixed(1)}%</strong></span>
            <span>Γ: {d.reflected.toFixed(1)}%</span>
            <span>Cpl: {d.coupled.toFixed(1)}%</span>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── Efficiency Chart (simple SVG sparkline) ── */

function TargetBandAreas({ bandsMhz = [], xScale, minF, maxF, top, height }) {
  return bandsMhz.map((band, index) => {
    const start = Math.max(minF, Number(band[0]) * 1e6);
    const stop = Math.min(maxF, Number(band[1]) * 1e6);
    if (!Number.isFinite(start) || !Number.isFinite(stop) || stop <= start) return null;
    return <rect key={`${band[0]}-${band[1]}-${index}`} x={xScale(start)} y={top}
      width={Math.max(1, xScale(stop) - xScale(start))} height={height}
      fill="rgba(18,104,196,.10)" stroke="rgba(18,104,196,.35)" strokeWidth="0.5" />;
  });
}

function EfficiencyChart({ sweepData, sweepsByPort, bandsMhz = [] }) {
  // Multi-port mode
  if (sweepsByPort && Object.keys(sweepsByPort).length > 0) {
    return <MultiPortEffChart sweepsByPort={sweepsByPort} bandsMhz={bandsMhz} />;
  }
  if (!sweepData?.efficiency) return null;
  const { frequencies, efficiency } = sweepData;
  const { total_pct } = efficiency;
  if (!frequencies || !total_pct || frequencies.length < 2) return null;

  const w = 400, h = 120;
  const pad = { top: 8, right: 8, bottom: 20, left: 35 };
  const iw = w - pad.left - pad.right, ih = h - pad.top - pad.bottom;

  const minF = frequencies[0], maxF = frequencies[frequencies.length - 1];
  const maxE = Math.max(...total_pct) * 1.1;

  const xScale = (f) => ((f - minF) / (maxF - minF)) * iw + pad.left;
  const yScale = (e) => h - pad.bottom - (e / maxE) * ih;

  const line = total_pct.map((e, i) => `${i === 0 ? 'M' : 'L'}${xScale(frequencies[i])},${yScale(e)}`).join(' ');

  return (
    <div>
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>总效率随频率变化</div>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', maxHeight: h }}>
        <TargetBandAreas bandsMhz={bandsMhz} xScale={xScale} minF={minF} maxF={maxF} top={pad.top} height={ih} />
        <path d={line} fill="none" stroke="#2ecc71" strokeWidth="2" />
        {frequencies.filter((_, i) => i % Math.max(1, Math.floor(frequencies.length / 10)) === 0).map((f, i) => (
          <text key={i} x={xScale(f)} y={h - 2} textAnchor="middle" fontSize={8} fill="var(--text-secondary)">
            {(f / 1e9).toFixed(1)}
          </text>
        ))}
        {[0, 25, 50, 75, 100].filter(v => v < maxE).map(v => (
          <text key={v} x={2} y={yScale(v) + 3} fontSize={8} fill="var(--text-secondary)">
            {v}%
          </text>
        ))}
      </svg>
      <div style={{ fontSize: 9, color: 'var(--text-secondary)', textAlign: 'center' }}>频率 (GHz)</div>
    </div>
  );
}

/* ── S11 Chart (simple SVG) ── */

function S11Chart({ sweepData, sweepsByPort, bandsMhz = [] }) {
  // Multi-port mode
  if (sweepsByPort && Object.keys(sweepsByPort).length > 0) {
    return <MultiPortS11Chart sweepsByPort={sweepsByPort} bandsMhz={bandsMhz} />;
  }
  if (!sweepData?.s11_db) return null;
  const { frequencies, s11_db, raw_db } = sweepData;
  if (!frequencies || frequencies.length < 2) return null;

  const w = 400, h = 120;
  const pad = { top: 8, right: 8, bottom: 20, left: 35 };
  const iw = w - pad.left - pad.right, ih = h - pad.top - pad.bottom;

  const minF = frequencies[0], maxF = frequencies[frequencies.length - 1];
  const allDb = [...s11_db, ...(raw_db || [])].filter(d => d != null && isFinite(d));
  const minDb = Math.min(-30, ...allDb.map(d => -Math.abs(d)));
  const maxDb = 0;

  const xScale = (f) => ((f - minF) / (maxF - minF)) * iw + pad.left;
  const yScale = (d) => h - pad.bottom - ((d - minDb) / (maxDb - minDb)) * ih;

  const matchLine = s11_db.map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(frequencies[i])},${yScale(-Math.abs(d))}`).join(' ');
  const rawLine = raw_db ? raw_db.map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(frequencies[i])},${yScale(-Math.abs(d))}`).join(' ') : '';

  return (
    <div>
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>回波损耗随频率变化</div>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', maxHeight: h }}>
        <TargetBandAreas bandsMhz={bandsMhz} xScale={xScale} minF={minF} maxF={maxF} top={pad.top} height={ih} />
        {rawLine && <path d={rawLine} fill="none" stroke="#ccc" strokeWidth="1" strokeDasharray="4,3" />}
        <path d={matchLine} fill="none" stroke="#e74c3c" strokeWidth="2" />
        {/* -10 dB reference line */}
        <line x1={pad.left} y1={yScale(-10)} x2={w - pad.right} y2={yScale(-10)} stroke="#f39c12" strokeWidth="0.5" strokeDasharray="3,3" />
        <text x={w - pad.right - 2} y={yScale(-10) - 2} textAnchor="end" fontSize={7} fill="#f39c12">-10dB</text>
        {frequencies.filter((_, i) => i % Math.max(1, Math.floor(frequencies.length / 10)) === 0).map((f, i) => (
          <text key={i} x={xScale(f)} y={h - 2} textAnchor="middle" fontSize={8} fill="var(--text-secondary)">
            {(f / 1e9).toFixed(1)}
          </text>
        ))}
        {[-5, -10, -15, -20, -30].filter(v => v > minDb).map(v => (
          <text key={v} x={2} y={yScale(v) + 3} fontSize={8} fill="var(--text-secondary)">
            {v}
          </text>
        ))}
      </svg>
      <div style={{ fontSize: 9, color: 'var(--text-secondary)', textAlign: 'center' }}>
        频率 (GHz) <span style={{ marginLeft: 8 }}>— 匹配后 <span style={{ color: '#e74c3c' }}>●</span></span>
        {raw_db && <span style={{ marginLeft: 8 }}>— Raw <span style={{ color: '#ccc' }}>┈</span></span>}
      </div>
    </div>
  );
}

/* ── Multi-port chart helpers ── */

function MultiPortS11Chart({ sweepsByPort, bandsMhz = [] }) {
  const entries = Object.entries(sweepsByPort).sort(([a], [b]) => Number(a) - Number(b));
  const first = entries[0][1];
  if (!first?.frequencies || first.frequencies.length < 2) return null;

  const w = 400, h = 150;
  const pad = { top: 12, right: 8, bottom: 24, left: 35 };
  const iw = w - pad.left - pad.right, ih = h - pad.top - pad.bottom;

  const allFreqs = entries.flatMap(([, d]) => d.frequencies || []).filter(f => f != null && isFinite(f));
  const minF = Math.min(...allFreqs), maxF = Math.max(...allFreqs);
  const allDb = entries.flatMap(([, d]) => [...(d.s11_db || []), ...(d.raw_db || [])]).filter(d => d != null && isFinite(d));
  const minDb = Math.min(-30, ...allDb.map(d => -Math.abs(d)));
  const maxDb = 0;

  const xScale = (f) => ((f - minF) / (maxF - minF)) * iw + pad.left;
  const yScale = (d) => h - pad.bottom - ((d - minDb) / (maxDb - minDb)) * ih;

  return (
    <div>
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>回波损耗 · 全端口</div>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', maxHeight: h }}>
        <TargetBandAreas bandsMhz={bandsMhz} xScale={xScale} minF={minF} maxF={maxF} top={pad.top} height={ih} />
        {/* -10 dB reference */}
        <line x1={pad.left} y1={yScale(-10)} x2={w - pad.right} y2={yScale(-10)} stroke="#f39c12" strokeWidth="0.5" strokeDasharray="3,3" />
        <text x={w - pad.right - 2} y={yScale(-10) - 2} textAnchor="end" fontSize={7} fill="#f39c12">-10dB</text>
        {entries.map(([piStr, d], idx) => {
          const color = COLORS[Number(piStr) % COLORS.length];
          const matchLine = d.s11_db?.map((v, i) => `${i === 0 ? 'M' : 'L'}${xScale(d.frequencies[i])},${yScale(-Math.abs(v))}`).join(' ') || '';
          return <path key={`s11-${piStr}`} d={matchLine} fill="none" stroke={color} strokeWidth="2" />;
        })}
        {allFreqs.filter((_, i) => i % Math.max(1, Math.floor(allFreqs.length / 10)) === 0).map((f, i) => (
          <text key={i} x={xScale(f)} y={h - 2} textAnchor="middle" fontSize={8} fill="var(--text-secondary)">
            {(f / 1e9).toFixed(1)}
          </text>
        ))}
        {[-5, -10, -15, -20, -30].filter(v => v > minDb).map(v => (
          <text key={v} x={2} y={yScale(v) + 3} fontSize={8} fill="var(--text-secondary)">{v}</text>
        ))}
      </svg>
      <div style={{ fontSize: 9, color: 'var(--text-secondary)', display: 'flex', flexWrap: 'wrap', gap: 6, justifyContent: 'center', marginTop: 2 }}>
        {entries.map(([piStr], idx) => (
          <span key={piStr}>
            <span style={{ color: COLORS[Number(piStr) % COLORS.length] }}>●</span> Port {Number(piStr) + 1}
          </span>
        ))}
      </div>
      <div style={{ fontSize: 9, color: 'var(--text-secondary)', textAlign: 'center' }}>频率 (GHz)</div>
      {bandsMhz.length > 0 && <div className="target-band-legend"><i />目标频段 {bandsMhz.map(band => `${band[0]}–${band[1]} MHz`).join(' · ')}</div>}
    </div>
  );
}

function MultiPortEffChart({ sweepsByPort, bandsMhz = [] }) {
  const entries = Object.entries(sweepsByPort).sort(([a], [b]) => Number(a) - Number(b));
  const first = entries[0][1];
  if (!first?.frequencies || first.frequencies.length < 2) return null;

  const w = 400, h = 150;
  const pad = { top: 12, right: 8, bottom: 24, left: 35 };
  const iw = w - pad.left - pad.right, ih = h - pad.top - pad.bottom;

  const allFreqs = entries.flatMap(([, d]) => d.frequencies || []).filter(f => f != null && isFinite(f));
  const minF = Math.min(...allFreqs), maxF = Math.max(...allFreqs);
  const maxE = Math.max(100, Math.max(...entries.flatMap(([, d]) => d.efficiency?.total_pct || [])) * 1.1);

  const xScale = (f) => ((f - minF) / (maxF - minF)) * iw + pad.left;
  const yScale = (e) => h - pad.bottom - (e / maxE) * ih;

  return (
    <div>
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>总效率 · 全端口</div>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', maxHeight: h }}>
        <TargetBandAreas bandsMhz={bandsMhz} xScale={xScale} minF={minF} maxF={maxF} top={pad.top} height={ih} />
        {entries.map(([piStr, d], idx) => {
          const color = COLORS[Number(piStr) % COLORS.length];
          const line = d.efficiency?.total_pct?.map((e, i) => `${i === 0 ? 'M' : 'L'}${xScale(d.frequencies[i])},${yScale(e)}`).join(' ') || '';
          return <path key={`eff-${piStr}`} d={line} fill="none" stroke={color} strokeWidth="2" />;
        })}
        {allFreqs.filter((_, i) => i % Math.max(1, Math.floor(allFreqs.length / 10)) === 0).map((f, i) => (
          <text key={i} x={xScale(f)} y={h - 2} textAnchor="middle" fontSize={8} fill="var(--text-secondary)">
            {(f / 1e9).toFixed(1)}
          </text>
        ))}
        {[0, 25, 50, 75, 100].filter(v => v < maxE).map(v => (
          <text key={v} x={2} y={yScale(v) + 3} fontSize={8} fill="var(--text-secondary)">{v}%</text>
        ))}
      </svg>
      <div style={{ fontSize: 9, color: 'var(--text-secondary)', display: 'flex', flexWrap: 'wrap', gap: 6, justifyContent: 'center', marginTop: 2 }}>
        {entries.map(([piStr], idx) => (
          <span key={piStr}>
            <span style={{ color: COLORS[Number(piStr) % COLORS.length] }}>●</span> Port {Number(piStr) + 1}
          </span>
        ))}
      </div>
      <div style={{ fontSize: 9, color: 'var(--text-secondary)', textAlign: 'center' }}>频率 (GHz)</div>
      {bandsMhz.length > 0 && <div className="target-band-legend"><i />目标频段 {bandsMhz.map(band => `${band[0]}–${band[1]} MHz`).join(' · ')}</div>}
    </div>
  );
}

/* ── Solution detail components for a selected solution ── */

function SolutionDetails({ solution, portIndex, portConfigs = [], selectedSeries = null, componentFilter = {} }) {
  const [alternativeAnalysis, setAlternativeAnalysis] = useState(null);
  const [alternativeBusy, setAlternativeBusy] = useState(false);
  const [alternativeError, setAlternativeError] = useState('');
  useEffect(() => {
    setAlternativeAnalysis(null);
    setAlternativeError('');
  }, [solution]);
  if (!solution) return null;

  const perPort = solution.per_port || {};
  const entries = Object.entries(perPort);
  if (entries.length === 0) return null;

  // Focus on selected port or show all
  const focusPorts = portIndex != null
    ? entries.filter(([k]) => parseInt(k) === portIndex)
    : entries;

  async function analyzeAlternatives(component, port) {
    const partNumber = component.part_number || component.part;
    if (!partNumber) return;
    setAlternativeBusy(true);
    setAlternativeError('');
    try {
      const config = portConfigs.find(item => item.port_index === port);
      const result = await api.componentAlternatives({
        part_number: partNumber,
        component_series: selectedSeries,
        component_filter: componentFilter,
        bands_mhz: config?.bands_mhz || [],
        num_band_points: 5,
        maximum_nominal_deviation_pct: 50,
        limit: 8,
      });
      setAlternativeAnalysis(result);
    } catch (error) {
      setAlternativeError(error.message);
    } finally {
      setAlternativeBusy(false);
    }
  }

  return (
    <div className="solution-engineering-detail">
      {focusPorts.map(([piStr, pm]) => {
        const pi = parseInt(piStr);
        const comps = pm.components || [];
        const counters = { L: 0, C: 0 };
        const bomRows = comps.slice().sort((a, b) => (a.position || 0) - (b.position || 0)).map(component => {
          const type = component.comp_type || component.type;
          const prefix = type === 'inductor' ? 'L' : 'C';
          counters[prefix] += 1;
          return { component, reference: `${prefix}${counters[prefix]}` };
        });
        return (
          <section key={pi} className="port-solution-block">
            <div className="port-solution-heading">
              <div><span style={{ background: COLORS[pi % COLORS.length] }} /><strong>端口 {pi + 1}</strong><small>{comps.length} 个匹配元件</small></div>
              <div className="port-kpis">
                <span>回波损耗<strong style={{ color: s11Color(pm.s11_db) }}>{pm.s11_db?.toFixed(1)} dB</strong></span>
                <span>失配效率<strong>{((pm.mismatch_efficiency || 0) * 100).toFixed(1)}%</strong></span>
                <span>耦合损耗<strong>{((pm.coupling_loss || 0) * 100).toFixed(1)}%</strong></span>
                <span>总效率<strong style={{ color: effColor((pm.total_efficiency || 0) * 100) }}>{((pm.total_efficiency || 0) * 100).toFixed(1)}%</strong></span>
              </div>
            </div>
            {comps.length > 0 && (
              <div className="bom-table-wrap"><table className="bom-table">
                <thead><tr><th>位号</th><th>连接方式</th><th>类型</th><th>厂商料号</th><th>标称值</th><th>厂商 / 系列</th><th>封装</th><th>容差</th><th>环境参数</th><th /></tr></thead>
                <tbody>{bomRows.map(({ component, reference }, index) => {
                  const type = component.comp_type || component.type;
                  const partNumber = component.part_number || component.part;
                  return <tr key={`${partNumber || 'ideal'}-${index}`}>
                    <td><b>{reference}</b></td>
                    <td>{component.connection_type === 'shunt' ? '并联到地' : '串联'}</td>
                    <td>{type === 'inductor' ? '电感' : '电容'}</td>
                    <td className="part-number">{partNumber || '理想元件'}</td>
                    <td>{component.value || `${component.nominal_value ?? '—'} ${component.nominal_unit || ''}`}</td>
                    <td>{[component.manufacturer, component.series].filter(Boolean).join(' / ') || '—'}</td>
                    <td>{component.package_code || '—'}</td>
                    <td title={component.metadata_provenance?.tolerance_pct ? `来源：${component.metadata_provenance.tolerance_pct}` : ''}>{component.tolerance_pct != null ? `±${Number(component.tolerance_pct).toFixed(2)}%` : '—'}</td>
                    <td title={component.environment_metadata?.source_document || '未提供料号级环境元数据'}>
                      {component.tempco_ppm_per_c != null || component.systematic_bias_pct != null
                        ? <><span>{component.tempco_ppm_per_c != null ? `${Number(component.tempco_ppm_per_c).toFixed(0)} ppm/°C` : 'TCR —'}</span><br /><small>{component.systematic_bias_pct != null ? `偏置 ${Number(component.systematic_bias_pct).toFixed(2)}%` : '偏置 —'}</small></>
                        : '全局回退'}
                    </td>
                    <td><button className="bom-action" disabled={!partNumber} onClick={() => analyzeAlternatives(component, pi)}>替代料</button></td>
                  </tr>;
                })}</tbody>
              </table></div>
            )}
          </section>
        );
      })}
      {alternativeBusy && <div className="bom-message">正在比较实测元件模型…</div>}
      {alternativeError && <div style={{ fontSize: 10, color: 'var(--accent-red)' }}>{alternativeError}</div>}
      {alternativeAnalysis && !alternativeBusy && (
        <div className="alternative-analysis">
          <h4>{alternativeAnalysis.reference.part_number} 的实测替代料</h4>
          <div className="bom-message">
            已在 {alternativeAnalysis.analysis_frequencies_hz.length} 个频点比较 {alternativeAnalysis.physically_evaluated} 个模型。S 矩阵 RMS 越低越接近，最终仍需核对厂商额定参数。
          </div>
          <table style={{ fontSize: 10 }}>
            <thead><tr><th>Part</th><th>Value</th><th>Series</th><th>Nominal Δ</th><th>S-matrix RMS</th><th>Maximum ΔS</th></tr></thead>
            <tbody>{alternativeAnalysis.alternatives.map(item => (
              <tr key={`${item.part_number}-${item.model_sha256 || item.series}`}>
                <td>{item.part_number}</td><td>{item.nominal_value} {item.nominal_unit}</td>
                <td>{item.series}</td><td>{item.nominal_deviation_pct.toFixed(2)}%</td>
                <td>{item.sparameter_rms_difference.toExponential(3)}</td>
                <td>{item.sparameter_maximum_difference.toExponential(3)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ── Main TuningPanel ── */

function normalizeComponentForManual(component, fallbackPort) {
  const type = component?.comp_type || component?.type || 'capacitor';
  const parsedValue = parseFloat(String(component?.value || '').replace(/[^\d.]/g, ''));
  const value = component?.nominal_value ?? (Number.isFinite(parsedValue) ? parsedValue : 1);
  const partNumber = component?.part_number || component?.part || '';
  return {
    comp_type: type,
    type,
    connection_type: component?.connection_type || 'series',
    value,
    nominal_value: value,
    nominal_unit: component?.nominal_unit || (type === 'inductor' ? 'nH' : 'pF'),
    port: component?.port ?? fallbackPort ?? 0,
    part_number: partNumber,
    use_ideal: component?.use_ideal ?? !partNumber,
  };
}

function componentsFromSolution(solution, portIndex) {
  const perPort = solution?.per_port || {};
  const entries = Object.entries(perPort)
    .filter(([pi]) => portIndex == null || Number(pi) === Number(portIndex));
  return Object.fromEntries(entries.map(([pi, pm]) => [
    Number(pi),
    (pm.components || []).map(c => normalizeComponentForManual(c, Number(pi))),
  ]));
}

function flattenManualComponents(componentsByPort) {
  return Object.entries(componentsByPort || {}).flatMap(([pi, comps]) =>
    (comps || []).map(c => ({
      comp_type: c.comp_type || c.type || 'capacitor',
      connection_type: c.connection_type || 'series',
      value: c.nominal_value ?? c.value ?? 1,
      port: Number(pi),
      use_ideal: c.use_ideal ?? !c.part_number,
      part_number: c.part_number || '',
    }))
  );
}

function TabButton({ id, activeTab, setActiveTab, children }) {
  return (
    <button
      className={`tab-button ${activeTab === id ? 'active' : ''}`}
      onClick={() => setActiveTab(id)}
    >
      {children}
    </button>
  );
}

export default function TuningPanel({
  loadedSNP, portConfigs, setPortConfigs,
  onRefreshSNP, snpFiles, dataDirs, setDataDirs,
  projectSnapshot, selectedSeries, componentFilter, componentCatalogReady = false,
  onOpenProjects,
}) {
  const [tuningMode, setTuningMode] = useState('fixed_lc');
  const [objective, setObjective] = useState('balanced');
  const [withinBandAverageWeight, setWithinBandAverageWeight] = useState(null);
  const [acrossBandAverageWeight, setAcrossBandAverageWeight] = useState(null);
  const [genericSynthesisLoss, setGenericSynthesisLoss] = useState({
    inductor_q: 30,
    inductor_q_reference_hz: 1e9,
    inductor_esr_ohm: 0,
    capacitor_esr_ohm: 0.4,
  });
  const [searchQuality, setSearchQuality] = useState('thorough');
  const [calibrationStatus, setCalibrationStatus] = useState(null);
  const [beamWidth, setBeamWidth] = useState(10);
  const [timeout, setTimeout_] = useState(120);
  const [bandPoints, setBandPoints] = useState(10);

  useEffect(() => {
    let active = true;
    api.getCalibrationStatus()
      .then(status => { if (active) setCalibrationStatus(status); })
      .catch(error => { if (active) setCalibrationStatus({ status: 'invalid', error: error.message }); });
    return () => { active = false; };
  }, []);
  const [isolationTargets, setIsolationTargets] = useState([]);
  const [tunerMdifPath, setTunerMdifPath] = useState('');
  const [frequencyConfigurations, setFrequencyConfigurations] = useState([
    { name: 'Set 1', bands_mhz: [[704, 746], [1920, 2170]], weight: 1 },
    { name: 'Set 2', bands_mhz: [[791, 862], [1920, 2170]], weight: 1 },
    { name: 'Set 3', bands_mhz: [[880, 960], [1920, 2170]], weight: 1 },
  ]);
  const [tunableFixedComponents, setTunableFixedComponents] = useState([
    { connection: 'series', kind: 'C', value: 2.8 },
    { connection: 'series', kind: 'L', value: 15 },
  ]);
  const [tunableAutoSynthesize, setTunableAutoSynthesize] = useState(false);
  const [switchStateOptions, setSwitchStateOptions] = useState({});
  const [switchMeasuredRefine, setSwitchMeasuredRefine] = useState(true);
  const requestedMeasuredComponents = portConfigs
    .filter(port => port.enabled)
    .some(port => Number(port.max_components ?? 2) > 0);
  const zeroComponentBareDut = tuningMode === 'fixed_lc' && !requestedMeasuredComponents;
  const componentSeriesRequired = !zeroComponentBareDut && tuningMode !== 'transmission_line' && !(
    tuningMode === 'switch' && !switchMeasuredRefine
  );
  const componentSeriesValid = selectedSeries === null || (
    tuningMode === 'fixed_lc'
      ? selectedSeries.length > 0
      : selectedSeries.some(item => item.startsWith('L::')) &&
        selectedSeries.some(item => item.startsWith('C::'))
  );
  const componentSeriesReady = !componentSeriesRequired || (
    componentCatalogReady && componentSeriesValid
  );
  const [switchMaxInputComponents, setSwitchMaxInputComponents] = useState(2);
  const [transmissionLineConfig, setTransmissionLineConfig] = useState({
    characteristic_impedance_min_ohm: 20,
    characteristic_impedance_max_ohm: 120,
    electrical_length_min_deg: 1,
    electrical_length_max_deg: 179,
    attenuation_db: 0,
    loss_frequency_exponent: 0.5,
    topologies: Object.keys(LINE_TOPOLOGY_LABELS),
    restarts: 10,
    iterations: 24,
    maximum_evaluations: 10000,
    microstrip: {
      enabled: false,
      substrate_name: 'FR-4 engineering model',
      relative_permittivity: 4.5,
      substrate_height_mm: 1.6,
      loss_tangent: 0.02,
      copper_thickness_um: 35,
      copper_resistivity_ohm_m: 1.68e-8,
      copper_roughness_rms_um: 0.15,
      minimum_width_mm: 0.1,
      maximum_width_mm: 10,
      width_tolerance_pct: 10,
      length_tolerance_pct: 0,
      substrate_height_tolerance_pct: 0,
      relative_permittivity_tolerance_pct: 0,
    },
    layout_blocks: [],
  });

  const [optimizing, setOptimizing] = useState(false);
  const [results, setResults] = useState(null);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const [error, setError] = useState(null);
  const [selectedSolution, setSelectedSolution] = useState(0);
  const [comparisonIndices, setComparisonIndices] = useState([0, 1, 2]);
  const [selectedPort, setSelectedPort] = useState(null);
  const [sweepData, setSweepData] = useState(null);
  const [sweepDataByPort, setSweepDataByPort] = useState({});
  const [activeTab, setActiveTab] = useState('summary');
  const [manualComponents, setManualComponents] = useState({});
  const [manualBusy, setManualBusy] = useState(false);
  const [manualError, setManualError] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const [activeJobId, setActiveJobId] = useState(null);
  const [jobProgress, setJobProgress] = useState(null);
  const [yieldConfig, setYieldConfig] = useState({
    samples: 200,
    minimum_total_efficiency: 0.5,
    minimum_average_total_efficiency: 0,
    minimum_return_loss_db: 6,
    default_tolerance_pct: 5,
    distribution: 'uniform',
    batch_correlation: 0,
    reference_temperature_c: 25,
    temperature_min_c: null,
    temperature_max_c: null,
    inductor_tempco_ppm_per_c: 0,
    capacitor_tempco_ppm_per_c: 0,
    inductor_bias_pct: 0,
    capacitor_bias_pct: 0,
  });
  const [yieldResult, setYieldResult] = useState(null);
  const [yieldBusy, setYieldBusy] = useState(false);
  const [yieldError, setYieldError] = useState(null);

  const timerRef = React.useRef(null);
  const manualRequestSequence = React.useRef(0);
  const manualDebounceRef = React.useRef(null);

  useEffect(() => {
    setInspectorCollapsed(Boolean(results));
    if (!results) {
      setComparisonIndices([]);
      return;
    }
    if (results.solutions?.length) {
      setComparisonIndices(current => {
        const valid = current.filter(index => index >= 0 && index < results.solutions.length).slice(0, 3);
        return valid.length ? valid : results.solutions.slice(0, 3).map((_, index) => index);
      });
    }
  }, [Boolean(results), results?.solutions?.length]);

  const toggleComparison = index => setComparisonIndices(current => {
    if (current.includes(index)) return current.length > 1 ? current.filter(value => value !== index) : current;
    return [...current, index].slice(-3);
  });

  useEffect(() => {
    if (optimizing) {
      const t0 = Date.now();
      timerRef.current = setInterval(() => setElapsed(((Date.now() - t0) / 1000).toFixed(1)), 200);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [optimizing]);

  useEffect(() => {
    if (!projectSnapshot) return;
    const request = projectSnapshot.tuning_request || {};
    const restoredSolutions = projectSnapshot.solutions || [];
    const restoredIndex = Math.min(
      projectSnapshot.selected_index || 0,
      Math.max(0, restoredSolutions.length - 1),
    );
    const requestMode = request.mode === 'tunable' ? 'tunable_c' : request.mode;
    if (requestMode && requestMode !== 'single' && requestMode !== 'joint') setTuningMode(requestMode);
    else setTuningMode('fixed_lc');
    if (request.objective) setObjective(request.objective);
    setWithinBandAverageWeight(request.within_band_average_weight ?? null);
    setAcrossBandAverageWeight(request.across_band_average_weight ?? null);
    setGenericSynthesisLoss(request.generic_synthesis_loss || {
      inductor_q: 30,
      inductor_q_reference_hz: 1e9,
      inductor_esr_ohm: 0,
      capacitor_esr_ohm: 0.4,
    });
    setSearchQuality(request.search_quality || inferSearchQuality(request.timeout_seconds || 120));
    if (request.beam_width) setBeamWidth(request.beam_width);
    if (request.timeout_seconds) setTimeout_(request.timeout_seconds);
    if (request.num_band_points) setBandPoints(request.num_band_points);
    setIsolationTargets(request.isolation_targets || []);
    if (request.tuner_mdif_path) setTunerMdifPath(request.tuner_mdif_path);
    if (request.frequency_configurations?.length) setFrequencyConfigurations(request.frequency_configurations);
    if (request.tunable_fixed_components?.length) setTunableFixedComponents(request.tunable_fixed_components);
    setTunableAutoSynthesize(Boolean(request.tunable_auto_synthesize));
    setSwitchStateOptions(request.switch_state_options || {});
    setSwitchMeasuredRefine(Boolean(request.switch_measured_refine));
    setSwitchMaxInputComponents(request.switch_max_input_components ?? 2);
    if (request.transmission_line) setTransmissionLineConfig(request.transmission_line);
    setResults({
      status: 'ok',
      restoration_mode: 'snapshot',
      solutions_count: restoredSolutions.length,
      solutions: restoredSolutions,
      best_solution: restoredSolutions[0] || null,
      best_avg_efficiency: restoredSolutions[0]?.avg_total_efficiency || 0,
      best_min_efficiency: restoredSolutions[0]?.min_total_efficiency || 0,
      best_score: restoredSolutions[0]?.system_score || 0,
      system_efficiency_pct: (restoredSolutions[0]?.system_power_balance?.system_efficiency || 0) * 100,
      total_time_s: restoredSolutions[0]?.total_time_s || 0,
      system_power_balance: restoredSolutions[0]?.system_power_balance || null,
      power_balance_chart: restoredSolutions[0]?.power_balance_chart || [],
    });
    setActiveTab('summary');
    setSelectedSolution(restoredIndex);
    const firstPort = request.ports?.find(port => port.enabled)?.port_index
      ?? restoredSolutions[restoredIndex]?.port_indices?.[0]
      ?? 0;
    setSelectedPort(firstPort);
    setManualComponents(componentsFromSolution(restoredSolutions[restoredIndex], null));
    if (restoredSolutions[restoredIndex]?.mode === 'switch') {
      loadStoredSwitchSweep(restoredSolutions[restoredIndex]);
    } else {
      setSweepData(null);
      setSweepDataByPort({});
    }
    setError(null);
    setActiveTab('summary');
  }, [projectSnapshot]);

  /* ── Run tuning ── */
  async function waitForTuningJob(started, maximumSeconds) {
    setActiveJobId(started.job_id);
    setJobProgress(started.progress);
    const deadline = Date.now() + maximumSeconds * 1000;
    let job = started;
    while (['queued', 'running', 'cancelling'].includes(job.status)) {
      if (Date.now() > deadline) throw new Error('Automatic synthesis job timed out');
      await new Promise(resolve => window.setTimeout(resolve, 500));
      job = await api.getTuningJob(started.job_id);
      setJobProgress(job.progress);
    }
    if (job.status === 'failed') throw new Error(job.error || 'Automatic synthesis failed');
    if (job.status === 'cancelled' && !job.result?.solutions?.length) {
      throw new Error('Optimization cancelled');
    }
    return job.result;
  }

  async function handleRun() {
    setOptimizing(true);
    setElapsed(0);
    setJobProgress(null);
    setResults(null);
    setSweepData(null);
    setSweepDataByPort({});
    setError(null);
    setYieldResult(null);
    setYieldError(null);

    try {
      const enabledPorts = portConfigs.filter(p => p.enabled);

      if (enabledPorts.length === 0) {
        throw new Error('At least one port must be enabled');
      }

      // Determine mode from tuningMode + port count
      let mode = 'joint';
      let extraParams = {};

      if (tuningMode === 'transmission_line') {
        if (enabledPorts.length !== 1) throw new Error('Transmission-line synthesis requires exactly one enabled port');
        mode = 'transmission_line';
        extraParams.transmission_line = transmissionLineConfig;
      } else if (tuningMode === 'grid_s2p') {
        mode = 'grid_s2p';
      } else if (tuningMode === 'tunable_c' || tuningMode === 'switch') {
        mode = tuningMode === 'tunable_c' ? 'tunable' : 'switch';
        if (tuningMode === 'tunable_c' && tunerMdifPath.trim()) {
          extraParams.tuner_mdif_path = tunerMdifPath.trim();
          extraParams.frequency_configurations = frequencyConfigurations;
          extraParams.tunable_fixed_components = tunableFixedComponents;
          extraParams.tunable_auto_synthesize = tunableAutoSynthesize;
        }
        if (tuningMode === 'switch') {
          if (!tunerMdifPath.trim()) throw new Error('Switch mode requires an MDIF file path');
          extraParams.tuner_mdif_path = tunerMdifPath.trim();
          extraParams.frequency_configurations = frequencyConfigurations;
          extraParams.switch_state_options = switchStateOptions;
          extraParams.switch_measured_refine = switchMeasuredRefine;
          extraParams.switch_max_input_components = switchMaxInputComponents;
        }
        // Generate band_state_map: each enabled port's bands become states
        // For single-port: {"Band1": [f_start, f_stop], "Band2": [...]}
        // For multi-port: {"P1_Band1": [...], "P2_Band1": [...]}
        const bsm = {};
        enabledPorts.forEach(p => {
          (p.bands_mhz || []).forEach((band, bi) => {
            const key = enabledPorts.length > 1
              ? `P${p.port_index + 1}_B${bi + 1}`
              : `B${bi + 1}`;
            bsm[key] = band;
          });
        });
        if (Object.keys(bsm).length > 0) {
          extraParams.band_state_map = bsm;
        }
      } else if (enabledPorts.length === 1) {
        mode = 'single';
      }

      const requestParams = {
        ports: enabledPorts.map(p => ({
          port_index: p.port_index,
          bands_mhz: p.bands_mhz,
          band_weights: (p.bands_mhz || []).map((_, index) => Number(p.band_weights?.[index] ?? 1)),
          port_weight: Number(p.port_weight ?? 1),
          max_components: p.max_components,
          allowed_topology_codes: p.allowed_topology_codes ?? null,
          enabled: true,
        })),
        objective: objective,
        within_band_average_weight: withinBandAverageWeight,
        across_band_average_weight: acrossBandAverageWeight,
        generic_synthesis_loss: genericSynthesisLoss,
        search_quality: searchQuality,
        mode: mode,
        ...extraParams,
        beam_width: beamWidth,
        timeout_seconds: timeout,
        num_band_points: bandPoints,
        isolation_targets: mode === 'joint' ? isolationTargets : [],
        component_series: selectedSeries,
        component_filter: componentFilter,
      };
      let res;
      if (mode === 'single' || mode === 'joint' || (mode === 'tunable' && tunableAutoSynthesize) || mode === 'switch' || mode === 'transmission_line') {
        const started = await api.startTuningJob(requestParams);
        res = await waitForTuningJob(started, timeout + 30);
      } else {
        res = await api.tuningOptimize(requestParams);
      }
      setResults(res);

      if (res.solutions?.length > 0) {
        const firstPort = enabledPorts[0].port_index;
        setSelectedPort(firstPort);
        setSelectedSolution(0);
        setManualComponents(componentsFromSolution(res.solutions[0], null));
        setActiveTab('curves');
        if (mode === 'switch') loadStoredSwitchSweep(res.solutions[0]);
        else await loadAllSweeps(0);
      }
    } catch (e) {
      if (e.message !== 'Optimization cancelled') setError(e.message);
    }
    setActiveJobId(null);
    setOptimizing(false);
  }

  async function handleCancelOptimization() {
    if (!activeJobId) return;
    try {
      const job = await api.cancelTuningJob(activeJobId);
      setJobProgress(job.progress);
    } catch (e) {
      setError(e.message);
    }
  }

  async function handleContinueOptimization() {
    const additionalSeconds = Math.max(5, Number(timeout) || 30);
    setOptimizing(true);
    setElapsed(0);
    setError(null);
    try {
      const started = await api.startTuningContinueJob(additionalSeconds);
      const res = await waitForTuningJob(started, additionalSeconds + 30);
      setResults(res);
      setSelectedSolution(0);
      const totalBudget = res.continuation?.total_timeout_seconds;
      if (totalBudget) setTimeout_(totalBudget);
      if (res.solutions?.length > 0) {
        const firstPort = portConfigs.find(port => port.enabled)?.port_index ?? 0;
        setSelectedPort(firstPort);
        setManualComponents(componentsFromSolution(res.solutions[0], null));
        setActiveTab('curves');
        await loadAllSweeps(0);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setActiveJobId(null);
      setOptimizing(false);
    }
  }

  async function loadAllSweeps(solutionIdx = 0) {
    const enabledPorts = portConfigs.filter(p => p.enabled);
    const newSweeps = {};
    let firstSweep = null;
    await Promise.all(enabledPorts.map(async (pc) => {
      try {
        const band = pc.bands_mhz[0];
        if (!band) return;
        const centerHz = (band[0] + band[1]) / 2 * 1e6;
        const points = Math.max(401, loadedSNP?.freq_count || 0);
        const res = await api.tuningSweep(pc.port_index, centerHz * 0.3, centerHz * 1.7, points, solutionIdx, true);
        newSweeps[pc.port_index] = { ...res, port_index: pc.port_index };
        if (!firstSweep) firstSweep = newSweeps[pc.port_index];
      } catch (e) {
        console.warn(`Sweep failed for port ${pc.port_index}:`, e.message);
      }
    }));
    setSweepDataByPort(newSweeps);
    if (firstSweep) setSweepData(firstSweep);
    else setSweepData(null);
  }

  function loadStoredSwitchSweep(solution) {
    const metrics = solution?.per_port?.[0] || solution?.per_port?.['0'];
    const frequencies = metrics?.band_freqs_hz || [];
    const returnLoss = metrics?.band_s11_db || [];
    const efficiency = metrics?.band_total_eff || [];
    const aggregated = new Map();
    frequencies.forEach((frequency, index) => {
      const current = aggregated.get(frequency) || { returnLoss: [], efficiency: [] };
      current.returnLoss.push(Number(returnLoss[index] || 0));
      current.efficiency.push(Number(efficiency[index] || 0));
      aggregated.set(frequency, current);
    });
    const sortedFrequencies = [...aggregated.keys()].sort((a, b) => a - b);
    const sweep = {
      frequencies: sortedFrequencies,
      s11_db: sortedFrequencies.map(frequency => Math.min(...aggregated.get(frequency).returnLoss)),
      efficiency: {
        total_pct: sortedFrequencies.map(frequency => {
          const values = aggregated.get(frequency).efficiency;
          return values.reduce((sum, value) => sum + value, 0) / values.length * 100;
        }),
      },
      source: 'stored_switch_active_bands',
    };
    setSweepData(sweep);
    setSweepDataByPort({});
  }

  const handleSelectSolution = async (idx) => {
    setSelectedSolution(idx);
    // Notify backend
    try { await api.tuningSelect(idx); } catch (e) {}
    if (results?.solutions?.[idx]) {
      const solution = results.solutions[idx];
      setResults(current => ({
        ...current,
        best_solution: solution,
        best_avg_efficiency: solution.avg_total_efficiency || 0,
        best_min_efficiency: solution.min_total_efficiency || 0,
        best_score: solution.system_score || 0,
        system_efficiency_pct: (solution.system_power_balance?.system_efficiency || 0) * 100,
        system_power_balance: solution.system_power_balance || null,
        power_balance_chart: solution.power_balance_chart || [],
      }));
      setManualComponents(componentsFromSolution(solution, null));
      if (solution.mode === 'switch') loadStoredSwitchSweep(solution);
      else await loadAllSweeps(idx);
    }
  };

  const handleSelectPort = async (portIdx) => {
    setSelectedPort(portIdx);
    const sol = results?.solutions?.[selectedSolution];
    setManualComponents(componentsFromSolution(sol, portIdx));
    await loadAllSweeps(selectedSolution);
  };

  async function handleYieldAnalysis() {
    setYieldBusy(true);
    setYieldError(null);
    try {
      const response = await api.tuningYield({
        ...yieldConfig,
        seed: 1,
        confidence_level: 0.95,
      });
      setYieldResult(response);
      const diagnosticsByIndex = new Map(
        (response.ranked_candidates || []).map(item => [item.solution_index, item]),
      );
      setResults(current => current ? ({
        ...current,
        solutions: (current.solutions || []).map((solution, index) => {
          const analysis = diagnosticsByIndex.get(index);
          return analysis ? {
            ...solution,
            search_diagnostics: { ...(solution.search_diagnostics || {}), yield_analysis: analysis },
          } : solution;
        }),
      }) : current);
    } catch (requestError) {
      setYieldError(requestError.message);
    } finally {
      setYieldBusy(false);
    }
  }

  async function evaluateManual(nextComponents) {
    const enabledPorts = portConfigs.filter(p => p.enabled);
    if (enabledPorts.length === 0) return;
    const sequence = ++manualRequestSequence.current;
    setManualBusy(true);
    setManualError(null);
    const flatComps = flattenManualComponents(nextComponents);
    const newSweeps = {};
    const failures = [];
    let firstSweep = null;
    await Promise.all(enabledPorts.map(async (pc) => {
      const band = pc.bands_mhz?.[0];
      if (!band) return;
      const centerHz = (band[0] + band[1]) / 2 * 1e6;
      try {
        const res = await api.manualTune({
          snp_filename: loadedSNP?.filename || '',
          target_frequency_hz: centerHz,
          input_port: pc.port_index,
          port_states: [],
          components: flatComps,
          sweep_start_hz: centerHz * 0.3,
          sweep_stop_hz: centerHz * 1.7,
          sweep_points: Math.max(401, loadedSNP?.freq_count || 0),
          use_snp_points: true,
        });
        if (res.sweep) {
          newSweeps[pc.port_index] = { ...res.sweep, port_index: pc.port_index, source: 'manual_realtime_full_snp' };
          if (!firstSweep) firstSweep = newSweeps[pc.port_index];
        }
      } catch (e) {
        failures.push(`端口 ${pc.port_index + 1}: ${e.message}`);
        console.warn(`Manual realtime tuning failed for port ${pc.port_index}:`, e.message);
      }
    }));
    if (sequence !== manualRequestSequence.current) return;
    setSweepDataByPort(newSweeps);
    setSweepData(firstSweep);
    if (failures.length) setManualError(failures.join(' · '));
    setManualBusy(false);
  }

  const handleManualComponentChange = (portIdx, componentIdx, patch) => {
    const next = { ...manualComponents };
    const comps = [...(next[portIdx] || [])];
    const current = comps[componentIdx] || normalizeComponentForManual({}, portIdx);
    const merged = { ...current, ...patch };
    if (patch.nominal_value != null || patch.comp_type != null || patch.type != null) {
      merged.use_ideal = true;
      merged.part_number = '';
    }
    const type = merged.comp_type || merged.type || 'capacitor';
    merged.type = type;
    merged.comp_type = type;
    merged.nominal_unit = type === 'inductor' ? 'nH' : 'pF';
    merged.value = merged.nominal_value ?? merged.value ?? 1;
    comps[componentIdx] = merged;
    next[portIdx] = comps;
    setManualComponents(next);
    window.clearTimeout(manualDebounceRef.current);
    manualDebounceRef.current = window.setTimeout(() => evaluateManual(next), 250);
  };

  const numEnabled = portConfigs.filter(p => p.enabled).length;
  const qualityPreset = SEARCH_QUALITY_PRESETS[searchQuality];
  const automaticDeepEligible = tuningMode === 'fixed_lc' && numEnabled >= 2 && numEnabled <= 3
    && portConfigs.filter(p => p.enabled).every(p => Number(p.max_components ?? 2) <= 2 && p.allowed_topology_codes == null);
  const solutions = results?.solutions || null;
  const selectedSolutionData = solutions?.[selectedSolution] || solutions?.[0];
  const activeBandsMhz = portConfigs.filter(port => port.enabled).flatMap(port => port.bands_mhz || []);
  const manualSolution = selectedSolutionData
    ? {
        ...selectedSolutionData,
        per_port: {
          ...selectedSolutionData.per_port,
          ...Object.fromEntries(Object.entries(manualComponents).map(([pi, comps]) => [
            pi,
            { ...(selectedSolutionData.per_port?.[pi] || {}), components: comps },
          ])),
        },
      }
    : null;

  return (
    <div className={`main-body ${inspectorCollapsed ? 'inspector-collapsed' : ''} ${results ? 'has-results' : ''}`}>
      <button className="inspector-rail-handle" onClick={() => setInspectorCollapsed(value => !value)} title={inspectorCollapsed ? '展开调谐配置' : '收起调谐配置'}>
        {inspectorCollapsed ? '›' : '‹'}
      </button>
      {/* ── Left sidebar ── */}
      <aside className="control-sidebar">
        <div className="inspector-heading">
          <div>
            <span className="eyebrow">SYNTHESIS SETUP</span>
            <h2>网络综合配置</h2>
            <p>按步骤定义网络、工作频段与优化目标</p>
          </div>
          <div className="setup-readiness" aria-label="配置完成度">
            <span className={loadedSNP ? 'done' : 'active'}>DUT</span>
            <span className={numEnabled > 0 ? 'done' : ''}>端口</span>
            <span className={componentSeriesReady ? 'done' : ''}>模型</span>
          </div>
        </div>
        <ModeSelector mode={tuningMode} setMode={setTuningMode} />

        <PortConfigPanel
          portConfigs={portConfigs}
          setPortConfigs={setPortConfigs}
          numPorts={loadedSNP?.num_ports || 0}
          showTopologyConstraints={tuningMode === 'fixed_lc'}
        />

        <ObjectiveSelector objective={objective} setObjective={setObjective} />

        <IsolationTargetEditor
          targets={isolationTargets}
          setTargets={setIsolationTargets}
          enabledPorts={portConfigs.filter(port => port.enabled)}
        />

        {tuningMode === 'tunable_c' && (
          <TunableConfigEditor
            mdifPath={tunerMdifPath} setMdifPath={setTunerMdifPath}
            configurations={frequencyConfigurations} setConfigurations={setFrequencyConfigurations}
            fixedComponents={tunableFixedComponents} setFixedComponents={setTunableFixedComponents}
            autoSynthesize={tunableAutoSynthesize} setAutoSynthesize={setTunableAutoSynthesize}
          />
        )}

        {tuningMode === 'switch' && (
          <TunableConfigEditor
            hardwareKind="switch"
            mdifPath={tunerMdifPath} setMdifPath={setTunerMdifPath}
            configurations={frequencyConfigurations} setConfigurations={setFrequencyConfigurations}
            fixedComponents={[]} setFixedComponents={() => {}}
            autoSynthesize={true} setAutoSynthesize={() => {}}
            stateOptions={switchStateOptions} setStateOptions={setSwitchStateOptions}
            measuredRefine={switchMeasuredRefine} setMeasuredRefine={setSwitchMeasuredRefine}
            maxInputComponents={switchMaxInputComponents} setMaxInputComponents={setSwitchMaxInputComponents}
          />
        )}

        {tuningMode === 'transmission_line' && (
          <TransmissionLineConfigEditor
            config={transmissionLineConfig}
            setConfig={setTransmissionLineConfig}
            availableFiles={snpFiles}
          />
        )}

        {/* Run area */}
        <div className="run-area">
          <button className="run-btn" disabled={optimizing || numEnabled === 0 || !componentSeriesReady || (projectSnapshot && !projectSnapshot.exact_recompute_available)}
            onClick={handleRun}>
            {optimizing ? 'Optimizing...' :
             numEnabled === 0 ? '请先启用一个端口' :
             componentSeriesRequired && !componentCatalogReady ? '元件数据源未就绪' :
             !componentSeriesReady ? (
               tuningMode === 'fixed_lc' ? '请选择实测元件系列' : '请选择电感与电容系列'
             ) :
             projectSnapshot && !projectSnapshot.input_verified ? '缺少已验证 DUT 数据' :
             projectSnapshot?.dependency_status?.some(item => !item.matches) ? '缺少已验证布局 / 夹具' :
             projectSnapshot?.component_library_status?.required && !projectSnapshot.component_library_status.matches ? '实测元件库版本不一致' :
             projectSnapshot && !projectSnapshot.exact_recompute_available ? '工程复算条件未满足' :
             tuningMode === 'switch' ? '生成开关匹配网络' :
             tuningMode === 'transmission_line' ? '生成传输线 / 枝节网络' :
             projectSnapshot ? '按保存配置重新计算' :
             numEnabled === 1 ? '生成匹配候选' :
             '开始多端口联合综合'}
          </button>
          <div className="run-timer">
            元件模型：{!componentSeriesRequired ? '当前模式不使用' : !componentCatalogReady ? '数据源未就绪' : selectedSeries === null ? '默认实测目录' : `已选择 ${selectedSeries.length} 个系列`}
          </div>
          <div className="run-timer">
            搜索预算：{qualityPreset?.label || '自定义'} · {timeout}s · {beamWidth} 个候选 · 每频段 {bandPoints} 点
          </div>
          {calibrationStatus && (
            <div className="run-timer" style={{ color: calibrationStatus.status === 'verified' ? 'var(--accent-green)' : 'var(--accent-red)' }}>
              搜索校准证据：{calibrationStatus.status === 'verified'
                ? `SHA-256 已验证 · 单端口 ${String(calibrationStatus.single_port?.artifact_sha256 || '').slice(0, 10)}… · 多端口 ${String(calibrationStatus.multiport?.artifact_sha256 || '').slice(0, 10)}… · 本机吞吐 1P/3P/4P ${Number(calibrationStatus.performance?.cases?.single_port?.heuristic_physical_evaluations_per_wall_second || 0).toFixed(1)}/${Number(calibrationStatus.performance?.cases?.multiport?.heuristic_physical_evaluations_per_wall_second || 0).toFixed(1)}/${Number(calibrationStatus.performance?.cases?.four_port?.heuristic_physical_evaluations_per_wall_second || 0).toFixed(1)} eval/s`
                : `不可用 · ${calibrationStatus.error || '工件校验失败'}`}
            </div>
          )}
          {projectSnapshot && (
            <div className={`project-restore-gate ${projectSnapshot.exact_recompute_available ? 'verified' : 'restricted'}`}>
              <strong>{projectSnapshot.exact_recompute_available ? '工程复算链已验证' : '可信快照审阅模式'}</strong>
              <span>
                {projectSnapshot.exact_recompute_available
                  ? 'DUT、布局依赖与实测元件库均通过内容哈希校验，可以按保存配置重新计算。'
                  : !projectSnapshot.input_verified
                    ? '原始 DUT 缺失或 SHA-256 不一致；当前只展示完整性验证后的快照结果。'
                    : projectSnapshot.dependency_status?.some(item => !item.matches)
                      ? '布局或去嵌夹具缺失/已变更；请在项目中心执行“查找并修复”。'
                      : projectSnapshot.component_library_status?.reason === 'snapshot_has_no_portable_manifest'
                        ? '该旧工程没有可移植元件清单，请在原环境验证后另存新快照。'
                        : '实测元件库内容与工程快照不一致，已禁止重新计算。'}
              </span>
            </div>
          )}
          {searchQuality === 'exhaustive' && !automaticDeepEligible && (
            <div className="run-timer" style={{ color: 'var(--accent-yellow)' }}>
              Automatic topology deep requires Fixed LC, 2–3 ports, ≤2 components/port, and Automatic topology on every port; this request will use the general thorough path.
            </div>
          )}
          {optimizing && (
            <>
              <div className="run-timer">Elapsed: {elapsed}s</div>
              {jobProgress && (
                <div className="run-timer">
                  {jobProgress.message || jobProgress.stage}
                  {jobProgress.total > 0 ? ` (${jobProgress.current}/${jobProgress.total})` :
                    jobProgress.current > 0 ? ` (${jobProgress.current})` : ''}
                  {jobProgress.budget_seconds > 0
                    ? ` · ${Number(jobProgress.elapsed_seconds || 0).toFixed(1)}/${Number(jobProgress.budget_seconds).toFixed(0)}s budget` : ''}
                </div>
              )}
              {activeJobId && (
                <button className="btn btn-xs btn-danger" onClick={handleCancelOptimization}>Cancel optimization</button>
              )}
            </>
          )}
        </div>

        {/* Advanced */}
        <div style={{ marginTop: 8 }}>
          <button className="advanced-toggle" onClick={() => {
            const el = document.getElementById('tune-advanced');
            if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
          }}>
            <span>高级搜索设置</span>
            <span>[+]</span>
          </button>
          <div id="tune-advanced" style={{ display: 'none', padding: 8 }}>
            <div className="form-group">
              <label>Within-band average weight</label>
              <input type="number" min="0" max="1" step="0.05"
                placeholder="Objective preset"
                value={withinBandAverageWeight ?? ''}
                onChange={e => setWithinBandAverageWeight(e.target.value === '' ? null : Number(e.target.value))} />
              <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 3 }}>
                0 = optimize the worst frequency; 1 = optimize the band average. Optenni radiation-efficiency tutorial: 0.5.
              </div>
            </div>
            <div className="form-group">
              <label>Across-band average weight</label>
              <input type="number" min="0" max="1" step="0.05"
                placeholder="0.1 default"
                value={acrossBandAverageWeight ?? ''}
                onChange={e => setAcrossBandAverageWeight(e.target.value === '' ? null : Number(e.target.value))} />
              <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 3 }}>
                0 = prioritize the weakest band; 1 = prioritize the average across bands.
              </div>
            </div>
            <div style={{ borderTop: '1px solid var(--border)', margin: '8px 0', paddingTop: 8 }}>
              <div style={{ fontSize: 10, fontWeight: 600, marginBottom: 5 }}>Generic topology-prior losses</div>
              <div className="form-group">
                <label>Inductor Q @ reference</label>
                <input type="number" min="0.01" step="1" value={genericSynthesisLoss.inductor_q}
                  onChange={e => setGenericSynthesisLoss({ ...genericSynthesisLoss, inductor_q: Number(e.target.value) })} />
              </div>
              <div className="form-group">
                <label>Q reference frequency (Hz)</label>
                <input type="number" min="1" step="1000000" value={genericSynthesisLoss.inductor_q_reference_hz}
                  onChange={e => setGenericSynthesisLoss({ ...genericSynthesisLoss, inductor_q_reference_hz: Number(e.target.value) })} />
              </div>
              <div className="form-group">
                <label>Inductor ESR (Ω)</label>
                <input type="number" min="0" step="0.01" value={genericSynthesisLoss.inductor_esr_ohm}
                  onChange={e => setGenericSynthesisLoss({ ...genericSynthesisLoss, inductor_esr_ohm: Number(e.target.value) })} />
              </div>
              <div className="form-group">
                <label>Capacitor ESR (Ω)</label>
                <input type="number" min="0" step="0.05" value={genericSynthesisLoss.capacitor_esr_ohm}
                  onChange={e => setGenericSynthesisLoss({ ...genericSynthesisLoss, capacitor_esr_ohm: Number(e.target.value) })} />
              </div>
              <div style={{ fontSize: 9, color: 'var(--text-secondary)' }}>
                Used only to synthesize the continuous topology prior. Final BOM ranking always uses measured S2P loss. Radiation Efficiency tutorial: Q 50 @ 1 GHz, capacitor ESR 0.3 Ω.
              </div>
            </div>
            <div className="form-group">
              <label>Search Quality</label>
              <select value={searchQuality} onChange={e => {
                const quality = e.target.value;
                setSearchQuality(quality);
                const preset = SEARCH_QUALITY_PRESETS[quality];
                if (preset) {
                  setTimeout_(preset.timeout);
                  setBeamWidth(preset.beamWidth);
                  setBandPoints(preset.bandPoints);
                }
              }}>
                {Object.entries(SEARCH_QUALITY_PRESETS).map(([value, preset]) => (
                  <option key={value} value={value}>{preset.label}</option>
                ))}
                <option value="custom">Custom engineering settings</option>
              </select>
              <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 3 }}>
                {qualityPreset?.desc || 'User-defined budget; quality is inferred from the actual limits.'}
              </div>
            </div>
            <div className="form-group">
              <label>Beam Width</label>
              <input type="number" value={beamWidth} onChange={e => { setBeamWidth(+e.target.value || 10); setSearchQuality('custom'); }} />
            </div>
            <div className="form-group">
              <label>Timeout (s)</label>
              <input type="number" value={timeout} onChange={e => { setTimeout_(+e.target.value || 60); setSearchQuality('custom'); }} />
            </div>
            <div className="form-group">
              <label>Band Points</label>
              <select value={bandPoints} onChange={e => { setBandPoints(+e.target.value); setSearchQuality('custom'); }}>
                {[3, 5, 7, 10, 15, 20].map(n => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
          </div>
        </div>

        {error && <div className="error-card" style={{ marginTop: 8 }}>{error}</div>}
      </aside>

      {/* ── Main workspace ── */}
      <main className="workspace-panel">
        {!results ? (
          <div className="empty-state">
            <section className="workspace-welcome">
              <div className="empty-visual"><span /><span /><span /></div>
              <div>
                <span className="eyebrow">MATCHING WORKSPACE</span>
                <h3>{loadedSNP ? 'DUT 已就绪，完成配置后生成候选' : '载入 Touchstone，开始网络综合'}</h3>
                <p>{loadedSNP
                  ? '右侧按顺序确认匹配方案、端口频段和优化目标。运行后，本区域会切换为频率曲线、Smith 圆图、候选列表与 BOM 工作台。'
                  : '从项目资源选择 SNP 文件，或打开已有工程。系统会读取端口、频率范围与参考阻抗，并建立可复算的调谐任务。'}</p>
                <div className="workspace-welcome-actions">
                  <button type="button" className="welcome-primary-action" onClick={onOpenProjects}>打开工程</button>
                  <button type="button" className="welcome-secondary-action" onClick={() => onRefreshSNP?.()}>刷新数据源</button>
                </div>
              </div>
            </section>
            <div className="workflow-steps">
              <div className={loadedSNP ? 'done' : 'active'}><b>1</b><span>载入 DUT<small>Touchstone / 效率</small></span></div>
              <div className={loadedSNP && numEnabled > 0 ? 'done' : loadedSNP ? 'active' : ''}><b>2</b><span>定义任务<small>端口 / 频段 / 目标</small></span></div>
              <div className={loadedSNP && numEnabled > 0 && componentSeriesReady ? 'active' : ''}><b>3</b><span>生成候选<small>实测模型 / 拓扑搜索</small></span></div>
              <div><b>4</b><span>工程审阅<small>曲线 / Smith / BOM</small></span></div>
            </div>
            <div className="workspace-readiness-grid">
              <article className={loadedSNP ? 'ready' : ''}>
                <span>01 · 输入数据</span>
                <strong>{loadedSNP?.filename || '等待选择 DUT'}</strong>
                <small>{loadedSNP ? `${loadedSNP.num_ports} 端口 · ${loadedSNP.freq_count || '—'} 个频点` : '支持 S1P / S2P / 多端口 Touchstone'}</small>
              </article>
              <article className={numEnabled > 0 ? 'ready' : ''}>
                <span>02 · 综合任务</span>
                <strong>{numEnabled > 0 ? `${numEnabled} 个端口参与匹配` : '尚未启用端口'}</strong>
                <small>{numEnabled > 0 ? `${tuningMode === 'fixed_lc' ? '固定 LC' : TUNING_MODES.find(item => item.name === tuningMode)?.label} · ${objective === 'balanced' ? '均衡优化' : OBJECTIVE_PRESETS.find(item => item.name === objective)?.label}` : '在左侧端口卡片中启用并配置频段'}</small>
              </article>
              <article className={componentSeriesReady ? 'ready' : ''}>
                <span>03 · 工程输出</span>
                <strong>排序候选与可制造 BOM</strong>
                <small>回波损耗、效率、Smith 圆图、网络拓扑与替代料</small>
              </article>
            </div>
          </div>
        ) : (
          <>
            <div className="result-toolbar-stack">
              <div className="result-command-bar">
                <div className="result-command-title">
                  <span className="eyebrow">ENGINEERING REVIEW</span>
                  <strong>{loadedSNP?.filename || '匹配结果'}</strong>
                  <small>{portConfigs.filter(port => port.enabled).map(port => (port.bands_mhz || []).map(band => `${band[0]}–${band[1]} MHz`).join(' + ')).join(' · ')}</small>
                </div>
                <div className="result-command-metrics">
                  <span><small>当前候选</small><b>#{selectedSolution + 1} / {results.solutions_count || solutions?.length || 0}</b></span>
                  <span><small>综合评分</small><b className="good">{scoreIsDb(selectedSolutionData?.efficiency_basis)
                    ? `${Number(selectedSolutionData?.system_score || 0).toFixed(2)} dB`
                    : `${(Number(selectedSolutionData?.system_score || selectedSolutionData?.balanced_score || 0) * 100).toFixed(1)}%`}</b></span>
                  <span><small>平均效率</small><b>{(Number(selectedSolutionData?.avg_total_efficiency || 0) * 100).toFixed(1)}%</b></span>
                  <span><small>拓扑</small><b className="topology-code">{selectedSolutionData?.search_diagnostics?.topology_code || selectedSolutionData?.topology_code || '自动'}</b></span>
                </div>
                <div className="result-command-actions">
                  <button type="button" disabled={selectedSolution <= 0} onClick={() => handleSelectSolution(selectedSolution - 1)} aria-label="上一个候选">‹</button>
                  <button type="button" disabled={!solutions || selectedSolution >= solutions.length - 1} onClick={() => handleSelectSolution(selectedSolution + 1)} aria-label="下一个候选">›</button>
                  <button type="button" className="command-save" onClick={onOpenProjects}>保存 / 导出</button>
                </div>
              </div>
              <div className="tab-strip">
                <TabButton id="summary" activeTab={activeTab} setActiveTab={setActiveTab}>候选排名</TabButton>
                <TabButton id="curves" activeTab={activeTab} setActiveTab={setActiveTab}>曲线与 Smith 圆图</TabButton>
                <TabButton id="topology" activeTab={activeTab} setActiveTab={setActiveTab}>网络与 BOM</TabButton>
                <TabButton id="power" activeTab={activeTab} setActiveTab={setActiveTab}>功率平衡</TabButton>
                <TabButton id="realtime" activeTab={activeTab} setActiveTab={setActiveTab}>实时调谐</TabButton>
              </div>
            </div>

            {(results.best_solution?.search_diagnostics?.search_truncated || results.continuation) && (
              <div className="workspace-card" style={{ marginBottom: 8, borderColor: results.best_solution?.search_diagnostics?.search_truncated ? 'var(--accent-orange)' : 'var(--accent-green)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                  <strong style={{ fontSize: 11 }}>
                    {results.best_solution?.search_diagnostics?.search_truncated
                      ? 'Partial anytime result'
                      : results.continuation?.checkpoint_reused
                        ? 'Continued from measured-search checkpoint'
                        : 'Continued search merged'}
                  </strong>
                  <span style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
                    {results.best_solution?.search_diagnostics?.ideal_evaluations ?? 0} ideal ·{' '}
                    {results.best_solution?.search_diagnostics?.physical_evaluations ?? 0} physical evaluations
                    {results.best_solution?.search_diagnostics?.termination_reason ? ` · ${results.best_solution.search_diagnostics.termination_reason}` : ''}
                  </span>
                  {results.best_solution?.search_diagnostics?.search_truncated && ['single', 'joint'].includes(results.mode) && (
                    <button className="btn btn-xs btn-primary" disabled={optimizing} onClick={handleContinueOptimization}>
                      Continue +{Math.max(5, Number(timeout) || 30)} s
                    </button>
                  )}
                </div>
                {results.continuation && (
                  <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 3 }}>
                    {results.continuation.checkpoint_reused
                      ? 'Reused loaded S2P models, ideal topology frontier, and exact physical-evaluation cache'
                      : 'Deterministic rerun + merge fallback'}: {results.continuation.previous_timeout_seconds}s → {results.continuation.total_timeout_seconds}s · retained {results.continuation.previous_candidates} old + {results.continuation.new_candidates} new candidates
                  </div>
                )}
              </div>
            )}

            {activeTab === 'summary' && (
              <>
            {/* System Summary */}
            {results.best_solution && (
              <div className="workspace-card result-overview">
                <div className="result-card-heading">
                  <div><span className="eyebrow">BEST CANDIDATE</span><h3>最佳方案概览</h3></div>
                  <div className="result-heading-actions">
                    <span className="solution-chip">#{selectedSolution + 1} / {results.solutions_count || 0}</span>
                    <button className="save-export-button" onClick={onOpenProjects}>保存与导出</button>
                  </div>
                </div>
                <div className="metric-row">
                  <div className="metric-tile">
                    <div className="metric-label">综合评分</div>
                    <div className={`metric-value ${(results.best_solution.avg_total_efficiency || 0) > 0.7 ? 'good' : 'warn'}`}>
                      {scoreIsDb(results.best_solution.efficiency_basis)
                        ? `${(results.best_solution.system_score || 0).toFixed(2)} dB`
                        : `${((results.best_solution.system_score || results.best_solution.balanced_score || 0) * 100).toFixed(1)}%`}
                    </div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-label">平均总效率</div>
                    <div className={`metric-value ${(results.best_solution.avg_total_efficiency || 0) > 0.7 ? 'good' : 'warn'}`}>
                      {((results.best_solution.avg_total_efficiency || 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-label">最差点效率</div>
                    <div className={`metric-value ${(results.best_solution.min_total_efficiency || 0) > 0.7 ? 'good' : 'warn'}`}>
                      {((results.best_solution.min_total_efficiency || 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  {loadedSNP?.num_ports > 1 && <div className="metric-tile">
                    <div className="metric-label">最大耦合损耗</div>
                    <div className={`metric-value ${(results.best_solution.max_coupling_loss || 1) < 0.03 ? 'good' : (results.best_solution.max_coupling_loss || 1) < 0.1 ? 'warn' : 'bad'}`}>
                      {((results.best_solution.max_coupling_loss || 0) * 100).toFixed(1)}%
                    </div>
                  </div>}
                  <div className="metric-tile">
                    <div className="metric-label">元件损耗</div>
                    <div className={`metric-value ${((results.best_solution.total_component_loss || results.best_solution.component_loss_total || 0)) < 0.05 ? 'good' : 'warn'}`}>
                      {((results.best_solution.total_component_loss || results.best_solution.component_loss_total || 0) * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-label">候选方案</div>
                    <div className="metric-value">{results.solutions_count || 0}</div>
                  </div>
                  {results.best_solution.isolation_targets?.length > 0 && (
                    <div className="metric-tile">
                      <div className="metric-label">Isolation</div>
                      <div className={`metric-value ${results.best_solution.isolation_constraints_passed ? 'good' : 'bad'}`}>
                        {results.best_solution.isolation_constraints_passed
                          ? 'PASS'
                          : `+${(results.best_solution.isolation_penalty_db || 0).toFixed(2)} dB`}
                      </div>
                    </div>
                  )}
                  <div className="metric-tile">
                    <div className="metric-label">搜索耗时</div>
                    <div className="metric-value">{(results.total_time_s || 0).toFixed(1)}s</div>
                  </div>
                </div>
              </div>
            )}

            {results.best_solution?.tunable_states && Object.keys(results.best_solution.tunable_states).length > 0 && (
              <div className="workspace-card">
                <h3>MDIF State Assignment</h3>
                <table style={{ fontSize: 11 }}>
                  <thead><tr><th>Configuration</th><th>Active bands</th><th>Selected state</th><th>Score</th></tr></thead>
                  <tbody>
                    {(results.best_solution.frequency_configurations || []).map(configuration => (
                      <tr key={configuration.name}>
                        <td>{configuration.name}</td>
                        <td>{(configuration.bands_mhz || []).map(band => `${band[0]}–${band[1]} MHz`).join(' + ')}</td>
                        <td style={{ fontWeight: 600, color: 'var(--accent-green)' }}>{configuration.state}</td>
                        <td>{Number(configuration.score_db).toFixed(2)} dB</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 5 }}>
                  {results.best_solution.efficiency_basis === 'rfmatch_core_switch_mdif_wave_power'
                    ? <>Coupled multi-port switch wave-power evaluation · maximum balance error {Number(results.best_solution.maximum_power_balance_error || 0).toExponential(2)}</>
                    : <>Physical S-parameter evaluation · maximum power-balance error {Number(results.best_solution.maximum_power_balance_error || 0).toExponential(2)}</>}
                </div>
              </div>
            )}

            {results.best_solution?.search_diagnostics?.mode === 'transmission_line_auto_synthesis' && (
              <div className="workspace-card">
                <h3>Synthesized Transmission-Line Network</h3>
                <table style={{ fontSize: 11 }}>
                  <thead><tr><th>Position (DUT outward)</th><th>Connection</th><th>Type</th><th>Z0</th><th>Electrical length</th><th>Width</th><th>Physical length</th><th>Manufacturing tolerance</th><th>Loss</th></tr></thead>
                  <tbody>
                    {(results.best_solution.per_port?.[results.best_solution.port_indices?.[0]]?.components || []).map((component, index) => (
                      <tr key={`${component.comp_type}-${index}`}>
                        <td>{index + 1}</td>
                        <td>{component.connection_type}</td>
                        <td>{component.comp_type}</td>
                        <td>{Number(component.characteristic_impedance_ohm ?? component.reference_impedance_ohm).toFixed(3)} Ω</td>
                        <td>{component.electrical_length_deg != null ? `${Number(component.electrical_length_deg).toFixed(3)}°` : 'measured'}</td>
                        <td>{component.width_m ? `${(Number(component.width_m) * 1e3).toFixed(4)} mm` : '—'}</td>
                        <td>{component.length_m ? `${(Number(component.length_m) * 1e3).toFixed(4)} mm` : '—'}</td>
                        <td>{component.manufacturing_tolerances_pct
                          ? `W ±${Number(component.manufacturing_tolerances_pct.trace_width).toFixed(1)}% · L ±${Number(component.manufacturing_tolerances_pct.physical_length).toFixed(1)}% · H ±${Number(component.manufacturing_tolerances_pct.substrate_height).toFixed(1)}% · εr ±${Number(component.manufacturing_tolerances_pct.relative_permittivity).toFixed(1)}%`
                          : '—'}</td>
                        <td>{component.attenuation_db != null ? `${Number(component.attenuation_db).toFixed(3)} dB` : 'from S2P'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 5 }}>
                  {results.best_solution.search_diagnostics.topology} ·{' '}
                  {results.best_solution.search_diagnostics.evaluations} evaluations ·{' '}
                  {Number(results.best_solution.search_diagnostics.elapsed_seconds).toFixed(2)} s ·{' '}
                  stopped: {results.best_solution.search_diagnostics.stopped_reason}
                  {results.best_solution.search_diagnostics.physical_microstrip ? ' · physical PCB geometry enabled' : ''}
                  {(results.best_solution.search_diagnostics.layout_blocks || []).length > 0
                    ? ` · ${results.best_solution.search_diagnostics.layout_blocks.length} measured layout block(s)` : ''}
                </div>
                {(results.best_solution.search_diagnostics.layout_blocks || []).map((block, index) => (
                  <div key={`${block.filename}-${index}`} style={{ fontSize: 9, color: block.passivity?.passive ? 'var(--text-secondary)' : 'var(--accent-red)', marginTop: 2 }}>
                      {block.location}: {block.filename} · {block.reverse_ports ? 'ports 2→1' : 'ports 1→2'} · {block.passivity?.renormalized ? `Z0 ${Number(block.passivity.native_reference_impedance_ohm).toFixed(2)}→${Number(block.passivity.reference_impedance_ohm).toFixed(2)} Ω` : `native Z0 ${Number(block.passivity?.reference_impedance_ohm || 50).toFixed(2)} Ω`} · SHA256 {String(block.sha256 || '').slice(0, 12)}… · max singular value {Number(block.passivity?.maximum_singular_value || 0).toFixed(6)}
                      {!block.passivity?.passive ? ' · NON-PASSIVE WARNING' : ''}
                      {block.passivity?.deembedded ? ` · de-embedded (${Object.keys(block.fixtures || {}).join('+')}) · cond ${Number(block.passivity.deembedding?.maximum_fixture_condition_number || 0).toExponential(2)} · residual ${Number(block.passivity.deembedding?.maximum_recascade_residual || 0).toExponential(2)}` : ''}
                    </div>
                  ))}
                <div style={{ fontSize: 9, color: 'var(--accent-green)', marginTop: 3 }}>
                  rfmatch_core physical node solver · maximum power-balance error{' '}
                  {Number(results.best_solution.maximum_power_balance_error || 0).toExponential(2)}
                </div>
              </div>
            )}

            {results.best_solution?.search_diagnostics?.mode === 'tunable_mdif_auto_synthesis' && (
              <div className="workspace-card">
                <h3>Synthesized Fixed Network</h3>
                <table style={{ fontSize: 11 }}>
                  <thead><tr><th>Position (DUT outward)</th><th>Connection</th><th>Type</th><th>Measured part</th><th>Value</th></tr></thead>
                  <tbody>
                    {(results.best_solution.search_diagnostics.fixed_network || []).map((component, index) => (
                      <tr key={`${component.part_number}-${index}`}>
                        <td>{index + 1}</td>
                        <td>{component.connection}</td>
                        <td>{component.kind}</td>
                        <td style={{ fontWeight: 600 }}>{component.part_number}</td>
                        <td>{component.value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 5 }}>
                  {results.best_solution.search_diagnostics.ideal_evaluations} ideal evaluations ·{' '}
                  {results.best_solution.search_diagnostics.exact_physical_evaluations} full physical candidates ·{' '}
                  {results.best_solution.search_diagnostics.tuner_state_precomputations} tuner states ·{' '}
                  {results.best_solution.search_diagnostics.ideal_frequency_points} coarse-search frequency points ·{' '}
                  {results.best_solution.search_diagnostics.component_models_loaded} measured models loaded
                </div>
              </div>
            )}

            {['switch_mdif_auto_synthesis', 'switch_mdif_measured_synthesis'].includes(results.best_solution?.search_diagnostics?.mode) && (
              <div className="workspace-card">
                <h3>Synthesized Switch Network</h3>
                <table style={{ fontSize: 11 }}>
                  <thead><tr><th>Location</th><th>Connection</th><th>Type</th><th>Part</th><th>Optimized value</th></tr></thead>
                  <tbody>
                    {(results.best_solution.search_diagnostics.branch_network || []).map(component => (
                      <tr key={`branch-${component.branch}`}>
                        <td>Throw {component.branch}</td>
                        <td>series to switch</td>
                        <td>{component.kind}</td>
                        <td>{component.part_number || 'ideal'}</td>
                        <td>{component.kind === 'L' ? `${(component.value_si * 1e9).toFixed(3)} nH` : `${(component.value_si * 1e12).toFixed(3)} pF`}</td>
                      </tr>
                    ))}
                    {(results.best_solution.search_diagnostics.input_network || []).map((component, index) => (
                      <tr key={`input-${index}`}>
                        <td>Shared input {index + 1}</td>
                        <td>{component.connection}</td>
                        <td>{component.kind}</td>
                        <td>{component.part_number || 'ideal'}</td>
                        <td>{component.kind === 'L' ? `${(component.value_si * 1e9).toFixed(3)} nH` : `${(component.value_si * 1e12).toFixed(3)} pF`}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 5 }}>
                  {results.best_solution.search_diagnostics.evaluations} candidate evaluations ·{' '}
                  {results.best_solution.search_diagnostics.physical_evaluations || 0} physical candidates ·{' '}
                  {results.best_solution.search_diagnostics.component_models_loaded || 0} measured models ·{' '}
                  {results.best_solution.search_diagnostics.switch_state_precomputations} switch states ·{' '}
                  {results.best_solution.search_diagnostics.active_frequency_points} active frequency points
                </div>
                <div style={{ fontSize: 9, color: 'var(--accent-green)', marginTop: 3 }}>
                  Full RFC/throw wave reconstruction separates DUT absorbed power from switch loss; maximum balance error {Number(results.best_solution.search_diagnostics.maximum_power_balance_error || 0).toExponential(2)}.
                  {Number(results.best_solution.search_diagnostics.maximum_switch_model_gain || 0) > 0 &&
                    ` Input MDIF non-passivity: ${Number(results.best_solution.search_diagnostics.maximum_switch_model_gain).toExponential(2)}.`}
                </div>
                {(results.best_solution.search_diagnostics.complexity_alternatives || []).length > 0 && (
                  <>
                    <h4 style={{ margin: '10px 0 4px', fontSize: 11 }}>Performance / BOM trade-offs</h4>
                    <table style={{ fontSize: 10 }}>
                      <thead><tr><th>Role</th><th>Input parts</th><th>Score</th><th>Δ best</th><th>Input topology</th><th>State mapping</th></tr></thead>
                      <tbody>
                        {results.best_solution.search_diagnostics.complexity_alternatives.map(alternative => (
                          <tr
                            key={alternative.input_component_count}
                            className={`clickable-row ${selectedSolution === alternative.solution_index ? 'selected' : ''}`}
                            onClick={() => handleSelectSolution(alternative.solution_index)}
                            title="Select this independently evaluated solution"
                          >
                            <td>{{
                              best_performance: 'Best performance',
                              simplest_bom: 'Simplest BOM',
                              performance_complexity_compromise: 'Compromise',
                            }[alternative.recommendation_role] || alternative.recommendation_role || 'Alternative'}</td>
                            <td>{alternative.input_component_count}</td>
                            <td>{Number(alternative.score_db).toFixed(3)} dB</td>
                            <td>{Number(alternative.score_delta_from_best_db || 0).toFixed(3)} dB</td>
                            <td>{alternative.input_network.length
                              ? alternative.input_network.map(item => `${item.connection} ${item.kind} ${item.part_number || ''}`).join(' → ')
                              : 'none'}</td>
                            <td>{Object.entries(alternative.state_by_configuration || {}).map(([name, state]) => `${name}: ${state}`).join(' · ')}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {results.best_solution.search_diagnostics.calibration_reference && (
                      <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 6 }}>
                        Design-role terminology is calibrated against the official Optenni 10.6 tutorial pages{' '}
                        {(results.best_solution.search_diagnostics.calibration_reference.verified_pages || []).join(', ')};
                        this is a reference benchmark, not proof for the current DUT or component catalog.
                      </div>
                    )}
                  </>
                )}
              </div>
            )}

            {results.best_solution?.search_diagnostics?.measured_physical_search === true && (
              <details className="workspace-card engineering-details">
                <summary>
                  <span><b>搜索与数值诊断</b><small>实测 S2P、搜索策略、模型加载和校准证据</small></span>
                  <i>展开工程详情</i>
                </summary>
                <div className="engineering-details-body">
                <div className="metric-row">
                  <div className="metric-tile"><div className="metric-label">Topology</div><div className="metric-value">{results.best_solution.search_diagnostics.topology_code || '—'}</div></div>
                  <div className="metric-tile"><div className="metric-label">Topology constraints</div><div className="metric-value" style={{ fontSize: 10 }}>{
                    Object.entries(results.best_solution.search_diagnostics.allowed_topology_codes_by_port || {})
                      .map(([port, codes]) => `P${Number(port) + 1}: ${(codes || []).join('/')}`)
                      .join(' · ') || 'Automatic'
                  }</div></div>
                  <div className="metric-tile"><div className="metric-label">Model backend</div><div className="metric-value">{(results.best_solution.search_diagnostics.component_model_backends || []).join(' + ') || '—'}</div></div>
                  <div className="metric-tile"><div className="metric-label">Physical evaluations</div><div className="metric-value">{results.best_solution.search_diagnostics.physical_evaluations || 0}</div></div>
                  <div className="metric-tile"><div className="metric-label">Models loaded</div><div className="metric-value">{results.best_solution.search_diagnostics.component_models_loaded || 0}</div></div>
                  <div className="metric-tile"><div className="metric-label">Frequency points</div><div className="metric-value">{results.best_solution.search_diagnostics.active_frequency_points || 0}</div></div>
                  <div className="metric-tile"><div className="metric-label">Balance error</div><div className="metric-value">{Number(results.best_solution.maximum_power_balance_error || 0).toExponential(2)}</div></div>
                  <div className="metric-tile"><div className="metric-label">Search profile</div><div className="metric-value">{results.best_solution.search_diagnostics.search_profile || 'standard'}</div></div>
                  <div className="metric-tile"><div className="metric-label">Requested quality</div><div className="metric-value">{results.best_solution.search_diagnostics.search_plan?.label || results.best_solution.search_diagnostics.search_quality_requested || 'auto'}</div></div>
                  <div className="metric-tile"><div className="metric-label">Execution strategy</div><div className="metric-value" style={{ fontSize: 10 }}>{results.best_solution.search_diagnostics.search_plan?.strategy || 'hierarchical_measured'}</div></div>
                  <div className="metric-tile"><div className="metric-label">Completion</div><div className="metric-value">{results.best_solution.search_diagnostics.search_truncated ? 'Time-budgeted partial' : 'Complete'}</div></div>
                  <div className="metric-tile" title="端口权重 × 频段权重；按端口依次列出每个频段的有效权重"><div className="metric-label">Effective priorities</div><div className="metric-value" style={{ fontSize: 9 }}>{priorityWeightLabel(results.best_solution)}</div></div>
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 6 }}>
                  Catalog: {results.best_solution.search_diagnostics.component_catalog_size?.inductors || 0} inductors +{' '}
                  {results.best_solution.search_diagnostics.component_catalog_size?.capacitors || 0} capacitors. Models are loaded lazily only for candidates reached by hierarchical refinement.
                </div>
                {results.best_solution.search_diagnostics.calibration_reference && (
                  <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 8, paddingTop: 8, borderTop: '1px solid var(--border-color)' }}>
                    Reference calibration only — not proof for this request&apos;s catalog:{' '}
                    {Number(results.best_solution.search_diagnostics.calibration_reference.reference_exact_top_k_recall || 0) * 100}% exact top-k recall on the official three-port benchmark ({results.best_solution.search_diagnostics.calibration_reference.scope?.catalog || 'reference catalog'}, max {results.best_solution.search_diagnostics.calibration_reference.scope?.maximum_components_per_port ?? '—'} component/port, per-port keep {results.best_solution.search_diagnostics.calibration_reference.scope?.per_port_keep ?? '—'}).
                    {' '}Evidence SHA-256 {String(results.best_solution.search_diagnostics.calibration_reference.artifact_sha256 || '').slice(0, 12)}…; claims are parsed from the artifact at runtime.
                    {results.best_solution.search_diagnostics.calibration_reference.four_port_scaling && (
                      <> Official four-element S4P scaling: {Number(results.best_solution.search_diagnostics.calibration_reference.four_port_scaling.reference_exact_top_k_recall || 0) * 100}% exact top-k recall across {results.best_solution.search_diagnostics.calibration_reference.four_port_scaling.scope?.exhaustive_candidates || '—'} exhaustive candidates with per-port keep {results.best_solution.search_diagnostics.calibration_reference.four_port_scaling.scope?.per_port_keep || '—'}; evidence {String(results.best_solution.search_diagnostics.calibration_reference.four_port_scaling.artifact_sha256 || '').slice(0, 12)}….</>
                    )}
                    {results.best_solution.search_diagnostics.calibration_reference.numerical_golden && (
                      <> The exact saved Optenni winner also matches displayed band efficiency within {Number(results.best_solution.search_diagnostics.calibration_reference.numerical_golden.maximum_efficiency_delta_from_rounded_ui_db).toFixed(3)} dB.</>
                    )}
                    {results.best_solution.search_diagnostics.calibration_reference.saved_winner_discovery && (
                      <> On its four-part saved-BOM grid, unrestricted automatic search rediscovers the exact three-port Optenni winner at rank {results.best_solution.search_diagnostics.calibration_reference.saved_winner_discovery.exact_saved_winner_rank}.</>
                    )}
                    {results.best_solution.search_diagnostics.calibration_reference.full_catalog_discovery && (
                      <> With the saved topology constrained over the full reference model set, coupled port-block search retains the exact Optenni BOM at rank {results.best_solution.search_diagnostics.calibration_reference.full_catalog_discovery.exact_saved_winner_rank} and finds a measured-model alternative {Number(results.best_solution.search_diagnostics.calibration_reference.full_catalog_discovery.best_score_improvement_db).toFixed(3)} dB better under the core objective.</>
                    )}
                    {results.best_solution.search_diagnostics.calibration_reference.automatic_full_catalog_discovery && (
                      <> Without topology hints, the 150-second product reference deep search ranks the saved topology {results.best_solution.search_diagnostics.calibration_reference.automatic_full_catalog_discovery.product_saved_topology_rank ?? results.best_solution.search_diagnostics.calibration_reference.automatic_full_catalog_discovery.saved_topology_rank} and exact BOM {results.best_solution.search_diagnostics.calibration_reference.automatic_full_catalog_discovery.product_exact_saved_winner_rank ?? results.best_solution.search_diagnostics.calibration_reference.automatic_full_catalog_discovery.exact_saved_winner_rank}; this remains reference evidence, not a guarantee for the current DUT.</>
                    )}
                    {results.best_solution.search_diagnostics.calibration_reference.product_full_catalog_discovery && (
                      <> On the official single-port Optimization Settings reference, a full 0402 procurement catalog ranks {results.best_solution.search_diagnostics.calibration_reference.product_full_catalog_discovery.topology} topology {results.best_solution.search_diagnostics.calibration_reference.product_full_catalog_discovery.topology_rank} with at most {Number(results.best_solution.search_diagnostics.calibration_reference.product_full_catalog_discovery.maximum_efficiency_delta_db).toFixed(3)} dB efficiency delta from the Optenni ideal network.</>
                    )}
                  </div>
                )}
                {results.best_solution.search_diagnostics.search_truncated && (
                  <div style={{ fontSize: 10, color: 'var(--accent-yellow)', marginTop: 8 }}>
                    Search stopped at the requested time budget and retained completed candidates: {results.best_solution.search_diagnostics.termination_reason || 'partial search'}.
                  </div>
                )}
                </div>
              </details>
            )}

            {results.best_solution?.search_diagnostics?.measured_physical_search === false && results.best_solution.search_diagnostics.measured_physical_fallback_reason && (
              <div className="workspace-card" style={{ borderColor: 'var(--accent-yellow)' }}>
                <h3>Measured Physical Search Fallback</h3>
                <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                  {results.best_solution.search_diagnostics.measured_physical_fallback_reason}
                </div>
              </div>
            )}

            {results.best_solution?.isolation_targets?.length > 0 && (
              <div className="workspace-card">
                <h3>Directed Isolation Constraints</h3>
                <table style={{ fontSize: 11 }}>
                  <thead><tr><th>Path</th><th>Band target</th><th>Worst</th><th>Average</th><th>Status</th></tr></thead>
                  <tbody>
                    {results.best_solution.isolation_targets.map((target, index) => (
                      <tr key={index}>
                        <td>S{target.destination_port + 1}{target.source_port + 1}</td>
                        <td>≤ {target.maximum_allowed_db.toFixed(1)} dB</td>
                        <td>{target.worst_transmission_db.toFixed(2)} dB</td>
                        <td>{target.average_transmission_db.toFixed(2)} dB</td>
                        <td style={{ color: target.passed ? 'var(--accent-green)' : 'var(--accent-red)', fontWeight: 600 }}>{target.passed ? 'PASS' : 'FAIL'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Single-port mode results */}
            {!results.best_solution && results.solutions && (
              <div className="workspace-card">
                <h3>Single-Port Results (sorted by efficiency)</h3>
                <div className="metric-row">
                  <div className="metric-tile">
                    <div className="metric-label">Avg η (enabled ports)</div>
                    <div className="metric-value good">{(results.best_avg_efficiency * 100 || 0).toFixed(1)}%</div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-label">Best Min η</div>
                    <div className="metric-value warn">{(results.best_min_efficiency * 100 || 0).toFixed(1)}%</div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-label">Score</div>
                    <div className="metric-value good">{(results.best_score * 100 || 0).toFixed(1)}%</div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-label">Solutions</div>
                    <div className="metric-value">{results.solutions_count}</div>
                  </div>
                </div>
              </div>
            )}

            {/* Results table */}
            {solutions && solutions.length > 0 && (
              <SolutionComparison solutions={solutions} indices={comparisonIndices} selectedIndex={selectedSolution} onSelect={handleSelectSolution} portConfigs={portConfigs} />
            )}
            {solutions && solutions.length > 0 && (
              <div className="workspace-card">
                <h3>候选方案排名
                  <span style={{ fontSize: 11, fontWeight: 400, color: 'var(--text-secondary)', marginLeft: 8 }}>
                    点击行可切换方案
                  </span>
                </h3>
                <ResultsTable
                  solutions={solutions}
                  onSelectSolution={handleSelectSolution}
                  selectedIndex={selectedSolution}
                  comparisonIndices={comparisonIndices}
                  onToggleComparison={toggleComparison}
                />
              </div>
            )}
            {solutions && solutions.length > 0 && (
              <div className="workspace-card">
                <h3>制造良率分析
                  <span style={{ fontSize: 11, fontWeight: 400, color: 'var(--text-secondary)', marginLeft: 8 }}>
                    实测 S2P 寄生 + L/C 数值公差 + 微带几何/材料公差 · 固定随机种子 · Wilson 95% 置信区间
                  </span>
                </h3>
                <>
                    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'end', marginBottom: 12 }}>
                      <label>Monte Carlo 样本数<input type="number" min="20" max="5000" value={yieldConfig.samples}
                        onChange={event => setYieldConfig({ ...yieldConfig, samples: Number(event.target.value) })} /></label>
                      <label>最低总效率 η (%)<input type="number" min="0" max="100" step="1"
                        value={yieldConfig.minimum_total_efficiency * 100}
                        onChange={event => setYieldConfig({ ...yieldConfig, minimum_total_efficiency: Number(event.target.value) / 100 })} /></label>
                      <label title="与 Optenni 一致：在 dB 域取算术平均，等价于线性域几何平均">最低平均效率 η (%)<input type="number" min="0" max="100" step="1"
                        value={yieldConfig.minimum_average_total_efficiency * 100}
                        onChange={event => setYieldConfig({ ...yieldConfig, minimum_average_total_efficiency: Number(event.target.value) / 100 })} /></label>
                      <label>最低回波损耗 (dB)<input type="number" min="0" step="0.5" value={yieldConfig.minimum_return_loss_db}
                        onChange={event => setYieldConfig({ ...yieldConfig, minimum_return_loss_db: Number(event.target.value) })} /></label>
                      <label>未知料号默认公差 (%)<input type="number" min="0.01" max="100" step="0.5" value={yieldConfig.default_tolerance_pct}
                        onChange={event => setYieldConfig({ ...yieldConfig, default_tolerance_pct: Number(event.target.value) })} /></label>
                      <label>公差分布<select value={yieldConfig.distribution}
                        onChange={event => setYieldConfig({ ...yieldConfig, distribution: event.target.value })}>
                        <option value="normal">正态分布（±3σ 截断）</option><option value="uniform">均匀分布</option>
                      </select></label>
                      <button className="btn btn-primary" disabled={yieldBusy} onClick={handleYieldAnalysis}>
                        {yieldBusy ? '正在分析…' : '计算并按良率排序'}
                      </button>
                    </div>
                    <details style={{ marginBottom: 12 }}>
                      <summary style={{ cursor: 'pointer', fontSize: 10, color: 'var(--text-secondary)' }}>
                        工艺偏置、批次相关性与温度环境
                      </summary>
                      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'end', marginTop: 8 }}>
                        <label title="对每个电感标称值施加同方向的系统性工艺偏移">
                          L 系统偏置 (%)<input type="number" min="-99" max="1000" step="0.1"
                            value={yieldConfig.inductor_bias_pct}
                            onChange={event => setYieldConfig({ ...yieldConfig, inductor_bias_pct: Number(event.target.value) })} />
                        </label>
                        <label title="对每个电容标称值施加同方向的系统性工艺偏移">
                          C 系统偏置 (%)<input type="number" min="-99" max="1000" step="0.1"
                            value={yieldConfig.capacitor_bias_pct}
                            onChange={event => setYieldConfig({ ...yieldConfig, capacitor_bias_pct: Number(event.target.value) })} />
                        </label>
                        <label title="同一块装配中各器件制造偏差的相关系数">
                          批次相关性 (%)<input type="number" min="0" max="100" step="5"
                            value={yieldConfig.batch_correlation * 100}
                            onChange={event => setYieldConfig({ ...yieldConfig, batch_correlation: Number(event.target.value) / 100 })} />
                        </label>
                        <label style={{ display: 'flex', gap: 5, alignItems: 'center', paddingBottom: 5 }}>
                          <input type="checkbox" checked={yieldConfig.temperature_min_c !== null}
                            onChange={event => setYieldConfig({
                              ...yieldConfig,
                              temperature_min_c: event.target.checked ? -40 : null,
                              temperature_max_c: event.target.checked ? 85 : null,
                            })} /> 启用温度扫描
                        </label>
                        {yieldConfig.temperature_min_c !== null && <>
                          <label>最低温度 °C<input type="number" min="-273.15" max="500" step="5"
                            value={yieldConfig.temperature_min_c}
                            onChange={event => setYieldConfig({ ...yieldConfig, temperature_min_c: Number(event.target.value) })} /></label>
                          <label>最高温度 °C<input type="number" min="-273.15" max="500" step="5"
                            value={yieldConfig.temperature_max_c}
                            onChange={event => setYieldConfig({ ...yieldConfig, temperature_max_c: Number(event.target.value) })} /></label>
                          <label>L 温漂 (ppm/°C)<input type="number" min="-10000" max="10000" step="10"
                            value={yieldConfig.inductor_tempco_ppm_per_c}
                            onChange={event => setYieldConfig({ ...yieldConfig, inductor_tempco_ppm_per_c: Number(event.target.value) })} /></label>
                          <label>C 温漂 (ppm/°C)<input type="number" min="-10000" max="10000" step="10"
                            value={yieldConfig.capacitor_tempco_ppm_per_c}
                            onChange={event => setYieldConfig({ ...yieldConfig, capacitor_tempco_ppm_per_c: Number(event.target.value) })} /></label>
                        </>}
                      </div>
                      <p className="hint-text" style={{ marginTop: 6 }}>
                        系统偏置先作用于同类器件；批次相关性通过 Gaussian Copula 保持所选边缘分布；每个装配样本只抽取一个温度，并在所有开关/调谐状态间共享。
                      </p>
                    </details>
                    {yieldError && <div className="error-message">{yieldError}</div>}
                    {yieldResult?.ranked_candidates?.length > 0 && (
                      <div style={{ overflowX: 'auto' }}><table className="results-table"><thead><tr>
                        <th>良率排名</th><th>候选</th><th>联合良率</th><th>最差状态</th><th>95% 置信区间</th><th>P5 裕量</th><th>样本数</th>
                      </tr></thead><tbody>{yieldResult.ranked_candidates.map(item => <tr key={item.solution_index}
                        className={selectedSolution === item.solution_index ? 'selected' : ''}
                        onClick={() => handleSelectSolution(item.solution_index)}>
                        <td>{item.yield_rank}</td><td>#{item.solution_index + 1}</td>
                        <td>{(item.yield_fraction * 100).toFixed(1)}%</td>
                        <td>{Object.keys(item.configuration_yield_fraction || {}).length > 0
                          ? `${(Math.min(...Object.values(item.configuration_yield_fraction)) * 100).toFixed(1)}%`
                          : '—'}</td>
                        <td>{(item.yield_confidence_interval[0] * 100).toFixed(1)}–{(item.yield_confidence_interval[1] * 100).toFixed(1)}%</td>
                        <td>{Number(item.score_percentiles_db?.['5'] || 0).toFixed(2)} dB</td><td>{item.samples}</td>
                      </tr>)}</tbody></table></div>
                    )}
                    {yieldResult?.ranked_candidates?.find(item => item.solution_index === selectedSolution)?.component_tolerances?.length > 0 && (
                      <div className="yield-variable-summary">
                        <strong>当前候选制造变量</strong>
                        <div>{yieldResult.ranked_candidates.find(item => item.solution_index === selectedSolution).component_tolerances.map((item, index) => (
                          <span key={`${item.position || item.part_number}-${item.variable || index}`}>
                            {item.part_number}{item.variable ? ` · ${{
                              trace_width: '线宽', physical_length: '线长', substrate_height: '板厚',
                              relative_permittivity: '介电常数', electrical_length: '电长度',
                            }[item.variable] || item.variable}` : ''} ±{Number(item.tolerance_pct).toFixed(2)}%
                          </span>
                        ))}</div>
                      </div>
                    )}
                    {yieldResult?.unsupported_candidates?.length > 0 && (
                      <p className="hint-text">有 {yieldResult.unsupported_candidates.length} 个候选无法分析：{yieldResult.unsupported_candidates[0].reason}</p>
                    )}
                </>
              </div>
            )}
              </>
            )}

            {/* Charts row */}
            {activeTab === 'curves' && sweepData && (
              <div className="analysis-charts">
                <div className="chart-grid">
                  <div className="chart-card">
                    <S11Chart sweepData={sweepData} sweepsByPort={sweepDataByPort} bandsMhz={activeBandsMhz} />
                  </div>
                  <div className="chart-card">
                    <EfficiencyChart sweepData={sweepData} sweepsByPort={sweepDataByPort} bandsMhz={activeBandsMhz} />
                  </div>
                </div>
                <div className="chart-card smith-card">
                  <EngineeringSmithChart sweepData={sweepData} sweepsByPort={sweepDataByPort} />
                </div>
              </div>
            )}

            {/* Power Balance */}
            {activeTab === 'power' && (results.system_power_balance || results.best_solution?.system_power_balance) && (
              <div className="workspace-card">
                <h3>Power Balance
                  <span style={{ fontSize: 11, fontWeight: 400, color: 'var(--text-secondary)', marginLeft: 8 }}>
                    System efficiency: {((results.system_power_balance?.system_efficiency || results.best_solution?.system_power_balance?.system_efficiency || 0) * 100).toFixed(1)}%
                  </span>
                </h3>
                <PowerBalanceBar
                  powerBalance={results.system_power_balance || results.best_solution?.system_power_balance}
                  chartData={results.power_balance_chart || results.best_solution?.power_balance_chart}
                />
              </div>
            )}

            {/* Per-port solution details */}
            {activeTab === 'topology' && solutions && solutions.length > 0 && (
              <div className="workspace-card topology-workbench">
                <div className="result-card-heading">
                  <div><span className="eyebrow">NETWORK & BILL OF MATERIALS</span><h3>匹配网络与物料清单</h3></div>
                  <span className="solution-chip">方案 #{selectedSolution + 1}</span>
                </div>
                <TopologySchematic
                  solution={selectedSolutionData}
                  portIndex={null}
                />
                {/* Port selector for solution details */}
                <div className="port-filter-bar"><span>查看端口</span>
                  {portConfigs.filter(p => p.enabled).map(pc => (
                    <button key={pc.port_index}
                      className={`btn btn-xs ${selectedPort === pc.port_index ? 'btn-primary' : ''}`}
                      onClick={() => handleSelectPort(pc.port_index)}>
                      端口 {pc.port_index + 1}
                    </button>
                  ))}
                </div>
                <SolutionDetails
                  solution={selectedSolutionData}
                  portIndex={selectedPort}
                  portConfigs={portConfigs}
                  selectedSeries={selectedSeries}
                  componentFilter={componentFilter}
                />
              </div>
            )}

            {activeTab === 'realtime' && solutions && solutions.length > 0 && (
              <div className="workspace-card">
                <h3>Realtime Tune
                  <span style={{ fontSize: 11, fontWeight: 400, color: 'var(--text-secondary)', marginLeft: 8 }}>
                    {manualBusy ? 'recomputing...' : 'edit a node to recompute curves'}
                  </span>
                </h3>
                <div className="realtime-layout">
                  <div className="realtime-curves">
                    {sweepData && <S11Chart sweepData={sweepData} sweepsByPort={sweepDataByPort} bandsMhz={activeBandsMhz} />}
                    {sweepData && <EfficiencyChart sweepData={sweepData} sweepsByPort={sweepDataByPort} bandsMhz={activeBandsMhz} />}
                  </div>
                  <TopologySchematic
                    solution={manualSolution}
                    portIndex={selectedPort}
                    editable
                    onComponentChange={handleManualComponentChange}
                  />
                </div>
                {manualError && <div className="error-card">手动重算失败：{manualError}</div>}
              </div>
            )}

            {results.warning && (
              <div className="error-card" style={{ background: 'rgba(255,193,7,0.08)', color: '#856404' }}>
                {results.warning}
              </div>
            )}
          </>
        )}
      </main>
      {results && inspectorCollapsed && (
        <aside className="result-inspector" aria-label="当前候选检查器">
          <div className="result-inspector-heading">
            <div>
              <span className="eyebrow">CIRCUIT INSPECTOR</span>
              <h2>当前候选</h2>
            </div>
            <span className="result-rank-badge">#{selectedSolution + 1}</span>
          </div>

          <section className="inspector-circuit-card">
            <div className="inspector-section-title">
              <span>匹配网络</span>
              <b>{selectedSolutionData?.search_diagnostics?.topology_code || selectedSolutionData?.topology_code || 'AUTO'}</b>
            </div>
            <TopologySchematic solution={selectedSolutionData} portIndex={selectedPort} />
          </section>

          <section className="inspector-kpi-grid">
            <div><span>综合评分</span><strong>{scoreIsDb(selectedSolutionData?.efficiency_basis)
              ? `${Number(selectedSolutionData?.system_score || 0).toFixed(2)} dB`
              : `${(Number(selectedSolutionData?.system_score || selectedSolutionData?.balanced_score || 0) * 100).toFixed(1)}%`}</strong></div>
            <div><span>平均效率</span><strong>{(Number(selectedSolutionData?.avg_total_efficiency || 0) * 100).toFixed(1)}%</strong></div>
            <div><span>最差效率</span><strong>{(Number(selectedSolutionData?.min_total_efficiency || 0) * 100).toFixed(1)}%</strong></div>
            <div><span>元件数量</span><strong>{Object.values(selectedSolutionData?.per_port || {}).reduce((total, port) => total + (port?.components?.length || 0), 0)}</strong></div>
          </section>

          <section className="inspector-candidate-list">
            <div className="inspector-section-title"><span>候选方案</span><small>{solutions?.length || 0} 个</small></div>
            <div className="candidate-list-scroll">
              {(solutions || []).slice(0, 8).map((solution, index) => (
                <button type="button" key={index} className={selectedSolution === index ? 'active' : ''} onClick={() => handleSelectSolution(index)}>
                  <span><b>#{index + 1}</b><small>{solution.search_diagnostics?.topology_code || solution.topology_code || 'AUTO'}</small></span>
                  <strong>{scoreIsDb(solution.efficiency_basis)
                    ? `${Number(solution.system_score || 0).toFixed(2)} dB`
                    : `${(Number(solution.system_score || solution.balanced_score || 0) * 100).toFixed(1)}%`}</strong>
                </button>
              ))}
            </div>
          </section>

          <div className="inspector-quick-actions">
            <button type="button" className={activeTab === 'curves' ? 'active' : ''} onClick={() => setActiveTab('curves')}>曲线 / Smith</button>
            <button type="button" className={activeTab === 'topology' ? 'active' : ''} onClick={() => setActiveTab('topology')}>网络 / BOM</button>
            <button type="button" onClick={() => setInspectorCollapsed(false)}>编辑综合任务</button>
          </div>
        </aside>
      )}
    </div>
  );
}
