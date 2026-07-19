import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

const STATE_OPTIONS = [
  { value: 'load',  label: '50 Ohm', cls: 'load' },
  { value: 'short', label: 'Short',  cls: 'short' },
  { value: 'open',  label: 'Open',   cls: 'open' },
];

const BAND_PRESETS = {
  'WiFi 2.4': [2400, 2500], 'WiFi 5': [5150, 5850], 'WiFi 6': [5925, 7125],
  'GPS L1': [1574, 1576], 'GPS L5': [1176, 1177], 'BT': [2400, 2480],
  'LTE B1': [1920, 2170], 'LTE B3': [1710, 1880], 'LTE B7': [2500, 2690],
  '5G n77': [3300, 4200], '5G n78': [3300, 3800], '5G n79': [4400, 5000],
  'UWB': [3100, 10600], 'NB-IoT': [700, 960], 'LoRa': [863, 870],
};

const MATCH_COLORS = ['#f26d5f', '#4aa3f0', '#3ecf9a', '#e8b04b', '#a47ef0', '#3cc6d0'];

function PortPanel({ portIndex, config, onChange, color }) {
  const u = (k, v) => onChange({ ...config, [k]: v });
  const band = config.band_mhz || [2400, 2500];

  return (
    <div className={`port-card ${config.use_matching ? 'matching' : ''}`}>
      {/* Header */}
      <div className="port-header">
        <span className="port-number" style={config.use_matching ? { color } : {}}>
          Port {portIndex + 1}
        </span>
        <label className="port-match-toggle">
          <input type="checkbox" checked={config.use_matching}
            onChange={e => u('use_matching', e.target.checked)} />
          <span style={{ color: config.use_matching ? 'var(--accent)' : 'var(--text-secondary)' }}>
            {config.use_matching ? 'Match' : 'No Match'}
          </span>
        </label>
      </div>

      {/* State buttons */}
      <div className="segmented-group">
        {STATE_OPTIONS.map(o => (
          <button key={o.value}
            className={`segmented-btn ${o.cls} ${config.state === o.value ? 'active' : ''}`}
            onClick={() => u('state', o.value)}>
            {o.label}
          </button>
        ))}
      </div>

      {/* Band */}
      <div className="band-input-row">
        <input type="number" value={band[0]}
          onChange={e => u('band_mhz', [parseFloat(e.target.value) || 0, band[1]])} />
        <span style={{ alignSelf: 'center', color: 'var(--text-secondary)', fontSize: 10 }}>~</span>
        <input type="number" value={band[1]}
          onChange={e => u('band_mhz', [band[0], parseFloat(e.target.value) || 0])} />
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2, marginBottom: 4 }}>
        {Object.entries(BAND_PRESETS).slice(0, 6).map(([name, range]) => (
          <button key={name} className={`preset-btn ${band[0] === range[0] && band[1] === range[1] ? 'active' : ''}`}
            onClick={() => u('band_mhz', range)}>{name}</button>
        ))}
      </div>

      {/* Matching settings */}
      {config.use_matching && (
        <>
          <div style={{ marginBottom: 4 }}>
            <label style={{ fontSize: 10, color: 'var(--text-secondary)', display: 'block', marginBottom: 2 }}>Max Components</label>
            <div className="pill-group">
              {[1, 2, 3, 4].map(n => (
                <button key={n} className={`pill-btn ${config.max_components === n ? 'active' : ''}`}
                  onClick={() => u('max_components', n)}>{n}</button>
              ))}
            </div>
          </div>
          <div className="comp-range-row">
            <div className="comp-range-group">
              <label>L (nH)</label>
              <div className="flex-row gap-1">
                <input type="number" step="0.1" value={config.l_min_nh}
                  onChange={e => u('l_min_nh', parseFloat(e.target.value) || 0.1)} />
                <span style={{ color: 'var(--text-secondary)', fontSize: 10 }}>~</span>
                <input type="number" step="0.1" value={config.l_max_nh}
                  onChange={e => u('l_max_nh', parseFloat(e.target.value) || 20)} />
              </div>
            </div>
            <div className="comp-range-group">
              <label>C (pF)</label>
              <div className="flex-row gap-1">
                <input type="number" step="0.1" value={config.c_min_pf}
                  onChange={e => u('c_min_pf', parseFloat(e.target.value) || 0.1)} />
                <span style={{ color: 'var(--text-secondary)', fontSize: 10 }}>~</span>
                <input type="number" step="0.1" value={config.c_max_pf}
                  onChange={e => u('c_max_pf', parseFloat(e.target.value) || 20)} />
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default function PerPortConfig({
  numPorts, portConfigs, setPortConfigs,
  optimizationGoal, setOptimizationGoal, compact,
  optimizationModes,  // list of {name, label, description}
  portEfficiency,     // {portIndex: {source, efficiency_min, ...}} or null
  onLoadEfficiency,   // (portIndex) => void
  onClearEfficiency,  // (portIndex) => void
}) {
  const [goalDescriptions, setGoalDescriptions] = useState(null);

  // Fetch optimization mode descriptions when modes change
  useEffect(() => {
    if (optimizationModes?.length) {
      const desc = {};
      optimizationModes.forEach(m => { desc[m.name] = m.description; });
      setGoalDescriptions(desc);
    }
  }, [optimizationModes]);

  if (!portConfigs.length) {
    return <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontStyle: 'italic' }}>Load an SNP file first</div>;
  }

  // Goal options: use backend modes if available, fallback to defaults
  const goalOptions = optimizationModes?.length > 0
    ? optimizationModes.map(m => ({ value: m.name, label: m.label }))
    : [
        { value: 'efficiency', label: 'Efficiency' },
        { value: 'return_loss', label: 'Return Loss' },
        { value: 'balanced', label: 'Balanced' },
        { value: 'worst_case', label: 'Worst-Case' },
        { value: 'average_focused', label: 'Average' },
        { value: 'low_cost', label: 'Low-Cost' },
      ];

  function updatePort(i, cfg) {
    const a = [...portConfigs]; a[i] = cfg; setPortConfigs(a);
  }

  return (
    <div>
      {/* Goal selector */}
      {compact && (
        <div>
          <div className="goal-selector" style={{ marginBottom: 4 }}>
            {goalOptions.map(o => (
              <button key={o.value} className={`goal-btn ${optimizationGoal === o.value ? 'active' : ''}`}
                onClick={() => setOptimizationGoal(o.value)}
                title={goalDescriptions?.[o.value] || ''}>
                {o.label}
              </button>
            ))}
          </div>
          {goalDescriptions?.[optimizationGoal] && (
            <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 6, padding: '0 2px' }}>
              {goalDescriptions[optimizationGoal]}
            </div>
          )}
        </div>
      )}

      {/* Port panels */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {portConfigs.map((cfg, i) => (
          <PortPanelWithEfficiency
            key={i}
            portIndex={i}
            config={cfg}
            onChange={c => updatePort(i, c)}
            color={MATCH_COLORS[i % MATCH_COLORS.length]}
            efficiencyData={portEfficiency?.[i] || null}
            onLoadEfficiency={onLoadEfficiency}
            onClearEfficiency={onClearEfficiency}
          />
        ))}
      </div>
    </div>
  );
}

/** Port panel with efficiency integration */
function PortPanelWithEfficiency({
  portIndex, config, onChange, color,
  efficiencyData, onLoadEfficiency, onClearEfficiency,
}) {
  const u = (k, v) => onChange({ ...config, [k]: v });
  const band = config.band_mhz || [2400, 2500];
  const [showEffInput, setShowEffInput] = useState(false);
  const [effPath, setEffPath] = useState('');

  return (
    <div className={`port-card ${config.use_matching ? 'matching' : ''}`}>
      {/* Header */}
      <div className="port-header">
        <span className="port-number" style={config.use_matching ? { color } : {}}>
          Port {portIndex + 1}
        </span>
        <label className="port-match-toggle">
          <input type="checkbox" checked={config.use_matching}
            onChange={e => u('use_matching', e.target.checked)} />
          <span style={{ color: config.use_matching ? 'var(--accent)' : 'var(--text-secondary)' }}>
            {config.use_matching ? 'Match' : 'No Match'}
          </span>
        </label>
      </div>

      {/* Efficiency status */}
      {efficiencyData && (
        <div style={{
          fontSize: 10, color: 'var(--accent-green)', marginBottom: 4,
          padding: '2px 6px', background: 'rgba(46,204,113,0.08)', borderRadius: 4,
        }}>
          η_rad: {(efficiencyData.efficiency_min * 100).toFixed(0)}–{(efficiencyData.efficiency_max * 100).toFixed(0)}%
          <button className="btn btn-sm" style={{ marginLeft: 6, fontSize: 9, padding: '0 4px' }}
            onClick={() => onClearEfficiency?.(portIndex)}>✕</button>
        </div>
      )}

      {/* State buttons */}
      <div className="segmented-group">
        {STATE_OPTIONS.map(o => (
          <button key={o.value}
            className={`segmented-btn ${o.cls} ${config.state === o.value ? 'active' : ''}`}
            onClick={() => u('state', o.value)}>
            {o.label}
          </button>
        ))}
      </div>

      {/* Band */}
      <div className="band-input-row">
        <input type="number" value={band[0]}
          onChange={e => u('band_mhz', [parseFloat(e.target.value) || 0, band[1]])} />
        <span style={{ alignSelf: 'center', color: 'var(--text-secondary)', fontSize: 10 }}>~</span>
        <input type="number" value={band[1]}
          onChange={e => u('band_mhz', [band[0], parseFloat(e.target.value) || 0])} />
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2, marginBottom: 4 }}>
        {Object.entries(BAND_PRESETS).slice(0, 6).map(([name, range]) => (
          <button key={name} className={`preset-btn ${band[0] === range[0] && band[1] === range[1] ? 'active' : ''}`}
            onClick={() => u('band_mhz', range)}>{name}</button>
        ))}
      </div>

      {/* Matching settings */}
      {config.use_matching && (
        <>
          <div style={{ marginBottom: 4 }}>
            <label style={{ fontSize: 10, color: 'var(--text-secondary)', display: 'block', marginBottom: 2 }}>Max Components</label>
            <div className="pill-group">
              {[1, 2, 3, 4].map(n => (
                <button key={n} className={`pill-btn ${config.max_components === n ? 'active' : ''}`}
                  onClick={() => u('max_components', n)}>{n}</button>
              ))}
            </div>
          </div>
          <div className="comp-range-row">
            <div className="comp-range-group">
              <label>L (nH)</label>
              <div className="flex-row gap-1">
                <input type="number" step="0.1" value={config.l_min_nh}
                  onChange={e => u('l_min_nh', parseFloat(e.target.value) || 0.1)} />
                <span style={{ color: 'var(--text-secondary)', fontSize: 10 }}>~</span>
                <input type="number" step="0.1" value={config.l_max_nh}
                  onChange={e => u('l_max_nh', parseFloat(e.target.value) || 20)} />
              </div>
            </div>
            <div className="comp-range-group">
              <label>C (pF)</label>
              <div className="flex-row gap-1">
                <input type="number" step="0.1" value={config.c_min_pf}
                  onChange={e => u('c_min_pf', parseFloat(e.target.value) || 0.1)} />
                <span style={{ color: 'var(--text-secondary)', fontSize: 10 }}>~</span>
                <input type="number" step="0.1" value={config.c_max_pf}
                  onChange={e => u('c_max_pf', parseFloat(e.target.value) || 20)} />
              </div>
            </div>
          </div>

          {/* Efficiency file loader */}
          <div style={{ marginTop: 4 }}>
            <button className="btn btn-sm" style={{ width: '100%', fontSize: 10 }}
              onClick={() => setShowEffInput(!showEffInput)}>
              {showEffInput ? 'Cancel' : efficiencyData ? 'Change η_rad' : 'Load η_rad'}
            </button>
            {showEffInput && (
              <div style={{ marginTop: 4, display: 'flex', gap: 4 }}>
                <input type="text" value={effPath}
                  onChange={e => setEffPath(e.target.value)}
                  placeholder="path/to/efficiency.txt"
                  style={{ flex: 1, fontSize: 10 }} />
                <button className="btn btn-sm btn-primary"
                  onClick={() => { onLoadEfficiency?.(portIndex, effPath); setShowEffInput(false); }}
                  disabled={!effPath.trim()}>Load</button>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
