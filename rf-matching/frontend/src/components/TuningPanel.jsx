import React, { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import TopologySchematic from './TopologySchematic';

const COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];

/* ── Objective presets (matching backend) ── */
const OBJECTIVE_PRESETS = [
  { name: 'average_efficiency', label: 'Best Average Efficiency', desc: 'Maximize average total efficiency across all ports and bands' },
  { name: 'worst_case', label: 'Best Worst-Case', desc: 'Maximize the minimum (worst) efficiency' },
  { name: 'balanced', label: 'Balanced', desc: 'Equal emphasis on average and worst-case' },
  { name: 'low_coupling', label: 'Low Coupling / MIMO Safe', desc: 'Heavily penalize inter-port coupling' },
  { name: 'low_cost', label: 'Low Component Count', desc: 'Penalize complex multi-component networks' },
];

/* ── Tuning modes ── */
const TUNING_MODES = [
  { name: 'fixed_lc', label: 'Fixed LC', desc: 'Traditional fixed component matching' },
  { name: 'tunable_c', label: 'Tunable Capacitor', desc: 'One or more variable capacitor positions' },
  { name: 'switch', label: 'Switch States', desc: 'Multi-state switch with per-branch LC' },
];

/* ── Band presets (matching backend) ── */
const BAND_PRESETS = {
  "GPS L1": [1574, 1576], "GPS L5": [1176, 1177],
  "WiFi 2.4GHz": [2400, 2500], "WiFi 5GHz": [5150, 5850],
  "LTE B1": [1920, 2170], "LTE B3": [1710, 1880],
  "LTE B7": [2500, 2690], "5G n77": [3300, 4200],
  "5G n78": [3300, 3800], "5G n79": [4400, 5000],
  "Bluetooth": [2400, 2480], "NB-IoT": [700, 960],
};

/* ── Sub-components ── */

function ModeSelector({ mode, setMode }) {
  return (
    <div className="card">
      <h3>1. Tuning Mode</h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {TUNING_MODES.map(m => (
          <button key={m.name}
            className={`btn btn-sm ${mode === m.name ? 'btn-primary' : ''}`}
            onClick={() => setMode(m.name)}
            style={{ textAlign: 'left', padding: '8px 10px' }}>
            <strong>{m.label}</strong>
            <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 2 }}>{m.desc}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

function BandEditor({ bands, setBands }) {
  const addBand = () => setBands([...bands, [2400, 2500]]);
  const removeBand = (i) => setBands(bands.filter((_, j) => j !== i));
  const updateBand = (i, idx, val) => {
    const nb = bands.map((b, j) => j === i ? [idx === 0 ? val : b[0], idx === 1 ? val : b[1]] : b);
    setBands(nb);
  };

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <span style={{ fontSize: 11, fontWeight: 600 }}>Bands:</span>
        <button className="btn btn-xs btn-primary" onClick={addBand}>+ Add</button>
        {/* Band presets dropdown */}
        <select style={{ fontSize: 10, padding: '1px 4px' }}
          onChange={e => {
            if (e.target.value) {
              const preset = BAND_PRESETS[e.target.value];
              if (preset) setBands([...bands, preset]);
            }
            e.target.value = '';
          }}>
          <option value="">Presets...</option>
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
          <button className="btn btn-xs btn-danger" onClick={() => removeBand(i)} style={{ padding: '1px 6px' }}>×</button>
          <span style={{ fontSize: 9, color: 'var(--text-secondary)' }}>
            {((b[0] + b[1]) / 2).toFixed(0)} MHz
          </span>
        </div>
      ))}
    </div>
  );
}

function PortConfigPanel({ portConfigs, setPortConfigs, numPorts }) {
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

  return (
    <div className="card">
      <h3>2. Port Configuration</h3>
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
            <strong style={{ fontSize: 12 }}>Port {i + 1}</strong>
            <label style={{ fontSize: 10, marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 4 }}>
              <input type="checkbox" checked={pc.enabled} onChange={e => updatePort(i, 'enabled', e.target.checked)} />
              Enable
            </label>
          </div>

          {pc.enabled && (
            <>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontSize: 10 }}>Max components:</span>
                {[1, 2, 3, 4].map(n => (
                  <button key={n}
                    className={`btn btn-xs ${pc.max_components === n ? 'btn-primary' : ''}`}
                    onClick={() => updatePort(i, 'max_components', n)}
                  >{n}</button>
                ))}
              </div>

              <BandEditor
                bands={pc.bands_mhz}
                setBands={(nb) => updatePort(i, 'bands_mhz', nb)}
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
    <div className="card">
      <h3>3. Objective</h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {OBJECTIVE_PRESETS.map(o => (
          <label key={o.name} style={{
            display: 'flex', alignItems: 'flex-start', gap: 6,
            padding: '4px 6px', borderRadius: 4, cursor: 'pointer',
            background: objective === o.name ? 'rgba(52,152,219,0.1)' : 'transparent',
            fontSize: 11,
          }}>
            <input type="radio" name="objective" checked={objective === o.name}
              onChange={() => setObjective(o.name)} style={{ marginTop: 2 }} />
            <div>
              <strong>{o.label}</strong>
              <div style={{ color: 'var(--text-secondary)', fontSize: 10 }}>{o.desc}</div>
            </div>
          </label>
        ))}
      </div>
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

function ResultsTable({ solutions, onSelectSolution, selectedIndex }) {
  const [sort, setSort] = useState({ key: 'score', dir: 'desc' });

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
            <th>#</th>
            <SortHeader label="Score" sortKey="score" currentSort={sort} onSort={handleSort} />
            <SortHeader label="Avg η" sortKey="avg_eff" currentSort={sort} onSort={handleSort} unit="%" />
            <SortHeader label="Min η" sortKey="min_eff" currentSort={sort} onSort={handleSort} unit="%" />
            <SortHeader label="Coupling" sortKey="coupling" currentSort={sort} onSort={handleSort} unit="%" />
            <SortHeader label="Comp.Loss" sortKey="comp_loss" currentSort={sort} onSort={handleSort} unit="%" />
            <SortHeader label="#Comps" sortKey="comp_count" currentSort={sort} onSort={handleSort} />
          </tr>
        </thead>
        <tbody>
          {sortedRows.map((r) => (
            <tr key={r.index}
              className={`clickable-row ${selectedIndex === r.index ? 'selected' : ''}`}
              onClick={() => onSelectSolution(r.index)}
              style={{ opacity: r.index < 5 ? 1 : 0.7 }}
            >
              <td style={{ fontSize: 10, color: 'var(--text-secondary)' }}>{r.index + 1}</td>
              <td style={{ fontWeight: 600, fontFamily: 'monospace', color: scoreColor(r.score) }}>
                {(r.score * 100).toFixed(1)}%
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
            </tr>
          ))}
        </tbody>
      </table>
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

function EfficiencyChart({ sweepData, sweepsByPort }) {
  // Multi-port mode
  if (sweepsByPort && Object.keys(sweepsByPort).length > 0) {
    return <MultiPortEffChart sweepsByPort={sweepsByPort} />;
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
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>Efficiency vs Frequency</div>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', maxHeight: h }}>
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
      <div style={{ fontSize: 9, color: 'var(--text-secondary)', textAlign: 'center' }}>Frequency (GHz)</div>
    </div>
  );
}

/* ── S11 Chart (simple SVG) ── */

function S11Chart({ sweepData, sweepsByPort }) {
  // Multi-port mode
  if (sweepsByPort && Object.keys(sweepsByPort).length > 0) {
    return <MultiPortS11Chart sweepsByPort={sweepsByPort} />;
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
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>Return Loss vs Frequency</div>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', maxHeight: h }}>
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
        Frequency (GHz) <span style={{ marginLeft: 8 }}>— Matched <span style={{ color: '#e74c3c' }}>●</span></span>
        {raw_db && <span style={{ marginLeft: 8 }}>— Raw <span style={{ color: '#ccc' }}>┈</span></span>}
      </div>
    </div>
  );
}

/* ── Multi-port chart helpers ── */

function MultiPortS11Chart({ sweepsByPort }) {
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
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>Return Loss vs Frequency (All Ports)</div>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', maxHeight: h }}>
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
      <div style={{ fontSize: 9, color: 'var(--text-secondary)', textAlign: 'center' }}>Frequency (GHz)</div>
    </div>
  );
}

function MultiPortEffChart({ sweepsByPort }) {
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
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>Efficiency vs Frequency (All Ports)</div>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', maxHeight: h }}>
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
      <div style={{ fontSize: 9, color: 'var(--text-secondary)', textAlign: 'center' }}>Frequency (GHz)</div>
    </div>
  );
}

/* ── Solution detail components for a selected solution ── */

function SolutionDetails({ solution, portIndex }) {
  if (!solution) return null;

  const perPort = solution.per_port || {};
  const entries = Object.entries(perPort);
  if (entries.length === 0) return null;

  // Focus on selected port or show all
  const focusPorts = portIndex != null
    ? entries.filter(([k]) => parseInt(k) === portIndex)
    : entries;

  return (
    <div>
      <h4>Per-Port Metrics</h4>
      {focusPorts.map(([piStr, pm]) => {
        const pi = parseInt(piStr);
        const comps = pm.components || [];
        return (
          <div key={pi} style={{
            marginBottom: 8, padding: 8, borderRadius: 6,
            background: 'rgba(0,0,0,0.02)', border: '1px solid rgba(0,0,0,0.06)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: COLORS[pi % COLORS.length] }} />
              <strong>Port {pi + 1}</strong>
            </div>
            <div style={{ fontSize: 11, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <span>RL: <strong style={{ color: s11Color(pm.s11_db) }}>{pm.s11_db?.toFixed(1)} dB</strong></span>
              <span>η_mismatch: <strong>{((pm.mismatch_efficiency || 0) * 100).toFixed(1)}%</strong></span>
              <span>Coupling: <strong>{((pm.coupling_loss || 0) * 100).toFixed(1)}%</strong></span>
              <span>η_total: <strong style={{ color: effColor((pm.total_efficiency || 0) * 100) }}>
                {((pm.total_efficiency || 0) * 100).toFixed(1)}%</strong></span>
            </div>
            {comps.length > 0 && (
              <div style={{ marginTop: 4, fontSize: 10 }}>
                <strong>Components:</strong>{' '}
                {comps.map((c, ci) => (
                  <span key={ci} style={{
                    display: 'inline-block', padding: '1px 6px', margin: '1px 2px',
                    borderRadius: 3, background: 'rgba(52,152,219,0.1)', fontSize: 10,
                  }}>
                    {c.type === 'inductor' ? 'L' : 'C'}: {c.value}
                  </span>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ── Main TuningPanel ── */

function normalizeComponentForManual(component, fallbackPort) {
  const type = component?.comp_type || component?.type || 'capacitor';
  const parsedValue = parseFloat(String(component?.value || '').replace(/[^\d.]/g, ''));
  const value = component?.nominal_value ?? (Number.isFinite(parsedValue) ? parsedValue : 1);
  return {
    comp_type: type,
    type,
    connection_type: component?.connection_type || 'series',
    value,
    nominal_value: value,
    nominal_unit: component?.nominal_unit || (type === 'inductor' ? 'nH' : 'pF'),
    port: component?.port ?? fallbackPort ?? 0,
    part_number: component?.part_number || component?.part || '',
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
      use_ideal: true,
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
}) {
  const [tuningMode, setTuningMode] = useState('fixed_lc');
  const [objective, setObjective] = useState('balanced');
  const [beamWidth, setBeamWidth] = useState(10);
  const [timeout, setTimeout_] = useState(120);
  const [bandPoints, setBandPoints] = useState(10);

  const [optimizing, setOptimizing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [selectedSolution, setSelectedSolution] = useState(0);
  const [selectedPort, setSelectedPort] = useState(null);
  const [sweepData, setSweepData] = useState(null);
  const [sweepDataByPort, setSweepDataByPort] = useState({});
  const [activeTab, setActiveTab] = useState('summary');
  const [manualComponents, setManualComponents] = useState({});
  const [manualBusy, setManualBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  const timerRef = React.useRef(null);

  useEffect(() => {
    if (optimizing) {
      const t0 = Date.now();
      timerRef.current = setInterval(() => setElapsed(((Date.now() - t0) / 1000).toFixed(1)), 200);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [optimizing]);

  /* ── Run tuning ── */
  async function handleRun() {
    setOptimizing(true);
    setResults(null);
    setSweepData(null);
    setSweepDataByPort({});
    setError(null);

    try {
      const enabledPorts = portConfigs.filter(p => p.enabled);

      if (enabledPorts.length === 0) {
        throw new Error('At least one port must be enabled');
      }

      // Determine mode from tuningMode + port count
      let mode = 'joint';
      let extraParams = {};

      if (tuningMode === 'tunable_c' || tuningMode === 'switch') {
        mode = tuningMode === 'tunable_c' ? 'tunable' : 'switch';
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

      // Use unified /api/tuning/optimize endpoint
      const res = await api.tuningOptimize({
        ports: enabledPorts.map(p => ({
          port_index: p.port_index,
          bands_mhz: p.bands_mhz,
          max_components: p.max_components,
          enabled: true,
        })),
        objective: objective,
        mode: mode,
        ...extraParams,
        beam_width: beamWidth,
        timeout_seconds: timeout,
        num_band_points: bandPoints,
      });
      setResults(res);

      if (res.solutions?.length > 0) {
        const firstPort = enabledPorts[0].port_index;
        setSelectedPort(firstPort);
        setSelectedSolution(0);
        setManualComponents(componentsFromSolution(res.solutions[0], null));
        setActiveTab('summary');
        await loadAllSweeps(0);
      }
    } catch (e) {
      setError(e.message);
    }
    setOptimizing(false);
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

  const handleSelectSolution = async (idx) => {
    setSelectedSolution(idx);
    // Notify backend
    try { await api.tuningSelect(idx); } catch (e) {}
    if (results?.solutions?.[idx]) {
      setManualComponents(componentsFromSolution(results.solutions[idx], null));
      await loadAllSweeps(idx);
    }
  };

  const handleSelectPort = async (portIdx) => {
    setSelectedPort(portIdx);
    const sol = results?.solutions?.[selectedSolution];
    setManualComponents(componentsFromSolution(sol, portIdx));
    await loadAllSweeps(selectedSolution);
  };

  async function evaluateManual(nextComponents) {
    const enabledPorts = portConfigs.filter(p => p.enabled);
    if (enabledPorts.length === 0) return;
    setManualBusy(true);
    const flatComps = flattenManualComponents(nextComponents);
    const newSweeps = {};
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
        console.warn(`Manual realtime tuning failed for port ${pc.port_index}:`, e.message);
      }
    }));
    setSweepDataByPort(newSweeps);
    setSweepData(firstSweep);
    setManualBusy(false);
  }

  const handleManualComponentChange = (portIdx, componentIdx, patch) => {
    const next = { ...manualComponents };
    const comps = [...(next[portIdx] || [])];
    const current = comps[componentIdx] || normalizeComponentForManual({}, portIdx);
    const merged = { ...current, ...patch };
    const type = merged.comp_type || merged.type || 'capacitor';
    merged.type = type;
    merged.comp_type = type;
    merged.nominal_unit = type === 'inductor' ? 'nH' : 'pF';
    merged.value = merged.nominal_value ?? merged.value ?? 1;
    comps[componentIdx] = merged;
    next[portIdx] = comps;
    setManualComponents(next);
    window.clearTimeout(handleManualComponentChange._timer);
    handleManualComponentChange._timer = window.setTimeout(() => evaluateManual(next), 250);
  };

  const numEnabled = portConfigs.filter(p => p.enabled).length;
  const solutions = results?.solutions || null;
  const selectedSolutionData = solutions?.[selectedSolution] || solutions?.[0];
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
    <div className="main-body">
      {/* ── Left sidebar ── */}
      <aside className="control-sidebar">
        <ModeSelector mode={tuningMode} setMode={setTuningMode} />

        <PortConfigPanel
          portConfigs={portConfigs}
          setPortConfigs={setPortConfigs}
          numPorts={loadedSNP?.num_ports || 0}
        />

        <ObjectiveSelector objective={objective} setObjective={setObjective} />

        {/* Run area */}
        <div className="run-area">
          <button className="run-btn" disabled={optimizing || numEnabled === 0}
            onClick={handleRun}>
            {optimizing ? 'Optimizing...' :
             numEnabled === 0 ? 'Enable a port' :
             numEnabled === 1 ? 'Tune Single Port' :
             'Joint Tune'}
          </button>
          {optimizing && (
            <div className="run-timer">Elapsed: {elapsed}s</div>
          )}
        </div>

        {/* Advanced */}
        <div style={{ marginTop: 8 }}>
          <button className="advanced-toggle" onClick={() => {
            const el = document.getElementById('tune-advanced');
            if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
          }}>
            <span>Advanced Settings</span>
            <span>[+]</span>
          </button>
          <div id="tune-advanced" style={{ display: 'none', padding: 8 }}>
            <div className="form-group">
              <label>Beam Width</label>
              <input type="number" value={beamWidth} onChange={e => setBeamWidth(+e.target.value || 10)} />
            </div>
            <div className="form-group">
              <label>Timeout (s)</label>
              <input type="number" value={timeout} onChange={e => setTimeout_(+e.target.value || 60)} />
            </div>
            <div className="form-group">
              <label>Band Points</label>
              <select value={bandPoints} onChange={e => setBandPoints(+e.target.value)}>
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
            <h3>Antenna Tuning — Optenni Style</h3>
            <p>
              Configure ports and objective in the left panel, then click Run.
            </p>
            <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 8 }}>
              {loadedSNP
                ? `${loadedSNP.filename} loaded (${loadedSNP.num_ports}P, ${loadedSNP.freq_count} freq points)`
                : 'Load an SNP file from the Data panel first.'}
            </p>
          </div>
        ) : (
          <>
            <div className="tab-strip">
              <TabButton id="summary" activeTab={activeTab} setActiveTab={setActiveTab}>Summary / Ranking</TabButton>
              <TabButton id="curves" activeTab={activeTab} setActiveTab={setActiveTab}>S11 / Efficiency</TabButton>
              <TabButton id="topology" activeTab={activeTab} setActiveTab={setActiveTab}>Topology</TabButton>
              <TabButton id="power" activeTab={activeTab} setActiveTab={setActiveTab}>Power Balance</TabButton>
              <TabButton id="realtime" activeTab={activeTab} setActiveTab={setActiveTab}>Realtime Tune</TabButton>
            </div>

            {activeTab === 'summary' && (
              <>
            {/* System Summary */}
            {results.best_solution && (
              <div className="workspace-card">
                <h3>System Summary</h3>
                <div className="metric-row">
                  <div className="metric-tile">
                    <div className="metric-label">Score</div>
                    <div className={`metric-value ${((results.best_solution.system_score || results.best_solution.balanced_score || 0)) > 0.7 ? 'good' : 'warn'}`}>
                      {((results.best_solution.system_score || results.best_solution.balanced_score || 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-label">System η</div>
                    <div className={`metric-value ${(results.system_efficiency_pct || 0) > 50 ? 'good' : (results.system_efficiency_pct || 0) > 20 ? 'warn' : 'bad'}`}>
                      {(results.system_efficiency_pct || 0).toFixed(1)}<span className="metric-unit">%</span>
                    </div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-label">Avg η (enabled ports)</div>
                    <div className={`metric-value ${(results.best_solution.avg_total_efficiency || 0) > 0.7 ? 'good' : 'warn'}`}>
                      {((results.best_solution.avg_total_efficiency || 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-label">Min η_total</div>
                    <div className={`metric-value ${(results.best_solution.min_total_efficiency || 0) > 0.7 ? 'good' : 'warn'}`}>
                      {((results.best_solution.min_total_efficiency || 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-label">Coupling Loss</div>
                    <div className={`metric-value ${(results.best_solution.max_coupling_loss || 1) < 0.03 ? 'good' : (results.best_solution.max_coupling_loss || 1) < 0.1 ? 'warn' : 'bad'}`}>
                      {((results.best_solution.max_coupling_loss || 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-label">Comp. Loss</div>
                    <div className={`metric-value ${((results.best_solution.total_component_loss || results.best_solution.component_loss_total || 0)) < 0.05 ? 'good' : 'warn'}`}>
                      {((results.best_solution.total_component_loss || results.best_solution.component_loss_total || 0) * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-label">Solutions</div>
                    <div className="metric-value">{results.solutions_count || 0}</div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-label">Time</div>
                    <div className="metric-value">{(results.total_time_s || 0).toFixed(1)}s</div>
                  </div>
                </div>
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
              <div className="workspace-card">
                <h3>Ranked Solutions
                  <span style={{ fontSize: 11, fontWeight: 400, color: 'var(--text-secondary)', marginLeft: 8 }}>
                    (click to select)
                  </span>
                </h3>
                <ResultsTable
                  solutions={solutions}
                  onSelectSolution={handleSelectSolution}
                  selectedIndex={selectedSolution}
                />
              </div>
            )}
              </>
            )}

            {/* Charts row */}
            {activeTab === 'curves' && sweepData && (
              <div className="chart-grid">
                <div className="chart-card">
                  <S11Chart sweepData={sweepData} sweepsByPort={sweepDataByPort} />
                </div>
                <div className="chart-card">
                  <EfficiencyChart sweepData={sweepData} sweepsByPort={sweepDataByPort} />
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
              <div className="workspace-card">
                <h3>Topology Diagram</h3>
                <TopologySchematic
                  solution={selectedSolutionData}
                  portIndex={null}
                />
                {/* Port selector for solution details */}
                <div style={{ display: 'flex', gap: 4, marginTop: 12, marginBottom: 8 }}>
                  {portConfigs.filter(p => p.enabled).map(pc => (
                    <button key={pc.port_index}
                      className={`btn btn-xs ${selectedPort === pc.port_index ? 'btn-primary' : ''}`}
                      onClick={() => handleSelectPort(pc.port_index)}>
                      Port {pc.port_index + 1}
                    </button>
                  ))}
                </div>
                <h3 style={{ marginTop: 14 }}>Solution Details</h3>
                <SolutionDetails
                  solution={selectedSolutionData}
                  portIndex={selectedPort}
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
                    {sweepData && <S11Chart sweepData={sweepData} sweepsByPort={sweepDataByPort} />}
                    {sweepData && <EfficiencyChart sweepData={sweepData} sweepsByPort={sweepDataByPort} />}
                  </div>
                  <TopologySchematic
                    solution={manualSolution}
                    portIndex={selectedPort}
                    editable
                    onComponentChange={handleManualComponentChange}
                  />
                </div>
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
    </div>
  );
}
