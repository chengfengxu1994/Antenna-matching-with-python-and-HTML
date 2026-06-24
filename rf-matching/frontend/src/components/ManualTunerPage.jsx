import React, { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import TopologyDiagram from './TopologyDiagram';

const EMPTY_COMP = { connection_type: 'series', comp_type: 'inductor', value: 10.0, use_ideal: true, port: 0 };

export default function ManualTunerPage({ loadedSNP, portConfigs, onBack }) {
  const [targetFreq, setTargetFreq] = useState(64e6);
  const [components, setComponents] = useState([{ ...EMPTY_COMP }]);
  const [result, setResult] = useState(null);
  const [sweepData, setSweepData] = useState(null);
  const [computing, setComputing] = useState(false);

  // Auto-compute on any change
  useEffect(() => {
    if (!loadedSNP || components.length === 0) return;
    const timer = setTimeout(() => compute(), 300);
    return () => clearTimeout(timer);
  }, [components, targetFreq, loadedSNP]);

  async function compute() {
    if (!loadedSNP) return;
    setComputing(true);
    try {
      const port_states = (portConfigs || []).map((p, i) => ({
        port_index: i,
        state: p.state || 'load',
      }));

      const res = await api.manualTune({
        snp_filename: loadedSNP.filename,
        target_frequency_hz: targetFreq,
        input_port: 0,
        port_states: port_states.filter(p => p.port_index !== 0),
        components: components.map(c => ({
          connection_type: c.connection_type,
          port: c.port || 0,
          comp_type: c.comp_type,
          value: c.value,
          use_ideal: c.use_ideal,
        })),
        sweep_start_hz: targetFreq * 0.5,
        sweep_stop_hz: targetFreq * 1.5,
        sweep_points: 200,
      });

      setResult(res);
      if (res.sweep) {
        setSweepData({
          frequencies: res.sweep.frequencies,
          s11_db: res.sweep.s11_db,
          raw_db: res.sweep.s11_db.map(() => 0), // placeholder
          s11_real: res.sweep.s11_real,
          s11_imag: res.sweep.s11_imag,
        });
      }
    } catch (e) {
      console.error('Manual tune failed:', e);
    }
    setComputing(false);
  }

  function addComponent() {
    setComponents([...components, { ...EMPTY_COMP }]);
  }

  function removeComponent(idx) {
    setComponents(components.filter((_, i) => i !== idx));
  }

  function updateComponent(idx, key, val) {
    const updated = [...components];
    updated[idx] = { ...updated[idx], [key]: val };
    setComponents(updated);
  }

  const numPorts = loadedSNP?.num_ports || 1;

  return (
    <div style={{display: 'flex', gap: 16, flex: 1}}>
      {/* Left: Controls */}
      <div style={{width: 340, minWidth: 340, display: 'flex', flexDirection: 'column', gap: 12}}>
        <div className="card">
          <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8}}>
            <h3 style={{margin: 0, border: 'none', padding: 0}}>Manual Tuning</h3>
            <button className="btn btn-sm" onClick={onBack}>Back</button>
          </div>
          <div className="form-group">
            <label>Target Frequency (Hz)</label>
            <input
              type="number"
              value={targetFreq}
              onChange={e => setTargetFreq(parseFloat(e.target.value) || 64e6)}
            />
            <span style={{fontSize: 11, color: 'var(--text-secondary)'}}>
              = {(targetFreq / 1e6).toFixed(2)} MHz
            </span>
          </div>
        </div>

        {/* Topology diagram */}
        <TopologyDiagram
          numPorts={numPorts}
          portConfigs={portConfigs}
          solution={result}
        />

        {/* Component editors */}
        <div className="card">
          <h3>Matching Components</h3>
          {components.map((comp, idx) => (
            <div key={idx} style={{
              padding: 8, marginBottom: 8,
              background: 'var(--bg-input)', borderRadius: 6,
              border: '1px solid var(--border)',
            }}>
              <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6}}>
                <span style={{fontSize: 12, fontWeight: 600}}>#{idx + 1}</span>
                <button
                  onClick={() => removeComponent(idx)}
                  style={{background: 'none', border: 'none', color: 'var(--accent-red)', cursor: 'pointer', fontSize: 14}}
                >
                  x
                </button>
              </div>
              <div style={{display: 'flex', gap: 4, marginBottom: 4}}>
                <select
                  value={comp.connection_type}
                  onChange={e => updateComponent(idx, 'connection_type', e.target.value)}
                  style={{flex: 1, padding: '4px 6px', fontSize: 12, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text)'}}
                >
                  <option value="series">Series</option>
                  <option value="shunt">Shunt</option>
                </select>
                <select
                  value={comp.comp_type}
                  onChange={e => updateComponent(idx, 'comp_type', e.target.value)}
                  style={{flex: 1, padding: '4px 6px', fontSize: 12, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text)'}}
                >
                  <option value="inductor">L (nH)</option>
                  <option value="capacitor">C (pF)</option>
                </select>
              </div>
              <div style={{display: 'flex', gap: 4, alignItems: 'center'}}>
                <input
                  type="number"
                  step="0.1"
                  value={comp.value}
                  onChange={e => updateComponent(idx, 'value', parseFloat(e.target.value) || 0)}
                  style={{flex: 1, padding: '4px 6px', fontSize: 12, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text)'}}
                />
                <span style={{fontSize: 11, color: 'var(--text-secondary)', minWidth: 24}}>
                  {comp.comp_type === 'inductor' ? 'nH' : 'pF'}
                </span>
              </div>
              <label style={{display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, marginTop: 4, cursor: 'pointer'}}>
                <input
                  type="checkbox"
                  checked={comp.use_ideal}
                  onChange={e => updateComponent(idx, 'use_ideal', e.target.checked)}
                />
                Ideal (ignore S2P model)
              </label>
            </div>
          ))}
          <button className="btn btn-sm" onClick={addComponent} style={{width: '100%'}}>
            + Add Component
          </button>
        </div>

        {/* Result summary */}
        {result && (
          <div className="card">
            <h3>Result</h3>
            <div style={{display: 'flex', gap: 12, flexWrap: 'wrap'}}>
              <div className="stat">
                <span className="stat-label">|S11|</span>
                <span className="stat-value" style={{color: result.s11_magnitude < 0.3 ? 'var(--accent-green)' : 'var(--accent-red)'}}>
                  {result.s11_magnitude?.toFixed(4)}
                </span>
              </div>
              <div className="stat">
                <span className="stat-label">Return Loss</span>
                <span className={`stat-value ${result.s11_db > 10 ? 'good' : 'bad'}`}>
                  {result.s11_db?.toFixed(1)} dB
                </span>
              </div>
              <div className="stat">
                <span className="stat-label">VSWR</span>
                <span className="stat-value">
                  {result.vswr?.toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Right: Charts */}
      <div style={{flex: 1, display: 'flex', flexDirection: 'column', gap: 12}}>
        {/* S11 Chart */}
        <div className="card" style={{flex: 1}}>
          <h3>S11 Frequency Sweep</h3>
          <div style={{height: 300}}>
            {result?.sweep ? (
              <S11MiniChart data={result.sweep} targetFreq={targetFreq} />
            ) : (
              <div style={{height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)'}}>
                {computing ? 'Computing...' : 'Configure components to see S11'}
              </div>
            )}
          </div>
        </div>

        {/* Smith Chart */}
        <div className="card" style={{flex: 1}}>
          <h3>Smith Chart</h3>
          <div style={{height: 300}}>
            {result?.sweep ? (
              <SmithMiniChart data={result.sweep} />
            ) : (
              <div style={{height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)'}}>
                Smith chart will appear here
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}


function S11MiniChart({ data, targetFreq }) {
  if (!data?.frequencies) return null;
  const W = 560, H = 280, pad = 40;
  const freqs = data.frequencies;
  const s11db = data.s11_db;

  const fMin = Math.min(...freqs);
  const fMax = Math.max(...freqs);
  const yMin = Math.min(-40, Math.min(...s11db));
  const yMax = Math.max(0, Math.max(...s11db));

  const xScale = f => pad + ((f - fMin) / (fMax - fMin)) * (W - 2 * pad);
  const yScale = v => pad + ((yMax - v) / (yMax - yMin)) * (H - 2 * pad);

  const points = freqs.map((f, i) => `${xScale(f)},${yScale(s11db[i])}`).join(' ');
  const targetX = xScale(targetFreq);

  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`}>
      {/* Grid */}
      <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#dee2e6" />
      <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#dee2e6" />
      {/* -10dB line */}
      {yMin < -10 && <line x1={pad} y1={yScale(-10)} x2={W - pad} y2={yScale(-10)} stroke="#198754" strokeDasharray="4" strokeOpacity={0.5} />}
      {/* Target freq */}
      <line x1={targetX} y1={pad} x2={targetX} y2={H - pad} stroke="#0d6efd" strokeDasharray="4" />
      <text x={targetX} y={pad - 5} textAnchor="middle" fontSize={10} fill="#0d6efd">{(targetFreq / 1e6).toFixed(0)} MHz</text>
      {/* S11 curve */}
      <polyline points={points} fill="none" stroke="#dc3545" strokeWidth={2} />
      {/* Axis labels */}
      <text x={W / 2} y={H - 5} textAnchor="middle" fontSize={10} fill="#6c757d">Frequency</text>
      <text x={5} y={H / 2} textAnchor="middle" fontSize={10} fill="#6c757d" transform={`rotate(-90, 5, ${H / 2})`}>S11 (dB)</text>
    </svg>
  );
}


function SmithMiniChart({ data }) {
  if (!data?.s11_real || !data?.s11_imag) return null;
  const W = 560, H = 280;
  const cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 20;

  const gamma = data.s11_real.map((re, i) => ({ re: data.s11_real[i], im: data.s11_imag[i] }));
  const points = gamma.map(g => `${cx + g.re * R},${cy - g.im * R}`).join(' ');

  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`}>
      {/* Smith circle */}
      <circle cx={cx} cy={cy} r={R} fill="none" stroke="#adb5bd" strokeWidth={1} />
      {/* Center dot */}
      <circle cx={cx} cy={cy} r={2} fill="#adb5bd" />
      {/* Gamma trace */}
      <polyline points={points} fill="none" stroke="#dc3545" strokeWidth={2} />
      {/* Start/end dots */}
      {gamma.length > 0 && (
        <>
          <circle cx={cx + gamma[0].re * R} cy={cy - gamma[0].im * R} r={4} fill="#0d6efd" />
          <circle cx={cx + gamma[gamma.length - 1].re * R} cy={cy - gamma[gamma.length - 1].im * R} r={4} fill="#198754" />
        </>
      )}
      {/* Labels */}
      <text x={cx} y={H - 5} textAnchor="middle" fontSize={10} fill="#6c757d">Re(Gamma)</text>
    </svg>
  );
}
