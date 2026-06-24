import React, { useState } from 'react';
import { api } from '../services/api';

export default function ManualTuner({ targetFreqHz, onResult, portStates, inputPort }) {
  const [components, setComponents] = useState([
    { comp_type: 'inductor', connection_type: 'series', value: 10, port: 0 },
  ]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  function addComponent() {
    setComponents([...components, {
      comp_type: 'inductor',
      connection_type: 'series',
      value: 10,
      port: 0,
    }]);
  }

  function removeComponent(idx) {
    setComponents(components.filter((_, i) => i !== idx));
  }

  function updateComponent(idx, field, value) {
    const updated = [...components];
    updated[idx] = { ...updated[idx], [field]: value };
    setComponents(updated);
  }

  async function handleTune() {
    setLoading(true);
    try {
      const req = {
        snp_filename: '',
        target_frequency_hz: targetFreqHz,
        input_port: inputPort,
        port_states: Object.entries(portStates).map(([port, state]) => ({
          port_index: parseInt(port),
          state: state,
        })),
        components: components.map(c => ({
          ...c,
          use_ideal: true,
        })),
        sweep_start_hz: targetFreqHz * 0.5,
        sweep_stop_hz: targetFreqHz * 1.5,
        sweep_points: 100,
      };
      const res = await api.manualTune(req);
      setResult(res);
      if (onResult) onResult(res);
    } catch (e) {
      alert('Tuning failed: ' + e.message);
    }
    setLoading(false);
  }

  return (
    <div className="card">
      <h3>Manual Tuning</h3>

      {components.map((c, i) => (
        <div key={i} className="flex-row" style={{marginBottom:8, flexWrap:'wrap'}}>
          <select
            value={c.connection_type}
            onChange={e => updateComponent(i, 'connection_type', e.target.value)}
            style={{width:80, padding:'4px 6px', fontSize:12, background:'var(--bg-input)', color:'var(--text)', border:'1px solid var(--border)', borderRadius:4}}
          >
            <option value="series">Series</option>
            <option value="shunt">Shunt</option>
          </select>
          <select
            value={c.comp_type}
            onChange={e => updateComponent(i, 'comp_type', e.target.value)}
            style={{width:90, padding:'4px 6px', fontSize:12, background:'var(--bg-input)', color:'var(--text)', border:'1px solid var(--border)', borderRadius:4}}
          >
            <option value="inductor">Inductor</option>
            <option value="capacitor">Capacitor</option>
          </select>
          <input
            type="number"
            value={c.value}
            onChange={e => updateComponent(i, 'value', parseFloat(e.target.value) || 0)}
            style={{width:80, padding:'4px 6px', fontSize:12, background:'var(--bg-input)', color:'var(--text)', border:'1px solid var(--border)', borderRadius:4}}
            step={c.comp_type === 'inductor' ? 0.1 : 0.5}
          />
          <span style={{fontSize:11, color:'var(--text-secondary)', minWidth:30}}>
            {c.comp_type === 'inductor' ? 'nH' : 'pF'}
          </span>
          <button className="btn btn-sm" onClick={() => removeComponent(i)} style={{color:'var(--accent-red)'}}>X</button>
        </div>
      ))}

      <div className="flex-row" style={{gap:8, marginBottom:8}}>
        <button className="btn btn-sm" onClick={addComponent}>+ Add Component</button>
        <button
          className="btn btn-sm btn-primary"
          onClick={handleTune}
          disabled={loading}
        >
          {loading ? 'Computing...' : 'Tune'}
        </button>
      </div>

      {result && (
        <div className="solution-stats">
          <div className="stat">
            <span className="stat-label">|S11|</span>
            <span className="stat-value" style={{color: result.s11_magnitude < 0.3 ? 'var(--accent-green)' : 'var(--accent-red)'}}>
              {result.s11_magnitude?.toFixed(4)}
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">Return Loss</span>
            <span className={`stat-value ${result.s11_db > 15 ? 'good' : ''}`}>
              {result.s11_db?.toFixed(1)} dB
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">VSWR</span>
            <span className="stat-value">
              {result.vswr?.toFixed(2)}
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">Zin</span>
            <span className="stat-value" style={{fontSize:12}}>
              {result.input_impedance_real?.toFixed(1)} + j{result.input_impedance_imag?.toFixed(1)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
