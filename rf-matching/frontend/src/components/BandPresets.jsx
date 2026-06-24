import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

const PRESET_GROUPS = {
  'GPS': ['GPS L1', 'GPS L5', 'GPS L1+L5'],
  'WiFi': ['WiFi 2.4GHz', 'WiFi 5GHz', 'WiFi 6GHz'],
  'Cellular': ['LTE B1', 'LTE B3', 'LTE B7', 'LTE B40', '5G n77', '5G n78', '5G n79'],
  'IoT': ['Bluetooth', 'NB-IoT', 'LoRa 868', 'ISM 915'],
  'UWB': ['UWB'],
};

export default function BandPresets({ bands, setBands }) {
  const [presets, setPresets] = useState({});
  const [newStart, setNewStart] = useState(2400);
  const [newEnd, setNewEnd] = useState(2500);

  useEffect(() => {
    api.getBandPresets().then(res => setPresets(res.presets || {})).catch(() => {});
  }, []);

  function addPreset(name) {
    const range = presets[name];
    if (!range) return;
    // Check if already added
    if (bands.some(b => b[0] === range[0] && b[1] === range[1])) return;
    setBands([...bands, range]);
  }

  function addCustom() {
    if (newStart >= newEnd) return;
    setBands([...bands, [newStart, newEnd]]);
  }

  function removeBand(index) {
    setBands(bands.filter((_, i) => i !== index));
  }

  return (
    <div className="card">
      <h3>Frequency Bands</h3>

      {/* Preset groups */}
      {Object.entries(PRESET_GROUPS).map(([group, names]) => (
        <div key={group} style={{marginBottom: 8}}>
          <div style={{fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4, fontWeight: 600}}>
            {group}
          </div>
          <div className="preset-grid">
            {names.map(name => {
              const range = presets[name];
              const isActive = range && bands.some(b => b[0] === range[0] && b[1] === range[1]);
              return (
                <button
                  key={name}
                  className={`preset-btn ${isActive ? 'active' : ''}`}
                  onClick={() => addPreset(name)}
                  title={range ? `${range[0]}-${range[1]} MHz` : ''}
                >
                  {name}
                </button>
              );
            })}
          </div>
        </div>
      ))}

      {/* Custom band */}
      <div style={{marginTop: 12, paddingTop: 8, borderTop: '1px solid var(--border)'}}>
        <div style={{fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4, fontWeight: 600}}>
          Custom Band (MHz)
        </div>
        <div style={{display: 'flex', gap: 4, marginBottom: 8}}>
          <input
            type="number"
            value={newStart}
            onChange={e => setNewStart(parseInt(e.target.value) || 0)}
            style={{flex: 1, padding: '4px 6px', fontSize: 12, background: 'var(--bg-input)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text)'}}
          />
          <span style={{alignSelf: 'center', color: 'var(--text-secondary)'}}>~</span>
          <input
            type="number"
            value={newEnd}
            onChange={e => setNewEnd(parseInt(e.target.value) || 0)}
            style={{flex: 1, padding: '4px 6px', fontSize: 12, background: 'var(--bg-input)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text)'}}
          />
          <button className="btn btn-sm" onClick={addCustom}>+</button>
        </div>
      </div>

      {/* Active bands */}
      {bands.length > 0 && (
        <div style={{marginTop: 4}}>
          <div style={{fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4, fontWeight: 600}}>
            Active Bands
          </div>
          {bands.map((b, i) => (
            <div key={i} style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              padding: '4px 8px', background: 'var(--bg-input)', borderRadius: 4,
              marginBottom: 4, fontSize: 12, border: '1px solid var(--border)',
            }}>
              <span>{b[0]} ~ {b[1]} MHz</span>
              <button
                onClick={() => removeBand(i)}
                style={{background: 'none', border: 'none', color: 'var(--accent-red)', cursor: 'pointer', fontSize: 14, padding: '0 4px'}}
              >
                x
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
