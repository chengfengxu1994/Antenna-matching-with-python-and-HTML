import React, { useState } from 'react';

export default function BandConfig({ bands, setBands }) {
  const [newStart, setNewStart] = useState(2400);
  const [newEnd, setNewEnd] = useState(2500);

  function addBand() {
    if (newStart >= newEnd) {
      alert('Start frequency must be less than end frequency');
      return;
    }
    setBands([...bands, [newStart, newEnd]]);
    setNewStart(newEnd + 100);
    setNewEnd(newEnd + 200);
  }

  function removeBand(index) {
    setBands(bands.filter((_, i) => i !== index));
  }

  return (
    <div className="card">
      <h3>📡 匹配频段 (Matching Bands)</h3>
      <div style={{fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8}}>
        指定频段内评估匹配效率 | Evaluate efficiency across bands
      </div>
      
      <div className="flex-row" style={{gap: 6, marginBottom: 8}}>
        <div style={{flex: 1}}>
          <label style={{fontSize: 11, color: 'var(--text-secondary)'}}>Start MHz</label>
          <input
            type="number"
            value={newStart}
            onChange={e => setNewStart(parseInt(e.target.value) || 0)}
            style={{width: '100%', padding: '6px 8px', background: 'var(--bg-input)', border: '1px solid var(--border)', borderRadius: 6, color: 'var(--text)', fontSize: 13}}
          />
        </div>
        <span style={{color: 'var(--text-secondary)', alignSelf: 'flex-end', paddingBottom: 8}}>~</span>
        <div style={{flex: 1}}>
          <label style={{fontSize: 11, color: 'var(--text-secondary)'}}>End MHz</label>
          <input
            type="number"
            value={newEnd}
            onChange={e => setNewEnd(parseInt(e.target.value) || 0)}
            style={{width: '100%', padding: '6px 8px', background: 'var(--bg-input)', border: '1px solid var(--border)', borderRadius: 6, color: 'var(--text)', fontSize: 13}}
          />
        </div>
        <button className="btn btn-sm" onClick={addBand} style={{alignSelf: 'flex-end', marginBottom: 0}}>
          + 添加
        </button>
      </div>

      {bands.length > 0 && (
        <div style={{display: 'flex', flexDirection: 'column', gap: 4}}>
          {bands.map((band, i) => (
            <div key={i} className="flex-row" style={{
              justifyContent: 'space-between',
              padding: '4px 8px',
              background: 'var(--bg-input)',
              borderRadius: 6,
              fontSize: 13
            }}>
              <span>{band[0]} ~ {band[1]} MHz ({((band[1]-band[0])/1000).toFixed(1)} GHz带宽)</span>
              <button
                className="btn btn-sm"
                onClick={() => removeBand(i)}
                style={{padding: '2px 6px', fontSize: 11, color: 'var(--accent-red)', borderColor: 'transparent', background: 'transparent'}}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {bands.length === 0 && (
        <div style={{fontSize: 12, color: 'var(--text-secondary)', fontStyle: 'italic', padding: '4px 0'}}>
          未指定频段，仅评估目标频率点 | No bands specified, evaluating target frequency only
        </div>
      )}
    </div>
  );
}
