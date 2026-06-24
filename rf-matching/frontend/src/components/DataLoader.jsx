import React from 'react';

export default function DataLoader({ dataDirs, setDataDirs, snpFiles, onLoadSNP, loadedSNP, onRefresh }) {
  return (
    <div>
      <h3>Data Setup</h3>
      <div className="form-group">
        <label>Component Library (Murata)</label>
        <input
          value={dataDirs.murata}
          onChange={e => setDataDirs({...dataDirs, murata: e.target.value})}
          placeholder="E:\RF matching\Murata"
        />
      </div>
      <div className="form-group">
        <label>SNP Directory</label>
        <input
          value={dataDirs.snp}
          onChange={e => setDataDirs({...dataDirs, snp: e.target.value})}
          placeholder="E:\RF matching\snp"
        />
      </div>
      <button className="btn btn-sm btn-primary" onClick={onRefresh}>Refresh</button>

      {loadedSNP && (
        <div style={{ marginTop: 6, fontSize: 11, color: 'var(--accent-green)', fontWeight: 600 }}>
          Loaded: {loadedSNP.filename}, {loadedSNP.num_ports} ports
        </div>
      )}

      <div className="file-list">
        {snpFiles.map(f => (
          <div key={f.filename}
            className={`file-item ${loadedSNP?.filename === f.filename ? 'selected' : ''}`}
            onClick={() => onLoadSNP(f.filename)}>
            <div className="file-name">{f.filename}</div>
            <div className="file-meta">
              {f.num_ports}P | {f.freq_count} pts
              | {(f.freq_min_hz / 1e6).toFixed(0)}-{(f.freq_max_hz / 1e6).toFixed(0)} MHz
            </div>
          </div>
        ))}
        {snpFiles.length === 0 && (
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', fontStyle: 'italic', padding: '6px 0' }}>
            No SNP files found
          </div>
        )}
      </div>
    </div>
  );
}
