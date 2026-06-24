import React, { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:8000';

export default function EfficiencyPanel() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [filePath, setFilePath] = useState('');
  const [inlineData, setInlineData] = useState('');
  const [showInline, setShowInline] = useState(false);
  const [error, setError] = useState(null);

  // Check status on mount
  useEffect(() => {
    checkStatus();
  }, []);

  async function checkStatus() {
    try {
      const res = await fetch(`${API_BASE}/api/efficiency/status`);
      const data = await res.json();
      setStatus(data);
    } catch (e) {
      // ignore
    }
  }

  async function handleLoadFile() {
    if (!filePath.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/efficiency/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filepath: filePath.trim() }),
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setStatus({ loaded: true, efficiency: data.efficiency });
      } else {
        setError(data.detail || 'Failed to load');
      }
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  }

  async function handleLoadInline() {
    if (!inlineData.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/efficiency/inline`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: inlineData, filename: 'pasted_data' }),
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setStatus({ loaded: true, efficiency: data.efficiency });
        setShowInline(false);
      } else {
        setError(data.detail || 'Failed to parse');
      }
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  }

  async function handleClear() {
    try {
      await fetch(`${API_BASE}/api/efficiency/clear`, { method: 'POST' });
      setStatus({ loaded: false });
    } catch (e) {
      // ignore
    }
  }

  return (
    <div className="card" style={{ fontSize: 13 }}>
      <h3 style={{ fontSize: 14, marginBottom: 8 }}>
        📡 Radiation Efficiency
        {status?.loaded && <span style={{ color: 'var(--accent-green)', marginLeft: 8, fontSize: 12 }}>✓ Loaded</span>}
      </h3>

      {status?.loaded ? (
        <div>
          <div style={{ padding: '6px 8px', background: 'var(--bg-input)', borderRadius: 6, marginBottom: 8 }}>
            <div>Source: <strong>{status.efficiency.source}</strong></div>
            <div>Freq: {(status.efficiency.freq_min_hz / 1e6).toFixed(0)} - {(status.efficiency.freq_max_hz / 1e6).toFixed(0)} MHz</div>
            <div>Points: {status.efficiency.num_points}</div>
            <div>η_rad: {(status.efficiency.efficiency_min * 100).toFixed(1)}% — {(status.efficiency.efficiency_max * 100).toFixed(1)}%</div>
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 8 }}>
            总效率 = η_rad(f) × (1 - |S11|²)。优化将同时考虑天线辐射效率。
          </div>
          <button className="btn btn-sm" onClick={handleClear} style={{ width: '100%' }}>
            Clear Efficiency Data
          </button>
        </div>
      ) : (
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 8 }}>
            加载天线辐射效率数据（来自 EM 仿真或测量）以启用总效率优化。
            <br />格式: 频率(MHz) 效率(0-1)
          </div>

          {/* File path input */}
          <div className="form-group" style={{ marginBottom: 8 }}>
            <label style={{ fontSize: 11 }}>文件路径</label>
            <div style={{ display: 'flex', gap: 4 }}>
              <input
                type="text"
                value={filePath}
                onChange={e => setFilePath(e.target.value)}
                placeholder="E:\path\to\efficiency.txt"
                style={{ flex: 1, fontSize: 12 }}
              />
              <button className="btn btn-sm btn-primary" onClick={handleLoadFile} disabled={loading}>
                {loading ? '...' : 'Load'}
              </button>
            </div>
          </div>

          {/* Or toggle inline paste */}
          <button
            className="btn btn-sm"
            onClick={() => setShowInline(!showInline)}
            style={{ width: '100%', marginBottom: showInline ? 8 : 0 }}
          >
            {showInline ? 'Cancel' : 'Paste Data Inline'}
          </button>

          {showInline && (
            <div>
              <textarea
                value={inlineData}
                onChange={e => setInlineData(e.target.value)}
                placeholder={"% Freq(MHz) eff(lin)\n700  0.833\n740  0.896\n780  0.937\n..."}
                style={{
                  width: '100%', height: 100, fontSize: 11, fontFamily: 'monospace',
                  background: 'var(--bg-input)', border: '1px solid var(--border)',
                  borderRadius: 4, padding: 6, resize: 'vertical',
                }}
              />
              <button className="btn btn-sm btn-primary" onClick={handleLoadInline} disabled={loading}
                      style={{ width: '100%' }}>
                Parse & Load
              </button>
            </div>
          )}

          {error && (
            <div style={{ color: 'var(--accent-red)', fontSize: 11, marginTop: 4 }}>{error}</div>
          )}
        </div>
      )}
    </div>
  );
}
