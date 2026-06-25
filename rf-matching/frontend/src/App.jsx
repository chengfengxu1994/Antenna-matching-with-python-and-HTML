import React, { useState, useEffect } from 'react';
import { api } from './services/api';
import DataLoader from './components/DataLoader';
import TuningPanel from './components/TuningPanel';
import ComponentSeriesSelector from './components/ComponentSeriesSelector';

export default function App() {
  /* Global state */
  const [backendOnline, setBackendOnline] = useState(false);
  const [snpFiles, setSnpFiles] = useState([]);
  const [loadedSNP, setLoadedSNP] = useState(null);
  const [dataDirs, setDataDirs] = useState({ snp: 'E:\\RF matching\\snp', murata: 'E:\\RF matching\\Murata' });
  const [selectedSeries, setSelectedSeries] = useState([]);
  const [portConfigs, setPortConfigs] = useState([]);
  const [portEfficiency, setPortEfficiency] = useState(null);
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => { initBackend(); }, []);

  async function initBackend() {
    try {
      await api.health(); setBackendOnline(true);
      await api.setDataDirs(dataDirs.snp, dataDirs.murata);
      const r = await api.listSNPFiles(); setSnpFiles(r.files || []);
      try {
        const effRes = await api.getEfficiencyStatus();
        if (effRes.loaded) {
          const effMap = {};
          if (effRes.global) effMap[-1] = effRes.global;
          if (effRes.per_port) {
            Object.entries(effRes.per_port).forEach(([k, v]) => { effMap[parseInt(k)] = v; });
          }
          setPortEfficiency(effMap);
        }
      } catch {}
    } catch { setBackendOnline(false); }
  }

  async function handleLoadSNP(filename) {
    try {
      const res = await api.loadSNP(filename);
      setLoadedSNP(res);
      const cfgs = Array.from({ length: res.num_ports }, (_, i) => ({
        port_index: i,
        state: 'load',
        enabled: i < 4,
        max_components: 2,
        bands_mhz: i === 0 ? [[2400, 2500]] : [[2400, 2500]],
      }));
      setPortConfigs(cfgs);
    } catch (e) {
      alert('Load failed: ' + e.message);
    }
  }

  return (
    <div className="layout-shell">
      <header className="app-header">
        <div className="header-left">
          <h1>Antenna Tuning Tool</h1>
          <span className={`status-badge ${backendOnline ? 'online' : 'offline'}`}>
            {backendOnline ? 'Connected' : 'Offline'}
          </span>
          {loadedSNP && (
            <span className="status-badge loaded">
              {loadedSNP.filename} ({loadedSNP.num_ports}P)
            </span>
          )}
        </div>
        <div className="header-right">
          <button className="header-btn" onClick={() => setShowSettings(!showSettings)}>
            Component Series
          </button>
        </div>
      </header>

      {showSettings && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 1000,
          background: 'rgba(0,0,0,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center',
        }} onClick={() => setShowSettings(false)}>
          <div style={{
            background: '#fff', borderRadius: 10, padding: 24,
            width: 500, maxHeight: '80vh', overflow: 'auto',
          }} onClick={e => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
              <h3 style={{ fontSize: 16, fontWeight: 600 }}>Component Series Selection</h3>
              <button onClick={() => setShowSettings(false)} style={{ background: 'none', border: 'none', fontSize: 20, cursor: 'pointer' }}>
                &times;
              </button>
            </div>
            <ComponentSeriesSelector selectedSeries={selectedSeries} setSelectedSeries={setSelectedSeries} />
          </div>
        </div>
      )}

      {/* Data sidebar + Tuning Panel */}
      <div className="app-main-shell">
        {/* Data sidebar (narrow) */}
        <aside className="data-sidebar">
          <DataLoader
            dataDirs={dataDirs} setDataDirs={setDataDirs}
            snpFiles={snpFiles} onLoadSNP={handleLoadSNP}
            loadedSNP={loadedSNP} onRefresh={initBackend}
          />
          {portEfficiency && (
            <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-secondary)' }}>
              <strong>Efficiency:</strong> Loaded
            </div>
          )}
        </aside>

        {/* Main tuning panel */}
        <div className="tuning-host">
          <TuningPanel
            loadedSNP={loadedSNP}
            portConfigs={portConfigs}
            setPortConfigs={setPortConfigs}
            onRefreshSNP={initBackend}
            snpFiles={snpFiles}
            dataDirs={dataDirs}
            setDataDirs={setDataDirs}
          />
        </div>
      </div>
    </div>
  );
}
