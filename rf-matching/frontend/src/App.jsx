import React, { useState, useEffect, useRef, useCallback } from 'react';
import { api } from './services/api';
import DataLoader from './components/DataLoader';
import PerPortConfig from './components/PerPortConfig';
import ResultsPanel from './components/ResultsPanel';
import PowerBalance from './components/PowerBalance';
import S11Chart from './components/S11Chart';
import SmithChart from './components/SmithChart';
import AllPortsChart from './components/AllPortsChart';

export default function App() {
  /* Global state */
  const [backendOnline, setBackendOnline] = useState(false);
  const [snpFiles, setSnpFiles] = useState([]);
  const [loadedSNP, setLoadedSNP] = useState(null);
  const [dataDirs, setDataDirs] = useState({ snp: 'E:\\RF matching\\snp', murata: 'E:\\RF matching\\Murata' });
  const [selectedSeries, setSelectedSeries] = useState([]);
  const [portConfigs, setPortConfigs] = useState([]);
  const [optimizationGoal, setOptimizationGoal] = useState('efficiency');
  const [optimizationModes, setOptimizationModes] = useState([]);
  const [jointResults, setJointResults] = useState(null);
  const [systemMetrics, setSystemMetrics] = useState(null);
  const [optimizing, setOptimizing] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState(null);

  /* Chart linkage */
  const [selectedPort, setSelectedPort] = useState(null);
  const [sweepData, setSweepData] = useState(null);
  const timerRef = useRef(null);

  /* Advanced panel */
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [beamWidth, setBeamWidth] = useState(10);
  const [timeout, setTimeout_] = useState(120);

  /* Efficiency */
  const [portEfficiency, setPortEfficiency] = useState(null); // {portIndex: data}

  useEffect(() => { initBackend(); }, []);
  useEffect(() => {
    if (optimizing) {
      const t0 = Date.now();
      timerRef.current = setInterval(() => setElapsed(((Date.now() - t0) / 1000).toFixed(1)), 200);
    } else {
      clearInterval(timerRef.current);
      setElapsed(0);
    }
    return () => clearInterval(timerRef.current);
  }, [optimizing]);

  async function initBackend() {
    try {
      await api.health(); setBackendOnline(true);
      await api.setDataDirs(dataDirs.snp, dataDirs.murata);
      const r = await api.listSNPFiles(); setSnpFiles(r.files || []);
      // Fetch optimization modes
      try {
        const modesRes = await api.getOptimizationModes();
        setOptimizationModes(modesRes.modes || []);
      } catch {}
      // Fetch efficiency status
      try {
        const effRes = await api.getEfficiencyStatus();
        if (effRes.loaded) {
          const effMap = {};
          if (effRes.global) effMap[-1] = effRes.global;
          if (effRes.per_port) {
            Object.entries(effRes.per_port).forEach(([k, v]) => {
              effMap[parseInt(k)] = v;
            });
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
      setJointResults(null); setSystemMetrics(null); setSweepData(null);
      setSelectedPort(null);
      const cfgs = Array.from({ length: res.num_ports }, (_, i) => ({
        port_index: i, state: 'load', use_matching: i < 4,
        max_components: 2, l_min_nh: 0.1, l_max_nh: 20, c_min_pf: 0.1, c_max_pf: 20,
        band_mhz: [2400, 2500], num_band_points: 5,
      }));
      setPortConfigs(cfgs);
      setError(null);
    } catch (e) { setError('Load failed: ' + e.message); }
  }

  async function handleLoadEfficiency(portIndex, filepath) {
    try {
      const res = await api.loadEfficiency(portIndex, filepath);
      const effMap = { ...(portEfficiency || {}) };
      effMap[portIndex] = res.efficiency;
      setPortEfficiency(effMap);
    } catch (e) {
      setError('Failed to load efficiency: ' + e.message);
    }
  }

  async function handleClearEfficiency(portIndex) {
    try {
      await api.clearEfficiency(portIndex);
      const effMap = { ...(portEfficiency || {}) };
      delete effMap[portIndex];
      if (portIndex < 0) {
        // Clear all
        setPortEfficiency(null);
      } else {
        setPortEfficiency(Object.keys(effMap).length > 0 ? effMap : null);
      }
    } catch (e) {
      // ignore
    }
  }

  async function handleOptimize() {
    setOptimizing(true); setJointResults(null); setSystemMetrics(null);
    setSweepData(null); setSelectedPort(null); setError(null);
    try {
      const res = await api.jointOptimize({
        snp_filename: loadedSNP?.filename || '',
        ports: portConfigs.map(p => ({
          port_index: p.port_index, state: p.state, use_matching: p.use_matching,
          max_components: p.max_components,
          l_min_nh: p.l_min_nh, l_max_nh: p.l_max_nh,
          c_min_pf: p.c_min_pf, c_max_pf: p.c_max_pf,
          band_mhz: p.band_mhz || [2400, 2500],
          num_band_points: p.num_band_points || 5,
        })),
        beam_width: beamWidth,
        timeout_seconds: timeout,
        optimization_goal: optimizationGoal,
      });
      setJointResults(res);
      setSystemMetrics(res.system_metrics || null);
      if (res.solutions_count > 0 && Object.keys(res.results_per_port || {}).length > 0) {
        const ports = Object.keys(res.results_per_port).map(Number);
        setSelectedPort(ports[0]);
      }
      if (res.warning) setError(res.warning);
    } catch (e) {
      setError('Optimization failed: ' + e.message);
    }
    setOptimizing(false);
  }

  const matchingCount = portConfigs.filter(p => p.use_matching).length;

  /* Handle port selection */
  const handleSelectPort = useCallback(async (portIndex) => {
    setSelectedPort(portIndex);
    // Trigger joint frequency sweep for the selected port
    if (loadedSNP && jointResults?.results_per_port?.[portIndex]) {
      try {
        const pr = jointResults.results_per_port[portIndex];
        const band = pr.band_mhz || [2400, 2500];
        const centerHz = ((band[0] + band[1]) / 2) * 1e6;
        const res = await api.jointSweep(portIndex, centerHz * 0.5, centerHz * 1.5, 200);
        setSweepData(res);
      } catch (e) {
        console.warn('Joint sweep failed:', e.message);
      }
    }
  }, [loadedSNP, jointResults]);

  /* Render */
  return (
    <div className="layout-shell">
      
      <header className="app-header">
        <div className="header-left">
          <h1>RF Matching Tool</h1>
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
            boxShadow: '0 8px 32px rgba(0,0,0,0.15)',
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

      
      <div className="main-body">

        
        <aside className="control-sidebar">
          
          <div className="panel-card">
            <DataLoader
              dataDirs={dataDirs} setDataDirs={setDataDirs}
              snpFiles={snpFiles} onLoadSNP={handleLoadSNP}
              loadedSNP={loadedSNP} onRefresh={initBackend}
            />
          </div>

          
          <div className="panel-card">
            <h3>Port Configuration</h3>
            <PerPortConfig
              numPorts={loadedSNP?.num_ports || 0}
              portConfigs={portConfigs} setPortConfigs={setPortConfigs}
              optimizationGoal={optimizationGoal} setOptimizationGoal={setOptimizationGoal}
              optimizationModes={optimizationModes}
              portEfficiency={portEfficiency}
              onLoadEfficiency={handleLoadEfficiency}
              onClearEfficiency={handleClearEfficiency}
              compact
            />
          </div>

          
          <div className="run-area">
            <button className="run-btn" disabled={optimizing || matchingCount < 2}
              onClick={handleOptimize}>
              {optimizing ? 'Optimizing...' : 'Run Joint Optimization'}
            </button>
            {optimizing && (
              <div className="run-timer">Elapsed: {elapsed}s</div>
            )}
            {!optimizing && matchingCount < 2 && (
              <div className="run-hint">Joint optimization requires at least 2 matching ports</div>
            )}
            {error && (
              <div className="error-card">{error}</div>
            )}
          </div>

          
          <div>
            <button className={`advanced-toggle ${showAdvanced ? 'open' : ''}`}
              onClick={() => setShowAdvanced(!showAdvanced)}>
              <span>Advanced Settings</span>
              <span>{showAdvanced ? '[-]' : '[+]'}</span>
            </button>
            {showAdvanced && (
              <div className="advanced-content">
                <div className="form-group">
                  <label>Beam Width (candidates per port)</label>
                  <input type="number" value={beamWidth} onChange={e => setBeamWidth(+e.target.value || 10)} min={1} />
                </div>
                <div className="form-group">
                  <label>Timeout (seconds)</label>
                  <input type="number" value={timeout} onChange={e => setTimeout_(+e.target.value || 120)} min={10} />
                </div>
                <div className="form-group">
                  <label>Component Series Filter</label>
                  <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                    {selectedSeries.length > 0
                      ? selectedSeries.join(', ')
                      : 'All series (configure in Component Series panel)'}
                  </div>
                </div>
              </div>
            )}
          </div>
        </aside>

        
        <main className="workspace-panel">
          {!jointResults ? (
            /* Empty state */
            <div className="empty-state">
              <h3>RF Matching Tool</h3>
              <p>
                {loadedSNP
                  ? 'Configure ports in the left panel and click "Run Joint Optimization" to begin.'
                  : 'Select an SNP file from the left panel to begin.'}
              </p>
              {!loadedSNP && snpFiles.length > 0 && (
                <p style={{ fontSize: 12, marginTop: 8, color: 'var(--text-secondary)' }}>
                  {snpFiles.length} SNP file{snpFiles.length > 1 ? 's' : ''} available
                </p>
              )}
            </div>
          ) : (
            <>
              {/* System Summary */}
              {systemMetrics && (
                <div className="workspace-card">
                  <h3>System Summary</h3>
                  <div className="metric-row">
                    <div className="metric-tile">
                      <div className="metric-label">Balanced Score</div>
                      <div className={`metric-value ${(systemMetrics.balanced_score || 0) > 0.7 ? 'good' : 'warn'}`}>
                        {((systemMetrics.balanced_score || 0) * 100).toFixed(1)}<span className="metric-unit">%</span>
                      </div>
                    </div>
                    <div className="metric-tile">
                      <div className="metric-label">Min System Efficiency</div>
                      <div className={`metric-value ${(systemMetrics.min_system_efficiency || 0) > 0.7 ? 'good' : 'warn'}`}>
                        {((systemMetrics.min_system_efficiency || 0) * 100).toFixed(1)}<span className="metric-unit">%</span>
                      </div>
                    </div>
                    <div className="metric-tile">
                      <div className="metric-label">Avg System Efficiency</div>
                      <div className={`metric-value ${(systemMetrics.avg_system_efficiency || 0) > 0.7 ? 'good' : 'warn'}`}>
                        {((systemMetrics.avg_system_efficiency || 0) * 100).toFixed(1)}<span className="metric-unit">%</span>
                      </div>
                    </div>
                    <div className="metric-tile">
                      <div className="metric-label">Min Total η</div>
                      <div className={`metric-value ${(systemMetrics.min_total_efficiency || 0) > 0.7 ? 'good' : 'warn'}`}>
                        {((systemMetrics.min_total_efficiency || 0) * 100).toFixed(1)}<span className="metric-unit">%</span>
                      </div>
                    </div>
                    <div className="metric-tile">
                      <div className="metric-label">Max Coupling Loss</div>
                      <div className={`metric-value ${(systemMetrics.max_coupling_loss || 1) < 0.03 ? 'good' : (systemMetrics.max_coupling_loss || 1) < 0.1 ? 'warn' : 'bad'}`}>
                        {((systemMetrics.max_coupling_loss || 0) * 100).toFixed(1)}<span className="metric-unit">%</span>
                      </div>
                    </div>
                    <div className="metric-tile">
                      <div className="metric-label">Comp. Loss</div>
                      <div className={`metric-value ${(systemMetrics.component_loss_total || 0) < 0.05 ? 'good' : 'warn'}`}>
                        {((systemMetrics.component_loss_total || 0) * 100).toFixed(2)}<span className="metric-unit">%</span>
                      </div>
                    </div>
                    <div className="metric-tile">
                      <div className="metric-label">Solutions</div>
                      <div className="metric-value">{jointResults.solutions_count || 0}</div>
                    </div>
                    <div className="metric-tile">
                      <div className="metric-label">Total Time</div>
                      <div className="metric-value">{(jointResults.total_time_s || 0).toFixed(1)}<span className="metric-unit">s</span></div>
                    </div>
                  </div>
                </div>
              )}

              {/* Results Table */}
              <div className="workspace-card">
                <ResultsPanel
                  jointResults={jointResults}
                  systemMetrics={systemMetrics}
                  onSelectPort={handleSelectPort}
                  selectedPort={selectedPort}
                />
              </div>

              {/* Charts */}
              <div className="chart-grid">
                <div className="chart-card">
                  <h4>Port Return Loss Summary</h4>
                  <div className="chart-container" style={{ height: 260 }}>
                    <AllPortsChart
                      jointResults={jointResults}
                      loadedSNP={loadedSNP}
                      selectedPort={selectedPort}
                    />
                  </div>
                </div>
                <div className="chart-card">
                  <h4>Reflection Magnitude Overview</h4>
                  <div className="chart-container" style={{ height: 260 }}>
                    <SmithChart
                      jointResults={jointResults}
                      loadedSNP={loadedSNP}
                      selectedPort={selectedPort}
                    />
                  </div>
                </div>
              </div>

              {/* Frequency sweep chart */}
              <div className="workspace-card" style={{padding: '8px 14px'}}>
                <S11Chart
                  sweepData={sweepData}
                  targetFreqHz={null}
                  bands={null}
                />
              </div>

              {/* Power Balance */}
              <div className="workspace-card">
                <h3>Power Balance</h3>
                <PowerBalance
                  resultsPerPort={jointResults.results_per_port}
                  powerBalance={null}  // computed from results_per_port
                />
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
