import React, { useState, useEffect } from 'react';
import { useRef } from 'react';
import { api } from './services/api';
import { isSameElectricalDut } from './utils/dataSource';
import DataLoader from './components/DataLoader';
import TuningPanel from './components/TuningPanel';
import ComponentSeriesSelector from './components/ComponentSeriesSelector';
import MultiScenarioPanel from './components/MultiScenarioPanel';
import ProjectManager from './components/ProjectManager';
import ManualTunerPage from './components/ManualTunerPage';
import ApplicationMenu from './components/ApplicationMenu';

const DEFAULT_DATA_DIRS = { snp: 'data/snp', murata: 'data/Murata', optenni: '', environment: '' };
const LAST_SNP_KEY = 'rfmatch.lastSnpFilename';

function loadSavedDataDirs() {
  try {
    return { ...DEFAULT_DATA_DIRS, ...JSON.parse(localStorage.getItem('rfmatch.dataDirs') || '{}') };
  } catch {
    return DEFAULT_DATA_DIRS;
  }
}

function loadSavedTheme() {
  try {
    return localStorage.getItem('rfmatch.theme') === 'light' ? 'light' : 'dark';
  } catch {
    return 'dark';
  }
}

function loadSavedLastSnp() {
  try {
    return localStorage.getItem(LAST_SNP_KEY) || '';
  } catch {
    return '';
  }
}

export default function App() {
  /* Global state */
  const [backendOnline, setBackendOnline] = useState(false);
  const [snpFiles, setSnpFiles] = useState([]);
  const [invalidSnpFiles, setInvalidSnpFiles] = useState([]);
  const [loadedSNP, setLoadedSNP] = useState(null);
  const [dataDirs, setDataDirs] = useState(loadSavedDataDirs);
  const [dataSourceStatus, setDataSourceStatus] = useState(null);
  const [dataSourceRevision, setDataSourceRevision] = useState(0);
  const [componentCatalogReady, setComponentCatalogReady] = useState(false);
  const [selectedSeries, setSelectedSeries] = useState(null);
  const [componentFilter, setComponentFilter] = useState({
    manufacturers: [], package_codes: [], voltage_codes: [], dielectrics: [],
    maximum_tolerance_pct: null,
    unknown_metadata_policy: 'include',
  });
  const [portConfigs, setPortConfigs] = useState([]);
  const [portEfficiency, setPortEfficiency] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [workspaceMode, setWorkspaceMode] = useState('single');
  const [showProjects, setShowProjects] = useState(false);
  const [projectSnapshot, setProjectSnapshot] = useState(null);
  const [manualWorkspace, setManualWorkspace] = useState(null);
  const [notice, setNotice] = useState(null);
  const [dataRailOpen, setDataRailOpen] = useState(true);
  const [theme, setTheme] = useState(loadSavedTheme);
  const initSequenceRef = useRef(0);

  useEffect(() => { initBackend(); }, []);
  useEffect(() => { localStorage.setItem('rfmatch.dataDirs', JSON.stringify(dataDirs)); }, [dataDirs]);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('rfmatch.theme', theme); } catch {}
  }, [theme]);
  useEffect(() => {
    if (!notice) return undefined;
    const timer = setTimeout(() => setNotice(null), 8000);
    return () => clearTimeout(timer);
  }, [notice]);

  async function initBackend() {
    const sequence = ++initSequenceRef.current;
    setComponentCatalogReady(false);
    let healthOk = false;
    try {
      await api.health();
      healthOk = true;
      if (sequence !== initSequenceRef.current) return;
      setBackendOnline(true);
      const sourceStatus = await api.setDataDirs(
        dataDirs.snp, dataDirs.murata, dataDirs.optenni, dataDirs.environment,
      );
      if (sequence !== initSequenceRef.current) return;
      setDataSourceStatus(sourceStatus);
      setComponentCatalogReady(sourceStatus?.status === 'ok');
      const snpResponse = await refreshSnpFiles(sequence);
      if (sequence !== initSequenceRef.current) return;
      if (!loadedSNP) {
        const savedFilename = loadSavedLastSnp();
        if (savedFilename && snpResponse.files?.some(file => file.filename === savedFilename)) {
          await handleLoadSNP(savedFilename, { restored: true });
        } else if (savedFilename) {
          try { localStorage.removeItem(LAST_SNP_KEY); } catch {}
        }
      }
      if (sequence !== initSequenceRef.current) return;
      setDataSourceRevision(current => current + 1);
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
    } catch (error) {
      if (sequence !== initSequenceRef.current) return;
      setBackendOnline(healthOk);
      setDataSourceStatus(null);
      setComponentCatalogReady(false);
      setNotice({
        type: 'error',
        message: healthOk
          ? `数据源配置失败：${error.message}`
          : `计算引擎连接失败：${error.message}`,
      });
    }
  }

  async function refreshSnpFiles(expectedInitSequence = null) {
    const response = await api.listSNPFiles();
    if (expectedInitSequence !== null && expectedInitSequence !== initSequenceRef.current) {
      return response;
    }
    setSnpFiles(response.files || []);
    setInvalidSnpFiles(response.invalid_files || []);
    return response;
  }

  async function handleLoadSNP(filename, { restored = false } = {}) {
    try {
      const res = await api.loadSNP(filename);
      const sameElectricalDut = isSameElectricalDut(loadedSNP, res);
      setLoadedSNP(res);
      try { localStorage.setItem(LAST_SNP_KEY, res.filename); } catch {}
      if (!sameElectricalDut) {
        setProjectSnapshot(null);
        setManualWorkspace(null);
        const cfgs = Array.from({ length: res.num_ports }, (_, i) => ({
          port_index: i,
          state: 'load',
          enabled: i < 4,
          max_components: 2,
          bands_mhz: i === 0 ? [[2400, 2500]] : [[2400, 2500]],
          band_weights: [1],
          port_weight: 1,
        }));
        setPortConfigs(cfgs);
      }
      setNotice({
        type: 'success',
        message: restored
          ? `已恢复上次使用的 DUT：${filename}`
          : sameElectricalDut
          ? `${filename} 网络数值未变化，已刷新来源信息并保留当前配置。`
          : `${filename} 已载入，可以开始配置匹配目标。`,
      });
    } catch (e) {
      setNotice({ type: 'error', message: `载入失败：${e.message}` });
    }
  }

  function handleProjectLoaded(snapshot) {
    setLoadedSNP({
      filename: snapshot.input_filename,
      num_ports: snapshot.num_ports,
      freq_count: snapshot.freq_count,
      freq_min_hz: snapshot.freq_min_hz,
      freq_max_hz: snapshot.freq_max_hz,
      sha256: snapshot.input_sha256,
      input_verified: snapshot.input_verified,
      restoration_mode: snapshot.restoration_mode,
    });
    const savedPorts = snapshot.tuning_request?.ports;
    setPortConfigs(savedPorts?.length ? savedPorts : Array.from({ length: snapshot.num_ports }, (_, i) => ({
      port_index: i, state: 'load', enabled: i === 0, max_components: 2,
      bands_mhz: [[2400, 2500]], band_weights: [1], port_weight: 1,
    })));
    setProjectSnapshot({ ...snapshot, loadedAt: Date.now() });
    setManualWorkspace(snapshot.manual_workspace || null);
    setSelectedSeries(snapshot.tuning_request?.component_series ?? null);
    setComponentFilter(snapshot.tuning_request?.component_filter || {
      manufacturers: [], package_codes: [], voltage_codes: [], dielectrics: [],
      maximum_tolerance_pct: null,
      unknown_metadata_policy: 'include',
    });
    setWorkspaceMode(snapshot.manual_workspace ? 'manual' : 'single');
    setShowProjects(false);
  }

  return (
    <div className="layout-shell">
      <header className="app-header">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32"><path d="M5 22c5-12 10-12 15 0M12 22c4-7 8-7 12 0M5 9h22M9 5v8M23 5v8" /></svg>
          </div>
          <div>
            <h1>RF Match Studio</h1>
            <span>射频匹配与天线调谐工作台</span>
          </div>
        </div>
        <div className="header-actions">
          <span className={`connection-state ${backendOnline ? 'online' : 'offline'}`}>
            <i />{backendOnline ? '计算引擎在线' : '计算引擎离线'}
          </span>
          {loadedSNP && <span className="window-document" title={loadedSNP.filename}>{loadedSNP.filename}</span>}
        </div>
      </header>

      <ApplicationMenu
        backendOnline={backendOnline}
        dataRailOpen={dataRailOpen}
        loadedSNP={loadedSNP}
        onOpenCatalog={() => setShowSettings(true)}
        onOpenProjects={() => setShowProjects(true)}
        onRefreshFiles={refreshSnpFiles}
        onToggleDataRail={() => setDataRailOpen(value => !value)}
        onToggleTheme={() => setTheme(value => value === 'dark' ? 'light' : 'dark')}
        onWorkspaceChange={setWorkspaceMode}
        theme={theme}
        workspaceMode={workspaceMode}
      />

      {showProjects && (
        <ProjectManager
          loadedSNP={loadedSNP}
          manualWorkspace={manualWorkspace}
          onLoaded={handleProjectLoaded}
          onClose={() => setShowProjects(false)}
        />
      )}

      {showSettings && (
        <div className="modal-backdrop" onClick={() => setShowSettings(false)}>
          <div className="settings-dialog" onClick={e => e.stopPropagation()}>
            <div className="dialog-heading">
              <div><span className="eyebrow">COMPONENT CATALOG</span><h2>元件系列与筛选</h2><p>限定优化器可使用的实测元件范围。</p></div>
              <button className="dialog-close" aria-label="关闭" onClick={() => setShowSettings(false)}>
                &times;
              </button>
            </div>
            <ComponentSeriesSelector
              selectedSeries={selectedSeries}
              setSelectedSeries={setSelectedSeries}
              componentFilter={componentFilter}
              setComponentFilter={setComponentFilter}
              enabled={componentCatalogReady}
            />
          </div>
        </div>
      )}

      {/* Data sidebar + Tuning Panel */}
      <div className={`app-main-shell ${dataRailOpen ? '' : 'data-rail-collapsed'}`}>
        {/* Data sidebar (narrow) */}
        <aside className="data-sidebar">
          <div className="sidebar-title"><span>项目资源</span><small>PROJECT</small></div>
          <DataLoader
            dataDirs={dataDirs} setDataDirs={setDataDirs}
            snpFiles={snpFiles} invalidSnpFiles={invalidSnpFiles} onLoadSNP={handleLoadSNP}
            loadedSNP={loadedSNP} onRefresh={refreshSnpFiles}
            onApplyDataDirs={initBackend}
            dataSourceStatus={dataSourceStatus}
          />
          {portEfficiency && (
            <div className="resource-status">
              <i /> 效率数据已载入
            </div>
          )}
        </aside>

        {/* Main tuning panel */}
        <div className="tuning-host">
          <div className={`workspace-mode-host ${workspaceMode === 'single' ? '' : 'hidden'}`}>
            <TuningPanel
              loadedSNP={loadedSNP}
              portConfigs={portConfigs}
              setPortConfigs={setPortConfigs}
              onRefreshSNP={refreshSnpFiles}
              snpFiles={snpFiles}
              dataDirs={dataDirs}
              setDataDirs={setDataDirs}
              projectSnapshot={projectSnapshot}
              selectedSeries={selectedSeries}
              componentFilter={componentFilter}
              componentCatalogReady={componentCatalogReady}
              onOpenProjects={() => setShowProjects(true)}
            />
          </div>
          <div className={`workspace-mode-host ${workspaceMode === 'manual' ? '' : 'hidden'}`}>
            <ManualTunerPage loadedSNP={loadedSNP} portConfigs={portConfigs}
              setPortConfigs={setPortConfigs}
              active={workspaceMode === 'manual'}
              dataSourceRevision={dataSourceRevision}
              componentCatalogReady={componentCatalogReady}
              restoredWorkspace={projectSnapshot?.manual_workspace}
              restorationKey={projectSnapshot?.loadedAt}
              onWorkspaceChange={setManualWorkspace}
              onBack={() => setWorkspaceMode('single')} />
          </div>
          <div className={`workspace-mode-host ${workspaceMode === 'multi' ? '' : 'hidden'}`}>
            <MultiScenarioPanel snpFiles={snpFiles}
              active={workspaceMode === 'multi'}
              dataSourceRevision={dataSourceRevision}
              componentCatalogReady={componentCatalogReady} />
          </div>
        </div>
      </div>
      {notice && (
        <button className={`app-notice ${notice.type}`} onClick={() => setNotice(null)}>
          <span>{notice.message}</span><b>×</b>
        </button>
      )}
      <footer className="status-bar">
        <div className="status-group">
          <span className="status-item">
            {workspaceMode === 'single' ? '单文件调谐' : workspaceMode === 'manual' ? '手动调谐' : '多场景联合'}
          </span>
          <span className="status-sep" />
          <span className={`status-item truncate ${loadedSNP ? 'mono' : ''}`} title={loadedSNP?.filename || ''}>
            {loadedSNP ? loadedSNP.filename : '未载入 DUT'}
          </span>
        </div>
        <div className="status-group">
          {loadedSNP && (
            <span className="status-item mono">
              {loadedSNP.num_ports} 端口{loadedSNP.freq_count ? ` · ${loadedSNP.freq_count} 频点` : ''}
            </span>
          )}
          {loadedSNP?.freq_min_hz != null && loadedSNP?.freq_max_hz != null && (
            <span className="status-item mono">
              {(loadedSNP.freq_min_hz / 1e6).toFixed(0)}–{(loadedSNP.freq_max_hz / 1e6).toFixed(0)} MHz
            </span>
          )}
          <span className="status-sep" />
          <span className={`status-item ${backendOnline ? 'online' : ''}`}>
            <i />{backendOnline ? '引擎在线' : '引擎离线'}
          </span>
        </div>
      </footer>
    </div>
  );
}
