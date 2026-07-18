import React, { useEffect, useRef, useState } from 'react';
import { api } from '../services/api';
import {
  CST_TREE_POLL_INTERVAL_MS,
  formatCstExportMessage,
  ingestionMethodLabel,
  shouldPollCstTree,
  sourceLabel,
} from '../utils/dataSource';

const TOUCHSTONE_ACCEPT = Array.from({ length: 64 }, (_, index) => `.s${index + 1}p`).join(',');

export default function DataLoader({
  dataDirs, setDataDirs, snpFiles, invalidSnpFiles = [], onLoadSNP,
  loadedSNP, onRefresh, onApplyDataDirs, dataSourceStatus,
}) {
  const importInputRef = useRef(null);
  const [importing, setImporting] = useState(false);
  const [importMessage, setImportMessage] = useState(null);
  const [importSource, setImportSource] = useState('CST');
  const [dragActive, setDragActive] = useState(false);
  const [watchState, setWatchState] = useState({ active: false, starting: false, pending: [] });
  const [autoLoadLatest, setAutoLoadLatest] = useState(true);
  const [cstConnection, setCstConnection] = useState({ loading: false, status: null, tree: null, error: null });
  const watchIdRef = useRef(null);
  const pollBusyRef = useRef(false);
  const cstTreePollBusyRef = useRef(false);
  const cstRequestRef = useRef(0);
  const callbacksRef = useRef({ onRefresh, onLoadSNP });

  useEffect(() => {
    callbacksRef.current = { onRefresh, onLoadSNP };
  }, [onRefresh, onLoadSNP]);

  useEffect(() => {
    if (!watchState.active || !watchIdRef.current) return undefined;
    let disposed = false;
    async function pollWatch() {
      if (disposed || pollBusyRef.current || !watchIdRef.current) return;
      pollBusyRef.current = true;
      try {
        const status = await api.getSnpWatchStatus(watchIdRef.current);
        if (disposed) return;
        setWatchState(current => ({ ...current, pending: status.pending || [], error: null }));
        if (status.ready?.length) {
          await callbacksRef.current.onRefresh?.();
          const latest = [...status.ready].sort((left, right) => right.mtime_ns - left.mtime_ns)[0];
          if (autoLoadLatest) await callbacksRef.current.onLoadSNP?.(latest.filename);
          setImportMessage({
            type: 'success',
            text: `检测到 ${status.ready.length} 个稳定结果${autoLoadLatest ? `，已载入 ${latest.filename}` : '，文件列表已刷新'}`,
          });
        }
        if (status.invalid?.length) {
          const latestInvalid = status.invalid[status.invalid.length - 1];
          setImportMessage({
            type: 'error',
            text: `${latestInvalid.filename} 写入已结束但校验失败：${latestInvalid.error}`,
          });
        }
      } catch (error) {
        if (!disposed) setWatchState(current => ({ ...current, active: false, error: error.message }));
      } finally {
        pollBusyRef.current = false;
      }
    }
    pollWatch();
    const timer = window.setInterval(pollWatch, 1200);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [watchState.active, autoLoadLatest]);

  useEffect(() => () => {
    if (watchIdRef.current) api.stopSnpWatch(watchIdRef.current).catch(() => {});
  }, []);

  useEffect(() => {
    if (!shouldPollCstTree(cstConnection)) return undefined;
    const timer = window.setInterval(() => {
      if (cstTreePollBusyRef.current) return;
      cstTreePollBusyRef.current = true;
      inspectCst(cstConnection.selected, true)
        .finally(() => { cstTreePollBusyRef.current = false; });
    }, CST_TREE_POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [
    cstConnection.tree?.solver_running,
    cstConnection.selected?.pid,
    cstConnection.selected?.path,
  ]);

  async function toggleWatch() {
    if (watchState.active && watchIdRef.current) {
      const watchId = watchIdRef.current;
      watchIdRef.current = null;
      setWatchState({ active: false, starting: false, pending: [] });
      await api.stopSnpWatch(watchId).catch(() => {});
      return;
    }
    if (watchIdRef.current) {
      await api.stopSnpWatch(watchIdRef.current).catch(() => {});
      watchIdRef.current = null;
    }
    setWatchState({ active: false, starting: true, pending: [] });
    setImportMessage(null);
    try {
      const session = await api.startSnpWatch(importSource, 1000);
      watchIdRef.current = session.watch_id;
      setWatchState({ ...session, active: true, starting: false, pending: [] });
    } catch (error) {
      setWatchState({ active: false, starting: false, pending: [], error: error.message });
      setImportMessage({ type: 'error', text: `无法监视导出目录：${error.message}` });
    }
  }

  async function inspectCst(project = null, quiet = false) {
    const requestId = ++cstRequestRef.current;
    if (!quiet) {
      setCstConnection(current => ({
        ...current,
        loading: true,
        error: null,
        ...(project ? { selected: project, tree: null } : { status: null, selected: null, tree: null }),
      }));
    }
    try {
      const status = project ? cstConnection.status : await api.getCstStatus(true);
      const selected = project || status.projects?.[0] || null;
      const tree = selected ? await api.getCstProjectTree(selected.pid, selected.path) : null;
      if (requestId !== cstRequestRef.current) return;
      setCstConnection(current => ({
        ...current, loading: false, status, selected, tree, error: null,
      }));
    } catch (error) {
      if (requestId !== cstRequestRef.current) return;
      setCstConnection(current => ({
        ...current, loading: false, error: error.message,
        ...(quiet ? {} : { selected: project, tree: null }),
      }));
    }
  }

  async function exportCstResult() {
    const project = cstConnection.selected;
    if (!project) return;
    setCstConnection(current => ({ ...current, exporting: true, error: null }));
    setImportMessage(null);
    try {
      const imported = await api.exportCstTouchstone(project.pid, project.path);
      await onRefresh?.();
      await onLoadSNP?.(imported.filename);
      setImportMessage({
        type: 'success',
        text: formatCstExportMessage(imported),
      });
      setCstConnection(current => ({ ...current, exporting: false, error: null }));
    } catch (error) {
      setCstConnection(current => ({ ...current, exporting: false, error: error.message }));
    }
  }

  async function importTouchstoneFiles(files) {
    const selected = Array.from(files || []);
    if (!selected.length) return;
    setImporting(true);
    setImportMessage(null);
    try {
      const importedFiles = [];
      for (const file of selected) {
        if (!/\.s\d+p$/i.test(file.name)) {
          throw new Error(`${file.name} 不是有效的 *.sNp 文件名`);
        }
        const content = await file.text();
        importedFiles.push(await api.importSNP(file.name, content, importSource));
      }
      await onRefresh?.();
      const imported = importedFiles[importedFiles.length - 1];
      await onLoadSNP?.(imported.filename);
      setImportMessage({
        type: 'success',
        text: importedFiles.length === 1
          ? `${imported.filename} · ${imported.num_ports} 端口 · ${imported.freq_count} 频点 · ${importSource}`
          : `已校验并导入 ${importedFiles.length} 个文件，当前载入 ${imported.filename}`,
      });
    } catch (error) {
      setImportMessage({ type: 'error', text: error.message });
    } finally {
      setImporting(false);
    }
  }

  async function handleTouchstoneImport(event) {
    const files = event.target.files;
    event.target.value = '';
    await importTouchstoneFiles(files);
  }

  async function handleDrop(event) {
    event.preventDefault();
    setDragActive(false);
    await importTouchstoneFiles(event.dataTransfer.files);
  }
  const referenceLabel = values => {
    const refs = values?.reference_impedances_ohm || [];
    if (!refs.length) return '';
    return refs.every(value => Math.abs(value - refs[0]) < 1e-9)
      ? `${refs[0]} Ω`
      : `[${refs.join(', ')}] Ω`;
  };
  const activeSourceLabel = sourceLabel(importSource);
  return (
    <div className="data-loader">
      <div className="resource-section-heading">
        <div><span className="resource-icon">S</span><strong>网络参数</strong></div>
        <button className="icon-button" title="刷新文件列表" onClick={() => onRefresh?.()}>↻</button>
      </div>

      <div className={`em-import-card ${dragActive ? 'drag-active' : ''}`}
        onDragEnter={event => { event.preventDefault(); setDragActive(true); }}
        onDragOver={event => event.preventDefault()}
        onDragLeave={event => {
          if (!event.currentTarget.contains(event.relatedTarget)) setDragActive(false);
        }}
        onDrop={handleDrop}>
        <div><span className="em-source-badge">EM</span><strong>CST / HFSS 结果</strong></div>
        <p>从求解器导出 Touchstone（*.sNp）。支持拖入单文件或一组参数扫描结果，并保留来源信息。</p>
        <div className="em-import-source">
          <label htmlFor="em-source">结果来源</label>
          <select id="em-source" value={importSource} disabled={watchState.active}
            onChange={event => setImportSource(event.target.value)}>
            <option value="CST">CST Studio Suite</option>
            <option value="HFSS">Ansys HFSS</option>
            <option value="VNA">VNA 实测</option>
            <option value="Touchstone">其他 Touchstone</option>
          </select>
        </div>
        <input ref={importInputRef} type="file" hidden multiple
          accept={TOUCHSTONE_ACCEPT}
          onChange={handleTouchstoneImport} />
        <button type="button" className="em-import-button" disabled={importing}
          onClick={() => importInputRef.current?.click()}>
          {importing ? '正在校验并导入…' : '选择或拖入 Touchstone'}
        </button>
        <small className="em-import-hint">自动校验端口数、频率单调性、矩阵完整性与参考阻抗</small>
        <div className={`em-watch-panel ${watchState.active ? 'active' : ''}`}>
          <div className="em-watch-heading">
            <span><i />{watchState.active ? `正在监视 ${activeSourceLabel} 导出目录` : `${activeSourceLabel} 结果自动同步`}</span>
            <button type="button" onClick={toggleWatch} disabled={watchState.starting || importing}>
              {watchState.starting ? '启动中…' : watchState.active ? '停止' : '开始监视'}
            </button>
          </div>
          {watchState.active && <>
            <label className="em-auto-load"><input type="checkbox" checked={autoLoadLatest}
              onChange={event => setAutoLoadLatest(event.target.checked)} /> 自动载入最新稳定结果</label>
            <small>{watchState.pending?.length
              ? `${watchState.pending.length} 个文件仍在写入，等待稳定…`
              : `已建立基线（${watchState.baseline_count} 个文件），等待 ${activeSourceLabel} 新结果`}</small>
          </>}
          {watchState.error && <small className="error">{watchState.error}</small>}
        </div>
        <div className={`cst-direct-panel ${cstConnection.status?.projects?.length ? 'connected' : ''}`}>
          <div className="cst-direct-heading">
            <span><i />CST 官方 Python 接口</span>
            <button type="button" onClick={() => inspectCst()} disabled={cstConnection.loading}>
              {cstConnection.loading ? '检测中…' : cstConnection.status ? '重新检测' : '检测直连'}
            </button>
          </div>
          {cstConnection.status && <small>
            {cstConnection.status.error
              ? 'CST 运行时已识别，但 Python 接口连接失败'
              : cstConnection.status.available
              ? `${cstConnection.status.runtime_version || cstConnection.status.installation?.display_name} · ${cstConnection.status.projects?.length || 0} 个打开工程`
              : '未发现兼容的 CST Studio Suite Python 运行时'}
          </small>}
          {cstConnection.status?.available && !cstConnection.status.error && !cstConnection.status.projects?.length &&
            <small>安装已识别；启动 CST Design Environment 并打开工程后可读取结果树。</small>}
          {cstConnection.status?.projects?.length > 0 && <>
            <select aria-label="CST 打开工程" value={`${cstConnection.selected?.pid}:${cstConnection.selected?.path}`}
              disabled={cstConnection.loading || cstConnection.exporting}
              onChange={event => inspectCst(cstConnection.status.projects.find(project => `${project.pid}:${project.path}` === event.target.value))}>
              {cstConnection.status.projects.map((project, index) =>
                <option key={`${project.pid}:${project.path}`} value={`${project.pid}:${project.path}`}>{project.path.split(/[\\/]/).pop()} · PID {project.pid}</option>)}
            </select>
            {cstConnection.tree && <details>
              <summary>{cstConnection.tree.tree_items.length} 个 S 参数结果节点{cstConnection.tree.solver_running ? ' · 求解中' : ''}</summary>
              {cstConnection.tree.tree_items.slice(0, 12).map(item => <span key={item}>{item}</span>)}
            </details>}
            {cstConnection.tree?.solver_running && <small className="cst-solver-refresh">求解状态每 2 秒自动刷新，完成后可直接导出。</small>}
            <button type="button" className="cst-export-button" onClick={exportCstResult}
              disabled={cstConnection.loading || cstConnection.exporting || cstConnection.tree?.solver_running || !cstConnection.tree?.tree_items?.length}>
              {cstConnection.exporting ? 'CST 正在导出…' : cstConnection.tree?.solver_running ? '等待求解完成' : '导出 S 参数并载入 DUT'}
            </button>
          </>}
          {(cstConnection.error || cstConnection.status?.error) && <small className="error">{cstConnection.error || cstConnection.status.error}</small>}
        </div>
        {importMessage && <div className={`em-import-message ${importMessage.type}`}>{importMessage.text}</div>}
      </div>

      <div className="source-path">
        <label>SNP 数据目录</label>
        <input value={dataDirs.snp} onChange={e => setDataDirs({...dataDirs, snp: e.target.value})} placeholder="data\\snp" />
      </div>

      {loadedSNP && (
        <div className="loaded-file-card">
          <span>当前 DUT</span>
          <strong>{loadedSNP.filename}</strong>
          <small>{loadedSNP.num_ports} 端口
          {loadedSNP.parameter_format ? ` · ${loadedSNP.parameter_format}` : ''}
          {referenceLabel(loadedSNP) ? ` · Z0 ${referenceLabel(loadedSNP)}` : ''}
          </small>
          {loadedSNP.provenance && <small className="loaded-source-provenance">
            {loadedSNP.provenance.source} · {ingestionMethodLabel(loadedSNP.provenance.ingestion_method)} · SHA 已绑定
          </small>}
          {loadedSNP.provenance_error && <small className="loaded-source-warning">来源记录不可用：{loadedSNP.provenance_error}</small>}
        </div>
      )}

      <div className="file-list">
        {snpFiles.map(f => (
          <button type="button" key={f.filename}
            className={`file-item ${loadedSNP?.filename === f.filename ? 'selected' : ''}`}
            onClick={() => onLoadSNP(f.filename)}>
            <div className="file-name"><span className="file-type">S{f.num_ports}P</span>{f.filename}</div>
            <div className="file-meta">
              {f.freq_count} 点 · {(f.freq_min_hz / 1e6).toFixed(0)}–{(f.freq_max_hz / 1e6).toFixed(0)} MHz
            </div>
          </button>
        ))}
        {snpFiles.length === 0 && (
          <div className="resource-empty">
            <strong>目录中没有 SNP 文件</strong>
            <span>修改上方路径后点击刷新</span>
          </div>
        )}
      </div>
      {invalidSnpFiles.length > 0 && (
        <details className="invalid-snp-files">
          <summary>{invalidSnpFiles.length} 个文件未通过校验</summary>
          {invalidSnpFiles.map(file => (
            <div key={file.filename}><strong>{file.filename}</strong><span>{file.error}</span></div>
          ))}
        </details>
      )}

      <details className="library-settings">
        <summary>元件数据源</summary>
        <div className="source-path">
          <label>Murata 元件库</label>
          <input value={dataDirs.murata} onChange={e => setDataDirs({...dataDirs, murata: e.target.value})} placeholder="data\\Murata" />
        </div>
        <div className="source-path">
          <label>实测 S2P 元件库</label>
          <input value={dataDirs.optenni || ''} onChange={e => setDataDirs({...dataDirs, optenni: e.target.value})} placeholder="Optenni / vendor library" />
        </div>
        <div className="source-path">
          <label>器件环境元数据（可选 JSON）</label>
          <input value={dataDirs.environment || ''}
            onChange={e => setDataDirs({...dataDirs, environment: e.target.value})}
            placeholder="留空时自动查找 component_environment.json" />
        </div>
        {dataSourceStatus?.environment_metadata && (
          <div className={`environment-metadata-status ${dataSourceStatus.environment_metadata.matched_components > 0 ? 'matched' : 'unmatched'}`}>
            <strong>{dataSourceStatus.environment_metadata.matched_components > 0
              ? '环境元数据已生效'
              : '文件有效，但尚无料号匹配'}</strong>
            <span>{dataSourceStatus.environment_metadata.matched_components} 个料号匹配</span>
            {dataSourceStatus.environment_metadata.unmatched_records > 0 &&
              <span>{dataSourceStatus.environment_metadata.unmatched_records} 条记录不在当前目录</span>}
            <small>SHA-256 {dataSourceStatus.environment_metadata.sha256?.slice(0, 12)}…</small>
          </div>
        )}
        <button className="btn btn-sm btn-primary" onClick={() => onApplyDataDirs?.()}>应用数据路径</button>
      </details>
    </div>
  );
}
