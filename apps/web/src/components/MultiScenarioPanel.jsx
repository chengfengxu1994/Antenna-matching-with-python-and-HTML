import React, { useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../services/api';
import { shouldLoadDataRevision } from '../utils/dataSource';
import { multiScenarioProgress, validateMultiScenarioInput } from '../utils/multiScenario';
import TopologySchematic from './TopologySchematic';

const COLORS = ['#4aa3f0', '#f26d5f', '#3ecf9a', '#e8894b', '#a47ef0', '#3cc6d0'];
const QUALITY_PROFILES = {
  quick: {timeout:15, beam:8, points:3},
  balanced: {timeout:45, beam:10, points:5},
  thorough: {timeout:120, beam:10, points:10},
  exhaustive: {timeout:150, beam:50, points:3},
};

function defaultScenario(filename) {
  return { snp_filename: filename, weight: 1, efficiency_filename: '', efficiency_kind: 'radiation' };
}

function Metric({ label, value, suffix = '' }) {
  return <div className="metric-tile"><div className="metric-label">{label}</div><div className="metric-value">{value}{suffix}</div></div>;
}

function ScenarioChart({ scenarios, field, title, percent = false }) {
  const all = scenarios?.flatMap(s => s.points || []) || [];
  if (!all.length) return null;
  const width = 760, height = 260, left = 52, right = 16, top = 20, bottom = 36;
  const xs = all.map(p => p.frequency_hz / 1e6);
  const rawYs = all.map(p => percent ? p[field] * 100 : p[field]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = percent ? 0 : Math.min(-40, Math.floor(Math.min(...rawYs) / 5) * 5);
  const y1 = percent ? 100 : 0;
  const sx = x => left + ((x - x0) / Math.max(x1 - x0, 1e-9)) * (width - left - right);
  const sy = y => top + ((y1 - y) / Math.max(y1 - y0, 1e-9)) * (height - top - bottom);
  const ticks = [0, .25, .5, .75, 1].map(t => y0 + t * (y1 - y0));
  return (
    <div className="workspace-card">
      <h3>{title}</h3>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        {ticks.map(v => <g key={v}>
          <line x1={left} x2={width-right} y1={sy(v)} y2={sy(v)} stroke="var(--border)" />
          <text x={left-7} y={sy(v)+4} textAnchor="end" fontSize="10" fill="var(--text-secondary)">{v.toFixed(0)}{percent ? '%' : ''}</text>
        </g>)}
        {scenarios.map((scenario, index) => {
          const points = (scenario.points || []).map(p => `${sx(p.frequency_hz/1e6)},${sy(percent ? p[field]*100 : p[field])}`).join(' ');
          return <polyline key={scenario.filename} points={points} fill="none" stroke={COLORS[index % COLORS.length]} strokeWidth="2" />;
        })}
        <text x={(left + width-right)/2} y={height-7} textAnchor="middle" fontSize="11" fill="var(--text-secondary)">频率 (MHz)</text>
      </svg>
      <div style={{display:'flex', gap:12, flexWrap:'wrap', fontSize:11}}>
        {scenarios.map((s, i) => <span key={s.filename} style={{color:COLORS[i % COLORS.length]}}>● {s.filename}</span>)}
      </div>
    </div>
  );
}

function ResultView({ result }) {
  if (!result) return (
    <div className="empty-state multi-empty">
      <span className="eyebrow">MULTI-SCENARIO</span>
      <h3>尚无联合结果</h3>
      <p>在右侧勾选至少两个实测场景，然后运行「联合优化」，或切换到「手动调谐」评估同一套匹配网络。</p>
    </div>
  );
  return <>
    <div className="workspace-card">
      <h3>{result.topology}</h3>
      <div className="metric-row">
        <Metric label="联合评分" value={(result.score_db ?? 10*Math.log10(result.score)).toFixed(2)} suffix=" dB" />
        <Metric label="平均总效率 η" value={(result.avg_total_efficiency * 100).toFixed(1)} suffix="%" />
        <Metric label="最差总效率 η" value={(result.min_total_efficiency * 100).toFixed(1)} suffix="%" />
        <Metric label="平均失配效率 η" value={(result.avg_mismatch_efficiency * 100).toFixed(1)} suffix="%" />
        <Metric label="最差回波损耗" value={result.min_return_loss_db.toFixed(1)} suffix=" dB" />
        {Number.isFinite(result.maximum_power_balance_error) && <Metric label="功率平衡误差" value={result.maximum_power_balance_error.toExponential(1)} />}
      </div>
      {result.search_plan && <div className="calibration-note" style={{marginTop:10}}>
        搜索：{result.search_plan.label} · {result.search_plan.strategy} · {result.search_diagnostics?.physical_evaluations ?? '—'} 次物理评估 · {result.search_diagnostics?.component_models_built ?? '—'} 个器件模型
        {result.search_diagnostics?.verification?.frequency_points ? ` · ${result.search_diagnostics.verification.frequency_points} 个频点独立验证` : ''}
        {result.search_diagnostics?.search_truncated ? ' · 已达到时间预算' : ''}
      </div>}
      <div style={{marginTop:12}}><TopologySchematic solution={{topology:result.topology, per_port:{0:{components:(result.components || []).map(c => ({...c, type:c.component_type, value:`${c.nominal_value}${c.nominal_unit || ''}`}))}}}} portIndex={0} /></div>
    </div>
    <div className="workspace-card">
      <h3>逐场景独立验证</h3>
      <div style={{overflowX:'auto'}}><table className="results-table"><thead><tr>
        <th>场景</th><th>权重</th><th>评分</th><th>效率输入</th><th>平均总效率</th><th>最差总效率</th><th>平均失配效率</th><th>最差 RL</th>
      </tr></thead><tbody>{result.scenarios.map(s => <tr key={s.filename}>
        <td>{s.filename}</td><td>{s.weight}</td><td>{Number.isFinite(s.score_db) ? `${s.score_db.toFixed(2)} dB` : '—'}</td><td>{s.efficiency_kind || '未提供'}</td>
        <td>{(s.avg_total_efficiency*100).toFixed(1)}%</td><td>{(s.min_total_efficiency*100).toFixed(1)}%</td>
        <td>{(s.avg_mismatch_efficiency*100).toFixed(1)}%</td><td>{s.min_return_loss_db.toFixed(1)} dB</td>
      </tr>)}</tbody></table></div>
    </div>
    <div className="chart-grid">
      <ScenarioChart scenarios={result.scenarios} field="s11_db" title="全部场景 S11" />
      <ScenarioChart scenarios={result.scenarios} field="total_efficiency" title="全部场景总效率" percent />
    </div>
  </>;
}

export default function MultiScenarioPanel({
  snpFiles, active = false, dataSourceRevision = 0, componentCatalogReady = false,
}) {
  const [mode, setMode] = useState('optimize');
  const [scenarios, setScenarios] = useState([]);
  const [efficiencyFiles, setEfficiencyFiles] = useState([]);
  const [componentCount, setComponentCount] = useState(2);
  const [topologies, setTopologies] = useState([]);
  const [selectedTopologies, setSelectedTopologies] = useState([]);
  const [manualTopology, setManualTopology] = useState('');
  const [bands, setBands] = useState([[2400, 2500]]);
  const [inputPort, setInputPort] = useState(0);
  const [objective, setObjective] = useState('balanced');
  const [searchQuality, setSearchQuality] = useState('balanced');
  const [beamWidth, setBeamWidth] = useState(10);
  const [timeout, setTimeoutValue] = useState(45);
  const [bandPoints, setBandPoints] = useState(5);
  const [verificationPoints, setVerificationPoints] = useState(41);
  const [realtimePoints, setRealtimePoints] = useState(81);
  const [manualComponents, setManualComponents] = useState([]);
  const [componentOptions, setComponentOptions] = useState({inductor:[], capacitor:[]});
  const [solutions, setSolutions] = useState([]);
  const [selectedSolution, setSelectedSolution] = useState(0);
  const [manualResult, setManualResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [activeJobId, setActiveJobId] = useState('');
  const [jobProgress, setJobProgress] = useState(null);
  const [liveStatus, setLiveStatus] = useState('idle');
  const [error, setError] = useState('');
  const liveRequestId = useRef(0);
  const optimizeRequestId = useRef(0);
  const efficiencyRevisionRef = useRef(0);
  const componentRevisionRef = useRef(0);
  const topologyCountRef = useRef(null);

  useEffect(() => {
    if (!shouldLoadDataRevision({
      active, revision: dataSourceRevision, lastRevision: efficiencyRevisionRef.current,
    })) return undefined;
    let disposed = false;
    api.listEfficiencyFiles().then(r => {
      if (!disposed) {
        setEfficiencyFiles(r.files || []);
        efficiencyRevisionRef.current = dataSourceRevision;
      }
    }).catch(loadError => { if (!disposed) setError(loadError.message); });
    return () => { disposed = true; };
  }, [active, dataSourceRevision]);

  useEffect(() => {
    if (!shouldLoadDataRevision({
      active, ready: componentCatalogReady, revision: dataSourceRevision,
      lastRevision: componentRevisionRef.current,
    })) return undefined;
    let disposed = false;
    Promise.all([api.searchComponents('inductor', '', 500), api.searchComponents('capacitor', '', 500)])
      .then(([l, c]) => {
        if (!disposed) {
          setComponentOptions({inductor:l.components || [], capacitor:c.components || []});
          componentRevisionRef.current = dataSourceRevision;
        }
      }).catch(loadError => { if (!disposed) setError(loadError.message); });
    return () => { disposed = true; };
  }, [active, componentCatalogReady, dataSourceRevision]);

  useEffect(() => {
    const available = new Set((snpFiles || []).map(f => f.filename));
    setScenarios(current => {
      const kept = current.filter(s => available.has(s.snp_filename));
      if (kept.length) return kept;
      return (snpFiles || []).slice(0, 2).map(f => defaultScenario(f.filename));
    });
  }, [snpFiles]);

  useEffect(() => {
    if (!active || topologyCountRef.current === componentCount) return undefined;
    let disposed = false;
    api.listTopologies(componentCount).then(r => {
      if (disposed) return;
      const exact = (r.topologies || []).filter(t => t.num_components === componentCount);
      topologyCountRef.current = componentCount;
      setTopologies(exact);
      setSelectedTopologies(current => {
        const valid = current.filter(name => exact.some(t => t.name === name));
        return valid.length ? valid : exact.slice(0, 1).map(t => t.name);
      });
      setManualTopology(current => exact.some(t => t.name === current) ? current : (exact[0]?.name || ''));
    }).catch(e => { if (!disposed) setError(e.message); });
    return () => { disposed = true; };
  }, [active, componentCount]);

  const selectedManualTopology = useMemo(() => topologies.find(t => t.name === manualTopology), [topologies, manualTopology]);
  useEffect(() => {
    if (!selectedManualTopology) return;
    setManualComponents(current => {
      const matches = current.length === selectedManualTopology.elements.length && current.every((item, index) =>
        item.component_type === selectedManualTopology.elements[index].component_type &&
        item.connection_type === selectedManualTopology.elements[index].connection_type
      );
      if (matches) return current;
      return selectedManualTopology.elements.map(element => {
      const first = componentOptions[element.component_type]?.[0];
      return {use_ideal: !first, part_number:first?.part_number || '', value:1, component_type:element.component_type, connection_type:element.connection_type};
      });
    });
  }, [selectedManualTopology, componentOptions]);

  const toggleScenario = filename => setScenarios(current => current.some(s => s.snp_filename === filename)
    ? current.filter(s => s.snp_filename !== filename) : [...current, defaultScenario(filename)]);
  const updateScenario = (filename, patch) => setScenarios(current => current.map(s => s.snp_filename === filename ? {...s, ...patch} : s));
  const updateBand = (index, side, value) => setBands(current => current.map((b, i) => i === index ? [side === 0 ? value : b[0], side === 1 ? value : b[1]] : b));
  const commonRequest = {scenarios, input_port:inputPort, bands_mhz:bands, objective, num_band_points:bandPoints};

  async function optimize() {
    if (!componentCatalogReady) return setError('实测器件库尚未就绪，请先在数据源设置中配置有效路径。');
    const validationError = validateMultiScenarioInput({
      scenarios, bands, inputPort, topologyNames:selectedTopologies, requireTopology:true,
    });
    if (validationError) return setError(validationError);
    const requestId = ++optimizeRequestId.current;
    setBusy(true); setError(''); setManualResult(null);
    try {
      const started = await api.startMultiScenarioJob({...commonRequest, component_count:componentCount,
        topology_names:selectedTopologies, search_quality:searchQuality,
        beam_width:beamWidth, timeout_seconds:timeout,
        verification_band_points:verificationPoints});
      if (requestId !== optimizeRequestId.current) return;
      setActiveJobId(started.job_id); setJobProgress(multiScenarioProgress(started));
      let job = started;
      while (requestId === optimizeRequestId.current && ['queued','running','cancelling'].includes(job.status)) {
        await new Promise(resolve => window.setTimeout(resolve, 450));
        job = await api.getTuningJob(started.job_id);
        if (requestId === optimizeRequestId.current) setJobProgress(multiScenarioProgress(job));
      }
      if (requestId !== optimizeRequestId.current) return;
      if (job.status === 'completed') {
        setSolutions(job.result?.solutions || []); setSelectedSolution(0);
      } else if (job.status === 'failed') {
        throw new Error(job.error || '联合优化失败。');
      }
    } catch (e) {
      if (requestId === optimizeRequestId.current) setError(e.message);
    } finally {
      if (requestId === optimizeRequestId.current) { setBusy(false); setActiveJobId(''); }
    }
  }

  async function cancelOptimization() {
    if (!activeJobId) return;
    try {
      const job = await api.cancelTuningJob(activeJobId);
      setJobProgress(multiScenarioProgress({...job, progress:{...(job.progress || {}), stage:'cancelling'}}));
    } catch (e) { setError(e.message); }
  }

  async function evaluateManual() {
    const validationError = validateMultiScenarioInput({scenarios, bands, inputPort, requireTopology:false});
    if (validationError) return setError(validationError);
    if (!manualTopology || !selectedManualTopology || manualComponents.length !== selectedManualTopology.num_components) return;
    const requestId = ++liveRequestId.current;
    setLiveStatus('updating'); setError('');
    try {
      const response = await api.multiScenarioManual({
        ...commonRequest,
        num_band_points: realtimePoints,
        topology_name:manualTopology,
        components:manualComponents,
      });
      if (requestId === liveRequestId.current) {
        setManualResult(response.result);
        setLiveStatus('ready');
      }
    } catch (e) {
      if (requestId === liveRequestId.current) {
        setLiveStatus('error');
        setError(e.message);
      }
    }
  }

  useEffect(() => () => { optimizeRequestId.current += 1; }, []);

  // Optenni-style Impedance Config: every edit applies one shared network to
  // every selected scenario and refreshes all curves after a short debounce.
  useEffect(() => {
    if (!active || mode !== 'manual' || scenarios.length < 2 || !manualTopology || !selectedManualTopology) return;
    if (manualComponents.length !== selectedManualTopology.num_components) return;
    setLiveStatus('waiting');
    const timer = window.setTimeout(() => evaluateManual(), 350);
    return () => {
      window.clearTimeout(timer);
      liveRequestId.current += 1;
    };
  }, [active, mode, scenarios, bands, inputPort, objective, manualTopology, manualComponents, realtimePoints, selectedManualTopology]);

  function useSolution(solution) {
    setMode('manual'); setComponentCount(solution.components.length); setManualTopology(solution.topology);
    setManualComponents(solution.components.map(c => ({use_ideal:false, part_number:c.part_number, value:c.nominal_value,
      component_type:c.component_type, connection_type:c.connection_type})));
    setManualResult(solution);
  }

  const displayed = mode === 'manual' ? manualResult : solutions[selectedSolution];
  return <div className="main-body">
    <aside className="control-sidebar" style={{width:360, minWidth:360}}>
      <div className="card"><h3>多场景共享网络</h3><div className="goal-selector">
        <button className={`goal-btn ${mode==='optimize'?'active':''}`} onClick={() => setMode('optimize')}>联合优化</button>
        <button className={`goal-btn ${mode==='manual'?'active':''}`} onClick={() => setMode('manual')}>手动调谐</button>
      </div></div>

      <div className="card"><h3>1. 测量场景（{scenarios.length}）</h3>
        <div className="checkbox-list" style={{maxHeight:150}}>{(snpFiles || []).map(file => <label className="checkbox-item" key={file.filename}>
          <input type="checkbox" checked={scenarios.some(s => s.snp_filename === file.filename)} onChange={() => toggleScenario(file.filename)} />
          <span>{file.filename} <small>({file.num_ports}P)</small></span>
        </label>)}</div>
        {scenarios.map(s => <div key={s.snp_filename} style={{padding:'7px 0', borderBottom:'1px solid var(--border)'}}>
          <div style={{fontSize:11, fontWeight:600, overflow:'hidden', textOverflow:'ellipsis'}}>{s.snp_filename}</div>
          <div style={{display:'grid', gridTemplateColumns:'58px 1fr 92px', gap:4, marginTop:4}}>
            <input title="场景权重" aria-label={`${s.snp_filename} 场景权重`} type="number" min="0" step="0.1" value={s.weight} onChange={e => updateScenario(s.snp_filename,{weight:+e.target.value})} />
            <select value={s.efficiency_filename} onChange={e => updateScenario(s.snp_filename,{efficiency_filename:e.target.value})}>
              <option value="">不使用效率文件</option>{efficiencyFiles.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
            <select value={s.efficiency_kind} disabled={!s.efficiency_filename} onChange={e => updateScenario(s.snp_filename,{efficiency_kind:e.target.value})}>
              <option value="radiation">辐射效率 η</option><option value="total">总效率 η</option>
            </select>
          </div>
        </div>)}
      </div>

      <div className="card"><h3>2. 共享拓扑</h3>
        <div className="form-group"><label>器件数量</label><select value={componentCount} onChange={e => setComponentCount(+e.target.value)}>{[1,2,3,4].map(n => <option key={n}>{n}</option>)}</select></div>
        {mode === 'optimize' ? <div className="checkbox-list" style={{maxHeight:170}}>{topologies.map(t => <label className="checkbox-item" key={t.name}>
          <input type="checkbox" checked={selectedTopologies.includes(t.name)} onChange={() => setSelectedTopologies(v => v.includes(t.name) ? v.filter(x => x!==t.name) : [...v,t.name])} />{t.name}
        </label>)}</div> : <>
          <div className="form-group"><label>拓扑</label><select value={manualTopology} onChange={e => setManualTopology(e.target.value)}>{topologies.map(t => <option key={t.name}>{t.name}</option>)}</select></div>
          {manualComponents.map((c, i) => <div key={`${manualTopology}-${i}`} style={{padding:7, marginBottom:6, background:'var(--bg-input)', borderRadius:5}}>
            <strong style={{fontSize:11}}>#{i+1} {c.connection_type} {c.component_type === 'inductor' ? 'L' : 'C'}</strong>
            <label className="checkbox-item"><input type="checkbox" checked={c.use_ideal} onChange={e => setManualComponents(v => v.map((x,j)=>j===i?{...x,use_ideal:e.target.checked}:x))} />使用理想值</label>
            {c.use_ideal ? <div style={{display:'flex',gap:4}}><input type="number" step="0.1" value={c.value} onChange={e => setManualComponents(v=>v.map((x,j)=>j===i?{...x,value:+e.target.value}:x))}/><span style={{fontSize:11}}>{c.component_type==='inductor'?'nH':'pF'}</span></div>
              : <><input list={`parts-${c.component_type}-${i}`} style={{width:'100%'}} value={c.part_number}
                  placeholder="输入或选择带 S2P 模型的精确料号"
                  onChange={e => setManualComponents(v=>v.map((x,j)=>j===i?{...x,part_number:e.target.value}:x))} />
                <datalist id={`parts-${c.component_type}-${i}`}>{(componentOptions[c.component_type] || []).map(p =>
                  <option key={p.part_number} value={p.part_number}>{p.nominal_value} {p.nominal_unit}</option>)}</datalist></>}
          </div>)}
        </>}
      </div>

      <div className="card"><h3>3. 优化目标与频段</h3>
        <div className="form-group"><label>联合目标</label><select value={objective} onChange={e=>setObjective(e.target.value)}><option value="balanced">平衡平均值与最差值</option><option value="average">最佳加权平均</option><option value="worst_case">最佳最差场景</option></select></div>
        {mode === 'optimize' && <div className="form-group"><label>搜索质量</label><select value={searchQuality} onChange={e=>{
          const quality=e.target.value; setSearchQuality(quality); const profile=QUALITY_PROFILES[quality];
          if (profile) { setTimeoutValue(profile.timeout); setBeamWidth(profile.beam); setBandPoints(profile.points); }
        }}><option value="quick">快速探索（15 秒）</option><option value="balanced">工程平衡（45 秒）</option><option value="thorough">约束精搜（120 秒）</option><option value="exhaustive">拓扑深搜（150 秒）</option><option value="custom">自定义</option></select></div>}
        <div className="form-group"><label>输入端口（从 1 开始）</label><input type="number" min="1" step="1" value={inputPort+1} onChange={e=>setInputPort(Math.max(0,+e.target.value-1))}/></div>
        {bands.map((b,i)=><div key={i} style={{display:'grid',gridTemplateColumns:'minmax(0,1fr) auto minmax(0,1fr) auto',alignItems:'center',gap:4,marginBottom:4}}>
          <input aria-label={`Band ${i+1} start MHz`} style={{width:'100%',minWidth:0}} type="number" value={b[0]} onChange={e=>updateBand(i,0,+e.target.value)}/>
          <span>–</span>
          <input aria-label={`Band ${i+1} stop MHz`} style={{width:'100%',minWidth:0}} type="number" value={b[1]} onChange={e=>updateBand(i,1,+e.target.value)}/>
          <button aria-label={`Remove band ${i+1}`} className="btn btn-sm" onClick={()=>setBands(v=>v.filter((_,j)=>j!==i))}>×</button>
        </div>)}
        <button className="btn btn-sm" onClick={()=>setBands(v=>[...v,[2400,2500]])}>+ 添加频段</button>
      </div>

      <button className="run-btn" disabled={busy || scenarios.length<2 || (mode === 'optimize' && !componentCatalogReady)} onClick={mode==='optimize'?optimize:evaluateManual}>{busy?'正在联合优化…':mode==='optimize' ? componentCatalogReady ? '优化共享匹配网络' : '实测器件库未就绪' : '立即刷新全部场景'}</button>
      {busy && jobProgress && <div className="calibration-note" style={{marginTop:8}} role="status">
        <div style={{display:'flex',justifyContent:'space-between',gap:8}}><strong>{jobProgress.label}</strong><span>{jobProgress.elapsed.toFixed(1)} s</span></div>
        <div style={{height:5,background:'var(--border)',borderRadius:4,margin:'6px 0',overflow:'hidden'}}><div style={{height:'100%',width:`${jobProgress.percent}%`,background:'var(--accent-blue)',transition:'width .2s'}} /></div>
        <div style={{display:'flex',justifyContent:'space-between',gap:8}}><span>{jobProgress.evaluations} 次物理评估</span><button className="btn btn-sm" onClick={cancelOptimization}>取消任务</button></div>
      </div>}
      {mode === 'manual' && <div style={{marginTop:7, padding:'7px 9px', borderRadius:5, fontSize:11,
        background:liveStatus==='error'?'rgba(220,53,69,.08)':'rgba(25,135,84,.08)',
        color:liveStatus==='error'?'var(--accent-red)':'var(--accent-green)'}}>
        {liveStatus === 'updating' ? `正在重算 ${scenarios.length} 个 SNP 场景…` :
         liveStatus === 'waiting' ? '检测到修改，准备刷新…' :
         liveStatus === 'ready' ? `实时 · 已更新 ${scenarios.length} 条 SNP 曲线` :
         liveStatus === 'error' ? '实时刷新失败，请检查器件选择' :
         '实时阻抗调谐已启用'}
      </div>}
      <div style={{marginTop:8}}><button className="advanced-toggle" onClick={e=>{const n=e.currentTarget.nextSibling;n.style.display=n.style.display==='none'?'block':'none'}}>高级设置 <span>[+]</span></button><div style={{display:'none',padding:8}}>
        <div className="form-group"><label>束宽</label><input type="number" value={beamWidth} onChange={e=>{setBeamWidth(+e.target.value);setSearchQuality('custom')}}/></div>
        <div className="form-group"><label>时间预算（秒）</label><input type="number" value={timeout} onChange={e=>{setTimeoutValue(+e.target.value);setSearchQuality('custom')}}/></div>
        <div className="form-group"><label>每频段搜索点数</label><input type="number" min="2" value={bandPoints} onChange={e=>{setBandPoints(+e.target.value);setSearchQuality('custom')}}/></div>
        <div className="form-group"><label>每频段验证点数</label><select value={verificationPoints} onChange={e=>setVerificationPoints(+e.target.value)}><option value="21">21（快速）</option><option value="41">41</option><option value="81">81（密集）</option></select></div>
        <div className="form-group"><label>实时曲线点数</label><select value={realtimePoints} onChange={e=>setRealtimePoints(+e.target.value)}><option value="41">41（快速）</option><option value="81">81</option><option value="161">161（平滑）</option></select></div>
      </div></div>
      {error && <div className="error-card" style={{marginTop:8}}>{error}</div>}
    </aside>

    <main className="workspace-panel">
      {mode === 'optimize' && solutions.length > 0 && <div className="workspace-card"><h3>联合方案排名</h3><div style={{overflowX:'auto'}}><table className="results-table"><thead><tr><th>#</th><th>拓扑</th><th>器件</th><th>评分</th><th>平均 η</th><th>最差 η</th><th></th></tr></thead><tbody>{solutions.map((s,i)=><tr key={`${s.topology}-${i}`} className={i===selectedSolution?'selected':''} onClick={()=>setSelectedSolution(i)}><td>{i+1}</td><td>{s.topology}</td><td>{s.components.map(c=>c.part_number).join(' + ')}</td><td>{(s.score_db ?? 10*Math.log10(s.score)).toFixed(2)} dB</td><td>{(s.avg_total_efficiency*100).toFixed(1)}%</td><td>{(s.min_total_efficiency*100).toFixed(1)}%</td><td><button className="btn btn-sm" onClick={e=>{e.stopPropagation();useSolution(s)}}>转到手动调谐</button></td></tr>)}</tbody></table></div></div>}
      <ResultView result={displayed} />
    </main>
  </div>;
}
