import React, { useState, useEffect, useRef } from 'react';
import { api } from '../services/api';
import ManualNetworkStrip from './ManualNetworkStrip';
import {
  clampFrequency,
  compareManualBom,
  gammaToImpedance,
  insertManualTopologyProbe,
  manualValueSliderConfig,
  manualResultMatchesDut,
  manualRefinementProgress,
  manualRefinementVariableCount,
  manualYieldProgress,
  moveArrayItem,
  retargetManualNetwork,
  summarizeManualBands,
  stepPreferredValue,
  toggleBoundedSelection,
  validateManualBands,
  validateManualNetwork,
  valueFromLogSlider,
} from '../utils/manualTuning';
import { shouldLoadDataRevision } from '../utils/dataSource';

const EMPTY_COMP = {
  connection_type: 'series', comp_type: 'inductor', value: 10.0,
  use_ideal: true, port: 0, characteristic_impedance_ohm: 50,
  electrical_length_deg: 45, reference_frequency_hz: 64e6,
  attenuation_db: 0, loss_frequency_exponent: 0.5,
};
const OVERLAY_COLORS = ['#7c3aed', '#0891b2', '#d97706', '#16a34a'];
const SENSITIVITY_PARAMETER_LABELS = {
  value: '标称值', characteristic_impedance_ohm: '特性阻抗', electrical_length_deg: '电长度',
};
const SENSITIVITY_DIRECTION_LABELS = {
  hold: '当前值附近最优', increase: '增大更优', decrease: '减小更优',
};
const TOPOLOGY_LOCATION_LABELS = { dut_side: 'DUT 侧', source_side: '端口侧' };
const TOPOLOGY_CONNECTION_LABELS = { series: '串联', shunt: '并联' };
const TOPOLOGY_COMPONENT_LABELS = { inductor: 'L', capacitor: 'C' };

export default function ManualTunerPage({
  loadedSNP, portConfigs, setPortConfigs, restoredWorkspace, restorationKey,
  onWorkspaceChange, onBack, active = false, dataSourceRevision = 0,
  componentCatalogReady = false,
}) {
  const [targetFreq, setTargetFreq] = useState(64e6);
  const [targetFreqText, setTargetFreqText] = useState('64');
  const [targetReturnLossDb, setTargetReturnLossDb] = useState(10);
  const [inputPort, setInputPort] = useState(0);
  const [components, setComponents] = useState([{ ...EMPTY_COMP }]);
  const [pastNetworks, setPastNetworks] = useState([]);
  const [futureNetworks, setFutureNetworks] = useState([]);
  const [portSessions, setPortSessions] = useState({});
  const [variants, setVariants] = useState([]);
  const [selectedVariantId, setSelectedVariantId] = useState(null);
  const [overlayVariantIds, setOverlayVariantIds] = useState([]);
  const [variantSweeps, setVariantSweeps] = useState({});
  const [variantSweepStatus, setVariantSweepStatus] = useState({});
  const [variantName, setVariantName] = useState('');
  const [result, setResult] = useState(null);
  const [computing, setComputing] = useState(false);
  const [error, setError] = useState('');
  const [catalogError, setCatalogError] = useState('');
  const [componentOptions, setComponentOptions] = useState({inductor: [], capacitor: []});
  const [selectedComponentIndex, setSelectedComponentIndex] = useState(0);
  const [refineObjective, setRefineObjective] = useState('balanced');
  const [refinePasses, setRefinePasses] = useState(4);
  const [activeRefineJobId, setActiveRefineJobId] = useState(null);
  const [refineProgress, setRefineProgress] = useState(null);
  const [refineSummary, setRefineSummary] = useState(null);
  const [refineError, setRefineError] = useState('');
  const [yieldSamples, setYieldSamples] = useState(200);
  const [yieldTolerancePct, setYieldTolerancePct] = useState(5);
  const [yieldDistribution, setYieldDistribution] = useState('uniform');
  const [activeYieldJobId, setActiveYieldJobId] = useState(null);
  const [yieldProgress, setYieldProgress] = useState(null);
  const [yieldAnalysis, setYieldAnalysis] = useState(null);
  const [yieldError, setYieldError] = useState('');
  const computeSequence = useRef(0);
  const variantRequestSequence = useRef(0);
  const activeVariantRequests = useRef({});
  const draggedComponentIndex = useRef(null);
  const componentRevisionRef = useRef(0);
  const refineSequence = useRef(0);
  const refineJobSignatureRef = useRef('');
  const yieldSequence = useRef(0);
  const yieldJobSignatureRef = useRef('');
  const [dragOverIndex, setDragOverIndex] = useState(null);
  const dutRevisionKey = loadedSNP?.network_sha256 || loadedSNP?.sha256;

  useEffect(() => {
    if (!loadedSNP) return;
    if (activeRefineJobId) api.cancelTuningJob(activeRefineJobId).catch(() => {});
    if (activeYieldJobId) api.cancelTuningJob(activeYieldJobId).catch(() => {});
    computeSequence.current += 1;
    variantRequestSequence.current += 1;
    activeVariantRequests.current = {};
    const minimum = Number(loadedSNP.freq_min_hz || 0);
    const maximum = Number(loadedSNP.freq_max_hz || 0);
    if (minimum > 0 && maximum >= minimum) {
      const center = (minimum + maximum) / 2;
      setTargetFreq(center);
      setTargetFreqText(String(Number((center / 1e6).toFixed(6))));
      setComponents([{ ...EMPTY_COMP, port: 0, reference_frequency_hz: center }]);
    } else {
      setComponents([{ ...EMPTY_COMP, port: 0 }]);
    }
    setInputPort(0);
    setPastNetworks([]);
    setFutureNetworks([]);
    setPortSessions({});
    setVariants([]);
    setSelectedVariantId(null);
    setOverlayVariantIds([]);
    setVariantSweeps({});
    setVariantSweepStatus({});
    setVariantName('');
    setResult(null);
    setError('');
    setSelectedComponentIndex(0);
    refineSequence.current += 1;
    setActiveRefineJobId(null);
    setRefineProgress(null);
    setRefineSummary(null);
    setRefineError('');
    yieldSequence.current += 1;
    setActiveYieldJobId(null);
    setYieldProgress(null);
    setYieldAnalysis(null);
    setYieldError('');
  }, [loadedSNP?.filename, dutRevisionKey]);

  useEffect(() => {
    if (!components.length) setSelectedComponentIndex(-1);
    else if (selectedComponentIndex < 0 || selectedComponentIndex >= components.length) {
      setSelectedComponentIndex(Math.max(0, components.length - 1));
    }
  }, [components.length, selectedComponentIndex]);

  useEffect(() => {
    if (active) return;
    computeSequence.current += 1;
    setComputing(false);
    if (activeRefineJobId) {
      refineSequence.current += 1;
      api.cancelTuningJob(activeRefineJobId).catch(() => {});
      setActiveRefineJobId(null);
      setRefineProgress(null);
    }
    if (activeYieldJobId) {
      yieldSequence.current += 1;
      api.cancelTuningJob(activeYieldJobId).catch(() => {});
      setActiveYieldJobId(null);
      setYieldProgress(null);
    }
  }, [active, activeRefineJobId, activeYieldJobId]);

  useEffect(() => {
    if (!shouldLoadDataRevision({
      active, ready: componentCatalogReady, revision: dataSourceRevision,
      lastRevision: componentRevisionRef.current,
    })) return undefined;
    let disposed = false;
    setCatalogError('');
    Promise.all([
      api.searchComponents('inductor', '', 500),
      api.searchComponents('capacitor', '', 500),
    ]).then(([inductors, capacitors]) => {
      if (disposed) return;
      setComponentOptions({
        inductor: inductors.components || [], capacitor: capacitors.components || [],
      });
      componentRevisionRef.current = dataSourceRevision;
    }).catch(loadError => { if (!disposed) setCatalogError(loadError.message); });
    return () => { disposed = true; };
  }, [active, componentCatalogReady, dataSourceRevision]);

  useEffect(() => {
    if (!loadedSNP || !restorationKey || !restoredWorkspace) return;
    const restoredPort = Number(restoredWorkspace.active_input_port || 0);
    const restoredTarget = clampFrequency(
      restoredWorkspace.target_frequency_hz,
      loadedSNP.freq_min_hz,
      loadedSNP.freq_max_hz,
    );
    const networks = restoredWorkspace.working_networks || {};
    const restoredSessions = {};
    Object.entries(networks).forEach(([port, network]) => {
      const portIndex = Number(port);
      restoredSessions[portIndex] = {
        components: retargetManualNetwork(network, portIndex),
        pastNetworks: [], futureNetworks: [],
      };
    });
    setInputPort(restoredPort);
    setTargetFreq(restoredTarget);
    setTargetFreqText(String(Number((restoredTarget / 1e6).toFixed(6))));
    setTargetReturnLossDb(Math.min(60, Math.max(1, Number(restoredWorkspace.target_return_loss_db || 10))));
    setComponents(restoredSessions[restoredPort]?.components || []);
    setPortSessions(restoredSessions);
    setPastNetworks([]);
    setFutureNetworks([]);
    setVariants(restoredWorkspace.variants || []);
    setSelectedVariantId(restoredWorkspace.selected_variant_id || null);
    setOverlayVariantIds(restoredWorkspace.overlay_variant_ids || []);
    setVariantSweeps({});
    setVariantSweepStatus({});
    setResult(null);
    setError('');
  }, [restorationKey]);

  const portStateSignature = JSON.stringify((portConfigs || []).map((config, index) => [
    Number(config.port_index ?? index), config.state || 'load',
  ]));

  // Auto-compute on any change
  useEffect(() => {
    if (!active || !loadedSNP) return;
    const sequence = ++computeSequence.current;
    const timer = setTimeout(() => compute(sequence), 300);
    return () => {
      clearTimeout(timer);
      computeSequence.current += 1;
    };
  }, [active, components, targetFreq, inputPort, loadedSNP?.filename, dutRevisionKey, portStateSignature]);

  useEffect(() => () => {
    variantRequestSequence.current += 1;
    activeVariantRequests.current = {};
  }, [active, loadedSNP?.filename, dutRevisionKey]);

  useEffect(() => {
    if (!active) return;
    overlayVariantIds.forEach(variantId => {
      if (variantSweeps[variantId] || variantSweepStatus[variantId]?.state === 'loading') return;
      const variant = variants.find(item => item.variant_id === variantId);
      if (variant) evaluateVariantSweep(variant);
    });
  }, [active, overlayVariantIds, variants, loadedSNP?.filename, dutRevisionKey]);

  useEffect(() => {
    if (!loadedSNP || !onWorkspaceChange) return;
    const workingNetworks = {};
    Object.entries(portSessions).forEach(([port, session]) => {
      workingNetworks[String(port)] = session.components || [];
    });
    workingNetworks[String(inputPort)] = components;
    onWorkspaceChange({
      schema_version: 1,
      active_input_port: inputPort,
      target_frequency_hz: targetFreq,
      target_return_loss_db: targetReturnLossDb,
      working_networks: workingNetworks,
      variants,
      selected_variant_id: selectedVariantId,
      overlay_variant_ids: overlayVariantIds,
    });
  }, [components, inputPort, targetFreq, targetReturnLossDb, portSessions, variants, selectedVariantId, overlayVariantIds, loadedSNP?.filename, dutRevisionKey]);

  function manualTunePayload({
    drivenPort = inputPort,
    frequency = targetFreq,
    network = components,
    terminationStates = null,
  } = {}) {
    const states = terminationStates || (portConfigs || []).map((config, index) => ({
      port_index: Number(config.port_index ?? index),
      state: config.state || 'load',
    })).filter(item => item.port_index !== drivenPort);
    return {
      snp_filename: loadedSNP.filename,
      expected_network_sha256: loadedSNP.network_sha256 || undefined,
      target_frequency_hz: frequency,
      input_port: drivenPort,
      port_states: states,
      components: network,
      sweep_start_hz: Number(loadedSNP.freq_min_hz || frequency * 0.5),
      sweep_stop_hz: Number(loadedSNP.freq_max_hz || frequency * 1.5),
      sweep_points: 200,
      use_snp_points: true,
    };
  }

  async function compute(sequence = ++computeSequence.current) {
    if (!active || !loadedSNP) return;
    const validationError = validateManualNetwork(components);
    if (validationError) {
      if (sequence === computeSequence.current) {
        setResult(null); setComputing(false); setError(validationError);
      }
      return;
    }
    setComputing(true);
    setError('');
    try {
      const res = await api.manualTune(manualTunePayload());

      if (sequence !== computeSequence.current || !manualResultMatchesDut(res, loadedSNP)) return;
      setResult(res);
    } catch (e) {
      if (sequence !== computeSequence.current) return;
      console.error('Manual tune failed:', e);
      setResult(null);
      setError(e?.response?.data?.detail || e?.message || 'Manual tuning failed');
    }
    if (sequence === computeSequence.current) setComputing(false);
  }

  async function evaluateVariantSweep(variant) {
    const requestSequence = ++variantRequestSequence.current;
    activeVariantRequests.current[variant.variant_id] = requestSequence;
    setVariantSweepStatus(current => ({
      ...current, [variant.variant_id]: { state: 'loading', error: '' },
    }));
    try {
      const response = await api.manualTune(manualTunePayload({
        drivenPort: variant.input_port,
        frequency: variant.target_frequency_hz,
        network: variant.components,
        terminationStates: variant.port_states,
      }));
      if (activeVariantRequests.current[variant.variant_id] !== requestSequence) return;
      setVariantSweeps(current => ({ ...current, [variant.variant_id]: response.sweep }));
      setVariantSweepStatus(current => ({
        ...current, [variant.variant_id]: {
          state: 'ready', error: '', evidence: 'current_dut_recomputed',
        },
      }));
    } catch (variantError) {
      if (activeVariantRequests.current[variant.variant_id] !== requestSequence) return;
      setVariantSweepStatus(current => ({
        ...current,
        [variant.variant_id]: {
          state: 'error', error: variantError?.message || '方案曲线重算失败',
        },
      }));
    }
  }

  async function startManualRefinement() {
    if (!active || !loadedSNP || activeRefineJobId || activeYieldJobId) return;
    const networkError = validateManualNetwork(components);
    if (networkError || bandValidationError) {
      setRefineError(networkError || bandValidationError);
      return;
    }
    if (!activeBandsMhz.length) {
      setRefineError('至少需要一个有效目标频段才能执行局部优化。');
      return;
    }
    if (!manualRefinementVariableCount(components)) {
      setRefineError('当前网络只有锁定的实测器件，没有可连续优化的参数。');
      return;
    }
    const sequence = ++refineSequence.current;
    refineJobSignatureRef.current = refineInputSignature;
    setRefineError('');
    setRefineSummary(null);
    setRefineProgress({ status: 'queued', stage: 'queued', current: 0, total: 0, fraction: 0 });
    try {
      const payload = manualTunePayload();
      const started = await api.startManualRefineJob({
        snp_filename: payload.snp_filename,
        expected_network_sha256: payload.expected_network_sha256,
        target_frequency_hz: payload.target_frequency_hz,
        input_port: payload.input_port,
        port_states: payload.port_states,
        components: payload.components,
        bands_mhz: activeBandsMhz,
        target_return_loss_db: targetReturnLossDb,
        objective: refineObjective,
        max_passes: refinePasses,
      });
      if (sequence !== refineSequence.current) return;
      setActiveRefineJobId(started.job_id);
      let job = started;
      while (sequence === refineSequence.current
        && ['queued', 'running', 'cancelling'].includes(job.status)) {
        await new Promise(resolve => window.setTimeout(resolve, 350));
        job = await api.getTuningJob(started.job_id);
        if (sequence === refineSequence.current) setRefineProgress(manualRefinementProgress(job));
      }
      if (sequence !== refineSequence.current) return;
      if (job.status === 'cancelled') throw new Error('局部优化已取消。');
      if (job.status === 'failed') throw new Error(job.error || '局部优化失败。');
      const refinement = job.result;
      if (!refinement?.result || !manualResultMatchesDut(refinement.result, loadedSNP)) {
        throw new Error('局部优化结果不属于当前 DUT，已拒绝应用。');
      }
      if (refineJobSignatureRef.current !== refineInputSignature) {
        throw new Error('优化期间网络或目标已改变，结果未应用。');
      }
      const baselineSweep = result?.sweep || null;
      commitComponents(refinement.components);
      setResult(refinement.result);
      setRefineSummary({ ...refinement, baseline_sweep: baselineSweep });
      setRefineProgress({
        status: 'completed', stage: 'complete', current: refinement.evaluations,
        total: refinement.evaluations, fraction: 1,
        bestWorstReturnLossDb: refinement.optimized?.worst_return_loss_db,
      });
    } catch (refinementError) {
      if (sequence === refineSequence.current) {
        setRefineError(refinementError?.response?.data?.detail || refinementError.message || '局部优化失败。');
      }
    } finally {
      if (sequence === refineSequence.current) setActiveRefineJobId(null);
    }
  }

  async function cancelManualRefinement() {
    if (!activeRefineJobId) return;
    const jobId = activeRefineJobId;
    refineSequence.current += 1;
    setActiveRefineJobId(null);
    setRefineProgress(null);
    setRefineError('局部优化已取消。');
    try { await api.cancelTuningJob(jobId); } catch (cancelError) {
      setRefineError(cancelError.message || '取消局部优化失败。');
    }
  }

  async function startManualYieldAnalysis() {
    if (!active || !loadedSNP || activeYieldJobId || activeRefineJobId) return;
    const networkError = validateManualNetwork(components);
    if (networkError || bandValidationError || !activeBandsMhz.length) {
      setYieldError(networkError || bandValidationError || '至少需要一个有效目标频段。');
      return;
    }
    const sequence = ++yieldSequence.current;
    yieldJobSignatureRef.current = refineInputSignature;
    setYieldError('');
    setYieldAnalysis(null);
    setYieldProgress({ status: 'queued', stage: 'queued', current: 0, total: yieldSamples, fraction: 0, yieldFraction: null });
    try {
      const payload = manualTunePayload();
      const started = await api.startManualYieldJob({
        snp_filename: payload.snp_filename,
        expected_network_sha256: payload.expected_network_sha256,
        input_port: payload.input_port,
        port_states: payload.port_states,
        components: payload.components,
        bands_mhz: activeBandsMhz,
        target_return_loss_db: targetReturnLossDb,
        samples: yieldSamples,
        seed: 1,
        distribution: yieldDistribution,
        confidence_level: 0.95,
        default_tolerance_pct: yieldTolerancePct,
        batch_correlation: 0,
      });
      if (sequence !== yieldSequence.current) return;
      setActiveYieldJobId(started.job_id);
      let job = started;
      while (sequence === yieldSequence.current
        && ['queued', 'running', 'cancelling'].includes(job.status)) {
        await new Promise(resolve => window.setTimeout(resolve, 350));
        job = await api.getTuningJob(started.job_id);
        if (sequence === yieldSequence.current) setYieldProgress(manualYieldProgress(job));
      }
      if (sequence !== yieldSequence.current) return;
      if (job.status === 'cancelled') throw new Error('制造良率分析已取消。');
      if (job.status === 'failed') throw new Error(job.error || '制造良率分析失败。');
      const analysis = job.result;
      if (!manualResultMatchesDut({ dut_identity: analysis?.dut_identity }, loadedSNP)) {
        throw new Error('良率结果不属于当前 DUT，已拒绝显示。');
      }
      if (yieldJobSignatureRef.current !== refineInputSignature) {
        throw new Error('分析期间网络或目标已改变，结果未应用。');
      }
      setYieldAnalysis(analysis);
      setYieldProgress({
        status: 'completed', stage: 'complete', current: analysis.samples,
        total: analysis.samples, fraction: 1, yieldFraction: analysis.yield_fraction,
      });
    } catch (analysisError) {
      if (sequence === yieldSequence.current) {
        setYieldError(analysisError?.response?.data?.detail || analysisError.message || '制造良率分析失败。');
      }
    } finally {
      if (sequence === yieldSequence.current) setActiveYieldJobId(null);
    }
  }

  async function cancelManualYieldAnalysis() {
    if (!activeYieldJobId) return;
    const jobId = activeYieldJobId;
    yieldSequence.current += 1;
    setActiveYieldJobId(null);
    setYieldProgress(null);
    setYieldError('制造良率分析已取消。');
    try { await api.cancelTuningJob(jobId); } catch (cancelError) {
      setYieldError(cancelError.message || '取消制造良率分析失败。');
    }
  }

  function applyTopologyProbe(probe) {
    if (!probe?.component) return;
    const insertionIndex = Math.max(0, Math.min(components.length, Number(probe.insertion_index) || 0));
    commitComponents(current => insertManualTopologyProbe(
      current, probe, inputPort, targetFreq,
    ));
    setSelectedComponentIndex(insertionIndex);
    setRefineError('');
  }

  function commitComponents(updater) {
    setSelectedVariantId(null);
    setRefineSummary(null);
    setRefineProgress(null);
    setComponents(current => {
      const next = typeof updater === 'function' ? updater(current) : updater;
      if (JSON.stringify(next) === JSON.stringify(current)) return current;
      setPastNetworks(history => [...history.slice(-29), current]);
      setFutureNetworks([]);
      return next;
    });
  }

  function addComponent() {
    commitComponents(current => [...current, {
      ...EMPTY_COMP, port: inputPort, reference_frequency_hz: targetFreq,
    }]);
  }

  function removeComponent(idx) {
    commitComponents(current => current.filter((_, i) => i !== idx));
  }

  function moveComponent(fromIndex, toIndex) {
    commitComponents(current => moveArrayItem(current, fromIndex, toIndex));
  }

  function finishComponentDrag() {
    draggedComponentIndex.current = null;
    setDragOverIndex(null);
  }

  function updateComponent(idx, key, val) {
    commitComponents(current => current.map((component, index) => index === idx
      ? { ...component, [key]: val, ...(key === 'value' ? { part_number: undefined } : {}) }
      : component));
  }

  function updateComponentType(idx, compType) {
    const isThroughLine = compType === 'transmission_line';
    const isStub = compType === 'open_stub' || compType === 'short_stub';
    commitComponents(current => current.map((component, index) => index === idx ? {
      ...component,
      comp_type: compType,
      connection_type: isThroughLine ? 'series' : isStub ? 'shunt' : component.connection_type,
      reference_frequency_hz: component.reference_frequency_hz || targetFreq,
      part_number: undefined,
    } : component));
  }

  function changeInputPort(nextPort) {
    if (nextPort === inputPort) return;
    const savedSession = portSessions[nextPort];
    setPortSessions(current => ({
      ...current,
      [inputPort]: {
        components: retargetManualNetwork(components, inputPort),
        pastNetworks: pastNetworks.map(network => retargetManualNetwork(network, inputPort)),
        futureNetworks: futureNetworks.map(network => retargetManualNetwork(network, inputPort)),
      },
    }));
    setInputPort(nextPort);
    setSelectedVariantId(null);
    setRefineSummary(null);
    setRefineProgress(null);
    setComponents(savedSession
      ? retargetManualNetwork(savedSession.components, nextPort)
      : [{ ...EMPTY_COMP, port: nextPort, reference_frequency_hz: targetFreq }]);
    setPastNetworks(savedSession
      ? savedSession.pastNetworks.map(network => retargetManualNetwork(network, nextPort))
      : []);
    setFutureNetworks(savedSession
      ? savedSession.futureNetworks.map(network => retargetManualNetwork(network, nextPort))
      : []);
  }

  function stepComponentValue(idx, direction) {
    commitComponents(current => current.map((component, index) => index === idx
      ? {
          ...component,
          value: Number(stepPreferredValue(component.value, direction).toPrecision(12)),
          part_number: undefined,
        }
      : component));
  }

  function resetNetwork() {
    commitComponents([]);
    setError('');
  }

  function undoNetwork() {
    if (!pastNetworks.length) return;
    const previous = pastNetworks[pastNetworks.length - 1];
    setPastNetworks(history => history.slice(0, -1));
    setFutureNetworks(history => [components, ...history].slice(0, 30));
    setComponents(previous);
    setRefineSummary(null);
    setRefineProgress(null);
  }

  function redoNetwork() {
    if (!futureNetworks.length) return;
    const next = futureNetworks[0];
    setFutureNetworks(history => history.slice(1));
    setPastNetworks(history => [...history.slice(-29), components]);
    setComponents(next);
    setRefineSummary(null);
    setRefineProgress(null);
  }

  function commitFrequencyText() {
    const parsedHz = Number(targetFreqText) * 1e6;
    const fallback = Number.isFinite(parsedHz) && parsedHz > 0 ? parsedHz : targetFreq;
    selectTargetFrequency(fallback);
  }

  function selectTargetFrequency(frequencyHz) {
    const clamped = clampFrequency(
      frequencyHz, loadedSNP.freq_min_hz, loadedSNP.freq_max_hz,
    );
    setTargetFreq(clamped);
    setTargetFreqText(String(Number((clamped / 1e6).toFixed(6))));
    setSelectedVariantId(null);
  }

  function freezeCurrentVariant() {
    if (!result || bandValidationError || variants.length >= 12) return;
    const variantId = `manual-${Date.now().toString(36)}`;
    const exactComponents = components.map((component, index) => ({
      ...component,
      ...(component.use_ideal === false && result.components?.[index]?.part_number
        ? { part_number: result.components[index].part_number } : {}),
    }));
    const name = variantName.trim() || `P${inputPort + 1} · ${(targetFreq / 1e6).toFixed(1)} MHz`;
    const frozen = {
      variant_id: variantId,
      name,
      input_port: inputPort,
      target_frequency_hz: targetFreq,
      target_return_loss_db: targetReturnLossDb,
      bands_mhz: activeBandsMhz.map(band => [...band]),
      band_weights: [...(activePortConfig?.band_weights || activeBandsMhz.map(() => 1))],
      band_metrics: bandSummaries.map(metric => ({ ...metric })),
      dut_identity: result.dut_identity ? { ...result.dut_identity } : null,
      components: exactComponents,
      port_states: (portConfigs || []).map((config, index) => ({
        port_index: Number(config.port_index ?? index), state: config.state || 'load',
      })).filter(config => config.port_index !== inputPort),
      metrics: {
        return_loss_db: Number(result.s11_db),
        return_loss_improvement_db: Number(result.return_loss_improvement_db),
        vswr: Number(result.vswr),
        input_impedance_real: Number(result.input_impedance_real),
        input_impedance_imag: Number(result.input_impedance_imag),
        maximum_power_balance_error: Number(result.maximum_power_balance_error || 0),
        numeric_core: result.numeric_core || 'rfmatch_core',
      },
      created_at: new Date().toISOString(),
    };
    setVariants(current => [...current, frozen]);
    if (result.sweep) {
      setVariantSweeps(current => ({ ...current, [variantId]: result.sweep }));
      setVariantSweepStatus(current => ({
        ...current, [variantId]: { state: 'ready', error: '', evidence: 'frozen' },
      }));
    }
    setSelectedVariantId(variantId);
    setVariantName('');
  }

  function loadVariant(variant) {
    setPortSessions(current => ({
      ...current,
      [inputPort]: {
        components: retargetManualNetwork(components, inputPort),
        pastNetworks, futureNetworks,
      },
    }));
    setInputPort(variant.input_port);
    setComponents(retargetManualNetwork(variant.components, variant.input_port));
    setPastNetworks([]);
    setFutureNetworks([]);
    selectTargetFrequency(variant.target_frequency_hz);
    setTargetReturnLossDb(Math.min(60, Math.max(1, Number(variant.target_return_loss_db || 10))));
    if (setPortConfigs) {
      const states = new Map((variant.port_states || []).map(item => [Number(item.port_index), item.state]));
      setPortConfigs(current => current.map((config, index) => {
        const portIndex = Number(config.port_index ?? index);
        return {
          ...config,
          state: states.get(portIndex) || config.state || 'load',
          ...(portIndex === variant.input_port && variant.bands_mhz?.length ? {
            bands_mhz: variant.bands_mhz.map(band => [...band]),
            band_weights: variant.bands_mhz.map((_, bandIndex) => Number(variant.band_weights?.[bandIndex] ?? 1)),
          } : {}),
        };
      }));
    }
    setSelectedVariantId(variant.variant_id);
  }

  function deleteVariant(variantId) {
    setVariants(current => current.filter(variant => variant.variant_id !== variantId));
    setOverlayVariantIds(current => current.filter(item => item !== variantId));
    setVariantSweeps(current => {
      const next = { ...current }; delete next[variantId]; return next;
    });
    setVariantSweepStatus(current => {
      const next = { ...current }; delete next[variantId]; return next;
    });
    activeVariantRequests.current[variantId] = ++variantRequestSequence.current;
    if (selectedVariantId === variantId) setSelectedVariantId(null);
  }

  function toggleVariantOverlay(variantId) {
    setOverlayVariantIds(current => toggleBoundedSelection(current, variantId, 4));
  }

  function setFrequencyFraction(fraction) {
    const minimum = Number(loadedSNP.freq_min_hz);
    const maximum = Number(loadedSNP.freq_max_hz);
    selectTargetFrequency(minimum + (maximum - minimum) * fraction);
  }

  function updateActiveBands(updater) {
    if (!setPortConfigs) return;
    setRefineSummary(null);
    setRefineProgress(null);
    setPortConfigs(current => current.map((config, index) => {
      if (Number(config.port_index ?? index) !== inputPort) return config;
      const currentBands = config.bands_mhz || [];
      const nextBands = typeof updater === 'function' ? updater(currentBands) : updater;
      const currentWeights = config.band_weights || [];
      return {
        ...config,
        bands_mhz: nextBands,
        band_weights: nextBands.map((_, bandIndex) => Number(currentWeights[bandIndex] ?? 1)),
      };
    }));
  }

  function updateManualBand(bandIndex, endpointIndex, value) {
    updateActiveBands(bands => bands.map((band, index) => index === bandIndex
      ? band.map((endpoint, endpointPosition) => endpointPosition === endpointIndex ? Number(value) : endpoint)
      : band));
  }

  function addManualBand() {
    const minimumMhz = Number(loadedSNP.freq_min_hz) / 1e6;
    const maximumMhz = Number(loadedSNP.freq_max_hz) / 1e6;
    const width = Math.max(1, (maximumMhz - minimumMhz) * 0.05);
    updateActiveBands(bands => {
      const lastStop = bands.length ? Math.max(...bands.map(band => Number(band[1]))) : targetFreq / 1e6 - width / 2;
      const start = Math.max(minimumMhz, Math.min(maximumMhz - width, lastStop));
      return [...bands, [Number(start.toFixed(6)), Number(Math.min(maximumMhz, start + width).toFixed(6))]];
    });
  }

  function removeManualBand(bandIndex) {
    if (!setPortConfigs) return;
    setRefineSummary(null);
    setRefineProgress(null);
    setPortConfigs(current => current.map((config, index) => {
      if (Number(config.port_index ?? index) !== inputPort) return config;
      return {
        ...config,
        bands_mhz: (config.bands_mhz || []).filter((_, itemIndex) => itemIndex !== bandIndex),
        band_weights: (config.band_weights || []).filter((_, itemIndex) => itemIndex !== bandIndex),
      };
    }));
  }

  function focusComponent(index) {
    setSelectedComponentIndex(index);
    document.getElementById(`manual-component-${index}`)?.scrollIntoView({
      behavior: 'smooth', block: 'center',
    });
  }

  const numPorts = loadedSNP?.num_ports || 1;
  const activePortConfig = (portConfigs || []).find((config, index) => (
    Number(config.port_index ?? index) === inputPort
  ));
  const activeBandsMhz = activePortConfig?.bands_mhz || [];
  const bandValidationError = validateManualBands(
    activeBandsMhz, loadedSNP?.freq_min_hz, loadedSNP?.freq_max_hz,
  );
  const bandSummaries = summarizeManualBands(result?.sweep, activeBandsMhz, targetReturnLossDb);
  const currentWorstBand = bandSummaries.filter(metric => metric.points && metric.worstReturnLossDb != null)
    .sort((left, right) => left.worstReturnLossDb - right.worstReturnLossDb)[0] || null;
  const variantComparisons = variants.map(variant => {
    const comparisonBands = variant.bands_mhz?.length ? variant.bands_mhz : activeBandsMhz;
    const comparisonTarget = Number(variant.target_return_loss_db || targetReturnLossDb);
    const metrics = variantSweeps[variant.variant_id]
      ? summarizeManualBands(variantSweeps[variant.variant_id], comparisonBands, comparisonTarget)
      : (variant.band_metrics || []);
    const worstBand = metrics.filter(metric => metric.points && metric.worstReturnLossDb != null)
      .sort((left, right) => left.worstReturnLossDb - right.worstReturnLossDb)[0] || null;
    return {
      variant,
      worstBand,
      bomDifference: compareManualBom(components, variant.components),
      evidence: variantSweepStatus[variant.variant_id]?.evidence
        || (variant.band_metrics?.length ? 'frozen' : 'point_snapshot'),
    };
  });
  const refinementVariableCount = manualRefinementVariableCount(components);
  const refineInputSignature = JSON.stringify({
    components, inputPort, bands: activeBandsMhz, targetReturnLossDb,
    dut: loadedSNP?.network_sha256 || loadedSNP?.filename,
  });

  useEffect(() => {
    if (!activeRefineJobId || !refineJobSignatureRef.current
      || refineJobSignatureRef.current === refineInputSignature) return;
    const jobId = activeRefineJobId;
    refineSequence.current += 1;
    setActiveRefineJobId(null);
    setRefineProgress(null);
    setRefineError('网络、频段或目标已改变；正在取消旧的局部优化。');
    api.cancelTuningJob(jobId).catch(() => {});
  }, [activeRefineJobId, refineInputSignature]);
  useEffect(() => {
    if (yieldJobSignatureRef.current === refineInputSignature) return;
    if (yieldAnalysis) setYieldAnalysis(null);
    if (!activeYieldJobId || !yieldJobSignatureRef.current) return;
    const jobId = activeYieldJobId;
    yieldSequence.current += 1;
    setActiveYieldJobId(null);
    setYieldProgress(null);
    setYieldError('网络、频段或门限已改变；正在取消旧的良率分析。');
    api.cancelTuningJob(jobId).catch(() => {});
  }, [activeYieldJobId, refineInputSignature, yieldAnalysis]);
  const variantOverlayCurves = overlayVariantIds.map((variantId, index) => {
    const variant = variants.find(item => item.variant_id === variantId);
    return variant && variantSweeps[variantId] ? {
      id: variantId,
      name: variant.name,
      port: variant.input_port,
      color: OVERLAY_COLORS[index],
      data: variantSweeps[variantId],
    } : null;
  }).filter(Boolean);
  const overlayCurves = [
    ...(refineSummary?.baseline_sweep ? [{
      id: '__refinement_baseline__', name: '优化前', port: inputPort,
      color: '#e8792e', data: refineSummary.baseline_sweep,
    }] : []),
    ...variantOverlayCurves,
  ];

  if (!loadedSNP) {
    return (
      <div className="manual-empty-state">
        <span className="eyebrow">MANUAL MATCHING</span>
        <h2>先载入 CST / HFSS Touchstone</h2>
        <p>从左侧导入或选择 DUT，然后在这里编辑理想元件、实测料号、传输线和开短路枝节。</p>
        <button className="btn btn-primary" onClick={onBack}>返回自动综合</button>
      </div>
    );
  }

  return (
    <div className="manual-tuner-page">
      <div className="manual-control-column">
        <div className="card manual-session-card">
          <div className="manual-card-heading">
            <div><span className="eyebrow">LIVE NETWORK EDITOR</span><h3 style={{margin: 0, border: 'none', padding: 0}}>手动调谐</h3></div>
            <button className="btn btn-sm" onClick={onBack}>返回综合</button>
          </div>
          <div className="manual-dut-summary"><strong>{loadedSNP.filename}</strong><span>{numPorts} 端口 · {(loadedSNP.freq_min_hz / 1e6).toFixed(1)}–{(loadedSNP.freq_max_hz / 1e6).toFixed(1)} MHz</span></div>
          <div className="manual-session-fields">
            <div className="form-group">
              <label htmlFor="manual-input-port">激励端口</label>
              <select id="manual-input-port" value={inputPort} onChange={event => changeInputPort(Number(event.target.value))}>
                {Array.from({ length: numPorts }, (_, port) => <option key={port} value={port}>端口 {port + 1}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="manual-target-frequency">目标频率</label>
              <div className="manual-frequency-field">
                <input id="manual-target-frequency" type="number"
                  value={targetFreqText}
                  min={loadedSNP.freq_min_hz / 1e6} max={loadedSNP.freq_max_hz / 1e6} step="0.1"
                  onChange={event => {
                    setTargetFreqText(event.target.value);
                    const nextHz = Number(event.target.value) * 1e6;
                    if (Number.isFinite(nextHz)
                      && nextHz >= Number(loadedSNP.freq_min_hz)
                      && nextHz <= Number(loadedSNP.freq_max_hz)) setTargetFreq(nextHz);
                  }}
                  onBlur={commitFrequencyText}
                  onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur(); }} />
                <span>MHz</span>
              </div>
            </div>
            <div className="form-group manual-goal-field">
              <label htmlFor="manual-return-loss-goal">回损目标</label>
              <div><input id="manual-return-loss-goal" type="number" min="1" max="60" step="1"
                value={targetReturnLossDb}
                onChange={event => {
                  setTargetReturnLossDb(Math.min(60, Math.max(1, Number(event.target.value) || 1)));
                  setRefineSummary(null);
                  setRefineProgress(null);
                }} />
                <span>dB</span></div>
            </div>
          </div>
          <div className="manual-frequency-presets" aria-label="频率快捷点">
            <button onClick={() => setFrequencyFraction(0)}>起点</button>
            <button onClick={() => setFrequencyFraction(.5)}>中心</button>
            <button onClick={() => setFrequencyFraction(1)}>终点</button>
          </div>
          <div className="manual-band-editor">
            <div className="manual-band-editor-heading"><strong>目标频段</strong>
              <button type="button" onClick={addManualBand} disabled={activeBandsMhz.length >= 8}>+ 添加频段</button></div>
            {activeBandsMhz.map((band, bandIndex) => <div className="manual-band-editor-row" key={bandIndex}>
              <b>B{bandIndex + 1}</b>
              <input type="number" step="0.1" value={band[0]}
                aria-label={`频段 ${bandIndex + 1} 起始频率`}
                onChange={event => updateManualBand(bandIndex, 0, event.target.value)} />
              <span>—</span>
              <input type="number" step="0.1" value={band[1]}
                aria-label={`频段 ${bandIndex + 1} 终止频率`}
                onChange={event => updateManualBand(bandIndex, 1, event.target.value)} />
              <i>MHz</i>
              <button type="button" aria-label={`删除频段 ${bandIndex + 1}`}
                onClick={() => removeManualBand(bandIndex)}>×</button>
            </div>)}
            {!activeBandsMhz.length && <p>未定义频段；当前仍可按单一频点调谐。</p>}
            {bandValidationError && <p className="error">{bandValidationError}</p>}
          </div>
          {numPorts > 1 && (
            <p className="manual-port-scope-note">
              网络元件跟随当前激励端口；其他端口按自动综合页的负载 / 开路 / 短路状态端接。
            </p>
          )}
        </div>

        <div className="card manual-network-card">
          <div className="manual-network-heading">
            <h3>匹配网络 <span>{components.length} 个单元</span></h3>
            <div><button title="撤销网络编辑" onClick={undoNetwork} disabled={!pastNetworks.length}>↶</button>
              <button title="恢复网络编辑" onClick={redoNetwork} disabled={!futureNetworks.length}>↷</button>
              <span className={`live-compute-state ${computing ? 'busy' : error ? 'error' : 'ready'}`}>
              {computing ? '正在重算' : error ? '计算异常' : '结果已同步'}
            </span><button onClick={resetNetwork} disabled={!components.length}>清空</button></div>
          </div>
          {components.length > 1 && (
            <p className="manual-order-note">物理顺序：编号从 DUT 向信号源排列；拖动或用箭头换位会立即重算。</p>
          )}
          {components.map((comp, idx) => (
            <div key={idx} id={`manual-component-${idx}`}
              className={`manual-component-card ${dragOverIndex === idx ? 'drag-over' : ''} ${selectedComponentIndex === idx ? 'selected' : ''}`}
              onFocusCapture={() => setSelectedComponentIndex(idx)}
              onClick={() => setSelectedComponentIndex(idx)}
              onDragOver={event => {
                event.preventDefault();
                event.dataTransfer.dropEffect = 'move';
                setDragOverIndex(idx);
              }}
              onDrop={event => {
                event.preventDefault();
                const source = draggedComponentIndex.current
                  ?? Number(event.dataTransfer.getData('text/plain'));
                moveComponent(source, idx);
                finishComponentDrag();
              }}>
              <div className="manual-component-heading">
                <span className="manual-component-title">
                  <span className="manual-drag-handle" draggable
                    title="拖动改变物理级联顺序"
                    onDragStart={event => {
                      draggedComponentIndex.current = idx;
                      setDragOverIndex(idx);
                      event.dataTransfer.effectAllowed = 'move';
                      event.dataTransfer.setData('text/plain', String(idx));
                    }}
                    onDragEnd={finishComponentDrag}>⠿</span>
                  <b>{idx + 1}</b>{comp.connection_type === 'series' ? '串联路径' : '并联到地'}
                </span>
                <span className="manual-order-controls">
                  <button aria-label={`元件 ${idx + 1} 上移`} title="上移（更靠近 DUT）"
                    disabled={idx === 0} onClick={() => moveComponent(idx, idx - 1)}>↑</button>
                  <button aria-label={`元件 ${idx + 1} 下移`} title="下移（更靠近信号源）"
                    disabled={idx === components.length - 1} onClick={() => moveComponent(idx, idx + 1)}>↓</button>
                  <button aria-label={`删除元件 ${idx + 1}`} title="删除元件"
                    onClick={() => removeComponent(idx)}>×</button>
                </span>
              </div>
              <div className="manual-component-selectors">
                <select value={inputPort} disabled
                  title="手动编辑器当前只计算激励端口；其他端口按端接状态处理"
                  aria-label={`元件 ${idx + 1} 端口（跟随激励端口）`}>
                  {Array.from({ length: numPorts }, (_, port) => <option key={port} value={port}>P{port + 1}</option>)}
                </select>
                <select value={comp.connection_type}
                  onChange={e => updateComponent(idx, 'connection_type', e.target.value)}
                  disabled={['transmission_line', 'open_stub', 'short_stub'].includes(comp.comp_type)}
                  aria-label={`元件 ${idx + 1} 连接方式`}>
                  <option value="series">串联</option>
                  <option value="shunt">并联</option>
                </select>
                <select value={comp.comp_type}
                  onChange={e => updateComponentType(idx, e.target.value)}
                  aria-label={`元件 ${idx + 1} 类型`}>
                  <option value="inductor">L (nH)</option>
                  <option value="capacitor">C (pF)</option>
                  <option value="resistor">R (ohm)</option>
                  <option value="transmission_line">传输线</option>
                  <option value="open_stub">开路枝节</option>
                  <option value="short_stub">短路枝节</option>
                </select>
              </div>
              {['inductor', 'capacitor', 'resistor'].includes(comp.comp_type) ? (
                <>
                  <div className="manual-value-stepper">
                    <button type="button" className="manual-e24-step" onClick={() => stepComponentValue(idx, -1)}
                      aria-label={`元件 ${idx + 1} 上一个 E24 标准值`}>−</button>
                    <div className="manual-value-field">
                      <input type="number" min="0" step="0.1" value={comp.value}
                        onChange={e => updateComponent(idx, 'value', parseFloat(e.target.value) || 0)}
                        aria-label={`元件 ${idx + 1} 数值`} />
                      <span>
                        {comp.comp_type === 'inductor' ? 'nH' : comp.comp_type === 'capacitor' ? 'pF' : 'ohm'}
                      </span>
                    </div>
                    <button type="button" className="manual-e24-step" onClick={() => stepComponentValue(idx, 1)}
                      aria-label={`元件 ${idx + 1} 下一个 E24 标准值`}>+</button>
                  </div>
                  {(() => {
                    const slider = manualValueSliderConfig(comp.comp_type, comp.value);
                    const unit = comp.comp_type === 'inductor' ? 'nH' : comp.comp_type === 'capacitor' ? 'pF' : 'Ω';
                    return <div className="manual-continuous-tune">
                      <div><span>{comp.use_ideal === false ? '标称目标' : '连续调谐'}</span>
                        <output>{Number(comp.value).toPrecision(4)} {unit}</output></div>
                      <input type="range" min={slider.minimumLog} max={slider.maximumLog}
                        step="0.005" value={slider.valueLog}
                        aria-label={`元件 ${idx + 1} 连续调谐`}
                        onChange={event => updateComponent(idx, 'value', valueFromLogSlider(event.target.value))} />
                      <div className="manual-slider-scale"><span>{Number(slider.minimum.toPrecision(3))}</span><span>对数刻度</span><span>{Number(slider.maximum.toPrecision(3))}</span></div>
                    </div>;
                  })()}
                  {comp.comp_type !== 'resistor' && (
                    <div className="manual-model-choice">
                      <label><input type="radio" checked={comp.use_ideal !== false}
                        onChange={() => updateComponent(idx, 'use_ideal', true)} /> 理想 {comp.comp_type === 'inductor' ? 'L' : 'C'}</label>
                      <label><input type="radio" checked={comp.use_ideal === false}
                        disabled={!componentCatalogReady}
                        onChange={() => updateComponent(idx, 'use_ideal', false)} /> 实测 S2P</label>
                      {comp.use_ideal === false && <>
                        <input list={`manual-parts-${comp.comp_type}-${idx}`}
                          aria-label={`元件 ${idx + 1} 精确料号`}
                          value={comp.part_number || ''}
                          placeholder="留空则按标称值自动选择最近料号"
                          onChange={event => updateComponent(idx, 'part_number', event.target.value)} />
                        <datalist id={`manual-parts-${comp.comp_type}-${idx}`}>
                          {(componentOptions[comp.comp_type] || []).map(part =>
                            <option key={part.part_number} value={part.part_number}>{part.nominal_value} {part.nominal_unit}</option>)}
                        </datalist>
                        {!computing && result?.components?.[idx]?.part_number && (() => {
                          const calculatedPart = result.components[idx].part_number;
                          const isExactPart = Boolean(comp.part_number)
                            && calculatedPart.localeCompare(comp.part_number, undefined, { sensitivity: 'accent' }) === 0;
                          return <small>当前计算料号：{calculatedPart}{isExactPart ? '（已锁定）' : '（自动匹配）'}</small>;
                        })()}
                      </>}
                    </div>
                  )}
                </>
              ) : (
                <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5}}>
                  {[
                    ['characteristic_impedance_ohm', '特性阻抗 Z0 (Ω)'],
                    ['electrical_length_deg', '电长度 (deg)'],
                    ['reference_frequency_hz', '参考频率 (Hz)'],
                    ['attenuation_db', '单程损耗 (dB)'],
                  ].map(([key, label]) => (
                    <label key={key} style={{fontSize: 10, color: 'var(--text-secondary)'}}>
                      {label}
                      <input type="number" min="0" step={key === 'reference_frequency_hz' ? 1000000 : 0.1}
                        value={comp[key]} onChange={e => updateComponent(idx, key, parseFloat(e.target.value) || 0)}
                        style={{width: '100%', marginTop: 2, padding: '4px 5px', fontSize: 11, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text)', boxSizing: 'border-box'}} />
                    </label>
                  ))}
                </div>
              )}
            </div>
          ))}
          <button className="btn btn-sm" onClick={addComponent} style={{width: '100%'}}>
            + 添加元件或传输线
          </button>
          {!components.length && <div className="manual-bare-dut-note">当前为裸 DUT；添加元件后曲线会自动重算。</div>}
        </div>

        <div className="card manual-refine-card">
          <div className="manual-card-heading">
            <div><span className="eyebrow">FIXED TOPOLOGY OPTIMIZER</span><h3>智能局部优化</h3></div>
            <b>{refinementVariableCount} 个连续变量</b>
          </div>
          <p>保持元件顺序、连接方式和实测料号不变，只优化理想值与传输线参数；结果由同一物理核心复算并可撤销。</p>
          <div className="manual-refine-controls">
            <label>优化目标<select value={refineObjective} aria-label="局部优化目标"
              onChange={event => {
                setRefineObjective(event.target.value);
                setRefineSummary(null);
                setRefineProgress(null);
              }} disabled={Boolean(activeRefineJobId)}>
              <option value="balanced">均衡最差点与平均值</option>
              <option value="worst">优先最差频点</option>
              <option value="average">优先频段平均</option>
            </select></label>
            <label>精修轮数<select value={refinePasses} aria-label="局部优化轮数"
              onChange={event => {
                setRefinePasses(Number(event.target.value));
                setRefineSummary(null);
                setRefineProgress(null);
              }} disabled={Boolean(activeRefineJobId)}>
              <option value={2}>快速 · 2</option><option value={4}>标准 · 4</option><option value={6}>深入 · 6</option>
            </select></label>
            {activeRefineJobId ? <button className="btn btn-sm manual-refine-cancel"
              onClick={cancelManualRefinement}>取消优化</button>
              : <button className="btn btn-primary" onClick={startManualRefinement}
                disabled={!refinementVariableCount || !activeBandsMhz.length || Boolean(bandValidationError) || Boolean(activeYieldJobId)}>优化当前拓扑</button>}
          </div>
          {refineProgress && <div className="manual-refine-progress">
            <div><span>{refineProgress.stage === 'complete' ? '优化完成'
              : refineProgress.stage === 'queued' ? '等待计算资源'
                : refineProgress.stage === 'manual_sensitivity' ? '正在分析参数灵敏度'
                  : refineProgress.stage === 'manual_topology_probe' ? '正在试探可追加拓扑'
                    : refineProgress.stage === 'manual_measured_probe' ? '正在粗筛真实 S2P 料号'
                      : refineProgress.stage === 'manual_full_verification' ? '正在执行真实料号全频验证' : '正在搜索连续参数'}</span>
              <b>{refineProgress.total ? `${refineProgress.current}/${refineProgress.total}` : '准备中'}</b></div>
            <div className="manual-refine-progress-track"><i style={{width: `${100 * refineProgress.fraction}%`}} /></div>
            {refineProgress.bestWorstReturnLossDb != null && <small>当前最差回损 {refineProgress.bestWorstReturnLossDb.toFixed(2)} dB</small>}
          </div>}
          {refineSummary && <div className={`manual-refine-summary ${refineSummary.improved ? 'improved' : 'unchanged'}`}>
            <strong>{refineSummary.improved ? '已应用更优连续参数' : '当前参数已是本轮局部最优'}</strong>
            <span>综合评分 {refineSummary.baseline.score_db.toFixed(2)} → {refineSummary.optimized.score_db.toFixed(2)} dB</span>
            <span>最差回损 {refineSummary.baseline.worst_return_loss_db.toFixed(2)} → {refineSummary.optimized.worst_return_loss_db.toFixed(2)} dB</span>
            <span>{refineSummary.evaluations} 次物理评估 · {refineSummary.variable_count} 个变量</span>
            <small>优化前网络已进入撤销栈，可使用上方 ↶ 恢复。</small>
            {Boolean(refineSummary.sensitivity?.length) && <div className="manual-sensitivity">
              <div className="manual-sensitivity-heading"><b>关键旋钮灵敏度</b><small>当前值 ±10% 的物理复算影响</small></div>
              {refineSummary.sensitivity.slice(0, 4).map((item, index) => <div
                className="manual-sensitivity-row" key={`${item.component_index}-${item.parameter}`}>
                <i>{index + 1}</i>
                <span><b>元件 {item.component_index + 1} · {SENSITIVITY_PARAMETER_LABELS[item.parameter] || item.parameter}</b>
                  <small>{Number(item.base_value).toPrecision(4)} {item.unit} · {SENSITIVITY_DIRECTION_LABELS[item.preferred_direction] || item.preferred_direction}</small></span>
                <strong>影响 {Number(item.score_impact_db).toFixed(2)} dB</strong>
              </div>)}
            </div>}
            {Boolean((refineSummary.optimized_full || refineSummary.optimized)?.bands?.length) && <div className="manual-bottlenecks">
              <div className="manual-sensitivity-heading"><b>频段瓶颈</b><small>优化后最差真实频点</small></div>
              {(refineSummary.optimized_full || refineSummary.optimized).bands.map((band, index) => <div className="manual-bottleneck-row" key={`${band.start_mhz}-${band.stop_mhz}`}>
                <span><b>B{index + 1} · {(band.worst_frequency_hz / 1e6).toFixed(2)} MHz</b>
                  <small>{band.worst_impedance_real_ohm == null ? '阻抗开路附近'
                    : `Z = ${band.worst_impedance_real_ohm.toFixed(1)} ${band.worst_impedance_imag_ohm < 0 ? '−' : '+'} j${Math.abs(band.worst_impedance_imag_ohm).toFixed(1)} Ω`}</small></span>
                <strong className={band.passes ? 'pass' : 'fail'}>{band.worst_return_loss_db.toFixed(2)} dB</strong>
              </div>)}
            </div>}
            {!Boolean((refineSummary.optimized_full || refineSummary.optimized)?.passes_all_bands) && <div className="manual-topology-probes">
              <div className="manual-sensitivity-heading"><b>追加单元件探针</b><small>{refineSummary.topology_probe_evaluations || 0} 理想 · {refineSummary.measured_probe_evaluations || 0} 实测粗筛 · {refineSummary.measured_full_verification_evaluations || 0} 全频验证</small></div>
              {refineSummary.topology_probes?.length ? refineSummary.topology_probes.map((probe, index) => {
                const measured = probe.measured_alternatives?.[0];
                return <div className="manual-topology-probe-row" key={`${probe.location}-${probe.connection_type}-${probe.component_type}`}>
                <i>{index + 1}</i>
                <span><b>{TOPOLOGY_LOCATION_LABELS[probe.location]} · {TOPOLOGY_CONNECTION_LABELS[probe.connection_type]} {TOPOLOGY_COMPONENT_LABELS[probe.component_type]}</b>
                  <small>{Number(probe.component.value).toPrecision(4)} {probe.component_type === 'inductor' ? 'nH' : 'pF'} · 预计最差回损 {probe.worst_return_loss_db.toFixed(2)} dB</small></span>
                <strong>+{probe.score_improvement_db.toFixed(2)} dB</strong>
                <div className="manual-probe-actions">
                  <button className="btn btn-sm" aria-label={`应用理想拓扑候选 ${index + 1}`} onClick={() => applyTopologyProbe(probe)}>应用理想</button>
                  {measured && <button className="btn btn-sm measured" aria-label={`应用实测拓扑候选 ${index + 1}`}
                    title={`${measured.part_number} · 最差回损 ${measured.worst_return_loss_db.toFixed(2)} dB`}
                    onClick={() => applyTopologyProbe({...probe, component: measured.component})}>应用实测</button>}
                </div>
                {measured ? <small className="manual-probe-measured">全频验证 {measured.part_number} · {measured.nominal_value} {measured.nominal_unit} · {measured.verification_points} 点 · 最差 {measured.worst_return_loss_db.toFixed(2)} dB{probe.measured_alternatives.length > 1 ? ` · 另有 ${probe.measured_alternatives.length - 1} 个已验证料号` : ''}</small>
                  : <small className="manual-probe-measured unavailable">当前目录没有保持改善的真实料号</small>}
              </div>}) : <p className="manual-probe-empty">本轮单元件粗探针未找到可验证的改善；建议改变现有拓扑顺序或转入自动综合。</p>}
            </div>}
          </div>}
          {refineError && <div className="manual-refine-error">{refineError}</div>}
        </div>

        <div className="card manual-yield-card">
          <div className="manual-card-heading">
            <div><span className="eyebrow">MANUFACTURING ROBUSTNESS</span><h3>制造良率</h3></div>
            <b>{components.length} 个物理位置</b>
          </div>
          <p>对当前精确网络执行确定性 Monte Carlo；真实料号采用目录公差，缺失元数据和理想元件采用默认公差。</p>
          <div className="manual-yield-controls">
            <label>样本数<select aria-label="良率样本数" value={yieldSamples} disabled={Boolean(activeYieldJobId)}
              onChange={event => { setYieldSamples(Number(event.target.value)); setYieldAnalysis(null); }}>
              <option value={100}>快速 · 100</option><option value={200}>标准 · 200</option><option value={500}>深入 · 500</option>
            </select></label>
            <label>默认公差<div><input aria-label="良率默认公差" type="number" min="0.1" max="50" step="0.5"
              value={yieldTolerancePct} disabled={Boolean(activeYieldJobId)}
              onChange={event => { setYieldTolerancePct(Math.max(.1, Math.min(50, Number(event.target.value) || .1))); setYieldAnalysis(null); }} /><span>%</span></div></label>
            <label>分布<select aria-label="良率分布" value={yieldDistribution} disabled={Boolean(activeYieldJobId)}
              onChange={event => { setYieldDistribution(event.target.value); setYieldAnalysis(null); }}>
              <option value="uniform">均匀边界</option><option value="normal">截断正态</option>
            </select></label>
            {activeYieldJobId ? <button className="btn btn-sm manual-refine-cancel" onClick={cancelManualYieldAnalysis}>取消分析</button>
              : <button className="btn btn-primary" onClick={startManualYieldAnalysis}
                disabled={Boolean(activeRefineJobId) || Boolean(bandValidationError) || !activeBandsMhz.length}>分析当前网络</button>}
          </div>
          {yieldProgress && <div className="manual-refine-progress">
            <div><span>{yieldProgress.stage === 'complete' ? '良率分析完成' : yieldProgress.stage === 'queued' ? '等待计算资源' : '正在抽样装配偏差'}</span>
              <b>{yieldProgress.total ? `${yieldProgress.current}/${yieldProgress.total}` : '准备中'}</b></div>
            <div className="manual-refine-progress-track"><i style={{width: `${100 * yieldProgress.fraction}%`}} /></div>
            {yieldProgress.yieldFraction != null && <small>当前通过率估计 {(100 * yieldProgress.yieldFraction).toFixed(1)}%</small>}
          </div>}
          {yieldAnalysis && <div className={`manual-yield-result ${yieldAnalysis.yield_fraction >= .95 ? 'good' : yieldAnalysis.yield_fraction >= .8 ? 'warn' : 'risk'}`}>
            <div className="manual-yield-hero"><span><small>制造通过率</small><strong>{(100 * yieldAnalysis.yield_fraction).toFixed(1)}%</strong></span>
              <span><small>95% 置信区间</small><b>{(100 * yieldAnalysis.yield_confidence_interval[0]).toFixed(1)}–{(100 * yieldAnalysis.yield_confidence_interval[1]).toFixed(1)}%</b></span></div>
            <div className="manual-yield-metrics">
              <span>标称最差回损<b>{yieldAnalysis.nominal_worst_return_loss_db.toFixed(2)} dB</b></span>
              <span>P5 最差回损<b>{yieldAnalysis.return_loss_percentiles_db['5'].toFixed(2)} dB</b></span>
              <span>通过样本<b>{yieldAnalysis.passed_samples}/{yieldAnalysis.samples}</b></span>
              <span>验证频点<b>{yieldAnalysis.frequency_points}</b></span>
            </div>
            {Boolean(yieldAnalysis.risk_components?.length) && <div className="manual-yield-risks">
              <div><b>最差样本风险排序</b><small>按该样本中的绝对偏差</small></div>
              {yieldAnalysis.risk_components.slice(0, 4).map(item => <span key={`${item.position}-${item.part_number}`}>
                <b>#{item.position} · {item.part_number}</b><small>公差 ±{item.tolerance_pct.toFixed(2)}% · 最差样本 {item.worst_deviation_pct >= 0 ? '+' : ''}{item.worst_deviation_pct.toFixed(2)}%</small>
              </span>)}
            </div>}
            <small className="manual-yield-evidence">固定 seed {yieldAnalysis.seed} · {yieldAnalysis.distribution === 'uniform' ? '均匀边界分布' : '截断正态分布'} · 功率闭合误差 {yieldAnalysis.maximum_nominal_power_balance_error.toExponential(2)}</small>
          </div>}
          {yieldError && <div className="manual-refine-error">{yieldError}</div>}
        </div>

        {result && (
          <div className="card manual-result-card">
            <h3>当前频点结果 <span>{(targetFreq / 1e6).toFixed(2)} MHz</span></h3>
            <div className="manual-result-stats">
              <div className="stat">
                <span className="stat-label">改善量</span>
                <span className="stat-value" style={{color: result.return_loss_improvement_db >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}}>
                  {result.return_loss_improvement_db >= 0 ? '+' : ''}{result.return_loss_improvement_db?.toFixed(1)} dB
                </span>
              </div>
              <div className="stat">
                <span className="stat-label">回波损耗 |S11|</span>
                <span className={`stat-value ${result.s11_db > 10 ? 'good' : 'bad'}`}>
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
                <span className="stat-label">输入阻抗</span>
                <span className="stat-value manual-zin">
                  {result.input_impedance_real?.toFixed(1)} {result.input_impedance_imag >= 0 ? '+' : '−'} j{Math.abs(result.input_impedance_imag)?.toFixed(1)} Ω
                </span>
              </div>
            </div>
            {activeBandsMhz.length && !bandValidationError ? (
              <div className="manual-band-summary">
                <div className="manual-band-summary-heading"><strong>目标频段检查</strong><span>门限：回波损耗 ≥ {targetReturnLossDb} dB</span></div>
                {bandSummaries.map(band => <div key={`${band.startMhz}-${band.stopMhz}`}
                  className={`manual-band-row ${band.passes ? 'pass' : 'fail'}`}>
                  <span>{band.startMhz}–{band.stopMhz} MHz</span>
                  {band.points ? <><b>最差 {band.worstReturnLossDb.toFixed(2)} dB</b>
                    <em>{band.marginDb >= 0 ? '+' : ''}{band.marginDb.toFixed(2)} dB 裕量</em></>
                    : <em>扫频内无采样点</em>}
                </div>)}
              </div>
            ) : <div className="manual-band-empty">{bandValidationError || '尚未定义目标频段；可直接在上方添加。'}</div>}
            <div className="manual-result-provenance">
              {result.numeric_core}
              {result.dut_identity?.filename ? ` · DUT ${result.dut_identity.filename}` : ''}
              {result.dut_identity?.network_sha256 ? ` · 网络 ${result.dut_identity.network_sha256.slice(0, 8)}…` : ''}
              {' · '}功率平衡误差 {result.maximum_power_balance_error?.toExponential(2)}
            </div>
          </div>
        )}
        <div className="card manual-freeze-card">
          <div className="manual-card-heading">
            <div><span className="eyebrow">DESIGN VARIANTS</span><h3>冻结当前方案</h3></div>
            <b>{variants.length}/12</b>
          </div>
          <p>保存端口、端接状态、精确元件与当前指标，用于工程内对比。</p>
          <div className="manual-freeze-row">
            <input value={variantName} maxLength={100} placeholder={`例如：P${inputPort + 1} 低 VSWR`}
              onChange={event => setVariantName(event.target.value)} />
            <button className="btn btn-primary" disabled={!result || Boolean(bandValidationError) || variants.length >= 12}
              onClick={freezeCurrentVariant}>冻结方案</button>
          </div>
        </div>
        {catalogError && <div className="card manual-error-card">器件目录读取失败：{catalogError}</div>}
        {error && <div className="card manual-error-card">{error}</div>}
      </div>

      <div className="manual-chart-column">
        <ManualNetworkStrip components={components} inputPort={inputPort}
          selectedIndex={selectedComponentIndex} onSelect={focusComponent} />
        {variants.length > 0 && (
          <section className="card manual-variants-card">
            <div className="manual-variants-heading">
              <div><span className="eyebrow">FROZEN COMPARISON</span><h3>手动方案对比</h3></div>
              <small>指标为冻结时快照；载入后用当前 DUT 重算</small>
            </div>
            <div className="manual-comparison-table-wrap">
              <table className="manual-comparison-table">
                <thead><tr><th>方案</th><th>最差频段</th><th>目标裕量</th><th>目标点</th><th>BOM 相对当前</th><th>证据</th></tr></thead>
                <tbody>
                  <tr className="current"><td><strong>当前编辑</strong><small>P{inputPort + 1}</small></td>
                    <td>{currentWorstBand ? `${currentWorstBand.worstReturnLossDb.toFixed(2)} dB` : '—'}</td>
                    <td><span className={currentWorstBand?.passes ? 'pass' : 'fail'}>{currentWorstBand ? `${currentWorstBand.marginDb >= 0 ? '+' : ''}${currentWorstBand.marginDb.toFixed(2)} dB` : '—'}</span></td>
                    <td>{result ? `${result.s11_db.toFixed(2)} dB · ${result.vswr.toFixed(2)} VSWR` : '—'}</td>
                    <td>比较基准</td><td>当前 DUT</td></tr>
                  {variantComparisons.map(({ variant, worstBand, bomDifference, evidence }) => (
                    <tr key={variant.variant_id} className={selectedVariantId === variant.variant_id ? 'selected' : ''}>
                      <td><strong>{variant.name}</strong><small>P{variant.input_port + 1} · {((variant.target_frequency_hz || 0) / 1e6).toFixed(2)} MHz</small></td>
                      <td>{worstBand ? `${worstBand.worstReturnLossDb.toFixed(2)} dB` : '—'}</td>
                      <td><span className={worstBand?.passes ? 'pass' : 'fail'}>{worstBand ? `${worstBand.marginDb >= 0 ? '+' : ''}${worstBand.marginDb.toFixed(2)} dB` : '—'}</span></td>
                      <td>{Number.isFinite(variant.metrics?.return_loss_db) ? `${variant.metrics.return_loss_db.toFixed(2)} dB · ${variant.metrics.vswr.toFixed(2)} VSWR` : '—'}</td>
                      <td><span className={bomDifference.isIdentical ? 'same' : 'changed'}>{bomDifference.summary}</span></td>
                      <td>{evidence === 'current_dut_recomputed' ? '当前 DUT 重算' : evidence === 'frozen' ? '冻结扫频' : '仅频点快照'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="manual-variant-grid">
              {variants.map(variant => (
                <article className={selectedVariantId === variant.variant_id ? 'selected' : ''}
                  key={variant.variant_id}>
                  <div className="manual-variant-title"><strong>{variant.name}</strong>
                    <span>P{variant.input_port + 1} · {(variant.target_frequency_hz / 1e6).toFixed(2)} MHz</span></div>
                  <div className="manual-variant-metrics">
                    <span><b>{variant.metrics.return_loss_db.toFixed(2)}</b> dB RL</span>
                    <span><b>{variant.metrics.vswr.toFixed(2)}</b> VSWR</span>
                    <span><b>{variant.metrics.input_impedance_real.toFixed(1)}</b>
                      {variant.metrics.input_impedance_imag >= 0 ? ' + j' : ' − j'}{Math.abs(variant.metrics.input_impedance_imag).toFixed(1)} Ω</span>
                  </div>
                  <p>{variant.components.length ? variant.components.map(component => {
                    const kind = component.comp_type === 'inductor' ? 'L' : component.comp_type === 'capacitor' ? 'C' : component.comp_type === 'resistor' ? 'R' : component.comp_type;
                    return component.part_number || `${component.connection_type === 'shunt' ? '∥' : '—'}${kind} ${component.value ?? component.electrical_length_deg}`;
                  }).join(' → ') : '裸 DUT'}</p>
                  <div className="manual-variant-actions">
                    <label className="manual-overlay-toggle" style={{'--overlay-color': OVERLAY_COLORS[Math.max(0, overlayVariantIds.indexOf(variant.variant_id))]}}>
                      <input type="checkbox" checked={overlayVariantIds.includes(variant.variant_id)}
                        disabled={!overlayVariantIds.includes(variant.variant_id) && overlayVariantIds.length >= 4}
                        onChange={() => toggleVariantOverlay(variant.variant_id)} /> 叠加曲线
                    </label>
                    <button className="btn btn-sm" onClick={() => loadVariant(variant)}>载入重算</button>
                    <button className="btn btn-sm" onClick={() => deleteVariant(variant.variant_id)}>删除</button>
                  </div>
                  {variantSweepStatus[variant.variant_id]?.state === 'loading' && <small className="manual-overlay-status">正在用当前 DUT 重算曲线…</small>}
                  {variantSweepStatus[variant.variant_id]?.state === 'error' && <small className="manual-overlay-status error">{variantSweepStatus[variant.variant_id].error}</small>}
                </article>
              ))}
            </div>
          </section>
        )}
        <div className="card manual-chart-card">
          <div className="manual-chart-heading"><h3>S11 频率响应 <small>点击或拖动选择目标频率</small></h3><div className="manual-chart-legend"><span className="raw">原始 DUT</span><span className="matched">当前网络</span>{overlayCurves.map(curve => <span className="overlay" style={{'--overlay-color': curve.color}} key={curve.id}>{curve.name} · P{curve.port + 1}</span>)}</div></div>
          <div className="manual-chart-canvas">
            {result?.sweep ? (
              <S11MiniChart data={result.sweep} targetFreq={targetFreq} bandsMhz={activeBandsMhz}
                targetReturnLossDb={targetReturnLossDb}
                overlays={overlayCurves} onTargetFrequencyChange={selectTargetFrequency} />
            ) : (
              <div style={{height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)'}}>
                {computing ? '正在计算…' : '修改网络后显示 S11'}
              </div>
            )}
          </div>
        </div>

        <div className="card manual-chart-card">
          <div className="manual-chart-heading"><h3>Smith 圆图</h3><div className="manual-chart-legend"><span className="raw">原始 DUT</span><span className="matched">当前网络</span>{overlayCurves.map(curve => <span className="overlay" style={{'--overlay-color': curve.color}} key={curve.id}>{curve.name}</span>)}</div></div>
          <div className="manual-chart-canvas smith">
            {result?.sweep ? (
              <SmithMiniChart data={result.sweep} targetFreq={targetFreq} overlays={overlayCurves}
                targetReturnLossDb={targetReturnLossDb}
                referenceImpedanceOhm={loadedSNP.reference_resistance || loadedSNP.reference_impedance || 50}
                onTargetFrequencyChange={selectTargetFrequency} />
            ) : (
              <div style={{height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)'}}>
                Smith 圆图将在计算后显示
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}


function S11MiniChart({ data, targetFreq, bandsMhz = [], targetReturnLossDb = 10, overlays = [], onTargetFrequencyChange }) {
  if (!data?.frequencies) return null;
  const W = 560, H = 280, pad = 40;
  const freqs = data.frequencies;
  const s11db = data.s11_db.map(value => -Math.abs(value));
  const rawDb = (data.raw_db || []).map(value => -Math.abs(value));
  const overlaySeries = overlays.map(curve => ({
    ...curve,
    frequencies: curve.data?.frequencies || [],
    values: (curve.data?.s11_db || []).map(value => -Math.abs(value)),
  })).filter(curve => curve.frequencies.length && curve.frequencies.length === curve.values.length);

  const fMin = Math.min(...freqs);
  const fMax = Math.max(...freqs);
  const overlayValues = overlaySeries.flatMap(curve => curve.values);
  const yMin = Math.min(-40, -targetReturnLossDb, Math.min(...s11db), ...(rawDb.length ? rawDb : [0]), ...(overlayValues.length ? overlayValues : [0]));
  const yMax = Math.max(0, Math.max(...s11db));
  const frequencySpan = Math.max(fMax - fMin, 1);
  const valueSpan = Math.max(yMax - yMin, 1);

  const xScale = f => pad + ((f - fMin) / frequencySpan) * (W - 2 * pad);
  const yScale = v => pad + ((yMax - v) / valueSpan) * (H - 2 * pad);

  const points = freqs.map((f, i) => `${xScale(f)},${yScale(s11db[i])}`).join(' ');
  const rawPoints = rawDb.length === freqs.length
    ? freqs.map((f, i) => `${xScale(f)},${yScale(rawDb[i])}`).join(' ')
    : '';
  const targetX = xScale(targetFreq);
  const targetIndex = freqs.reduce((best, frequency, index) => (
    Math.abs(frequency - targetFreq) < Math.abs(freqs[best] - targetFreq) ? index : best
  ), 0);
  const targetY = yScale(s11db[targetIndex]);
  const xTicks = [fMin, (fMin + fMax) / 2, fMax];
  const yTicks = [0, -10, -20, -30, -40].filter(value => value >= yMin);
  const visibleBands = (bandsMhz || []).map((band, index) => ({
    index,
    start: Math.max(fMin, Number(band?.[0]) * 1e6),
    stop: Math.min(fMax, Number(band?.[1]) * 1e6),
  })).filter(band => Number.isFinite(band.start) && Number.isFinite(band.stop) && band.stop >= band.start);

  function setTargetFromPointer(event) {
    const bounds = event.currentTarget.getBoundingClientRect();
    const viewX = ((event.clientX - bounds.left) / Math.max(bounds.width, 1)) * W;
    const fraction = Math.min(1, Math.max(0, (viewX - pad) / (W - 2 * pad)));
    onTargetFrequencyChange(fMin + fraction * frequencySpan);
  }

  return (
    <svg className="manual-interactive-chart" width="100%" height="100%" viewBox={`0 0 ${W} ${H}`}
      onPointerDown={event => {
        event.currentTarget.setPointerCapture?.(event.pointerId);
        setTargetFromPointer(event);
      }}
      onPointerMove={event => { if (event.buttons === 1) setTargetFromPointer(event); }}>
      {visibleBands.map(band => <g key={`${band.start}-${band.stop}`}>
        <rect x={xScale(band.start)} y={pad} width={Math.max(1, xScale(band.stop) - xScale(band.start))}
          height={H - 2 * pad} fill="#1268c4" opacity={0.055} />
        <text x={(xScale(band.start) + xScale(band.stop)) / 2} y={pad + 10}
          textAnchor="middle" fontSize={8} fill="#4776a4">频段 {band.index + 1}</text>
      </g>)}
      {/* Grid */}
      <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#dee2e6" />
      <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#dee2e6" />
      {xTicks.map(value => <g key={value}>
        <line x1={xScale(value)} y1={H - pad} x2={xScale(value)} y2={H - pad + 4} stroke="#aeb8c5" />
        <text x={xScale(value)} y={H - pad + 15} textAnchor="middle" fontSize={9} fill="#7b8796">{(value / 1e6).toFixed(0)}</text>
      </g>)}
      {yTicks.map(value => <g key={value}>
        <line x1={pad} y1={yScale(value)} x2={W - pad} y2={yScale(value)} stroke="#edf0f3" />
        <text x={pad - 6} y={yScale(value) + 3} textAnchor="end" fontSize={9} fill="#7b8796">{value}</text>
      </g>)}
      {/* Return-loss goal */}
      {yMin <= -targetReturnLossDb && <g>
        <line x1={pad} y1={yScale(-targetReturnLossDb)} x2={W - pad} y2={yScale(-targetReturnLossDb)} stroke="#198754" strokeDasharray="4" strokeOpacity={0.55} />
        <text x={W - pad - 2} y={yScale(-targetReturnLossDb) - 4} textAnchor="end" fontSize={8} fill="#198754">目标 −{targetReturnLossDb} dB</text>
      </g>}
      {/* Target freq */}
      <line x1={targetX} y1={pad} x2={targetX} y2={H - pad} stroke="#0d6efd" strokeDasharray="4" />
      <text x={targetX} y={pad - 5} textAnchor="middle" fontSize={10} fill="#0d6efd">{(targetFreq / 1e6).toFixed(0)} MHz</text>
      {rawPoints && <polyline points={rawPoints} fill="none" stroke="#8a96a6" strokeWidth={1.5} strokeDasharray="5 4" />}
      {overlaySeries.map(curve => <polyline key={curve.id}
        points={curve.frequencies.map((frequency, index) => `${xScale(frequency)},${yScale(curve.values[index])}`).join(' ')}
        fill="none" stroke={curve.color} strokeWidth={1.8} strokeOpacity={0.9} />)}
      <polyline points={points} fill="none" stroke="#dc3545" strokeWidth={2} />
      <circle cx={targetX} cy={targetY} r={4} fill="#fff" stroke="#0d6efd" strokeWidth={2} />
      {/* Axis labels */}
      <text x={W / 2} y={H - 5} textAnchor="middle" fontSize={10} fill="#6c757d">频率 (MHz)</text>
      <text x={5} y={H / 2} textAnchor="middle" fontSize={10} fill="#6c757d" transform={`rotate(-90, 5, ${H / 2})`}>S11 (dB)</text>
    </svg>
  );
}


function SmithMiniChart({
  data, targetFreq, overlays = [], targetReturnLossDb = 10,
  referenceImpedanceOhm = 50, onTargetFrequencyChange,
}) {
  if (!data?.s11_real || !data?.s11_imag) return null;
  const W = 720, H = 300;
  const cx = 232, cy = H / 2, R = 126;
  const resistances = [0, 0.2, 0.5, 1, 2, 5];
  const reactances = [0.2, 0.5, 1, 2, 5];

  const gamma = data.s11_real.map((re, i) => ({ re: data.s11_real[i], im: data.s11_imag[i] }));
  const points = gamma.map(g => `${cx + g.re * R},${cy - g.im * R}`).join(' ');
  const rawGamma = (data.raw_real || []).map((re, i) => ({ re, im: data.raw_imag[i] }));
  const rawPoints = rawGamma.map(g => `${cx + g.re * R},${cy - g.im * R}`).join(' ');
  const overlaySeries = overlays.map(curve => ({
    ...curve,
    gamma: (curve.data?.s11_real || []).map((re, index) => ({
      re, im: curve.data?.s11_imag?.[index] || 0,
    })),
  })).filter(curve => curve.gamma.length);
  const targetIndex = (data.frequencies || []).reduce((best, frequency, index, values) => (
    Math.abs(frequency - targetFreq) < Math.abs(values[best] - targetFreq) ? index : best
  ), 0);
  const targetGamma = gamma[targetIndex];
  const impedance = targetGamma
    ? gammaToImpedance(targetGamma.re, targetGamma.im, referenceImpedanceOhm) : null;
  const targetMagnitude = impedance?.gammaMagnitude ?? 1;
  const targetVswr = targetMagnitude < 1 ? (1 + targetMagnitude) / Math.max(1 - targetMagnitude, 1e-12) : Infinity;
  const targetReturnLoss = -20 * Math.log10(Math.max(targetMagnitude, 1e-15));
  const goalGammaRadius = Math.min(1, 10 ** (-targetReturnLossDb / 20));

  function selectNearestFrequency(event) {
    if (!onTargetFrequencyChange || !gamma.length) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const viewX = ((event.clientX - bounds.left) / Math.max(bounds.width, 1)) * W;
    const viewY = ((event.clientY - bounds.top) / Math.max(bounds.height, 1)) * H;
    const clickedRe = (viewX - cx) / R;
    const clickedIm = (cy - viewY) / R;
    const nearest = gamma.reduce((best, point, index) => {
      const distance = (point.re - clickedRe) ** 2 + (point.im - clickedIm) ** 2;
      return distance < best.distance ? { index, distance } : best;
    }, { index: 0, distance: Infinity });
    onTargetFrequencyChange(data.frequencies[nearest.index]);
  }

  return (
    <svg className="manual-interactive-chart" width="100%" height="100%" viewBox={`0 0 ${W} ${H}`}
      role="img" aria-label="可点击选频的标准阻抗 Smith 圆图" onPointerDown={selectNearestFrequency}>
      <defs><clipPath id="manual-smith-unit-circle"><circle cx={cx} cy={cy} r={R} /></clipPath></defs>
      <circle cx={cx} cy={cy} r={R} fill="#fbfcfe" stroke="#6f7f92" strokeWidth={1.4} />
      <g clipPath="url(#manual-smith-unit-circle)" fill="none" stroke="#dce3eb" strokeWidth={0.8}>
        <line x1={cx - R} y1={cy} x2={cx + R} y2={cy} />
        {resistances.map(value => <circle key={`r-${value}`}
          cx={cx + R * value / (1 + value)} cy={cy} r={R / (1 + value)} />)}
        {reactances.flatMap(value => [value, -value]).map(value => <circle key={`x-${value}`}
          cx={cx + R} cy={cy - R / value} r={R / Math.abs(value)} />)}
      </g>
      <g fill="#8a96a5" fontSize={8} textAnchor="middle">
        {resistances.map(value => <text key={`rl-${value}`}
          x={cx + R * (value - 1) / (value + 1)} y={cy + 12}>{value}</text>)}
        <text x={cx + R} y={cy + 12}>∞</text>
      </g>
      <circle cx={cx} cy={cy} r={R * goalGammaRadius} fill="none" stroke="#198754"
        strokeWidth={1.1} strokeDasharray="4 3" opacity={0.68} />
      <text x={cx} y={cy - R * goalGammaRadius - 5} textAnchor="middle" fontSize={8} fill="#198754">
        {targetReturnLossDb} dB 目标圈
      </text>
      {/* Gamma trace */}
      <g clipPath="url(#manual-smith-unit-circle)">
        {rawPoints && <polyline points={rawPoints} fill="none" stroke="#8a96a6" strokeWidth={1.3} strokeDasharray="5 4" />}
        {overlaySeries.map(curve => <polyline key={curve.id}
          points={curve.gamma.map(point => `${cx + point.re * R},${cy - point.im * R}`).join(' ')}
          fill="none" stroke={curve.color} strokeWidth={1.8} strokeOpacity={0.9} />)}
        <polyline points={points} fill="none" stroke="#dc3545" strokeWidth={2.2} />
      </g>
      {targetGamma && <><circle cx={cx + targetGamma.re * R} cy={cy - targetGamma.im * R}
        r={6} fill="#fff" stroke="#0d6efd" strokeWidth={2} />
        <circle cx={cx + targetGamma.re * R} cy={cy - targetGamma.im * R} r={2} fill="#0d6efd" /></>}
      {/* Start/end dots */}
      {gamma.length > 0 && (
        <>
          <circle cx={cx + gamma[0].re * R} cy={cy - gamma[0].im * R} r={4} fill="#0d6efd" />
          <circle cx={cx + gamma[gamma.length - 1].re * R} cy={cy - gamma[gamma.length - 1].im * R} r={4} fill="#198754" />
        </>
      )}
      <g transform="translate(410 42)">
        <text x="0" y="0" fill="#273d54" fontSize="12" fontWeight="700">目标点工程读数</text>
        <text x="0" y="22" fill="#758497" fontSize="9">频率</text>
        <text x="118" y="22" textAnchor="end" fill="#2f4d6b" fontSize="10" fontWeight="700">{(Number(data.frequencies?.[targetIndex]) / 1e6).toFixed(3)} MHz</text>
        <text x="0" y="44" fill="#758497" fontSize="9">反射系数 Γ</text>
        <text x="118" y="44" textAnchor="end" fill="#2f4d6b" fontSize="10">{targetMagnitude.toFixed(4)} ∠ {impedance?.gammaAngleDeg.toFixed(1) ?? '—'}°</text>
        <text x="0" y="66" fill="#758497" fontSize="9">回波损耗</text>
        <text x="118" y="66" textAnchor="end" fill={targetReturnLoss >= targetReturnLossDb ? '#198754' : '#b64350'} fontSize="10" fontWeight="700">{targetReturnLoss.toFixed(2)} dB</text>
        <text x="0" y="88" fill="#758497" fontSize="9">VSWR</text>
        <text x="118" y="88" textAnchor="end" fill="#2f4d6b" fontSize="10">{Number.isFinite(targetVswr) ? targetVswr.toFixed(3) : '∞'}</text>
        <text x="0" y="110" fill="#758497" fontSize="9">输入阻抗</text>
        <text x="118" y="110" textAnchor="end" fill="#2f4d6b" fontSize="10" fontWeight="700">
          {impedance ? `${impedance.resistanceOhm.toFixed(2)} ${impedance.reactanceOhm >= 0 ? '+' : '−'} j${Math.abs(impedance.reactanceOhm).toFixed(2)} Ω` : '开路极限'}
        </text>
        <line x1="0" y1="128" x2="118" y2="128" stroke="#e0e6ed" />
        <text x="0" y="147" fill="#8a96a5" fontSize="8">点击轨迹附近可选择最近频点</text>
        <text x="0" y="162" fill="#8a96a5" fontSize="8">参考阻抗 Z0 = {Number(referenceImpedanceOhm).toFixed(1)} Ω</text>
      </g>
    </svg>
  );
}
