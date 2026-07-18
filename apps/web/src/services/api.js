/**
 * API service for communicating with the RF Matching backend.
 */
const API_BASE = '/api';

async function request(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const config = {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  };
  const res = await fetch(url, config);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => request('/health'),

  getCalibrationStatus: () => request('/calibration/status'),

  setDataDirs: (snpDir, murataDir, optenniComponentDir = '', environmentMetadataPath = '') =>
    request('/config/dirs', {
      method: 'POST',
      body: JSON.stringify({
        snp_dir: snpDir,
        murata_dir: murataDir,
        optenni_component_dir: optenniComponentDir,
        environment_metadata_path: environmentMetadataPath,
      }),
    }),

  listSNPFiles: () => request('/snp/list'),

  listEfficiencyFiles: () => request('/efficiency-files/list'),

  loadSNP: (filename) =>
    request('/snp/load?filename=' + encodeURIComponent(filename), { method: 'POST' }),

  importSNP: (filename, content, source = 'Touchstone') =>
    request('/snp/import', {
      method: 'POST',
      body: JSON.stringify({ filename, content, source }),
    }),

  startSnpWatch: (source = 'CST', stableMs = 1000) =>
    request(`/snp/watch/start?source=${encodeURIComponent(source)}&stable_ms=${stableMs}`, { method: 'POST' }),

  getSnpWatchStatus: (watchId) =>
    request(`/snp/watch/status?watch_id=${encodeURIComponent(watchId)}`),

  stopSnpWatch: (watchId) =>
    request(`/snp/watch/stop?watch_id=${encodeURIComponent(watchId)}`, { method: 'POST' }),

  getCstStatus: (force = false) => request(`/cst/status?force=${force}`),

  getCstProjectTree: (pid, projectPath) =>
    request(`/cst/project-tree?pid=${pid}&project_path=${encodeURIComponent(projectPath)}`),

  exportCstTouchstone: (pid, projectPath) =>
    request(`/cst/export-touchstone?pid=${pid}&project_path=${encodeURIComponent(projectPath)}`, { method: 'POST' }),

  getSParams: (freqHz = 64e6) => request(`/snp/s-params?freq_hz=${freqHz}`),

  listComponents: (type, limit = 200) =>
    request(`/components/list?comp_type=${type || ''}&limit=${limit}`),

  searchComponents: (type, query = '', limit = 200) =>
    request(`/components/search?comp_type=${type || ''}&q=${encodeURIComponent(query)}&limit=${limit}`),

  listTopologies: (maxComponents = 4) =>
    request(`/topologies/list?max_components=${maxComponents}`),

  getComponentSeries: () => request('/component-series'),
  getComponentDetail: (partNumber) => request(`/components/detail?part_number=${encodeURIComponent(partNumber)}`),
  componentAlternatives: (params) => request('/components/alternatives', {
    method: 'POST',
    body: JSON.stringify(params),
  }),
  previewComponentLibrary: (params) => request('/component-library/preview', {
    method: 'POST',
    body: JSON.stringify(params),
  }),
  getBandPresets: () => request('/band-presets'),

  getResults: (limit = 50) => request(`/optimize/results?limit=${limit}`),

  frequencySweep: (startHz, stopHz, numPoints, solutionIndex) =>
    request(
      `/snp/frequency-sweep?start_hz=${startHz}&stop_hz=${stopHz}&num_points=${numPoints}&solution_index=${solutionIndex}`
    ),

  manualTune: (params) =>
    request('/manual-tune', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  multiScenarioOptimize: (params) =>
    request('/multi-scenario/optimize', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  startMultiScenarioJob: (params) =>
    request('/multi-scenario/jobs', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  multiScenarioManual: (params) =>
    request('/multi-scenario/manual', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  /* ─── Efficiency ─── */
  getOptimizationModes: () => request('/optimization-modes'),

  loadEfficiency: (portIndex, filepath) =>
    request('/efficiency/load', {
      method: 'POST',
      body: JSON.stringify({ port_index: portIndex, filepath }),
    }),

  loadEfficiencyInline: (portIndex, content) =>
    request('/efficiency/inline', {
      method: 'POST',
      body: JSON.stringify({ port_index: portIndex, content }),
    }),

  getEfficiencyStatus: () => request('/efficiency/status'),

  clearEfficiency: (portIndex = -1) =>
    request('/efficiency/clear', {
      method: 'POST',
      body: JSON.stringify({ port_index: portIndex }),
    }),

  /* ─── Tuning API (unified, Optenni-style) ─── */

  tuningOptimize: (params) =>
    request('/tuning/optimize', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  tuningTunable: (params) =>
    request('/tuning/optimize', {
      method: 'POST',
      body: JSON.stringify({ ...params, mode: 'tunable' }),
    }),

  tuningSweep: (portIndex, startHz, stopHz, numPoints = 201, solutionIndex = 0, useSnpPoints = true) =>
    request(
      `/tuning/sweep?port_index=${portIndex}&start_hz=${startHz}&stop_hz=${stopHz}&num_points=${numPoints}&solution_index=${solutionIndex}&use_snp_points=${useSnpPoints}`
    ),

  tuningPowerBalance: (freqHz = 2.45e9) =>
    request(`/tuning/power-balance?frequency_hz=${freqHz}`),

  tuningSelect: (solutionIndex) =>
    request(`/tuning/select?solution_index=${solutionIndex}`, { method: 'POST' }),

  tuningStatus: () => request('/tuning/status'),

  tuningYield: (params) =>
    request('/tuning/yield', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  tuningContinue: (additionalSeconds) =>
    request('/tuning/continue', {
      method: 'POST',
      body: JSON.stringify({ additional_seconds: additionalSeconds }),
    }),

  startTuningContinueJob: (additionalSeconds) =>
    request('/tuning/continue/jobs', {
      method: 'POST',
      body: JSON.stringify({ additional_seconds: additionalSeconds }),
    }),

  /* ─── Project snapshots ─── */
  listProjects: () => request('/projects'),

  projectReportUrl: (projectId) =>
    `${API_BASE}/projects/${encodeURIComponent(projectId)}/report`,

  projectPdfReportUrl: (projectId) =>
    `${API_BASE}/projects/${encodeURIComponent(projectId)}/report.pdf`,

  projectBomUrl: (projectId) =>
    `${API_BASE}/projects/${encodeURIComponent(projectId)}/bom.csv`,

  projectSnapshotUrl: (projectId) =>
    `${API_BASE}/projects/${encodeURIComponent(projectId)}/snapshot.json`,

  saveProject: (name, projectId = null, manualWorkspace = null) =>
    request('/projects/save', {
      method: 'POST',
      body: JSON.stringify({
        name, project_id: projectId, manual_workspace: manualWorkspace,
      }),
    }),

  importProject: (document, conflictPolicy = 'copy') =>
    request('/projects/import', {
      method: 'POST',
      body: JSON.stringify({ document, conflict_policy: conflictPolicy }),
    }),

  relinkProject: (projectId, applyMatches = true) =>
    request('/projects/relink', {
      method: 'POST',
      body: JSON.stringify({ project_id: projectId, apply_matches: applyMatches }),
    }),

  startTuningJob: (params) =>
    request('/tuning/jobs', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  startManualRefineJob: (params) =>
    request('/manual-refine/jobs', {
      method: 'POST',
      body: JSON.stringify(params),
    }),
  startManualYieldJob: (params) =>
    request('/manual-yield/jobs', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  getTuningJob: (jobId) => request(`/tuning/jobs/${jobId}`),

  cancelTuningJob: (jobId) =>
    request(`/tuning/jobs/${jobId}/cancel`, { method: 'POST' }),

  loadProject: (projectId, verifyInput = true) =>
    request('/projects/load', {
      method: 'POST',
      body: JSON.stringify({ project_id: projectId, verify_input: verifyInput }),
    }),

};
