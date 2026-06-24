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

  setDataDirs: (snpDir, murataDir) =>
    request('/config/dirs', {
      method: 'POST',
      body: JSON.stringify({ snp_dir: snpDir, murata_dir: murataDir }),
    }),

  listSNPFiles: () => request('/snp/list'),

  loadSNP: (filename) =>
    request('/snp/load?filename=' + encodeURIComponent(filename), { method: 'POST' }),

  getSParams: (freqHz = 64e6) => request(`/snp/s-params?freq_hz=${freqHz}`),

  listComponents: (type, limit = 200) =>
    request(`/components/list?comp_type=${type || ''}&limit=${limit}`),

  listTopologies: (maxComponents = 4) =>
    request(`/topologies/list?max_components=${maxComponents}`),

  runOptimization: (params) =>
    request('/optimize', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  jointOptimize: (params) =>
    request('/joint-optimize', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  jointSweep: (portIndex, startHz, stopHz, numPoints = 200) =>
    request(
      `/joint-optimize/sweep?port_index=${portIndex}&start_hz=${startHz}&stop_hz=${stopHz}&num_points=${numPoints}`
    ),

  multiPortOptimize: (params) =>
    request('/multipass', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  sweepAllPorts: (startHz, stopHz, numPoints) =>
    request(`/multipass/sweep-all?start_hz=${startHz}&stop_hz=${stopHz}&num_points=${numPoints}`),

  getComponentSeries: () => request('/component-series'),
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
};
