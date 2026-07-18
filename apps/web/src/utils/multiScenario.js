export const MULTI_SCENARIO_STAGE_LABELS = {
  queued: '等待执行',
  ideal_screen: '理想拓扑预筛',
  physical_screen: '实物器件预筛',
  physical_refine: '实物器件精搜',
  verification: '独立密集验证',
  complete: '优化完成',
  cancelling: '正在取消',
  cancelled: '已取消',
};

export function validateMultiScenarioInput({ scenarios, bands, inputPort, topologyNames, requireTopology }) {
  if (!Array.isArray(scenarios) || scenarios.length < 2) return '请至少选择两个测量场景。';
  if (scenarios.some(item => !Number.isFinite(Number(item.weight)) || Number(item.weight) < 0)) {
    return '场景权重必须是大于或等于 0 的数字。';
  }
  if (!scenarios.some(item => Number(item.weight) > 0)) return '至少一个场景的权重必须大于 0。';
  if (!Number.isInteger(Number(inputPort)) || Number(inputPort) < 0) return '输入端口必须是大于或等于 1 的整数。';
  if (!Array.isArray(bands) || bands.length === 0) return '请至少设置一个频段。';
  for (let index = 0; index < bands.length; index += 1) {
    const [start, stop] = bands[index] || [];
    if (!Number.isFinite(Number(start)) || !Number.isFinite(Number(stop)) || Number(start) <= 0 || Number(stop) <= Number(start)) {
      return `频段 ${index + 1} 无效：起点和终点应为正数，且终点必须大于起点。`;
    }
  }
  if (requireTopology && (!Array.isArray(topologyNames) || topologyNames.length === 0)) return '请至少选择一种拓扑。';
  return '';
}

export function multiScenarioProgress(job) {
  const progress = job?.progress || {};
  const current = Number(progress.current) || 0;
  const total = Number(progress.total) || 0;
  const fraction = total > 0 ? Math.min(1, current / total) : Number(progress.budget_fraction) || 0;
  return {
    label: MULTI_SCENARIO_STAGE_LABELS[progress.stage] || progress.message || '正在计算',
    percent: Math.max(0, Math.min(100, Math.round(fraction * 100))),
    elapsed: Number(progress.elapsed_seconds) || 0,
    evaluations: Number(progress.physical_evaluations) || 0,
  };
}
