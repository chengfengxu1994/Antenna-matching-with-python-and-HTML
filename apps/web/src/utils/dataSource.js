const SOURCE_LABELS = {
  CST: 'CST',
  HFSS: 'HFSS',
  VNA: 'VNA',
  Touchstone: 'Touchstone',
};

export function sourceLabel(source) {
  return SOURCE_LABELS[source] || String(source || 'Touchstone');
}

export function ingestionMethodLabel(method) {
  return {
    cst_python_bridge: 'CST 直连',
    directory_watch: '目录同步',
    file_import: '手动导入',
  }[method] || '文件导入';
}

export const CST_TREE_POLL_INTERVAL_MS = 2000;

export function shouldPollCstTree(connection) {
  return Boolean(connection?.tree?.solver_running && connection?.selected);
}

export function formatCstExportMessage(result) {
  const prefix = result?.replaced_existing
    ? result.content_changed === false
      ? 'CST 结果内容未变化，已重新载入'
      : '已刷新 CST 新修订并重新载入'
    : '已从 CST 直出并载入';
  return `${prefix} ${result.filename} · ${result.num_ports} 端口 · ${result.freq_count} 频点`;
}

export function isSameElectricalDut(current, next) {
  const currentKey = current?.network_sha256 || current?.sha256;
  const nextKey = next?.network_sha256 || next?.sha256;
  return Boolean(
    current?.filename && current.filename === next?.filename
    && currentKey && currentKey === nextKey
  );
}

export function shouldLoadDataRevision({ active, revision, lastRevision, ready = true }) {
  return Boolean(active && ready && Number(revision) > 0 && revision !== lastRevision);
}

export function reconcileSeriesSelection(current, availableIds, defaults = []) {
  const available = new Set(availableIds || []);
  const fallback = (defaults || []).filter(item => available.has(item));
  if (current === null) return fallback;
  const valid = (current || []).filter(item => available.has(item));
  return valid.length ? valid : fallback;
}
