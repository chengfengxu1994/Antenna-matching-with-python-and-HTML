import test from 'node:test';
import assert from 'node:assert/strict';

import {
  formatCstExportMessage,
  ingestionMethodLabel,
  isSameElectricalDut,
  shouldLoadDataRevision,
  reconcileSeriesSelection,
  shouldPollCstTree,
  sourceLabel,
} from '../src/utils/dataSource.js';

test('solver and ingestion sources use precise user-facing labels', () => {
  assert.equal(sourceLabel('CST'), 'CST');
  assert.equal(sourceLabel('HFSS'), 'HFSS');
  assert.equal(sourceLabel('VNA'), 'VNA');
  assert.equal(ingestionMethodLabel('cst_python_bridge'), 'CST 直连');
  assert.equal(ingestionMethodLabel('directory_watch'), '目录同步');
  assert.equal(ingestionMethodLabel('file_import'), '手动导入');
  assert.equal(ingestionMethodLabel('future_method'), '文件导入');
});

test('component-series selection cannot leak across catalog revisions', () => {
  assert.deepEqual(
    reconcileSeriesSelection(['L::old', 'C::kept'], ['L::new', 'C::kept'], ['L::new', 'C::kept']),
    ['C::kept'],
  );
  assert.deepEqual(
    reconcileSeriesSelection(['L::old'], ['L::new', 'C::new'], ['L::new', 'C::new']),
    ['L::new', 'C::new'],
  );
  assert.deepEqual(
    reconcileSeriesSelection(null, ['L::new'], ['L::new', 'C::missing']),
    ['L::new'],
  );
});

test('deferred panels load each ready data-source revision only when active', () => {
  assert.equal(shouldLoadDataRevision({ active: false, ready: true, revision: 1, lastRevision: 0 }), false);
  assert.equal(shouldLoadDataRevision({ active: true, ready: false, revision: 1, lastRevision: 0 }), false);
  assert.equal(shouldLoadDataRevision({ active: true, ready: true, revision: 0, lastRevision: 0 }), false);
  assert.equal(shouldLoadDataRevision({ active: true, ready: true, revision: 1, lastRevision: 0 }), true);
  assert.equal(shouldLoadDataRevision({ active: true, ready: true, revision: 1, lastRevision: 1 }), false);
});

test('electrical DUT identity prefers semantic network SHA over raw file bytes', () => {
  const current = { filename: 'dut.s1p', sha256: 'raw-a', network_sha256: 'network-a' };
  assert.equal(isSameElectricalDut(current, {
    filename: 'dut.s1p', sha256: 'raw-b', network_sha256: 'network-a',
  }), true);
  assert.equal(isSameElectricalDut(current, {
    filename: 'dut.s1p', sha256: 'raw-b', network_sha256: 'network-b',
  }), false);
  assert.equal(isSameElectricalDut(current, {
    filename: 'other.s1p', sha256: 'raw-a', network_sha256: 'network-a',
  }), false);
});

test('CST tree polling only runs for a selected project during solving', () => {
  assert.equal(shouldPollCstTree({ tree: { solver_running: true }, selected: { pid: 1 } }), true);
  assert.equal(shouldPollCstTree({ tree: { solver_running: false }, selected: { pid: 1 } }), false);
  assert.equal(shouldPollCstTree({ tree: { solver_running: true }, selected: null }), false);
});

test('CST export feedback distinguishes new, changed, and unchanged results', () => {
  const result = { filename: 'antenna.s1p', num_ports: 1, freq_count: 101 };
  assert.match(formatCstExportMessage(result), /^已从 CST 直出并载入/);
  assert.match(formatCstExportMessage({ ...result, replaced_existing: true, content_changed: true }), /^已刷新 CST 新修订并重新载入/);
  assert.match(formatCstExportMessage({ ...result, replaced_existing: true, content_changed: false }), /^CST 结果内容未变化，已重新载入/);
});
