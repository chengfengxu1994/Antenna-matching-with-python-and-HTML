import test from 'node:test';
import assert from 'node:assert/strict';

import {
  clampFrequency,
  compareManualBom,
  moveArrayItem,
  retargetManualNetwork,
  stepPreferredValue,
  toggleBoundedSelection,
  validateManualNetwork,
  manualResultMatchesDut,
  manualRefinementProgress,
  manualRefinementVariableCount,
  manualYieldProgress,
  insertManualTopologyProbe,
  manualValueSliderConfig,
  gammaToImpedance,
  summarizeManualBands,
  validateManualBands,
  valueFromLogSlider,
} from '../src/utils/manualTuning.js';

test('network components move without mutating their source order', () => {
  const original = ['series-L', 'shunt-C', 'series-R'];
  assert.deepEqual(moveArrayItem(original, 0, 2), ['shunt-C', 'series-R', 'series-L']);
  assert.deepEqual(moveArrayItem(original, 2, 1), ['series-L', 'series-R', 'shunt-C']);
  assert.deepEqual(moveArrayItem(original, -1, 1), original);
  assert.deepEqual(original, ['series-L', 'shunt-C', 'series-R']);
});

test('E24 stepping moves to adjacent preferred values across decades', () => {
  assert.equal(stepPreferredValue(2.2, 1), 2.4);
  assert.equal(stepPreferredValue(2.2, -1), 2.0);
  assert.equal(stepPreferredValue(9.1, 1), 10);
  assert.equal(stepPreferredValue(1.0, -1), 0.91);
});

test('retargeting creates an independent network for the selected port', () => {
  const original = [{ comp_type: 'inductor', value: 5.6, port: 0 }];
  const retargeted = retargetManualNetwork(original, 2);
  assert.deepEqual(retargeted, [{ comp_type: 'inductor', value: 5.6, port: 2 }]);
  assert.notEqual(retargeted[0], original[0]);
  assert.equal(original[0].port, 0);
});

test('frequency cursor is clamped to the measured sweep', () => {
  assert.equal(clampFrequency(2.45e9, 2.4e9, 2.5e9), 2.45e9);
  assert.equal(clampFrequency(2.3e9, 2.4e9, 2.5e9), 2.4e9);
  assert.equal(clampFrequency(2.6e9, 2.4e9, 2.5e9), 2.5e9);
});

test('curve overlay selection is unique, toggleable, and bounded', () => {
  assert.deepEqual(toggleBoundedSelection(['a', 'b'], 'a', 4), ['b']);
  assert.deepEqual(toggleBoundedSelection(['a', 'b'], 'c', 4), ['a', 'b', 'c']);
  assert.deepEqual(toggleBoundedSelection(['a', 'b', 'c', 'd'], 'e', 4), ['a', 'b', 'c', 'd']);
  assert.deepEqual(toggleBoundedSelection(['a', 'a'], 'b', 4), ['a', 'b']);
});

test('manual-network validation identifies the exact invalid element', () => {
  assert.match(validateManualNetwork([{comp_type:'inductor', value:0}]), /元件 1/);
  assert.match(validateManualNetwork([{comp_type:'transmission_line', characteristic_impedance_ohm:0,
    reference_frequency_hz:1e9, electrical_length_deg:45, attenuation_db:0}]), /特性阻抗/);
  assert.equal(validateManualNetwork([{comp_type:'capacitor', value:2.2, use_ideal:false, part_number:'C1'}]), '');
});

test('manual result identity rejects a stale CST revision', () => {
  const result = {dut_identity:{filename:'test.s1p', network_sha256:'new'}};
  assert.equal(manualResultMatchesDut(result, {filename:'test.s1p', network_sha256:'new'}), true);
  assert.equal(manualResultMatchesDut(result, {filename:'test.s1p', network_sha256:'old'}), false);
});

test('continuous manual slider uses logarithmic engineering ranges', () => {
  const config = manualValueSliderConfig('inductor', 10);
  assert.equal(config.minimum, 0.1);
  assert.equal(config.maximum, 100);
  assert.equal(valueFromLogSlider(1), 10);
  assert.equal(manualValueSliderConfig('resistor', 5000).maximum, 50000);
});

test('manual band summary reports worst-point margin against the 10 dB goal', () => {
  const summary = summarizeManualBands({
    frequencies: [2.4e9, 2.45e9, 2.5e9, 2.6e9],
    s11_db: [12, 9, 15, 20],
  }, [[2400, 2500], [2550, 2650]]);
  assert.equal(summary[0].points, 3);
  assert.equal(summary[0].worstReturnLossDb, 9);
  assert.equal(summary[0].marginDb, -1);
  assert.equal(summary[0].passes, false);
  assert.equal(summary[1].passes, true);
});

test('manual band validation rejects reversed, out-of-sweep, and overlapping bands', () => {
  assert.match(validateManualBands([[2500, 2400]], 2e9, 3e9), /终止频率/);
  assert.match(validateManualBands([[1900, 2100]], 2e9, 3e9), /DUT 扫频范围/);
  assert.match(validateManualBands([[2100, 2300], [2200, 2400]], 2e9, 3e9), /重叠/);
  assert.equal(validateManualBands([[2100, 2200], [2200, 2400]], 2e9, 3e9), '');
});

test('reflection coefficient converts to physical impedance at the selected Z0', () => {
  assert.deepEqual(gammaToImpedance(0, 0, 50), {
    resistanceOhm: 50, reactanceOhm: 0, gammaMagnitude: 0, gammaAngleDeg: 0,
  });
  const openish = gammaToImpedance(0.5, 0, 50);
  assert.equal(openish.resistanceOhm, 150);
  assert.equal(openish.reactanceOhm, 0);
  assert.equal(gammaToImpedance(1, 0, 50), null);
});

test('manual BOM comparison respects physical position, connection, and exact part identity', () => {
  const current = [
    {connection_type:'series', comp_type:'inductor', value:5.6},
    {connection_type:'shunt', comp_type:'capacitor', part_number:'GJM15-2P2'},
  ];
  assert.equal(compareManualBom(current, current.map(item => ({...item}))).isIdentical, true);
  const changed = compareManualBom(current, [
    {connection_type:'series', comp_type:'inductor', value:6.8},
    {connection_type:'series', comp_type:'capacitor', part_number:'GJM15-2P2'},
    {connection_type:'series', comp_type:'resistor', value:1},
  ]);
  assert.deepEqual({changed:changed.changed, added:changed.added, removed:changed.removed},
    {changed:2, added:1, removed:0});
});

test('manual refinement counts only continuous knobs and normalizes job progress', () => {
  assert.equal(manualRefinementVariableCount([
    {comp_type:'inductor', use_ideal:true},
    {comp_type:'capacitor', use_ideal:false},
    {comp_type:'transmission_line'},
  ]), 3);
  assert.deepEqual(manualRefinementProgress({status:'running', progress:{
    stage:'manual_refinement', current:5, total:10, best_worst_return_loss_db:12.5,
  }}), {
    status:'running', stage:'manual_refinement', current:5, total:10,
    fraction:0.5, bestWorstReturnLossDb:12.5,
  });
  assert.equal(manualRefinementProgress({status:'running', progress:{
    stage:'manual_sensitivity', current:11, total:12,
  }}).stage, 'manual_sensitivity');
});

test('manual yield progress exposes bounded live yield estimate', () => {
  assert.deepEqual(manualYieldProgress({status:'running', progress:{
    stage:'manual_yield', current:50, total:200, yield_fraction:1.2,
  }}), {
    status:'running', stage:'manual_yield', current:50, total:200,
    fraction:0.25, yieldFraction:1,
  });
});

test('manual topology probe inserts an independent ideal component at the physical position', () => {
  const current = [{comp_type:'inductor', connection_type:'series', value:5.6, port:0}];
  const next = insertManualTopologyProbe(current, {
    insertion_index:0,
    component:{comp_type:'capacitor', connection_type:'shunt', value:2.2, use_ideal:true},
  }, 1, 2.45e9);
  assert.equal(current.length, 1);
  assert.deepEqual(next.map(item => item.comp_type), ['capacitor', 'inductor']);
  assert.equal(next[0].port, 1);
  assert.equal(next[0].reference_frequency_hz, 2.45e9);
  const measured = insertManualTopologyProbe(current, {
    insertion_index:1,
    component:{comp_type:'inductor', connection_type:'series', value:5.6,
      use_ideal:false, part_number:'LQ_TEST'},
  }, 0, 1e9);
  assert.equal(measured[1].part_number, 'LQ_TEST');
  assert.equal(measured[1].use_ideal, false);
});
