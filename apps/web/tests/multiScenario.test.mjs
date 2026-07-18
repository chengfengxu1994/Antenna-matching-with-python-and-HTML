import assert from 'node:assert/strict';
import test from 'node:test';
import { multiScenarioProgress, validateMultiScenarioInput } from '../src/utils/multiScenario.js';

const valid = { scenarios: [{weight: 1}, {weight: 0.5}], bands: [[2400, 2500]], inputPort: 0, topologyNames: ['L'], requireTopology: true };

test('multi-scenario validation accepts a valid engineering request', () => {
  assert.equal(validateMultiScenarioInput(valid), '');
});

test('multi-scenario validation rejects reversed bands and zero total weight', () => {
  assert.match(validateMultiScenarioInput({...valid, bands:[[2500, 2400]]}), /频段 1/);
  assert.match(validateMultiScenarioInput({...valid, scenarios:[{weight:0}, {weight:0}]}), /至少一个场景/);
});

test('multi-scenario progress maps backend stage and clamps percentage', () => {
  assert.deepEqual(multiScenarioProgress({progress:{stage:'verification', current:3, total:2, elapsed_seconds:4.2, physical_evaluations:17}}), {
    label:'独立密集验证', percent:100, elapsed:4.2, evaluations:17,
  });
});
