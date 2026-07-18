export const E24_BASE_VALUES = [
  1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
  3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1,
];

export function retargetManualNetwork(components, port) {
  return (components || []).map(component => ({ ...component, port }));
}

export function stepPreferredValue(value, direction) {
  const numeric = Number(value);
  const stepDirection = Math.sign(Number(direction));
  if (!Number.isFinite(numeric) || numeric <= 0 || !stepDirection) return numeric;

  const exponent = Math.floor(Math.log10(numeric));
  const candidates = [];
  for (let decade = exponent - 2; decade <= exponent + 2; decade += 1) {
    const scale = 10 ** decade;
    E24_BASE_VALUES.forEach(base => candidates.push(base * scale));
  }
  candidates.sort((left, right) => left - right);
  const epsilon = Math.max(Math.abs(numeric) * 1e-9, Number.EPSILON);
  if (stepDirection > 0) {
    return candidates.find(candidate => candidate > numeric + epsilon) ?? numeric * 10;
  }
  const lower = candidates.filter(candidate => candidate < numeric - epsilon);
  return lower.length ? lower[lower.length - 1] : numeric / 10;
}

export function clampFrequency(frequencyHz, minimumHz, maximumHz) {
  const numeric = Number(frequencyHz);
  const minimum = Number(minimumHz);
  const maximum = Number(maximumHz);
  if (!Number.isFinite(numeric) || !Number.isFinite(minimum) || !Number.isFinite(maximum)) {
    return minimum;
  }
  return Math.min(maximum, Math.max(minimum, numeric));
}

export function toggleBoundedSelection(values, value, maximum = 4) {
  const current = Array.from(new Set(values || []));
  if (current.includes(value)) return current.filter(item => item !== value);
  if (current.length >= maximum) return current;
  return [...current, value];
}

export function moveArrayItem(values, fromIndex, toIndex) {
  const current = Array.isArray(values) ? [...values] : [];
  if (!Number.isInteger(fromIndex) || !Number.isInteger(toIndex)
    || fromIndex < 0 || toIndex < 0
    || fromIndex >= current.length || toIndex >= current.length
    || fromIndex === toIndex) return current;
  const [item] = current.splice(fromIndex, 1);
  current.splice(toIndex, 0, item);
  return current;
}

export function validateManualNetwork(components) {
  for (let index = 0; index < (components || []).length; index += 1) {
    const component = components[index] || {};
    const label = `元件 ${index + 1}`;
    if (['inductor', 'capacitor', 'resistor'].includes(component.comp_type)) {
      if (!Number.isFinite(Number(component.value)) || Number(component.value) <= 0) {
        return `${label}的数值必须大于 0。`;
      }
      if (component.use_ideal === false && component.comp_type === 'resistor') {
        return `${label}的电阻暂不支持实测 S2P 模式。`;
      }
    }
    if (['transmission_line', 'open_stub', 'short_stub'].includes(component.comp_type)) {
      if (!Number.isFinite(Number(component.characteristic_impedance_ohm))
        || Number(component.characteristic_impedance_ohm) <= 0) return `${label}的特性阻抗必须大于 0。`;
      if (!Number.isFinite(Number(component.reference_frequency_hz))
        || Number(component.reference_frequency_hz) <= 0) return `${label}的参考频率必须大于 0。`;
      if (!Number.isFinite(Number(component.electrical_length_deg))
        || Number(component.electrical_length_deg) < 0) return `${label}的电长度不能为负数。`;
      if (!Number.isFinite(Number(component.attenuation_db))
        || Number(component.attenuation_db) < 0) return `${label}的损耗不能为负数。`;
    }
  }
  return '';
}

export function manualResultMatchesDut(result, loadedSNP) {
  if (!result?.dut_identity || !loadedSNP) return false;
  const expected = loadedSNP.network_sha256;
  return result.dut_identity.filename === loadedSNP.filename
    && (!expected || result.dut_identity.network_sha256 === expected);
}

export function manualValueSliderConfig(componentType, value) {
  const defaults = {
    inductor: [0.1, 100],
    capacitor: [0.1, 100],
    resistor: [1, 1000],
  };
  const [defaultMinimum, defaultMaximum] = defaults[componentType] || [0.1, 1000];
  const numeric = Number(value);
  const positive = Number.isFinite(numeric) && numeric > 0 ? numeric : defaultMinimum;
  const minimum = Math.min(defaultMinimum, positive / 10);
  const maximum = Math.max(defaultMaximum, positive * 10);
  return {
    minimum,
    maximum,
    minimumLog: Math.log10(minimum),
    maximumLog: Math.log10(maximum),
    valueLog: Math.log10(positive),
  };
}

export function valueFromLogSlider(position) {
  const numeric = 10 ** Number(position);
  return Number.isFinite(numeric) && numeric > 0
    ? Number(numeric.toPrecision(6)) : 0;
}

export function summarizeManualBands(sweep, bandsMhz, targetReturnLossDb = 10) {
  const frequencies = sweep?.frequencies || [];
  const returnLoss = sweep?.s11_db || [];
  if (!frequencies.length || frequencies.length !== returnLoss.length) return [];
  return (bandsMhz || []).map((band, index) => {
    const startMhz = Number(band?.[0]);
    const stopMhz = Number(band?.[1]);
    if (!Number.isFinite(startMhz) || !Number.isFinite(stopMhz) || stopMhz < startMhz) return null;
    const values = returnLoss.filter((value, pointIndex) => {
      const frequencyMhz = Number(frequencies[pointIndex]) / 1e6;
      return frequencyMhz >= startMhz && frequencyMhz <= stopMhz && Number.isFinite(Number(value));
    }).map(Number);
    if (!values.length) return {
      index, startMhz, stopMhz, points: 0, averageReturnLossDb: null,
      worstReturnLossDb: null, marginDb: null, passes: false,
    };
    const averageReturnLossDb = values.reduce((sum, value) => sum + value, 0) / values.length;
    const worstReturnLossDb = Math.min(...values);
    return {
      index, startMhz, stopMhz, points: values.length,
      averageReturnLossDb, worstReturnLossDb,
      marginDb: worstReturnLossDb - targetReturnLossDb,
      passes: worstReturnLossDb >= targetReturnLossDb,
    };
  }).filter(Boolean);
}

export function validateManualBands(bandsMhz, minimumFrequencyHz, maximumFrequencyHz) {
  const minimumMhz = Number(minimumFrequencyHz) / 1e6;
  const maximumMhz = Number(maximumFrequencyHz) / 1e6;
  for (let index = 0; index < (bandsMhz || []).length; index += 1) {
    const start = Number(bandsMhz[index]?.[0]);
    const stop = Number(bandsMhz[index]?.[1]);
    if (!Number.isFinite(start) || !Number.isFinite(stop)) return `频段 ${index + 1} 必须填写有效起止频率。`;
    if (stop < start) return `频段 ${index + 1} 的终止频率不能小于起始频率。`;
    if (start < minimumMhz || stop > maximumMhz) {
      return `频段 ${index + 1} 必须位于 DUT 扫频范围 ${Number(minimumMhz.toPrecision(8))}–${Number(maximumMhz.toPrecision(8))} MHz 内。`;
    }
  }
  const sorted = (bandsMhz || []).map((band, index) => ({
    index, start: Number(band[0]), stop: Number(band[1]),
  })).sort((left, right) => left.start - right.start);
  for (let index = 1; index < sorted.length; index += 1) {
    if (sorted[index].start < sorted[index - 1].stop) {
      return `频段 ${sorted[index - 1].index + 1} 与频段 ${sorted[index].index + 1} 重叠，请合并或调整。`;
    }
  }
  return '';
}

export function gammaToImpedance(real, imaginary, referenceImpedanceOhm = 50) {
  const re = Number(real);
  const im = Number(imaginary);
  const z0 = Number(referenceImpedanceOhm);
  const denominator = (1 - re) ** 2 + im ** 2;
  if (![re, im, z0, denominator].every(Number.isFinite) || z0 <= 0 || denominator < 1e-15) return null;
  return {
    resistanceOhm: z0 * (1 - re ** 2 - im ** 2) / denominator,
    reactanceOhm: z0 * (2 * im) / denominator,
    gammaMagnitude: Math.hypot(re, im),
    gammaAngleDeg: Math.atan2(im, re) * 180 / Math.PI,
  };
}

function manualComponentSignature(component) {
  if (!component) return null;
  const type = String(component.comp_type || component.component_type || '');
  const connection = String(component.connection_type || 'series');
  const measuredPart = String(component.part_number || '').trim().toLocaleLowerCase();
  if (measuredPart) return [connection, type, 'part', measuredPart];
  if (['transmission_line', 'open_stub', 'short_stub'].includes(type)) {
    return [
      connection, type,
      Number(component.characteristic_impedance_ohm),
      Number(component.electrical_length_deg),
      Number(component.reference_frequency_hz),
      Number(component.attenuation_db || 0),
    ];
  }
  return [connection, type, 'ideal', Number(component.value)];
}

export function compareManualBom(referenceComponents, candidateComponents) {
  const reference = referenceComponents || [];
  const candidate = candidateComponents || [];
  const maximum = Math.max(reference.length, candidate.length);
  const changes = [];
  for (let index = 0; index < maximum; index += 1) {
    if (index >= reference.length) changes.push({ position: index, kind: 'added' });
    else if (index >= candidate.length) changes.push({ position: index, kind: 'removed' });
    else if (JSON.stringify(manualComponentSignature(reference[index]))
      !== JSON.stringify(manualComponentSignature(candidate[index]))) {
      changes.push({ position: index, kind: 'changed' });
    }
  }
  const added = changes.filter(change => change.kind === 'added').length;
  const removed = changes.filter(change => change.kind === 'removed').length;
  const changed = changes.filter(change => change.kind === 'changed').length;
  return {
    added, removed, changed, changes,
    isIdentical: changes.length === 0,
    summary: changes.length ? `改 ${changed} · 增 ${added} · 减 ${removed}` : '与当前 BOM 相同',
  };
}

export function manualRefinementVariableCount(components) {
  return (components || []).reduce((count, component) => {
    const type = String(component?.comp_type || '').toLowerCase();
    if (['inductor', 'capacitor', 'resistor'].includes(type)) {
      return count + (component.use_ideal === false ? 0 : 1);
    }
    return count + (['transmission_line', 'open_stub', 'short_stub'].includes(type) ? 2 : 0);
  }, 0);
}

export function manualRefinementProgress(job) {
  const progress = job?.progress || {};
  const current = Math.max(0, Number(progress.current || 0));
  const total = Math.max(0, Number(progress.total || 0));
  return {
    status: job?.status || 'idle',
    stage: progress.stage || job?.status || 'idle',
    current,
    total,
    fraction: total > 0 ? Math.min(1, current / total) : 0,
    bestWorstReturnLossDb: Number.isFinite(Number(progress.best_worst_return_loss_db))
      ? Number(progress.best_worst_return_loss_db) : null,
  };
}

export function manualYieldProgress(job) {
  const progress = job?.progress || {};
  const current = Math.max(0, Number(progress.current || 0));
  const total = Math.max(0, Number(progress.total || 0));
  const estimate = Number(progress.yield_fraction);
  return {
    status: job?.status || 'idle',
    stage: progress.stage || job?.status || 'idle',
    current,
    total,
    fraction: total > 0 ? Math.min(1, current / total) : 0,
    yieldFraction: Number.isFinite(estimate) ? Math.max(0, Math.min(1, estimate)) : null,
  };
}

export function insertManualTopologyProbe(components, probe, inputPort, targetFrequencyHz) {
  if (!probe?.component) return [...(components || [])];
  const source = [...(components || [])];
  const insertionIndex = Math.max(0, Math.min(
    source.length, Number.isInteger(Number(probe.insertion_index)) ? Number(probe.insertion_index) : 0,
  ));
  source.splice(insertionIndex, 0, {
    ...probe.component,
    port: inputPort,
    reference_frequency_hz: probe.component.reference_frequency_hz || targetFrequencyHz,
  });
  return source;
}
