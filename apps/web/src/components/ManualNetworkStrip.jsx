import React from 'react';

const TYPE_LABELS = {
  inductor: 'L', capacitor: 'C', resistor: 'R',
  transmission_line: 'TL', open_stub: 'OPEN', short_stub: 'SHORT',
};

function valueLabel(component) {
  if (component.part_number) return component.part_number;
  if (component.comp_type === 'inductor') return `${component.value} nH`;
  if (component.comp_type === 'capacitor') return `${component.value} pF`;
  if (component.comp_type === 'resistor') return `${component.value} Ω`;
  return `${component.electrical_length_deg ?? 0}° @ ${Number(component.reference_frequency_hz || 0) / 1e6} MHz`;
}

export default function ManualNetworkStrip({ components, inputPort, selectedIndex, onSelect }) {
  return (
    <section className="manual-network-strip" aria-label="当前物理信号链">
      <div className="manual-strip-heading">
        <div><span className="eyebrow">PHYSICAL SIGNAL PATH</span><strong>当前物理信号链</strong></div>
        <small>DUT → 信号源 · P{inputPort + 1}</small>
      </div>
      <div className="manual-strip-flow">
        <div className="manual-strip-terminal dut"><b>DUT</b><span>P{inputPort + 1}</span></div>
        {(components || []).map((component, index) => (
          <React.Fragment key={`${index}-${component.comp_type}-${component.connection_type}`}>
            <span className="manual-strip-link" aria-hidden="true">→</span>
            <button type="button"
              className={`manual-strip-component ${component.connection_type} ${selectedIndex === index ? 'selected' : ''}`}
              aria-label={`编辑元件 ${index + 1} ${TYPE_LABELS[component.comp_type] || component.comp_type}`}
              onClick={() => onSelect(index)}>
              <i>{index + 1}</i>
              <b>{component.connection_type === 'shunt' ? '∥ ' : ''}{TYPE_LABELS[component.comp_type] || component.comp_type}</b>
              <span>{valueLabel(component)}</span>
              {component.connection_type === 'shunt' && <em aria-hidden="true">⏚</em>}
            </button>
          </React.Fragment>
        ))}
        <span className="manual-strip-link" aria-hidden="true">→</span>
        <div className="manual-strip-terminal source"><b>PORT</b><span>50 Ω</span></div>
      </div>
      {!components?.length && <p>当前为裸 DUT，端口直接接入 50 Ω 信号源。</p>}
    </section>
  );
}
