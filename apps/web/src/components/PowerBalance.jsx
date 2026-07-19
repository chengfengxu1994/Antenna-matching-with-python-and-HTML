import React from 'react';

const POWER_COLORS = {
  reflected: '#e74c3c',      // red
  coupled: '#f39c12',        // orange
  component_loss: '#9b59b6', // purple
  antenna_loss: '#e67e22',   // amber
  radiated: '#2ecc71',       // green
};

const POWER_LABELS = {
  reflected: 'Reflected',
  coupled: 'Coupled',
  component_loss: 'Comp. Loss',
  antenna_loss: 'Ant. Loss',
  radiated: 'Radiated',
};

/**
 * Power Balance stacked bar chart.
 * Shows where input power goes for each port.
 */
export default function PowerBalance({ resultsPerPort, powerBalance }) {
  if (!resultsPerPort || Object.keys(resultsPerPort).length === 0) {
    return (
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', textAlign: 'center', padding: 20 }}>
        No power balance data available
      </div>
    );
  }

  // Build rows from per-port results
  const portIndices = Object.keys(resultsPerPort).map(Number).sort((a, b) => a - b);
  const rows = portIndices.map(pi => {
    const pr = resultsPerPort[pi];
    // Use power_balance if available, else estimate from other fields
    const pb = pr.power_balance || powerBalance?.[pi] || null;
    if (pb) {
      return {
        portIndex: pi,
        reflected: pb.reflected || 0,
        coupled: pb.coupled || 0,
        componentLoss: pb.component_loss || 0,
        antennaLoss: pb.antenna_loss || 0,
        radiated: pb.radiated || 0,
      };
    }
    // Fallback estimate
    const s11Mag = pr.s11_magnitude || 0;
    const reflected = s11Mag * s11Mag;
    const coupling = pr.coupling_loss || 0;
    const radiated = Math.max(0, 1 - reflected - coupling);
    return {
      portIndex: pi,
      reflected,
      coupled: coupling,
      componentLoss: 0,
      antennaLoss: 0,
      radiated,
    };
  });

  const maxTotal = 1.0; // Normalised to input power = 1

  return (
    <div>
      <div style={{ marginBottom: 8, fontSize: 11, color: 'var(--text-secondary)' }}>
        Power distribution per port (normalised to 100% input power)
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {rows.map(r => {
          const total = r.reflected + r.coupled + r.componentLoss + r.antennaLoss + r.radiated;
          const norm = total > 0 ? 1 / total : 1;
          const segments = [
            { key: 'reflected', value: r.reflected * norm, color: POWER_COLORS.reflected },
            { key: 'coupled', value: r.coupled * norm, color: POWER_COLORS.coupled },
            { key: 'componentLoss', value: r.componentLoss * norm, color: POWER_COLORS.component_loss },
            { key: 'antennaLoss', value: r.antennaLoss * norm, color: POWER_COLORS.antenna_loss },
            { key: 'radiated', value: r.radiated * norm, color: POWER_COLORS.radiated },
          ].filter(s => s.value > 0.005); // Only show segments > 0.5%

          return (
            <div key={r.portIndex} style={{ fontSize: 11 }}>
              <div style={{
                display: 'flex', justifyContent: 'space-between', marginBottom: 2,
              }}>
                <strong>Port {r.portIndex + 1}</strong>
                <span style={{ color: POWER_COLORS.radiated }}>
                  {(r.radiated * 100).toFixed(1)}% radiated
                </span>
              </div>
              <div style={{
                height: 22, borderRadius: 4, overflow: 'hidden',
                display: 'flex', background: 'var(--bg-input)',
              }}>
                {segments.map(s => (
                  <div key={s.key} style={{
                    width: `${s.value * 100}%`,
                    backgroundColor: s.color,
                    height: '100%',
                    transition: 'width 0.3s ease',
                    position: 'relative',
                  }} title={`${POWER_LABELS[s.key] || s.key}: ${(s.value * 100).toFixed(1)}%`} />
                ))}
              </div>
              <div style={{
                display: 'flex', gap: 10, marginTop: 2, flexWrap: 'wrap',
              }}>
                {segments.map(s => (
                  <span key={s.key} style={{ fontSize: 9, display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                    <span style={{
                      width: 8, height: 8, borderRadius: 2,
                      backgroundColor: s.color, display: 'inline-block',
                    }} />
                    {POWER_LABELS[s.key] || s.key}: {(s.value * 100).toFixed(1)}%
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
