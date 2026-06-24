import React from 'react';

const COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];

/** S11 dB values are negative; larger magnitude = better match */
function s11Color(s11dB) {
  const absVal = Math.abs(s11dB);
  if (absVal > 15) return 'var(--accent-green)';
  if (absVal > 8) return 'var(--accent-yellow)';
  return 'var(--accent-red)';
}

function effColor(eff) {
  if (eff > 80) return 'var(--accent-green)';
  if (eff > 50) return 'var(--accent-yellow)';
  return 'var(--accent-red)';
}

/** Render a component tag handling both joint-optimize and multipass formats */
function renderComponent(c, idx) {
  // multipass format: { connection_type, nominal_value, nominal_unit, part_number }
  // joint-optimize format: { part, type: 'inductor'|'capacitor', value: '10nH' }
  const isJoint = !c.connection_type && (c.type === 'inductor' || c.type === 'capacitor');
  let cls = 'series';
  let connLabel = '';
  if (isJoint) {
    // Use component type for styling: inductor = blue, capacitor = green
    cls = c.type === 'inductor' ? 'series' : 'shunt';
    connLabel = c.type === 'inductor' ? 'L' : 'C';
  } else {
    cls = c.connection_type === 'shunt' ? 'shunt' : 'series';
    connLabel = c.connection_type === 'shunt' ? 'shunt' : (c.connection_type === 'series' ? 'series' : '');
  }
  const valueStr = isJoint
    ? (c.value || '')
    : (c.nominal_value != null ? `${c.nominal_value}${c.nominal_unit || ''}` : c.value || '');
  return (
    <span key={idx} className={`component-tag ${cls}`}>
      {connLabel}: {valueStr}
    </span>
  );
}

export default function ResultsPanel({ jointResults, onSelectPort, selectedPort }) {
  if (!jointResults) return null;

  const { results_per_port = {}, ports_optimized = [] } = jointResults;

  if (ports_optimized.length === 0) {
    return <div style={{ fontSize: 13, color: 'var(--text-secondary)', textAlign: 'center', padding: 20 }}>
      No matching ports configured
    </div>;
  }

  /* Build table rows from results_per_port */
  const rows = Object.entries(results_per_port).map(([piStr, pr]) => {
    const pi = parseInt(piStr);
    return {
      portIndex: pi,
      isOptimized: ports_optimized.includes(pi),
      color: COLORS[pi % COLORS.length],
      s11_db: pr.s11_db != null ? pr.s11_db : (pr.best_s11_db != null ? pr.best_s11_db : null),
      s11_magnitude: pr.s11_magnitude != null ? pr.s11_magnitude : null,
      efficiency_pct: pr.efficiency_pct != null ? pr.efficiency_pct :
        (pr.mismatch_efficiency != null ? pr.mismatch_efficiency * 100 : null),
      // coupling_loss is a power ratio (0-1); convert to percentage
      coupling_loss_pct: pr.coupling_loss != null ? pr.coupling_loss * 100 : null,
      total_efficiency: pr.total_efficiency != null ? pr.total_efficiency : null,
      components: pr.components || (pr.solutions?.[0]?.components || []),
      band_mhz: pr.band_mhz || null,
      solutions_count: pr.solutions_count || null,
    };
  }).sort((a, b) => a.portIndex - b.portIndex);

  return (
    <div>
      <h3>Per-Port Results ({ports_optimized.length} ports optimized)</h3>
      <div className="results-section">
        <table>
          <thead>
            <tr>
              <th>Port</th>
              <th>RL (dB)</th>
              <th>|S11|</th>
              <th>Efficiency</th>
              <th>Coupling Loss</th>
              <th>Components</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.portIndex}
                className={`clickable-row ${selectedPort === r.portIndex ? 'selected' : ''}`}
                onClick={() => onSelectPort(r.portIndex)}>
                <td>
                  <span style={{
                    display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
                    background: r.color, marginRight: 6, verticalAlign: 'middle',
                  }}/>
                  <span style={{ fontWeight: r.isOptimized ? 700 : 400 }}>
                    Port {r.portIndex + 1}
                  </span>
                  {r.isOptimized && <span style={{ fontSize: 10, color: 'var(--accent-green)', marginLeft: 4 }}>M</span>}
                </td>
                <td style={{ fontWeight: 600, fontFamily: 'monospace', color: s11Color(r.s11_db || 0) }}>
                  {r.s11_db != null ? r.s11_db.toFixed(1) + ' dB' : '-'}
                </td>
                <td style={{ fontFamily: 'monospace', fontSize: 11 }}>
                  {r.s11_magnitude != null ? r.s11_magnitude.toFixed(4) : '-'}
                </td>
                <td style={{ fontWeight: 600, fontFamily: 'monospace', color: effColor(r.efficiency_pct || 0) }}>
                  {r.efficiency_pct != null ? r.efficiency_pct.toFixed(1) + '%' : '-'}
                </td>
                <td style={{ fontFamily: 'monospace' }}>
                  {r.coupling_loss_pct != null ? r.coupling_loss_pct.toFixed(1) + '%' : '-'}
                </td>
                <td>
                  {r.components.length > 0 ? (
                    r.components.slice(0, 3).map((c, ci) => renderComponent(c, ci))
                  ) : (
                    <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>-</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 && (
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', textAlign: 'center', padding: 20 }}>
            No results available
          </div>
        )}
      </div>
      {jointResults.warning && (
        <div style={{ marginTop: 8, fontSize: 11, color: 'var(--accent-yellow)', background: 'rgba(255,193,7,0.08)', padding: '6px 10px', borderRadius: 4 }}>
          {jointResults.warning}
        </div>
      )}
    </div>
  );
}
