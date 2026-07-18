import React, { useState } from 'react';

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

function scoreColor(score) {
  if (score > 0.7) return 'var(--accent-green)';
  if (score > 0.4) return 'var(--accent-yellow)';
  return 'var(--accent-red)';
}

/** Render a component tag handling both joint-optimize and multipass formats */
function renderComponent(c, idx) {
  const isJoint = !c.connection_type && (c.type === 'inductor' || c.type === 'capacitor');
  let cls = 'series';
  let connLabel = '';
  if (isJoint) {
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

/** Sortable column header */
function SortHeader({ label, sortKey, currentSort, onSort, unit }) {
  const isActive = currentSort.key === sortKey;
  const arrow = isActive ? (currentSort.dir === 'asc' ? ' ▲' : ' ▼') : '';
  return (
    <th onClick={() => onSort(sortKey)}
        style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap' }}
        title={`Sort by ${label}`}>
      {label}{unit && <span style={{ fontSize: 9, fontWeight: 400 }}> ({unit})</span>}{arrow}
    </th>
  );
}

export default function ResultsPanel({
  jointResults, systemMetrics, onSelectPort, selectedPort,
}) {
  const [sort, setSort] = useState({ key: 'score', dir: 'desc' });

  if (!jointResults) return null;

  const { results_per_port = {}, ports_optimized = [] } = jointResults;

  if (ports_optimized.length === 0) {
    return <div style={{ fontSize: 13, color: 'var(--text-secondary)', textAlign: 'center', padding: 20 }}>
      No matching ports configured
    </div>;
  }

  function handleSort(key) {
    setSort(prev => ({
      key,
      dir: prev.key === key && prev.dir === 'desc' ? 'asc' : 'desc',
    }));
  }

  /* Build table rows from results_per_port */
  const rows = Object.entries(results_per_port).map(([piStr, pr]) => {
    const pi = parseInt(piStr);
    // Combined score from system_metrics or compute from components
    const s11DB = pr.s11_db != null ? pr.s11_db : (pr.best_s11_db != null ? pr.best_s11_db : null);
    const mismatchEff = pr.efficiency_pct != null ? pr.efficiency_pct :
      (pr.mismatch_efficiency != null ? pr.mismatch_efficiency * 100 : null);
    const totalEff = pr.total_efficiency != null ? pr.total_efficiency * 100 : mismatchEff;
    const couplingLossPct = pr.coupling_loss != null ? pr.coupling_loss * 100 : null;
    const compLossPct = pr.component_loss != null ? pr.component_loss * 100 : null;
    const radEff = pr.radiated_efficiency != null ? pr.radiated_efficiency * 100 : null;

    // Estimate score for this port individually (0-1)
    // Higher is better: use a composite
    const normScore = totalEff != null
      ? Math.min(totalEff / 100, 1.0) * 0.6
        + (s11DB != null ? Math.min(Math.abs(s11DB) / 20, 1.0) * 0.2 : 0)
        - (couplingLossPct != null ? couplingLossPct / 100 * 0.2 : 0)
      : (s11DB != null ? Math.min(Math.abs(s11DB) / 20, 1.0) : 0);

    return {
      portIndex: pi,
      isOptimized: ports_optimized.includes(pi),
      color: COLORS[pi % COLORS.length],
      s11_db: s11DB,
      s11_magnitude: pr.s11_magnitude != null ? pr.s11_magnitude : null,
      efficiency_pct: totalEff,
      mismatch_efficiency_pct: mismatchEff,
      coupling_loss_pct: couplingLossPct,
      component_loss_pct: compLossPct,
      radiated_efficiency_pct: radEff,
      total_efficiency_linear: pr.total_efficiency || null,
      score: Math.max(0, Math.min(1, normScore)),
      components: pr.components || (pr.solutions?.[0]?.components || []),
      band_mhz: pr.band_mhz || null,
      solutions_count: pr.solutions_count || null,
    };
  });

  // Sort
  const sortedRows = [...rows].sort((a, b) => {
    let va, vb;
    switch (sort.key) {
      case 'port': va = a.portIndex; vb = b.portIndex; break;
      case 'score': va = a.score; vb = b.score; break;
      case 's11': va = a.s11_db != null ? -Math.abs(a.s11_db) : -999; vb = b.s11_db != null ? -Math.abs(b.s11_db) : -999; break;
      case 'eff': va = a.efficiency_pct != null ? a.efficiency_pct : -1; vb = b.efficiency_pct != null ? b.efficiency_pct : -1; break;
      case 'coupling': va = a.coupling_loss_pct != null ? a.coupling_loss_pct : 999; vb = b.coupling_loss_pct != null ? b.coupling_loss_pct : 999; break;
      case 'compLoss': va = a.component_loss_pct != null ? a.component_loss_pct : 999; vb = b.component_loss_pct != null ? b.component_loss_pct : 999; break;
      default: va = a.score; vb = b.score;
    }
    return sort.dir === 'desc' ? (vb - va) : (va - vb);
  });

  return (
    <div>
      <h3>Per-Port Results ({ports_optimized.length} ports optimized)</h3>
      <div className="results-section" style={{ overflowX: 'auto' }}>
        <table style={{ minWidth: 700 }}>
          <thead>
            <tr>
              <SortHeader label="Port" sortKey="port" currentSort={sort} onSort={handleSort} />
              <SortHeader label="Score" sortKey="score" currentSort={sort} onSort={handleSort} />
              <SortHeader label="RL" sortKey="s11" currentSort={sort} onSort={handleSort} unit="dB" />
              <SortHeader label="η_total" sortKey="eff" currentSort={sort} onSort={handleSort} unit="%" />
              <SortHeader label="η_rad" currentSort={sort} onSort={handleSort} unit="%" sortKey="port" />
              <SortHeader label="Coupling" sortKey="coupling" currentSort={sort} onSort={handleSort} unit="%" />
              <SortHeader label="Comp.Loss" sortKey="compLoss" currentSort={sort} onSort={handleSort} unit="%" />
              <th>Components</th>
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((r) => (
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
                <td style={{ fontWeight: 600, fontFamily: 'monospace', color: scoreColor(r.score) }}>
                  {r.score.toFixed(3)}
                </td>
                <td style={{ fontWeight: 600, fontFamily: 'monospace', color: s11Color(r.s11_db || 0) }}>
                  {r.s11_db != null ? r.s11_db.toFixed(1) + ' dB' : '-'}
                </td>
                <td style={{ fontWeight: 600, fontFamily: 'monospace', color: effColor(r.efficiency_pct || 0) }}>
                  {r.efficiency_pct != null ? r.efficiency_pct.toFixed(1) + '%' : '-'}
                </td>
                <td style={{ fontFamily: 'monospace', fontSize: 11 }}>
                  {r.radiated_efficiency_pct != null ? r.radiated_efficiency_pct.toFixed(1) + '%' : '-'}
                </td>
                <td style={{ fontFamily: 'monospace', color: (r.coupling_loss_pct || 0) > 20 ? 'var(--accent-red)' : 'var(--text-secondary)' }}>
                  {r.coupling_loss_pct != null ? r.coupling_loss_pct.toFixed(1) + '%' : '-'}
                </td>
                <td style={{ fontFamily: 'monospace', color: (r.component_loss_pct || 0) > 10 ? 'var(--accent-red)' : 'var(--text-secondary)' }}>
                  {r.component_loss_pct != null ? r.component_loss_pct.toFixed(2) + '%' : '-'}
                </td>
                <td style={{ maxWidth: 160 }}>
                  {r.components.length > 0 ? (
                    r.components.slice(0, 4).map((c, ci) => renderComponent(c, ci))
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

      {/* System metrics summary */}
      {systemMetrics && (
        <div style={{
          marginTop: 8, display: 'flex', gap: 10, flexWrap: 'wrap',
          fontSize: 11, padding: '6px 10px', background: 'rgba(0,0,0,0.02)', borderRadius: 4,
        }}>
          <span>System Score: <strong style={{ color: scoreColor(systemMetrics.balanced_score || 0) }}>
            {((systemMetrics.balanced_score || 0) * 100).toFixed(1)}%</strong></span>
          <span>Min η_total: <strong>{(systemMetrics.min_total_efficiency * 100 || 0).toFixed(1)}%</strong></span>
          <span>Avg η_total: <strong>{(systemMetrics.avg_total_efficiency * 100 || 0).toFixed(1)}%</strong></span>
          <span>Max Coupling: <strong>{(systemMetrics.max_coupling_loss * 100 || 0).toFixed(1)}%</strong></span>
          <span>Comp. Loss: <strong>{(systemMetrics.component_loss_total * 100 || 0).toFixed(2)}%</strong></span>
        </div>
      )}

      {jointResults.warning && (
        <div style={{ marginTop: 8, fontSize: 11, color: 'var(--accent-yellow)', background: 'rgba(255,193,7,0.08)', padding: '6px 10px', borderRadius: 4 }}>
          {jointResults.warning}
        </div>
      )}
    </div>
  );
}
