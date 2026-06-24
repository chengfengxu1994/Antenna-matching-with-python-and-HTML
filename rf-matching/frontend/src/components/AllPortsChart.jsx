import React from 'react';

const COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];

/**
 * Compact S11 overview chart for joint-optimize results.
 * Shows each port's S11 dB as a horizontal bar with raw vs matched.
 */
export default function AllPortsChart({ jointResults, loadedSNP, selectedPort }) {
  if (!jointResults) return null;

  const { results_per_port = {}, ports_optimized = [] } = jointResults;
  const N = loadedSNP?.num_ports || Object.keys(results_per_port).length || 2;

  // Build data from joint results
  const portData = Array.from({ length: N }, (_, pi) => {
    const pr = results_per_port[pi];
    const isOpt = ports_optimized.includes(pi);
    const color = COLORS[pi % COLORS.length];
    const s11db = pr?.s11_db != null ? pr.s11_db : (pr?.best_s11_db != null ? pr.best_s11_db : null);
    const eff = pr?.efficiency_pct != null ? pr.efficiency_pct :
      (pr?.mismatch_efficiency != null ? pr.mismatch_efficiency * 100 : null);
    return { portIndex: pi, color, isOptimized: isOpt, s11_db: s11db, efficiency: eff, data: pr };
  });

  // SVG dimensions
  const W = 320, H = N * 36 + 30;
  const pad = { left: 48, right: 16, top: 20, bottom: 10 };
  const barW = W - pad.left - pad.right;

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: 400 }}>
      {/* Y-axis labels (port numbers) */}
      {portData.map((pd, i) => (
        <text key={`label-${i}`} x={pad.left - 6} y={pad.top + i * 36 + 18}
          textAnchor="end" fontSize={11} fontWeight={pd.isOptimized ? 700 : 400}
          fill={pd.color}>
          P{pd.portIndex + 1}
        </text>
      ))}

      {/* Bars */}
      {portData.map((pd, i) => {
        const y = pad.top + i * 36 + 4;
        const bh = 20;
        const isSel = selectedPort === pd.portIndex;

        if (!pd.isOptimized || pd.s11_db == null) {
          return (
            <g key={`bar-${i}`}>
              <rect x={pad.left} y={y} width={barW} height={bh}
                fill="var(--bg-input)" rx={3} stroke={isSel ? pd.color : 'var(--border)'}
                strokeWidth={isSel ? 2 : 1}/>
              <text x={pad.left + 4} y={y + bh / 2 + 4} fontSize={10} fill="var(--text-secondary)">
                Not matched
              </text>
            </g>
          );
        }

        // S11 dB is negative (e.g. -25 dB good, -3 dB poor)
        // Use absolute value for bar width
        const absS11 = Math.abs(pd.s11_db);
        const maxS11 = 40; // 0 to 40 dB scale
        const frac = Math.min(Math.max(absS11 / maxS11, 0), 1);
        const barFillW = Math.max(frac * barW, 4);
        const barColor = absS11 > 15 ? 'var(--accent-green)' : absS11 > 8 ? 'var(--accent-yellow)' : 'var(--accent-red)';

        return (
          <g key={`bar-${i}`}>
            {/* Background track */}
            <rect x={pad.left} y={y} width={barW} height={bh}
              fill="var(--bg-input)" rx={3} stroke={isSel ? pd.color : 'var(--border)'}
              strokeWidth={isSel ? 2 : 1}/>
            {/* Filled bar */}
            <rect x={pad.left} y={y} width={barFillW} height={bh}
              fill={barColor} rx={3} opacity={0.7}/>
            {/* -10 dB reference line */}
            <line x1={pad.left + (10 / maxS11) * barW} y1={y} x2={pad.left + (10 / maxS11) * barW} y2={y + bh}
              stroke="var(--accent-green)" strokeDasharray="3" strokeWidth={0.8} opacity={0.5}/>
            {/* Label */}
            <text x={pad.left + 6} y={y + bh / 2 + 4} fontSize={10} fontWeight={600} fill={pd.color}>
              RL {pd.s11_db.toFixed(1)} dB
              {pd.efficiency != null && ` | ${pd.efficiency.toFixed(0)}%`}
            </text>
          </g>
        );
      })}

      {/* Legend */}
      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={9} fill="var(--text-secondary)">
        Each bar shows return loss; wider = better match
      </text>
    </svg>
  );
}
