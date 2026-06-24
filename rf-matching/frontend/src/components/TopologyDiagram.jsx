import React from 'react';

/**
 * Renders a topology diagram as SVG.
 * Shows the DUT (SNP) as a rounded rectangle with ports,
 * each port's termination state, and matching networks.
 */
export default function TopologyDiagram({ numPorts, portConfigs, solution, width = 600, height = 300 }) {
  const W = width;
  const H = height;
  const pad = 40;
  const dutW = 160;
  const dutH = Math.max(120, numPorts * 50 + 40);
  const dutX = (W - dutW) / 2;
  const dutY = (H - dutH) / 2;

  const portSpacing = dutH / (numPorts + 1);

  const stateColors = {
    'load': '#198754',
    'short': '#dc3545',
    'open': '#ffc107',
    'component': '#0d6efd',
    'matching': '#0d6efd',
  };

  const stateLabels = {
    'load': '50 Ohm',
    'short': 'Short',
    'open': 'Open',
    'component': 'Match',
    'matching': 'Match',
  };

  return (
    <div className="topology-diagram">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{fontFamily: 'monospace'}}>
        {/* DUT body */}
        <rect
          x={dutX} y={dutY}
          width={dutW} height={dutH}
          rx={12} ry={12}
          fill="#f0f4ff" stroke="#0d6efd" strokeWidth={2}
        />
        <text
          x={dutX + dutW / 2} y={dutY + 20}
          textAnchor="middle" fontSize={13} fontWeight="bold" fill="#0d6efd"
        >
          DUT
        </text>
        <text
          x={dutX + dutW / 2} y={dutY + 35}
          textAnchor="middle" fontSize={10} fill="#6c757d"
        >
          {numPorts}-port
        </text>

        {/* Ports */}
        {Array.from({ length: numPorts }, (_, i) => {
          const cfg = portConfigs?.[i] || {};
          const state = cfg.state || 'load';
          const useMatching = cfg.use_matching || false;
          const portY = dutY + portSpacing * (i + 1);

          // Left side ports
          const pinX = dutX;
          const labelX = pinX - 20;
          const extX = pinX - 80;

          const color = useMatching ? stateColors['matching'] : stateColors[state] || '#6c757d';
          const label = useMatching ? 'Match' : (stateLabels[state] || state);

          return (
            <g key={i}>
              {/* Pin line */}
              <line x1={extX + 60} y1={portY} x2={pinX} y2={portY}
                stroke={color} strokeWidth={2} />
              {/* Pin dot */}
              <circle cx={pinX} cy={portY} r={4} fill={color} />
              {/* Port label */}
              <text x={extX + 30} y={portY - 8}
                textAnchor="middle" fontSize={11} fontWeight="bold" fill={color}>
                P{i + 1}
              </text>
              {/* State label */}
              <text x={extX + 30} y={portY + 12}
                textAnchor="middle" fontSize={9} fill={color}>
                {label}
              </text>
              {/* Matching network indicator */}
              {useMatching && (
                <g>
                  <rect x={extX - 10} y={portY - 14} width={36} height={28}
                    rx={4} fill="rgba(13,110,253,0.1)" stroke="#0d6efd" strokeWidth={1} strokeDasharray="3,2" />
                  <text x={extX + 8} y={portY + 3}
                    textAnchor="middle" fontSize={8} fill="#0d6efd">
                    L/C
                  </text>
                </g>
              )}
              {/* Frequency annotation */}
              {cfg.target_frequency_hz && (
                <text x={extX + 30} y={portY + 24}
                  textAnchor="middle" fontSize={8} fill="#adb5bd">
                  {(cfg.target_frequency_hz / 1e6).toFixed(0)}MHz
                </text>
              )}
            </g>
          );
        })}

        {/* Solution annotation (if available) */}
        {solution && (
          <g>
            <text x={dutX + dutW / 2} y={dutY + dutH + 20}
              textAnchor="middle" fontSize={11} fill="#198754" fontWeight="bold">
              Best: |S11|={solution.s11_magnitude?.toFixed(3) || 'N/A'}
            </text>
            {solution.topology && (
              <text x={dutX + dutW / 2} y={dutY + dutH + 35}
                textAnchor="middle" fontSize={10} fill="#6c757d">
                {solution.topology}
              </text>
            )}
          </g>
        )}
      </svg>
    </div>
  );
}
