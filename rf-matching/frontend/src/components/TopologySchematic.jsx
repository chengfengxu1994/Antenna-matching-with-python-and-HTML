import React from 'react';

function formatValue(component) {
  if (!component) return '';
  if (component.value) return component.value;
  if (component.nominal_value == null) return '';
  return `${component.nominal_value}${component.nominal_unit || ''}`;
}

function componentLabel(component, index) {
  const type = component?.comp_type || component?.type || 'capacitor';
  const prefix = type === 'inductor' ? 'L' : 'C';
  return `${prefix}${index + 1}: ${formatValue(component)}`;
}

function editableNode(component, index, x, y, onChange) {
  const type = component?.comp_type || component?.type || 'capacitor';
  const parsedValue = parseFloat(String(component?.value || '').replace(/[^\d.]/g, ''));
  const value = component?.nominal_value ?? (Number.isFinite(parsedValue) ? parsedValue : 1);
  return (
    <foreignObject x={x - 58} y={y - 38} width="116" height="76">
      <div className="schematic-editor-node">
        <div className="schematic-editor-row">
          <select value={type} onChange={e => onChange(index, { comp_type: e.target.value, type: e.target.value })}>
            <option value="capacitor">C</option>
            <option value="inductor">L</option>
          </select>
          <select
            value={component?.connection_type || 'series'}
            onChange={e => onChange(index, { connection_type: e.target.value })}
          >
            <option value="series">Series</option>
            <option value="shunt">Shunt</option>
          </select>
        </div>
        <div className="schematic-editor-row">
          <input
            type="number"
            value={Number.isFinite(value) ? value : 1}
            min="0"
            step={type === 'inductor' ? '0.1' : '0.1'}
            onChange={e => onChange(index, { nominal_value: parseFloat(e.target.value) || 0 })}
          />
          <span>{type === 'inductor' ? 'nH' : 'pF'}</span>
        </div>
      </div>
    </foreignObject>
  );
}

function SeriesSymbol({ x, y, type }) {
  if (type === 'inductor') {
    return (
      <path
        d={`M ${x - 30} ${y} c 6 -12 12 -12 18 0 c 6 12 12 12 18 0 c 6 -12 12 -12 18 0`}
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
      />
    );
  }
  return (
    <g stroke="currentColor" strokeWidth="2">
      <line x1={x - 6} y1={y - 17} x2={x - 6} y2={y + 17} />
      <line x1={x + 6} y1={y - 17} x2={x + 6} y2={y + 17} />
    </g>
  );
}

function ShuntSymbol({ x, y, type }) {
  const gy = y + 82;
  return (
    <g stroke="currentColor" strokeWidth="2" fill="none">
      <line x1={x} y1={y} x2={x} y2={y + 24} />
      {type === 'inductor' ? (
        <path d={`M ${x} ${y + 24} c -12 6 -12 12 0 18 c 12 6 12 12 0 18`} />
      ) : (
        <>
          <line x1={x - 17} y1={y + 34} x2={x + 17} y2={y + 34} />
          <line x1={x - 17} y1={y + 46} x2={x + 17} y2={y + 46} />
        </>
      )}
      <line x1={x} y1={y + 58} x2={x} y2={gy - 12} />
      <line x1={x - 18} y1={gy - 12} x2={x + 18} y2={gy - 12} />
      <line x1={x - 12} y1={gy - 6} x2={x + 12} y2={gy - 6} />
      <line x1={x - 6} y1={gy} x2={x + 6} y2={gy} />
    </g>
  );
}

export default function TopologySchematic({
  solution,
  portIndex,
  editable = false,
  onComponentChange,
}) {
  const perPort = solution?.per_port || {};
  const entries = Object.entries(perPort)
    .filter(([pi]) => portIndex == null || Number(pi) === Number(portIndex))
    .sort(([a], [b]) => Number(a) - Number(b));

  if (!solution || entries.length === 0) {
    return <div className="empty-state compact">No topology available.</div>;
  }

  const width = 980;
  const height = Math.max(300, entries.length * 135 + 110);
  const dutX = 430;
  const dutY = 52;
  const dutW = 170;
  const dutH = height - 104;

  return (
    <div className="schematic-wrap">
      <svg className="schematic-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Matching topology">
        <rect x={dutX} y={dutY} width={dutW} height={dutH} rx="9" fill="#9beacb" stroke="#111" strokeWidth="1.5" />
        {entries.map(([piStr, pm], row) => {
          const pi = Number(piStr);
          const leftSide = pi % 2 === 0;
          const y = dutY + 70 + row * 128;
          const portX = leftSide ? 74 : width - 74;
          const dutPinX = leftSide ? dutX : dutX + dutW;
          const lineStart = leftSide ? portX + 34 : portX - 34;
          const comps = (pm.components || []).slice().sort((a, b) => (a.position || 0) - (b.position || 0));
          const series = comps.filter(c => (c.connection_type || 'series') === 'series');
          const shunts = comps.filter(c => c.connection_type === 'shunt');
          const baseXs = leftSide ? [185, 325] : [795, 665];

          return (
            <g key={pi}>
              <text x={portX - (leftSide ? 4 : -4)} y={y - 31} textAnchor={leftSide ? 'middle' : 'middle'} fontSize="18" fill="#111">
                Port {pi + 1}
              </text>
              <polygon
                points={leftSide ? `${portX},${y - 20} ${portX},${y + 20} ${portX + 38},${y}` : `${portX},${y - 20} ${portX},${y + 20} ${portX - 38},${y}`}
                fill="#f2f000"
                stroke="#111"
              />
              <line x1={lineStart} y1={y} x2={dutPinX} y2={y} stroke="#111" strokeWidth="1.5" />
              <text x={dutPinX + (leftSide ? 12 : -12)} y={y + 8} textAnchor={leftSide ? 'start' : 'end'} fontSize="18" fill="#045">
                [{pi + 1}]
              </text>

              {series.map((c, ci) => {
                const x = baseXs[ci % baseXs.length];
                const type = c.comp_type || c.type;
                return (
                  <g key={`s-${ci}`} className="schematic-component">
                    <SeriesSymbol x={x} y={y} type={type} />
                    <text x={x} y={y - 38} textAnchor="middle" fontSize="18">{componentLabel(c, ci)}</text>
                    {c.part_number || c.part ? <text x={x} y={y - 17} textAnchor="middle" fontSize="16">({c.part_number || c.part})</text> : null}
                  </g>
                );
              })}

              {shunts.map((c, ci) => {
                const x = baseXs[(series.length + ci) % baseXs.length];
                const type = c.comp_type || c.type;
                return (
                  <g key={`p-${ci}`} className="schematic-component">
                    <ShuntSymbol x={x} y={y} type={type} />
                    <text x={x + (leftSide ? 26 : -26)} y={y + 46} textAnchor={leftSide ? 'start' : 'end'} fontSize="18">{componentLabel(c, series.length + ci)}</text>
                    {c.part_number || c.part ? <text x={x + (leftSide ? 26 : -26)} y={y + 68} textAnchor={leftSide ? 'start' : 'end'} fontSize="16">({c.part_number || c.part})</text> : null}
                  </g>
                );
              })}

              {editable && comps.map((c, ci) => {
                const x = baseXs[ci % baseXs.length];
                const yEdit = (c.connection_type || 'series') === 'series' ? y - 2 : y + 58;
                return (
                  <g key={`edit-${ci}`}>
                    {editableNode(c, ci, x, yEdit, (index, patch) => onComponentChange?.(pi, index, patch))}
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
