import React from 'react';

const COLORS = ['#e34f46', '#2779c8', '#159b7d', '#d4881f', '#8759bd', '#178d9c'];

function validSweep(sweep) {
  return sweep?.frequencies?.length > 1 &&
    sweep.s11_real?.length === sweep.frequencies.length &&
    sweep.s11_imag?.length === sweep.frequencies.length;
}

function tracePath(sweep, cx, cy, radius, prefix = 's11') {
  const real = sweep[`${prefix}_real`] || [];
  const imag = sweep[`${prefix}_imag`] || [];
  return real.map((value, index) => {
    const x = cx + Number(value) * radius;
    const y = cy - Number(imag[index]) * radius;
    return `${index ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(' ');
}

function marker(sweep, index, cx, cy, radius, color, key) {
  if (!validSweep(sweep)) return null;
  const x = cx + sweep.s11_real[index] * radius;
  const y = cy - sweep.s11_imag[index] * radius;
  return <circle key={key} cx={x} cy={y} r="3.5" fill={color} stroke="#fff" strokeWidth="1.5" />;
}

export default function EngineeringSmithChart({ sweepData, sweepsByPort = {} }) {
  const entries = Object.keys(sweepsByPort).length
    ? Object.entries(sweepsByPort).sort(([a], [b]) => Number(a) - Number(b))
    : (validSweep(sweepData) ? [[String(sweepData.port_index || 0), sweepData]] : []);

  if (!entries.length) {
    return <div className="smith-unavailable">当前扫频结果没有复数 S 参数，无法绘制可信 Smith 轨迹。</div>;
  }

  const width = 920;
  const height = 360;
  const cx = 405;
  const cy = height / 2;
  const radius = 154;
  const resistances = [0, 0.2, 0.5, 1, 2, 5];
  const reactances = [0.2, 0.5, 1, 2, 5];
  const firstSweep = entries[0][1];
  const startGHz = firstSweep.frequencies[0] / 1e9;
  const stopGHz = firstSweep.frequencies[firstSweep.frequencies.length - 1] / 1e9;

  return (
    <div className="engineering-smith">
      <div className="smith-heading">
        <div><span className="eyebrow">COMPLEX REFLECTION COEFFICIENT</span><strong>Smith Chart · 匹配后 Γ 轨迹</strong></div>
        <span>{startGHz.toFixed(3)}–{stopGHz.toFixed(3)} GHz</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="基于复数 S11 扫频数据的 Smith Chart">
        <defs><clipPath id="smith-unit-circle"><circle cx={cx} cy={cy} r={radius} /></clipPath></defs>
        <circle cx={cx} cy={cy} r={radius} fill="#fbfcfe" stroke="#738296" strokeWidth="1.4" />
        <g clipPath="url(#smith-unit-circle)" fill="none" stroke="#dbe2ea" strokeWidth="0.8">
          <line x1={cx - radius} y1={cy} x2={cx + radius} y2={cy} />
          {resistances.map(value => (
            <circle key={`r-${value}`} cx={cx + radius * value / (1 + value)} cy={cy} r={radius / (1 + value)} />
          ))}
          {reactances.flatMap(value => [value, -value]).map(value => (
            <circle key={`x-${value}`} cx={cx + radius} cy={cy - radius / value} r={radius / Math.abs(value)} />
          ))}
        </g>
        <g fill="#8190a3" fontSize="9" textAnchor="middle">
          <text x={cx} y={cy + 13}>1.0</text><text x={cx + radius * 2 / 3} y={cy + 13}>5.0</text>
          <text x={cx - radius} y={cy + 13}>0</text><text x={cx + radius} y={cy + 13}>∞</text>
        </g>
        {entries.map(([port, sweep]) => {
          const color = COLORS[Number(port) % COLORS.length];
          const last = sweep.frequencies.length - 1;
          return (
            <g key={port} clipPath="url(#smith-unit-circle)">
              {sweep.raw_real?.length === sweep.frequencies.length && <path d={tracePath(sweep, cx, cy, radius, 'raw')} fill="none" stroke="#aeb8c5" strokeWidth="1" strokeDasharray="4 4" opacity=".72" />}
              <path d={tracePath(sweep, cx, cy, radius)} fill="none" stroke={color} strokeWidth="2.3" strokeLinejoin="round" />
              {marker(sweep, 0, cx, cy, radius, color, `${port}-start`)}
              {marker(sweep, last, cx, cy, radius, color, `${port}-end`)}
            </g>
          );
        })}
        <g transform="translate(610 72)">
          <text x="0" y="0" fill="#334258" fontSize="12" fontWeight="700">轨迹说明</text>
          {entries.map(([port], index) => <g key={port} transform={`translate(0 ${24 + index * 24})`}><line x1="0" y1="0" x2="24" y2="0" stroke={COLORS[Number(port) % COLORS.length]} strokeWidth="3" /><text x="34" y="4" fill="#536176" fontSize="11">端口 {Number(port) + 1} · 匹配后</text></g>)}
          <g transform={`translate(0 ${34 + entries.length * 24})`}><line x1="0" y1="0" x2="24" y2="0" stroke="#aeb8c5" strokeDasharray="4 4" /><text x="34" y="4" fill="#7b8797" fontSize="10">匹配前 DUT</text></g>
          <g transform={`translate(0 ${65 + entries.length * 24})`} fill="#6d798a" fontSize="10"><circle cx="3" cy="0" r="3" fill="#2779c8" /><text x="14" y="4">起点 {startGHz.toFixed(3)} GHz</text><circle cx="3" cy="22" r="3" fill="#2779c8" /><text x="14" y="26">终点 {stopGHz.toFixed(3)} GHz</text></g>
          <text x="0" y={125 + entries.length * 24} fill="#8994a4" fontSize="9">轨迹直接使用扫频 API 返回的</text>
          <text x="0" y={139 + entries.length * 24} fill="#8994a4" fontSize="9">S11 实部与虚部，不做幅度近似。</text>
        </g>
      </svg>
    </div>
  );
}
