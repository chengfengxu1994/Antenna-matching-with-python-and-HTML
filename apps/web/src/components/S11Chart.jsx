import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, ReferenceArea, Legend } from 'recharts';

export default function S11Chart({ sweepData, targetFreqHz, bands, bandsMhz }) {
  if (!sweepData) {
    return (
      <div className="card">
        <h3>S11 Frequency Sweep</h3>
        <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
          Select a result to view S11 sweep
        </div>
      </div>
    );
  }

  const data = sweepData.frequencies.map((f, i) => ({
    freq: f / 1e6,
    // Product APIs expose positive return loss. Plot conventional signed S11 dB.
    s11_db: -Math.abs(sweepData.s11_db?.[i] ?? 0),
    raw_db: -Math.abs(sweepData.raw_db?.[i] ?? 0),
  }));

  const targetMHz = targetFreqHz ? targetFreqHz / 1e6 : null;

  return (
    <div className="card">
      <h3>S11 Frequency Sweep</h3>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8 }}>
        Solid: Matched | Dashed: Raw DUT
      </div>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="freq"
              label={{ value: 'Frequency (MHz)', position: 'bottom', fill: 'var(--text-secondary)' }}
              stroke="var(--text-secondary)"
              fontSize={11}
            />
            <YAxis
              label={{ value: 'S11 (dB)', angle: -90, position: 'left', fill: 'var(--text-secondary)' }}
              stroke="var(--text-secondary)"
              fontSize={11}
            />
            <Tooltip
              contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 6 }}
              formatter={(value, name) => [value?.toFixed(2) + ' dB', name === 's11_db' ? 'Matched' : 'Raw DUT']}
              labelFormatter={(v) => v.toFixed(2) + ' MHz'}
            />
            <Legend
              formatter={(value) => value === 's11_db' ? 'Matched S11' : 'Raw DUT S11'}
              wrapperStyle={{ fontSize: 12 }}
            />
            {targetMHz && (
              <ReferenceLine
                x={targetMHz}
                stroke="var(--accent)"
                strokeDasharray="5 5"
                label={{ value: `${targetMHz.toFixed(1)} MHz`, fill: 'var(--accent)', fontSize: 11 }}
              />
            )}
            <ReferenceLine
              y={-10}
              stroke="var(--accent-green)"
              strokeDasharray="3 3"
              strokeOpacity={0.5}
            />
            {(bandsMhz || bands || []).map((band, i) => (
              <ReferenceArea key={i}
                x1={band[0]} x2={band[1]}
                fill="var(--accent)" fillOpacity={0.06}
                stroke="var(--accent)" strokeOpacity={0.2}
              />
            ))}
            <Line type="monotone" dataKey="raw_db" stroke="var(--text-secondary)" dot={false} strokeWidth={1}
              strokeDasharray="4 4" name="raw_db" />
            <Line type="monotone" dataKey="s11_db" stroke="var(--accent)" dot={false} strokeWidth={2} name="s11_db" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
