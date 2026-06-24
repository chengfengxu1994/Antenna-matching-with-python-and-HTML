import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend, Area, AreaChart,
  ScatterChart, Scatter, ZAxis, Cell
} from 'recharts';

const API_BASE = 'http://localhost:8000';

export default function BandwidthPotentialPanel({ loadedSNP }) {
  const [loading, setLoading] = useState(false);
  const [bpData, setBpData] = useState(null);
  const [multiData, setMultiData] = useState(null);
  const [threshold, setThreshold] = useState(-6.0);
  const [viewMode, setViewMode] = useState('single'); // 'single' | 'multi' | 'qfactor'
  const [error, setError] = useState(null);

  const runAnalysis = useCallback(async (mode = 'single') => {
    if (!loadedSNP) return;
    setLoading(true);
    setError(null);

    try {
      if (mode === 'multi') {
        const res = await fetch(`${API_BASE}/api/bandwidth-potential`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            s11_threshold_db: threshold,
            thresholds_db: [-3.0, -6.0, -10.0],
            max_points: 200,
          }),
        });
        const data = await res.json();
        if (data.status === 'ok') {
          setMultiData(data.results);
          setViewMode('multi');
        }
      } else {
        const res = await fetch(`${API_BASE}/api/bandwidth-potential`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            s11_threshold_db: threshold,
            max_points: 300,
          }),
        });
        const data = await res.json();
        if (data.status === 'ok') {
          setBpData(data.result);
          setViewMode('single');
        }
      }
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  }, [loadedSNP, threshold]);

  if (!loadedSNP) {
    return (
      <div className="card">
        <h3>📡 Bandwidth Potential 带宽潜力评估</h3>
        <div style={{height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)'}}>
          请先加载 SNP 文件
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12}}>
        <h3 style={{margin: 0}}>📡 Bandwidth Potential 带宽潜力评估</h3>
        <div style={{display: 'flex', gap: 8, alignItems: 'center'}}>
          <label style={{fontSize: 12, color: 'var(--text-secondary)'}}>
            S11 阈值:
            <select
              value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value))}
              style={{marginLeft: 4, padding: '2px 6px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-card)'}}
            >
              <option value={-3}>-3 dB</option>
              <option value={-6}>-6 dB</option>
              <option value={-10}>-10 dB</option>
              <option value={-15}>-15 dB</option>
              <option value={-20}>-20 dB</option>
            </select>
          </label>
          <button
            className="btn btn-primary"
            onClick={() => runAnalysis('single')}
            disabled={loading}
            style={{fontSize: 12, padding: '4px 12px'}}
          >
            {loading ? '计算中...' : '分析'}
          </button>
          <button
            className="btn"
            onClick={() => runAnalysis('multi')}
            disabled={loading}
            style={{fontSize: 12, padding: '4px 12px'}}
          >
            带宽图 (多阈值)
          </button>
          {bpData && (
            <button
              className="btn"
              onClick={() => setViewMode(viewMode === 'qfactor' ? 'single' : 'qfactor')}
              style={{fontSize: 12, padding: '4px 12px'}}
            >
              Q 因子
            </button>
          )}
        </div>
      </div>

      <div style={{fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8}}>
        对每个中心频率，综合最优 2 元件 L 型匹配网络，测量可获得的阻抗带宽。
        揭示天线的"隐藏带宽"潜力。
      </div>

      {error && <div style={{color: 'var(--error, #ef4444)', marginBottom: 8}}>Error: {error}</div>}

      {/* Summary stats */}
      {bpData && viewMode === 'single' && (
        <div style={{display: 'flex', gap: 16, marginBottom: 12, padding: '8px 12px', background: 'var(--bg-secondary, #f8f9fa)', borderRadius: 6, fontSize: 13}}>
          <div>
            <span style={{color: 'var(--text-secondary)'}}>最大潜力: </span>
            <strong style={{color: 'var(--accent)'}}>{bpData.max_potential_pct?.toFixed(1)}%</strong>
            <span style={{color: 'var(--text-secondary)', marginLeft: 4}}>
              @ {(bpData.max_potential_freq_hz / 1e6)?.toFixed(0)} MHz
            </span>
          </div>
          <div>
            <span style={{color: 'var(--text-secondary)'}}>平均潜力: </span>
            <strong>{bpData.avg_potential_pct?.toFixed(1)}%</strong>
          </div>
          <div>
            <span style={{color: 'var(--text-secondary)'}}>阈值: </span>
            <strong>{bpData.s11_threshold_db} dB</strong>
          </div>
        </div>
      )}

      {/* Single threshold bandwidth potential chart */}
      {bpData && viewMode === 'single' && (
        <SingleThresholdChart data={bpData} />
      )}

      {/* Q factor chart */}
      {bpData && viewMode === 'qfactor' && (
        <QFactorChart data={bpData} />
      )}

      {/* Multi-threshold bandwidth map */}
      {multiData && viewMode === 'multi' && (
        <MultiThresholdChart data={multiData} />
      )}

      {/* Loading overlay */}
      {loading && (
        <div style={{height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)'}}>
          <div style={{textAlign: 'center'}}>
            <div style={{fontSize: 24, marginBottom: 8}}>⏳</div>
            <div>正在计算带宽潜力...</div>
            <div style={{fontSize: 11, marginTop: 4, opacity: 0.6}}>对每个频率点综合最优匹配网络</div>
          </div>
        </div>
      )}
    </div>
  );
}

function SingleThresholdChart({ data }) {
  const chartData = (data.points || [])
    .filter(p => p.relative_bandwidth_pct > 0)
    .map(p => ({
      freq: p.center_freq_hz / 1e6,
      bw_pct: p.relative_bandwidth_pct,
      bw_mhz: p.absolute_bandwidth_hz / 1e6,
      f_low: p.f_low_hz / 1e6,
      f_high: p.f_high_hz / 1e6,
      q_factor: p.q_factor,
      s11_db: p.s11_at_center_db,
    }));

  return (
    <div className="chart-container" style={{height: 350}}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{top: 5, right: 20, left: 10, bottom: 20}}>
          <defs>
            <linearGradient id="bwGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--accent, #6366f1)" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="var(--accent, #6366f1)" stopOpacity={0.02}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="freq"
            label={{ value: '中心频率 (MHz)', position: 'bottom', fill: 'var(--text-secondary)', offset: 0 }}
            stroke="var(--text-secondary)"
            fontSize={11}
            tickFormatter={v => v.toFixed(0)}
          />
          <YAxis
            label={{ value: '相对带宽 (%)', angle: -90, position: 'left', fill: 'var(--text-secondary)' }}
            stroke="var(--text-secondary)"
            fontSize={11}
          />
          <Tooltip
            contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 12 }}
            formatter={(value, name) => {
              if (name === 'bw_pct') return [value?.toFixed(1) + '%', '相对带宽'];
              if (name === 'bw_mhz') return [value?.toFixed(1) + ' MHz', '绝对带宽'];
              return [value, name];
            }}
            labelFormatter={v => `中心: ${v?.toFixed(1)} MHz`}
          />
          <Legend
            formatter={(value) => {
              if (value === 'bw_pct') return '相对带宽 (%)';
              if (value === 'bw_mhz') return '绝对带宽 (MHz)';
              return value;
            }}
            wrapperStyle={{fontSize: 11}}
          />
          <Area
            type="monotone"
            dataKey="bw_pct"
            stroke="var(--accent, #6366f1)"
            fill="url(#bwGradient)"
            strokeWidth={2}
            dot={false}
            name="bw_pct"
          />
          <Line
            type="monotone"
            dataKey="bw_mhz"
            stroke="var(--accent-green, #10b981)"
            dot={false}
            strokeWidth={1.5}
            strokeDasharray="4 4"
            name="bw_mhz"
            yAxisId={0}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function QFactorChart({ data }) {
  const chartData = (data.points || [])
    .filter(p => p.q_factor != null && p.q_factor < 1000)
    .map(p => ({
      freq: p.center_freq_hz / 1e6,
      q: p.q_factor,
    }));

  return (
    <div className="chart-container" style={{height: 350}}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{top: 5, right: 20, left: 10, bottom: 20}}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="freq"
            label={{ value: '频率 (MHz)', position: 'bottom', fill: 'var(--text-secondary)' }}
            stroke="var(--text-secondary)"
            fontSize={11}
            tickFormatter={v => v.toFixed(0)}
          />
          <YAxis
            label={{ value: 'Q 因子', angle: -90, position: 'left', fill: 'var(--text-secondary)' }}
            stroke="var(--text-secondary)"
            fontSize={11}
          />
          <Tooltip
            contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 12 }}
            formatter={(value) => [value?.toFixed(1), 'Q 因子']}
            labelFormatter={v => `${v?.toFixed(1)} MHz`}
          />
          <Line
            type="monotone"
            dataKey="q"
            stroke="var(--accent-orange, #f59e0b)"
            dot={false}
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function MultiThresholdChart({ data }) {
  // Merge data from multiple thresholds into one chart
  const allFreqs = new Set();
  const dataByThreshold = {};

  for (const [threshold, result] of Object.entries(data)) {
    dataByThreshold[threshold] = {};
    for (const p of (result.points || [])) {
      const freq = p.center_freq_hz / 1e6;
      allFreqs.add(freq);
      dataByThreshold[threshold][freq] = p.relative_bandwidth_pct;
    }
  }

  const sortedFreqs = [...allFreqs].sort((a, b) => a - b);
  const chartData = sortedFreqs.map(freq => {
    const row = { freq };
    for (const threshold of Object.keys(dataByThreshold)) {
      row[`t${threshold}`] = dataByThreshold[threshold][freq] || 0;
    }
    return row;
  });

  const colors = {
    '-3': '#10b981',    // green - relaxed
    '-6': '#6366f1',    // purple - standard
    '-10': '#ef4444',   // red - strict
  };

  return (
    <div>
      <div style={{fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, textAlign: 'center'}}>
        不同 S11 阈值下的带宽潜力图 (匹配越好要求越严格 → 带宽越窄)
      </div>
      <div className="chart-container" style={{height: 350}}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{top: 5, right: 20, left: 10, bottom: 20}}>
            <defs>
              {Object.keys(data).map(threshold => (
                <linearGradient key={threshold} id={`grad-${threshold}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={colors[threshold] || '#6366f1'} stopOpacity={0.2}/>
                  <stop offset="95%" stopColor={colors[threshold] || '#6366f1'} stopOpacity={0.02}/>
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="freq"
              label={{ value: '中心频率 (MHz)', position: 'bottom', fill: 'var(--text-secondary)' }}
              stroke="var(--text-secondary)"
              fontSize={11}
              tickFormatter={v => v.toFixed(0)}
            />
            <YAxis
              label={{ value: '相对带宽 (%)', angle: -90, position: 'left', fill: 'var(--text-secondary)' }}
              stroke="var(--text-secondary)"
              fontSize={11}
            />
            <Tooltip
              contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 12 }}
              formatter={(value, name) => [value?.toFixed(1) + '%', `S11 ${name.replace('t', '')} dB`]}
              labelFormatter={v => `${v?.toFixed(1)} MHz`}
            />
            <Legend
              formatter={(value) => `S11 = ${value.replace('t', '')} dB`}
              wrapperStyle={{fontSize: 11}}
            />
            {Object.keys(data).map(threshold => (
              <Area
                key={threshold}
                type="monotone"
                dataKey={`t${threshold}`}
                stroke={colors[threshold] || '#6366f1'}
                fill={`url(#grad-${threshold})`}
                strokeWidth={2}
                dot={false}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
