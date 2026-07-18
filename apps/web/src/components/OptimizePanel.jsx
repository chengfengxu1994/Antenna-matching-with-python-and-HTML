import React from 'react';

export default function OptimizePanel({ params, setParams, onOptimize, optimizing, numBandPoints, setNumBandPoints, hasBands }) {
  return (
    <div className="card">
      <h3>⚙️ 优化参数 (Optimization Settings)</h3>
      
      <div className="form-group">
        <label>目标频率 (Target Frequency)</label>
        <div className="flex-row" style={{gap: 6}}>
          <input
            type="number"
            value={params.targetFrequencyHz}
            onChange={e => setParams({...params, targetFrequencyHz: parseFloat(e.target.value) || 64e6})}
            style={{flex: 2}}
          />
          <span style={{fontSize: 12, color: 'var(--text-secondary)', flex: 1}}>
            = {(params.targetFrequencyHz / 1e6).toFixed(2)} MHz
          </span>
        </div>
      </div>

      <div className="form-group">
        <label>匹配元件数 (Max Components)</label>
        <div style={{display: 'flex', gap: 4}}>
          {[1,2,3,4].map(n => (
            <button
              key={n}
              className={`btn btn-sm ${params.maxComponents === n ? 'btn-primary' : ''}`}
              onClick={() => setParams({...params, maxComponents: n})}
              style={{flex: 1}}
            >
              {n}
            </button>
          ))}
        </div>
      </div>

      <div className="form-group">
        <label>输入端口 (Input Port)</label>
        <select
          value={params.inputPort}
          onChange={e => setParams({...params, inputPort: parseInt(e.target.value)})}
        >
          {[0,1,2,3,4,5].map(n => (
            <option key={n} value={n}>Port {n+1}</option>
          ))}
        </select>
      </div>

      {hasBands && (
        <div className="form-group">
          <label>频段评估点数 (Points per Band)</label>
          <select
            value={numBandPoints}
            onChange={e => setNumBandPoints(parseInt(e.target.value))}
          >
            {[3,5,7,10,15,20].map(n => (
              <option key={n} value={n}>{n} 点</option>
            ))}
          </select>
        </div>
      )}

      <div className="form-group">
        <label>搜索宽度 (Beam Width)</label>
        <input
          type="number"
          value={params.beamWidth}
          onChange={e => setParams({...params, beamWidth: parseInt(e.target.value) || 10})}
        />
      </div>

      <div className="form-group">
        <label>超时时间 (Timeout seconds)</label>
        <input
          type="number"
          value={params.timeoutSeconds}
          onChange={e => setParams({...params, timeoutSeconds: parseFloat(e.target.value) || 60})}
        />
      </div>

      <button
        className="btn btn-success"
        onClick={onOptimize}
        disabled={optimizing}
        style={{width:'100%', marginTop: 12, padding: '12px 16px', fontSize: 15, fontWeight: 600}}
      >
        {optimizing ? '⏳ 优化中...' : '🚀 开始匹配优化'}
      </button>
    </div>
  );
}
