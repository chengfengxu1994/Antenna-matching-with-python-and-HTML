import React from 'react';

const STATE_OPTIONS = [
  { value: 'component', label: '匹配网络', color: 'var(--accent)' },
  { value: 'open', label: '开路', color: 'var(--accent-yellow)' },
  { value: 'short', label: '短路', color: 'var(--accent-red)' },
  { value: 'load', label: '50Ω负载', color: 'var(--accent-green)' },
];

export default function PortConfig({ numPorts, portStates, setPortStates, inputPort }) {
  function toggleState(port) {
    const states = { ...portStates };
    const current = states[port] || 'load';
    const idx = STATE_OPTIONS.findIndex(o => o.value === current);
    const nextIdx = (idx + 1) % STATE_OPTIONS.length;
    states[port] = STATE_OPTIONS[nextIdx].value;
    setPortStates(states);
  }

  function getStateInfo(state) {
    return STATE_OPTIONS.find(o => o.value === state) || STATE_OPTIONS[3];
  }

  return (
    <div className="card">
      <h3>🔌 端口配置 ({numPorts} 端口)</h3>
      <div style={{marginBottom: 8, fontSize: 12, color: 'var(--text-secondary)'}}>
        输入端口: Port {inputPort + 1} (始终接匹配网络)
      </div>
      <div className="port-state-grid">
        {Array.from({length: numPorts}, (_, i) => {
          const state = portStates[i] || (i === inputPort ? 'component' : 'load');
          const info = getStateInfo(state);
          return (
            <div
              key={i}
              className="port-state-item"
              onClick={() => i !== inputPort && toggleState(i)}
              style={{
                cursor: i === inputPort ? 'default' : 'pointer',
                borderLeft: `3px solid ${info.color}`,
              }}
            >
              <span style={{fontWeight:600}}>P{i+1}</span>
              <span style={{color: info.color}}>{info.label}</span>
            </div>
          );
        })}
      </div>
      <div style={{fontSize:11, color:'var(--text-secondary)', marginTop:8}}>
        点击切换: 匹配网络 → 开路 → 短路 → 50Ω负载
      </div>
    </div>
  );
}
