import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

export default function TopologySelector({ selected, setSelected, maxComponents }) {
  const [topologies, setTopologies] = useState([]);

  useEffect(() => {
    api.listTopologies(maxComponents).then(res => {
      setTopologies(res.topologies || []);
    }).catch(() => {});
  }, [maxComponents]);

  function toggle(name) {
    if (selected.includes(name)) {
      setSelected(selected.filter(n => n !== name));
    } else {
      setSelected([...selected, name]);
    }
  }

  return (
    <div className="card">
      <h3>Matching Topologies ({topologies.length})</h3>
      <div style={{fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8}}>
        {selected.length === 0 ? 'All topologies will be tried' : `${selected.length} selected`}
      </div>
      <div style={{maxHeight: 220, overflowY: 'auto'}}>
        {topologies.map(t => (
          <div
            key={t.name}
            className={`topology-item ${selected.includes(t.name) ? 'selected' : ''}`}
            onClick={() => toggle(t.name)}
          >
            <div style={{fontWeight:500, fontSize:12}}>{t.name}</div>
            <div style={{color:'var(--text-secondary)', fontSize:11}}>
              {t.num_components} components | {t.elements.map(e => e.connection_type[0].toUpperCase()).join('-')}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
