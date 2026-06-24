import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

export default function ComponentSeriesSelector({ selectedSeries, setSelectedSeries }) {
  const [series, setSeries] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadSeries();
  }, []);

  async function loadSeries() {
    setLoading(true);
    try {
      const res = await api.getComponentSeries();
      setSeries(res);
      // Auto-select defaults if nothing selected
      if (!selectedSeries || selectedSeries.length === 0) {
        const defaults = [];
        // Default inductors
        for (const s of Object.keys(res.inductor_series || {})) {
          if (s.includes('LQP03HQ') || s.includes('LQP02HQ')) defaults.push(s);
        }
        // Default capacitors
        for (const s of Object.keys(res.capacitor_series || {})) {
          if (s.includes('GJM')) defaults.push(s);
        }
        if (defaults.length > 0) setSelectedSeries(defaults);
      }
    } catch (e) {
      console.error('Failed to load component series:', e);
    }
    setLoading(false);
  }

  function toggleSeries(name) {
    if (selectedSeries.includes(name)) {
      setSelectedSeries(selectedSeries.filter(s => s !== name));
    } else {
      setSelectedSeries([...selectedSeries, name]);
    }
  }

  function selectAll(type) {
    if (!series) return;
    const src = type === 'L' ? series.inductor_series : series.capacitor_series;
    const names = Object.keys(src || {});
    const newSel = [...new Set([...selectedSeries, ...names])];
    setSelectedSeries(newSel);
  }

  function clearAll(type) {
    if (!series) return;
    const src = type === 'L' ? series.inductor_series : series.capacitor_series;
    const names = new Set(Object.keys(src || {}));
    setSelectedSeries(selectedSeries.filter(s => !names.has(s)));
  }

  if (loading) return <div style={{fontSize: 12, color: 'var(--text-secondary)'}}>Loading series...</div>;
  if (!series) return null;

  const totalSelected = selectedSeries.length;
  const totalAvailable = Object.keys({...series.inductor_series, ...series.capacitor_series}).length;

  return (
    <div className="card">
      <h3>Component Library ({totalSelected}/{totalAvailable} series)</h3>

      {/* Inductors */}
      <div style={{marginBottom: 12}}>
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4}}>
          <label style={{fontSize: 13, fontWeight: 600, color: 'var(--text)'}}>Inductors (L)</label>
          <div style={{display: 'flex', gap: 4}}>
            <button className="btn btn-sm" onClick={() => selectAll('L')}>All</button>
            <button className="btn btn-sm" onClick={() => clearAll('L')}>None</button>
          </div>
        </div>
        <div className="checkbox-list">
          {Object.entries(series.inductor_series || {}).map(([name, count]) => (
            <label key={name} className="checkbox-item">
              <input
                type="checkbox"
                checked={selectedSeries.includes(name)}
                onChange={() => toggleSeries(name)}
              />
              <span>{name}</span>
              <span style={{color: 'var(--text-secondary)', fontSize: 11}}>({count})</span>
            </label>
          ))}
        </div>
      </div>

      {/* Capacitors */}
      <div>
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4}}>
          <label style={{fontSize: 13, fontWeight: 600, color: 'var(--text)'}}>Capacitors (C)</label>
          <div style={{display: 'flex', gap: 4}}>
            <button className="btn btn-sm" onClick={() => selectAll('C')}>All</button>
            <button className="btn btn-sm" onClick={() => clearAll('C')}>None</button>
          </div>
        </div>
        <div className="checkbox-list">
          {Object.entries(series.capacitor_series || {}).map(([name, count]) => (
            <label key={name} className="checkbox-item">
              <input
                type="checkbox"
                checked={selectedSeries.includes(name)}
                onChange={() => toggleSeries(name)}
              />
              <span>{name}</span>
              <span style={{color: 'var(--text-secondary)', fontSize: 11}}>({count})</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );
}
