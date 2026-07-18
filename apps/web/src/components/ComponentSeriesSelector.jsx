import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { reconcileSeriesSelection } from '../utils/dataSource';

export default function ComponentSeriesSelector({
  selectedSeries, setSelectedSeries, componentFilter, setComponentFilter, enabled = true,
}) {
  const [series, setSeries] = useState(null);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState('');
  const [preview, setPreview] = useState(null);
  const [previewError, setPreviewError] = useState('');
  const [loadError, setLoadError] = useState('');

  useEffect(() => {
    if (!enabled) return;
    loadSeries();
  }, [enabled]);

  useEffect(() => {
    if (!enabled || selectedSeries === null) return undefined;
    const timer = window.setTimeout(async () => {
      try {
        const result = await api.previewComponentLibrary({
          component_series: selectedSeries,
          component_filter: componentFilter,
        });
        setPreview(result);
        setPreviewError('');
      } catch (error) {
        setPreviewError(error.message);
      }
    }, 180);
    return () => window.clearTimeout(timer);
  }, [enabled, selectedSeries, componentFilter]);

  async function loadSeries() {
    setLoading(true);
    setLoadError('');
    setSeries(null);
    setPreview(null);
    try {
      const res = await api.getComponentSeries();
      setSeries(res);
      const available = (res.series || []).map(item => item.id);
      setSelectedSeries(current => reconcileSeriesSelection(
        current, available, res.default_selected || [],
      ));
    } catch (e) {
      console.error('Failed to load component series:', e);
      setLoadError(e.message);
    }
    setLoading(false);
  }

  function toggleSeries(name) {
    const selected = selectedSeries || [];
    if (selected.includes(name)) {
      setSelectedSeries(selected.filter(s => s !== name));
    } else {
      setSelectedSeries([...selected, name]);
    }
  }

  function selectAll(type) {
    if (!series) return;
    const names = matchingItems(type)
      .map(item => item.id);
    const newSel = [...new Set([...(selectedSeries || []), ...names])];
    setSelectedSeries(newSel);
  }

  function clearAll(type) {
    if (!series) return;
    const names = new Set(matchingItems(type).map(item => item.id));
    setSelectedSeries((selectedSeries || []).filter(s => !names.has(s)));
  }

  function matchingItems(type) {
    const componentType = type === 'L' ? 'inductor' : 'capacitor';
    const needle = query.trim().toLowerCase();
    return (series?.series || []).filter(item => {
      if (item.component_type !== componentType) return false;
      if (!needle) return true;
      return [item.name, item.manufacturer, item.package_code]
        .some(value => String(value || '').toLowerCase().includes(needle));
    });
  }

  if (!enabled) return <div className="catalog-unavailable">元件数据源尚未就绪。请先在项目资源中应用有效的元件库路径。</div>;
  if (loading) return <div style={{fontSize: 12, color: 'var(--text-secondary)'}}>正在载入元件系列…</div>;
  if (loadError) return <div className="catalog-unavailable">元件系列载入失败：{loadError}<button className="btn btn-sm" onClick={loadSeries}>重试</button></div>;
  if (!series) return null;

  const selected = selectedSeries || [];
  const items = series.series || [];
  const inductors = matchingItems('L');
  const capacitors = matchingItems('C');
  const totalSelected = selected.length;
  const totalAvailable = items.length;
  const facets = series.facets || {};

  function updateFilter(field, value) {
    setComponentFilter({ ...componentFilter, [field]: value });
  }

  function toggleFilterValue(field, value) {
    const current = componentFilter[field] || [];
    updateFilter(field, current.includes(value)
      ? current.filter(item => item !== value)
      : [...current, value]);
  }

  return (
    <div className="component-library-panel">
      <div className="catalog-toolbar">
        <div><strong>实测元件系列</strong><span>{totalSelected} / {totalAvailable} 已选</span></div>
        <input className="form-input" value={query} onChange={event => setQuery(event.target.value)} placeholder="搜索厂商、系列或封装…" />
      </div>

      <details className="procurement-filters">
        <summary><span>采购与物料约束</span><small>厂商、封装、耐压、容差和元数据策略</small></summary>
        <div className="procurement-body">
        <label style={{ display: 'block', fontSize: 11, marginBottom: 4 }}>Manufacturers</label>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginBottom: 8 }}>
          {(facets.manufacturers || []).map(value => (
            <button
              key={value}
              className={`btn btn-sm ${(componentFilter.manufacturers || []).includes(value) ? 'active' : ''}`}
              onClick={() => toggleFilterValue('manufacturers', value)}
            >{value}</button>
          ))}
        </div>
        <label style={{ display: 'block', fontSize: 11, marginBottom: 4 }}>Packages</label>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginBottom: 8 }}>
          {(facets.package_codes || []).map(value => (
            <button
              key={value}
              className={`btn btn-sm ${(componentFilter.package_codes || []).includes(value) ? 'active' : ''}`}
              onClick={() => toggleFilterValue('package_codes', value)}
            >{value}</button>
          ))}
        </div>
        {(facets.dielectrics || []).length > 0 && <>
          <label style={{ display: 'block', fontSize: 11, marginBottom: 4 }}>Dielectric (part-number inferred)</label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginBottom: 8 }}>
            {facets.dielectrics.map(value => (
              <button
                key={value}
                className={`btn btn-sm ${(componentFilter.dielectrics || []).includes(value) ? 'active' : ''}`}
                onClick={() => toggleFilterValue('dielectrics', value)}
              >{value}</button>
            ))}
          </div>
        </>}
        {(facets.voltage_codes || []).length > 0 && (
          <label style={{ display: 'block', fontSize: 11, marginBottom: 8 }}>
            Rated-voltage codes (part-number inferred; Ctrl/Cmd for multiple)
            <select
              className="form-input"
              multiple
              size={Math.min(4, facets.voltage_codes.length)}
              value={componentFilter.voltage_codes || []}
              onChange={event => updateFilter(
                'voltage_codes', Array.from(event.target.selectedOptions, option => option.value),
              )}
            >
              {facets.voltage_codes.map(value => <option key={value} value={value}>{value}</option>)}
            </select>
          </label>
        )}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <label style={{ fontSize: 11 }}>
            Max tolerance (%)
            <input
              className="form-input"
              type="number"
              min="0.01"
              max="100"
              step="0.1"
              value={componentFilter.maximum_tolerance_pct ?? ''}
              onChange={event => updateFilter(
                'maximum_tolerance_pct', event.target.value === '' ? null : Number(event.target.value),
              )}
              placeholder={facets.tolerance_available ? 'Any' : 'Metadata unavailable'}
            />
          </label>
          <label style={{ fontSize: 11 }}>
            Missing metadata
            <select
              className="form-input"
              value={componentFilter.unknown_metadata_policy || 'include'}
              onChange={event => updateFilter('unknown_metadata_policy', event.target.value)}
            >
              <option value="include">Include and flag</option>
              <option value="exclude">Exclude (strict)</option>
            </select>
          </label>
        </div>
        <div style={{ marginTop: 7, fontSize: 10, color: 'var(--text-secondary)' }}>
          Directory-derived metadata is labelled separately from authoritative database fields.
        </div>
        </div>
      </details>

      <div className="series-columns">
      {/* Inductors */}
      <section className="series-column">
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4}}>
          <label style={{fontSize: 13, fontWeight: 600, color: 'var(--text)'}}>电感 <small>Inductors (L) · {inductors.length}</small></label>
          <div style={{display: 'flex', gap: 4}}>
            <button className="btn btn-sm" onClick={() => selectAll('L')}>全选</button>
            <button className="btn btn-sm" onClick={() => clearAll('L')}>清空</button>
          </div>
        </div>
        <div className="checkbox-list">
          {inductors.map(item => (
            <label key={item.id} className="checkbox-item">
              <input
                type="checkbox"
                checked={selected.includes(item.id)}
                onChange={() => toggleSeries(item.id)}
              />
              <span>{item.name}</span>
              <span style={{color: 'var(--text-secondary)', fontSize: 11}}>({item.count})</span>
              {(item.manufacturer || item.package_code) && (
                <span style={{color: 'var(--text-secondary)', fontSize: 10}}>
                  {[item.manufacturer, item.package_code].filter(Boolean).join(' · ')}
                </span>
              )}
            </label>
          ))}
        </div>
      </section>

      {/* Capacitors */}
      <section className="series-column">
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4}}>
          <label style={{fontSize: 13, fontWeight: 600, color: 'var(--text)'}}>电容 <small>Capacitors (C) · {capacitors.length}</small></label>
          <div style={{display: 'flex', gap: 4}}>
            <button className="btn btn-sm" onClick={() => selectAll('C')}>全选</button>
            <button className="btn btn-sm" onClick={() => clearAll('C')}>清空</button>
          </div>
        </div>
        <div className="checkbox-list">
          {capacitors.map(item => (
            <label key={item.id} className="checkbox-item">
              <input
                type="checkbox"
                checked={selected.includes(item.id)}
                onChange={() => toggleSeries(item.id)}
              />
              <span>{item.name}</span>
              <span style={{color: 'var(--text-secondary)', fontSize: 11}}>({item.count})</span>
              {(item.manufacturer || item.package_code) && (
                <span style={{color: 'var(--text-secondary)', fontSize: 10}}>
                  {[item.manufacturer, item.package_code].filter(Boolean).join(' · ')}
                </span>
              )}
            </label>
          ))}
        </div>
      </section>
      </div>
      {totalSelected === 0 && (
        <div style={{ marginTop: 8, color: 'var(--accent-orange)', fontSize: 11 }}>
          Select at least one measured component family. Tunable and switch synthesis may require both L and C.
        </div>
      )}
      {preview && (
        <div style={{ marginTop: 8, fontSize: 11, color: preview.valid_for_measured_search ? 'var(--text-secondary)' : 'var(--accent-orange)' }}>
          Preview: {preview.inductors} inductors / {preview.capacitors} capacitors
          {preview.filter_statistics?.included_with_unknown > 0
            ? ` · ${preview.filter_statistics.included_with_unknown} included with unknown metadata`
            : ''}
          {preview.filter_statistics?.excluded_unknown > 0
            ? ` · ${preview.filter_statistics.excluded_unknown} unknown excluded`
            : ''}
        </div>
      )}
      {previewError && <div style={{ marginTop: 8, color: 'var(--accent-orange)', fontSize: 11 }}>{previewError}</div>}
    </div>
  );
}
