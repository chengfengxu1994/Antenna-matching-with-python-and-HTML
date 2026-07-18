import React, { useEffect, useRef, useState } from 'react';
import { api } from '../services/api';


export default function ProjectManager({ loadedSNP, manualWorkspace, onLoaded, onClose }) {
  const importInputRef = useRef(null);
  const canSave = Boolean(loadedSNP && loadedSNP.input_verified !== false);
  const [projects, setProjects] = useState([]);
  const [name, setName] = useState(loadedSNP?.filename?.replace(/\.s\d+p$/i, '') || 'RF Matching Project');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [repairResults, setRepairResults] = useState({});

  async function refresh() {
    try {
      const response = await api.listProjects();
      setProjects(response.projects || []);
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => { refresh(); }, []);

  async function save() {
    if (!canSave || !name.trim()) return;
    setBusy(true); setError(''); setMessage('');
    try {
      const result = await api.saveProject(name.trim(), null, manualWorkspace);
      setMessage(`已保存 ${result.name} · 格式 v${result.schema_version} · ${result.solutions_count} 个自动候选 · ${manualWorkspace?.variants?.length || 0} 个手动方案`);
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function open(project, verifyInput = true) {
    setBusy(true); setError(''); setMessage('');
    try {
      const result = await api.loadProject(project.project_id, verifyInput);
      onLoaded(result);
      setMessage(`${result.input_verified ? '已校验并打开' : '已进入只读快照模式'} ${result.name} · 格式 v${result.schema_version}${result.migrated_from_version ? `（从 v${result.migrated_from_version} 迁移）` : ''}${result.input_verified ? '' : '。候选结果可审阅，重新计算已禁用。'}`);
    } catch (e) {
      if (e.message.includes('Project input is unavailable')) {
        setError('找不到工程引用的原始 Touchstone。请切换到包含该文件的数据目录，或使用“仅查看快照”。');
      } else if (e.message.includes('SHA-256 does not match')) {
        setError('当前 Touchstone 与保存工程的 SHA-256 不一致。为避免误算，已拒绝载入；可仅查看可信快照。');
      } else if (e.message.includes('dependency is unavailable or changed')) {
        setError('布局或去嵌夹具依赖缺失/已变更。请恢复原文件，或仅查看可信快照。');
      } else {
        setError(e.message);
      }
    } finally {
      setBusy(false);
    }
  }

  async function importSnapshot(event) {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    setError(''); setMessage('');
    if (file.size > 25 * 1024 * 1024) {
      setError('工程 JSON 超过 25 MB，已拒绝导入。');
      return;
    }
    setBusy(true);
    try {
      const text = await file.text();
      const document = JSON.parse(text);
      if (!document || Array.isArray(document) || typeof document !== 'object') {
        throw new Error('工程文件必须是 JSON 对象');
      }
      const result = await api.importProject(document, 'copy');
      const action = {
        imported: '已验证并导入',
        unchanged: '本地已有完全相同的工程',
        copied: '检测到同 ID 的不同工程，已安全创建副本',
        replaced: '已替换',
      }[result.status] || '导入完成';
      setMessage(`${action}：${result.name} · 格式 v${result.schema_version} · ${result.solutions_count} 个候选方案。打开时将校验原始 Touchstone 与外部依赖。`);
      await refresh();
    } catch (e) {
      setError(e instanceof SyntaxError ? '工程 JSON 语法无效，未导入任何内容。' : e.message);
    } finally {
      setBusy(false);
    }
  }

  async function repair(project) {
    setBusy(true); setError(''); setMessage('');
    try {
      const result = await api.relinkProject(project.project_id, true);
      setRepairResults(current => ({ ...current, [project.project_id]: result }));
      if (result.status === 'ready') {
        const restored = await api.loadProject(project.project_id, true);
        onLoaded(restored);
        setMessage(restored.exact_recompute_available
          ? `已按 SHA-256 找齐 ${result.total_count} 个工程文件并恢复精确复算：${restored.name}`
          : `工程文件已全部匹配，但当前实测元件库与快照版本不一致；结果可审阅，重新计算仍保持禁用。`);
      } else {
        setMessage(`已匹配 ${result.matched_count}/${result.total_count} 个工程文件。请将缺失文件放入当前 SNP 数据目录后刷新并重试。`);
      }
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <section className="project-dialog project-center" onClick={e => e.stopPropagation()} aria-label="项目与导出">
        <div className="project-dialog-header">
          <div>
            <span className="eyebrow">REPRODUCIBLE ENGINEERING HANDOFF</span>
            <h2>项目保存与工程导出</h2>
            <p>带输入 SHA-256、版本信息、优化约束和候选结果的完整快照</p>
          </div>
          <button className="dialog-close" onClick={onClose} aria-label="Close">×</button>
        </div>

        <div className="project-save-row">
          <input value={name} onChange={e => setName(e.target.value)} maxLength={200}
            placeholder="项目名称" disabled={!canSave || busy} />
          <button className="btn btn-primary" onClick={save} disabled={!canSave || !name.trim() || busy}>
            保存当前工程快照
          </button>
          <input
            ref={importInputRef}
            className="visually-hidden"
            type="file"
            accept=".json,.rfmatch.json,application/json"
            onChange={importSnapshot}
          />
          <button className="btn" onClick={() => importInputRef.current?.click()} disabled={busy}>
            导入工程 JSON
          </button>
        </div>
        {!loadedSNP && <div className="project-note">请先载入 Touchstone 文件。</div>}
        {loadedSNP?.input_verified === false && <div className="project-note">当前为只读快照，需载入并校验原始 Touchstone 后才能保存新工程或重新计算。</div>}
        {message && <div className="project-message success">{message}</div>}
        {error && <div className="project-message error">{error}</div>}

        <div className="project-list">
          {projects.length === 0 && <div className="project-empty">还没有保存的工程快照。</div>}
          {projects.map(project => (
            <article className={`project-item ${project.status}`} key={project.project_id}>
              <div className="project-item-main">
                <strong>{project.name}</strong>
                <span>{project.input_filename || project.project_id}</span>
                <small>
                  {project.status === 'valid'
                    ? `格式 v${project.schema_version}${project.migrated_from_version ? ` ← v${project.migrated_from_version}` : ''} · ${project.solutions_count} 个自动方案 · ${project.manual_variants_count || 0} 个手动方案${project.imported_from_project_id ? ` · 导入自 ${project.imported_from_project_id}` : ''} · ${new Date(project.updated_at).toLocaleString()}`
                    : `快照无效 · ${project.error}`}
                </small>
              </div>
              <div className="project-item-actions">
                <a
                  className={`btn btn-sm ${project.status !== 'valid' ? 'disabled' : ''}`}
                  href={project.status === 'valid' ? api.projectReportUrl(project.project_id) : undefined}
                  aria-disabled={project.status !== 'valid'}
                  onClick={event => { if (project.status !== 'valid') event.preventDefault(); }}
                >
                  HTML 报告
                </a>
                <a
                  className={`btn btn-sm ${project.status !== 'valid' ? 'disabled' : ''}`}
                  href={project.status === 'valid' ? api.projectPdfReportUrl(project.project_id) : undefined}
                  aria-disabled={project.status !== 'valid'}
                  onClick={event => { if (project.status !== 'valid') event.preventDefault(); }}
                >
                  PDF 报告
                </a>
                <a
                  className={`btn btn-sm ${project.status !== 'valid' ? 'disabled' : ''}`}
                  href={project.status === 'valid' ? api.projectBomUrl(project.project_id) : undefined}
                  aria-disabled={project.status !== 'valid'}
                  onClick={event => { if (project.status !== 'valid') event.preventDefault(); }}
                >
                  BOM CSV
                </a>
                <a
                  className={`btn btn-sm ${project.status !== 'valid' ? 'disabled' : ''}`}
                  href={project.status === 'valid' ? api.projectSnapshotUrl(project.project_id) : undefined}
                  aria-disabled={project.status !== 'valid'}
                  onClick={event => { if (project.status !== 'valid') event.preventDefault(); }}
                  title="包含完整配置、候选结果、软件版本和 SHA-256 完整性摘要"
                >
                  工程 JSON
                </a>
                <button className="btn btn-sm" disabled={busy || project.status !== 'valid'} onClick={() => open(project)}>
                  打开
                </button>
                <button className="btn btn-sm" disabled={busy || project.status !== 'valid'} onClick={() => open(project, false)}>
                  仅查看快照
                </button>
                <button className="btn btn-sm btn-accent" disabled={busy || project.status !== 'valid'} onClick={() => repair(project)}>
                  查找并修复
                </button>
              </div>
              {repairResults[project.project_id] && (
                <div className="project-relink-status">
                  <strong>文件完整性匹配</strong>
                  {repairResults[project.project_id].targets.map(target => (
                    <div className={target.matched ? 'matched' : 'missing'} key={target.key}>
                      <span>{target.role === 'dut_touchstone' ? 'DUT' : target.role}</span>
                      <code>{target.original_filename}</code>
                      <b>{target.matched ? `✓ ${target.linked_filename}` : '缺失'}</b>
                    </div>
                  ))}
                  <small>搜索目录：{repairResults[project.project_id].data_directory}</small>
                </div>
              )}
            </article>
          ))}
        </div>

        <div className="project-note">
          工程 JSON 带文档级 SHA-256 完整性摘要（不是身份数字签名）。导入只恢复可信快照；重新打开时会核对原始 Touchstone 和布局/夹具依赖，精确扫频需按保存配置重新运行。
        </div>
      </section>
    </div>
  );
}
