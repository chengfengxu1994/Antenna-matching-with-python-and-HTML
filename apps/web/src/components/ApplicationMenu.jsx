import React, { useEffect, useRef, useState } from 'react';

const WORKSPACES = [
  { id: 'single', label: '单文件调谐', hint: '自动综合与候选方案' },
  { id: 'multi', label: '多场景联合', hint: '多个 DUT 共用同一网络' },
  { id: 'manual', label: '手动调谐', hint: '固定拓扑与交互式复算' },
];

function MenuItem({ checked, children, disabled = false, hint, onSelect }) {
  const checkable = typeof checked === 'boolean';
  return (
    <button
      type="button"
      role={checkable ? 'menuitemcheckbox' : 'menuitem'}
      aria-checked={checkable ? checked : undefined}
      className="application-menu-item"
      disabled={disabled}
      onClick={() => !disabled && onSelect?.()}
    >
      <span className="application-menu-check" aria-hidden="true">{checked ? '✓' : ''}</span>
      <span className="application-menu-copy"><strong>{children}</strong>{hint && <small>{hint}</small>}</span>
    </button>
  );
}

export default function ApplicationMenu({
  backendOnline, dataRailOpen, loadedSNP, onOpenCatalog, onOpenProjects,
  onRefreshFiles, onToggleDataRail, onToggleTheme, onWorkspaceChange,
  theme, workspaceMode,
}) {
  const [openMenu, setOpenMenu] = useState(null);
  const barRef = useRef(null);

  useEffect(() => {
    if (!openMenu) return undefined;
    const closeFromPointer = event => {
      if (!barRef.current?.contains(event.target)) setOpenMenu(null);
    };
    const closeFromKeyboard = event => {
      if (event.key === 'Escape') setOpenMenu(null);
    };
    document.addEventListener('pointerdown', closeFromPointer);
    document.addEventListener('keydown', closeFromKeyboard);
    return () => {
      document.removeEventListener('pointerdown', closeFromPointer);
      document.removeEventListener('keydown', closeFromKeyboard);
    };
  }, [openMenu]);

  const select = callback => {
    setOpenMenu(null);
    callback?.();
  };
  const menuButtonProps = id => ({
    type: 'button',
    'aria-haspopup': 'menu',
    'aria-expanded': openMenu === id,
    className: openMenu === id ? 'active' : '',
    onClick: () => setOpenMenu(current => current === id ? null : id),
    onPointerEnter: () => { if (openMenu) setOpenMenu(id); },
  });

  return (
    <div className="application-menu-bar" ref={barRef}>
      <nav className="application-menus" aria-label="应用菜单">
        <div className="application-menu">
          <button {...menuButtonProps('file')}>文件(F)</button>
          {openMenu === 'file' && (
            <div className="application-menu-popup" role="menu" aria-label="文件">
              <MenuItem onSelect={() => select(onOpenProjects)} hint="打开、保存与导出工程">项目中心…</MenuItem>
              <MenuItem onSelect={() => select(onRefreshFiles)} hint="重新扫描 Touchstone 文件">刷新项目资源</MenuItem>
              <div className="application-menu-separator" role="separator" />
              <MenuItem disabled hint={loadedSNP ? `${loadedSNP.num_ports} 端口` : '请先从项目资源载入'}>
                {loadedSNP?.filename || '尚未载入 DUT'}
              </MenuItem>
            </div>
          )}
        </div>
        <div className="application-menu">
          <button {...menuButtonProps('workspace')}>工作区(W)</button>
          {openMenu === 'workspace' && (
            <div className="application-menu-popup workspace-menu" role="menu" aria-label="工作区">
              {WORKSPACES.map(workspace => (
                <MenuItem key={workspace.id} checked={workspaceMode === workspace.id} hint={workspace.hint}
                  onSelect={() => select(() => onWorkspaceChange(workspace.id))}>
                  {workspace.label}
                </MenuItem>
              ))}
            </div>
          )}
        </div>
        <div className="application-menu">
          <button {...menuButtonProps('view')}>查看(V)</button>
          {openMenu === 'view' && (
            <div className="application-menu-popup" role="menu" aria-label="查看">
              <MenuItem checked={dataRailOpen} onSelect={() => select(onToggleDataRail)}>项目资源窗格</MenuItem>
              <div className="application-menu-separator" role="separator" />
              <MenuItem checked={theme === 'light'} onSelect={() => select(theme === 'light' ? undefined : onToggleTheme)}>浅色主题</MenuItem>
              <MenuItem checked={theme === 'dark'} onSelect={() => select(theme === 'dark' ? undefined : onToggleTheme)}>深色主题</MenuItem>
            </div>
          )}
        </div>
        <div className="application-menu">
          <button {...menuButtonProps('tools')}>工具(T)</button>
          {openMenu === 'tools' && (
            <div className="application-menu-popup" role="menu" aria-label="工具">
              <MenuItem onSelect={() => select(onOpenCatalog)} hint="限定可用的实测器件">元件库与筛选…</MenuItem>
              <div className="application-menu-separator" role="separator" />
              <MenuItem disabled hint="运行状态也显示在底部状态栏">
                计算引擎：{backendOnline ? '在线' : '离线'}
              </MenuItem>
            </div>
          )}
        </div>
      </nav>
      <div className="application-menu-context" aria-live="polite">
        <span>{WORKSPACES.find(item => item.id === workspaceMode)?.label}</span>
        {loadedSNP && <strong title={loadedSNP.filename}>{loadedSNP.filename}</strong>}
      </div>
    </div>
  );
}
