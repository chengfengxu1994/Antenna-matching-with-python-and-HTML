# 📡 天线匹配软件 (Antenna Matching Tool)

Murata LC元件库 + 多频段效率评估 + 多拓扑优化

> A tooling for antenna matching, powered by Python+FastAPI backend and React frontend.

## 一键启动

### Windows (双击即可)
- **`启动.bat`** — 生产模式 (推荐，单窗口，浏览器自动打开)
- **`启动-开发模式.bat`** — 开发模式 (前后端分离，支持热更新)

### 通用方式 (Windows/Mac/Linux)
```bash
# 生产模式
python start.py

# 开发模式 (需要 Node.js)
python start.py --dev
```

启动后浏览器打开: **http://localhost:8000**

## 目录结构
```
E:\RF matching\
├── 启动.bat              ← 双击启动 (生产模式)
├── 启动-开发模式.bat     ← 双击启动 (开发模式)
├── start.py              ← Python启动脚本
├── Murata/               ← Murata元件S参数库 (ZIP + SQLite数据库)
├── snp/                  ← 天线SNP测试文件
└── rf-matching/
    ├── backend/
    │   ├── api/server.py      ← FastAPI后端
    │   ├── engine/            ← 匹配引擎
    │   │   ├── optimizer.py   ← 优化器 (beam search + 分析预计算)
    │   │   ├── network.py     ← S参数网络运算
    │   │   ├── topology.py    ← 拓扑定义 (L/Pi/T/Ladder)
    │   │   ├── touchstone.py  ← SNP文件解析
    │   │   ├── component_lib.py ← 元件库管理
    │   │   ├── cost_function.py ← Optenni式统一评分函数
    │   │   ├── multiport_optimizer.py ← 联合多端口优化器
    │   │   ├── efficiency_data.py ← 辐射效率数据解析
    │   │   └── murata_db.py   ← Murata SQLite数据库
    │   └── tests/             ← 测试和基准
    └── frontend/
        ├── src/               ← React源码
        │   ├── components/    ← UI组件
        │   └── services/      ← API客户端
        └── dist/              ← 构建产物
```

## 功能特性
- **多拓扑优化**: L-Network / Pi-Network / T-Network / Ladder-N
- **多频段效率评估**: 指定多个频段，评估匹配后全频段效率
- **S11曲线对比**: 匹配前后S11全频段对比图
- **Smith Chart**: 阻抗匹配可视化
- **手动调谐**: 手动指定元件值查看效果
- **端口配置**: 支持多端口DUT (open/short/load/component)
- **联合多端口优化**: 考虑S21耦合的联合匹配
- **功率平衡分析**: 反射/耦合/元件损耗/辐射占比可视化
- **Murata元件数据库**: 29K+ 真实元件S参数，支持快速查询
- **辐射效率集成**: 支持每端口独立η_rad文件加载
- **多目标优化**: 6种优化模式 (效率/回损/平衡/最差/平均/低成本)

## 依赖
- Python 3.8+ (numpy, fastapi, uvicorn, pydantic)
- Node.js 16+ (仅开发模式需要)
