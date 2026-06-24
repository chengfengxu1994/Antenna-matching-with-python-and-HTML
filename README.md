# 📡 天线匹配软件 (Antenna Matching Tool)

Murata LC元件库 + 多频段效率评估 + 多拓扑优化

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
├── Murata/               ← Murata元件S参数库 (ZIP)
├── snp/                  ← 天线SNP测试文件
└── rf-matching/
    ├── backend/
    │   ├── api/server.py      ← FastAPI后端
    │   ├── engine/            ← 匹配引擎
    │   │   ├── optimizer.py   ← 优化器 (beam search + 分析预计算)
    │   │   ├── network.py     ← S参数网络运算
    │   │   ├── topology.py    ← 拓扑定义 (L/Pi/T/Ladder)
    │   │   ├── touchstone.py  ← SNP文件解析
    │   │   └── component_lib.py ← 元件库管理
    │   └── tests/             ← 测试和基准
    └── frontend/
        ├── src/               ← React源码
        └── dist/              ← 构建产物
```

## 功能特性
- **多拓扑优化**: L-Network / Pi-Network / T-Network / Ladder-N
- **多频段效率评估**: 指定多个频段，评估匹配后全频段效率
- **S11曲线对比**: 匹配前后S11全频段对比图
- **Smith Chart**: 阻抗匹配可视化
- **手动调谐**: 手动指定元件值查看效果
- **端口配置**: 支持多端口DUT (open/short/load/component)

## 依赖
- Python 3.8+ (numpy, fastapi, uvicorn, pydantic)
- Node.js 16+ (仅开发模式需要)
