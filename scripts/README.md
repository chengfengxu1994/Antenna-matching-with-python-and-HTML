# Scripts

- `rebuild_full_db.py`：合并 Murata 与 Optenni 元件数据库。
- `verify_optenni.py`：检查 Optenni 元件标称值与 S2P 推导值。
- `cross_validate.py`：交叉核对 Murata 阻抗数据与 S2P。
- `check_reference_cases.py`：检查本机 Optenni 官方教程输入是否可用于对标。
- `validate_optenni_golden.py`：校验用户从 Optenni 导出的逐频点 golden CSV。
- `validate_optenni_switch_export.py`：把 Optenni 的 SP2T/SP3T 分配置曲线 CSV 与
  full-network measured 基准逐频比较，输出 S11/total-efficiency 的最大与 RMS 误差。
- `run_optenni_baseline.py`：运行确定性的官方教程基线并记录输入哈希、耗时和候选值；
  switch 案例同时固化教程 PDF 哈希、页码及最佳/简化/降 BOM 三套官方参考拓扑。
- `run_search_recall_benchmark.py`：在小型 L/C S2P 目录上用 exhaustive 校准分层搜索，
  记录精确料号/拓扑 top-k recall、得分差、评估数、耗时和峰值跟踪内存。
- `run_multiport_search_recall_benchmark.py`：在官方三天线输入和 Optenni 实测元件小目录上，
  校准独立端口 shortlist 经过全耦合联合排名后的 top-k recall。
- `run_four_port_search_recall_benchmark.py`：在官方四阵元 S4P 上穷举 6,561 个
  联合实测 S2P 候选，校准每端口 shortlist 宽度并记录召回率、吞吐、内存与环境指纹。
- `run_optenni_multiport_live_golden.py`：用运行中的 Optenni 4.3 官方三天线保存最优解
  的同一 DUT、精确 BOM 和实测 S2P 复算频段效率与 1.24 GHz 功率平衡。
- `extract_optenni_opr_manifest.py`：只读解析 Optenni 4.x `.opr` XML，提取内嵌 DUT
  元数据、频段/目标、候选排序、拓扑和真实器件 BOM；不把核心复算描述为 Optenni 曲线。
- `run_optenni_saved_winner_discovery.py`：只向搜索提供官方三天线 DUT、频段和 saved BOM
  所属的四个实测料号，分别验证固定拓扑离散搜索与完全自动拓扑搜索能否召回精确首选网络。
- `run_product_saved_winner_constrained.py`：从统一 `run_tuning_joint` 产品服务入口加载完整
  41L+104C 参考目录，验证目录元数据、逐端口拓扑约束、耦合精修、结果转换和诊断的端到端召回。
- `run_product_saved_winner_automatic.py`：从同一产品服务入口使用 150 秒超深度档位，不传
  任何拓扑提示，验证完整目录自动拓扑发现、精确 BOM 排名、预算余量和功率闭合。
- `run_product_optimization_settings.py`：用官方单端口 Optimization Settings DUT、完整
  Coilcraft 0402HP + AVX ACCU-P 0402 实测系列和 Balanced 档位，验证 PCSL 拓扑、采购值
  对理想 Optenni 值的偏差，以及最低/平均效率 dB 口径的跨软件一致性。
- `run_product_multi_scenario.py`：从产品多阻抗优化器加载官方三种 DUT 状态、完整 43 个
  Coilcraft 0402CS 与 882 个 Murata GJM15 模型；先用连续理想综合分配拓扑预算，再对全部
  16 个二元件拓扑做物理粗筛、目录邻域精修，并在 82 个独立频点上重排候选。
- `run_environmental_yield_benchmark.py`：在有 Optenni 原生容差证据的 PCSL DUT/网络上，
  对比独立 ±2% uniform 控制组和批次相关 + 温度漂移工程假设，并明确记录证据边界。
- `run_product_joint_timeout_benchmark.py`：在官方三端口与完整 0402HP/GQM18 扫描目录上
  验证产品软时间预算、截断来源、候选质量、懒加载数量与功率守恒。
- `run_transmission_line_benchmark.py`：用闭式公式验证无损线相位、有损线传递功率、
  四分之一波阻抗变换与开/短路 Stub 导纳，并生成物理防回退产物。
- `run_microstrip_geometry_benchmark.py`：验证 PCB 微带几何、频散、线宽反解、铜/介质
  损耗、制造约束、灵敏度和损耗感知自动综合。
- `run_layout_block_benchmark.py`：验证固定 EM/VNA S2P 在 DUT/连接器两侧的级联顺序、
  被动性、互易性、插损、逐频功率和功率闭合。
- `run_six_component_search_benchmark.py`：在 8192 个六元件真实 S2P 物理候选上用
  exhaustive 校准渐进拓扑束搜索和两阶段离散邻域，记录精确料号/拓扑 top-k recall、
  得分差、物理评估数、时间与内存。
- `run_real_catalog_six_component_budget.py`：在 Optenni 官方 DUT 与完整 0402HP/GQM18
  目录上运行 2/8/20 秒六元件产品预算曲线，验证部分结果、anytime 单调质量、懒加载和功率守恒。
- `run_quality_gate.py`：统一运行编译、API、核心、适配器、引擎、参考输入和前端构建检查。
- `set_utf8.ps1`、`run_with_utf8.bat`：Windows UTF-8 环境辅助脚本。

脚本默认使用 `data/Murata`，也遵循项目的 `RFMATCH_*` 环境变量。

搜索召回率 P0 门禁：

```powershell
python scripts/run_search_recall_benchmark.py --minimum-recall 0.95
python scripts/run_multiport_search_recall_benchmark.py --minimum-recall 0.95
python scripts/run_four_port_search_recall_benchmark.py --minimum-recall 0.95
python scripts/run_product_saved_winner_constrained.py
python scripts/run_product_saved_winner_automatic.py
python scripts/run_product_optimization_settings.py
python scripts/run_product_multi_scenario.py
python scripts/run_product_joint_timeout_benchmark.py --timeout-seconds 10
python scripts/run_transmission_line_benchmark.py
python scripts/run_microstrip_geometry_benchmark.py
python scripts/run_layout_block_benchmark.py
python scripts/run_six_component_search_benchmark.py
python scripts/run_real_catalog_six_component_budget.py
```

开关跨软件验证可使用一个带 `configuration` 列的合并文件：

```powershell
python scripts/validate_optenni_switch_export.py optenni-switch.csv `
  --report artifacts/benchmarks/optenni-switch-cross-validation.json
```

也可为三个配置分别导出文件；文件本身没有配置列时按顺序指定名称：

```powershell
python scripts/validate_optenni_switch_export.py set1.csv set2.csv set3.csv `
  --configuration "Set 1" --configuration "Set 2" --configuration "Set 3"
```

频率列接受 Hz/MHz/GHz，S11 使用 dB；total efficiency 可使用线性值、百分数或 dB。
# Optenni tolerance inference

`python scripts/analyze_optenni_tolerance.py` parses the native Optenni 4.3
100-evaluation export, extracts the exact nominal PCSL values from the exported
network ABCD matrix, and fits the L/C scale used by every sample. It uses only
NumPy and reports bounded-uniform goodness-of-fit, scale correlation and curve
reconstruction error.
# Numeric input baselines

Verify that RI, MA and DB encodings of the same non-reciprocal 75-ohm
multi-port DUT remain identical after parsing and loss-aware physical matching.
The same baseline also round-trips per-port `[50, 90] Ω` renormalization and
proves the physical evaluator does not collapse it to one scalar impedance:

```powershell
python scripts/run_touchstone_format_benchmark.py
```
