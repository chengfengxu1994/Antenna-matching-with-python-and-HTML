# Optenni Lab 对标路线与验收口径

目标不是复制 Optenni Lab 的所有界面，而是先在天线匹配主流程上达到可验证的专业结果，再扩展到更复杂的 RF 自动化。Optenni 官方将核心能力概括为自动拓扑综合、单/多端口联合优化、真实元件、可调器件、多数据配置、效率/隔离目标和容差分析。本项目据此划分优先级。

官方能力参考：

- [Matching circuit synthesis](https://optenni.com/overview-of-optenni-lab/features/matching-circuit-synthesis-and-optimization/matching-circuit-synthesis/)
- [Optimization](https://optenni.com/overview-of-optenni-lab/features/matching-circuit-synthesis-and-optimization/optimization/)
- [Component library](https://optenni.com/overview-of-optenni-lab/features/matching-circuit-synthesis-and-optimization/component-library/)
- [Data configurations](https://optenni.com/overview-of-optenni-lab/features/matching-circuit-synthesis-and-optimization/data-configurations/)
- [Tolerance analysis](https://optenni.com/overview-of-optenni-lab/features/postprocessing-tools/tolerance-analysis/)

## 能力矩阵

| 能力 | 当前状态 | 目标验收 | 优先级 |
|---|---|---|---|
| Touchstone 与参考阻抗 | 产品与独立核心共用严格解析器；已认证三端口以上规范行顺序、非互易矩阵、二端口 `12_21/21_12`、Full/Lower/Upper、跨行 `[Reference]`、RI/MA/DB、75 Ω 及逐端口正实数 Z0 的解析、重归一化和物理功率基线 | Touchstone 2.0 本身不支持复数参考；如增加非文件型复数参考环境，须作为独立数据契约认证 Kurokawa power-wave 口径 | P0 |
| 单端口自动拓扑综合 | 产品与兼容入口已统一到 `run_tuning_single` + measured core；拓扑白名单由核心强制，旧路由在 OpenAPI 标记 deprecated，Web 不再暴露旧求解调用 | 继续扩展官方 quick-start、optimization-settings 的逐频点 golden 与搜索召回认证 | P0 |
| 总效率目标 | 已支持辐射效率输入、显式带内/跨频段平均权重和效率感知高密度扫频 | 同时计入失配、元件损耗、天线效率；口径逐点一致 | P0 |
| 真实离散元件 | 支持完整 Optenni 目录发现、系列选择及厂商/封装/最大容差/电压代码/介质参数过滤；默认 GQM18/0402HP，预览与搜索共用筛选链路，结果与 BOM 记录来源统计、目录及模型哈希 | 接入厂商验证的额定电压/介质数据源，并扩展更多系列的 Optenni 固定电路逐点 golden | P0 |
| 多频段优化 | 带内平均/最差、跨频段平均权重、逐端口优先级和逐频段优先级均可配置；有效优先级贯通连续拓扑种子、真实 S2P 排名、项目与报告 | 增加更多官方逐频导出，校准具体业务权重下的搜索召回 | P0 |
| 多端口联合匹配 | 官方 3 端口已有理想/真实 S2P 基线；支持每端口 0–2 元件、Optenni 对齐拓扑、定向 Sij 隔离目标和耦合精修；官方 4 阵元 S4P 已完成 6,561 候选穷举召回/性能校准 | 扩展到完整真实目录、0–2 元件和 5+ 端口 | P1 |
| 多阻抗环境 | 官方三环境已完成理想/真实 S2P 基准与产品闭环 | 扩展到多端口多状态、容差联合排序 | P1 |
| 容差与良率 | 固定 LC、开关、MDIF 可调和物理传输线方案均支持可复现 Monte Carlo、Wilson 区间与 P5 裕量；共享器件/板级制造变量跨状态共用；L/C 支持系统偏置、批次相关性、共享温度及精确料号 sidecar；微带显式扰动线宽、线长、板厚和介电常数后重算阻抗/频散/损耗，`msWidthToler=20%` 的 Optenni UWB OPR 设置证据已固化；Optenni 100 样本原生 L/C 导出已固化 | 获取 Optenni UWB 的权威逐频容差包络，以及板厂/温箱实测材料批次数据 | P1 |
| 可调电容/开关 | variable-C 与 SP2T/SP3T 的 MDIF 物理模型、固定网络/料号/状态联合自动综合、实测精修、API/UI、后台进度/取消与功率守恒已完成 | 导出 Optenni 全部 switch-state 曲线做跨软件数值对齐 | P2 |
| PCB/传输线模型 | 核心、手动调谐与统一自动综合支持有损均匀线、开/短路 Stub、7 类组合；PCB 模式增加微带几何/频散/损耗/线宽反解；产品 UI 可有序级联 EM/VNA S2P、翻转端口、重归一化 Z0，并用左右二端口 fixture 去嵌入；项目/报告追踪全部 SHA-256、变换、条件数和残差 | 增加阻焊/共面/带状线、校准标准求解、复数/逐端口 Z0 与 Optenni/EM 跨软件认证 | P2 |
| 手动工程调谐 | 理想/实测 L/C、R、有损线和开/短路 Stub 共用物理核心；逐激励端口保留独立网络与撤销栈，支持 DUT 向信号源的物理级联拖拽/上下移动及撤销、E24 快调、S11 拖动选频、Smith 目标标记和实测最近料号重算；最多 12 个命名方案可冻结对比，并经强校验、SHA-256 覆盖的 v2 工程与 HTML/PDF 报告完整保存/恢复；最多 4 个冻结方案可用当前 DUT 重算并在 S11/Smith 叠加，同名 DUT 内容变更使缓存失效 | 增加非端接多端口同时手调 | P2 |
| 报告与项目保存 | v2 JSON、完整性校验、原子保存、输入哈希恢复、v1 完整性优先的确定性迁移、自包含 HTML 和原生分页 PDF 追溯报告及候选良率/置信区间已完成 | 增加更长期的多版本迁移回归样本和报告签名 | P1 |
| 阵列/波束赋形 | 未实现 | 主匹配链路稳定后独立立项 | P3 |
| EM/VNA 实时联动 | 支持严格 Touchstone 拖入/批量导入、稳定目录监视、来源哈希；CST 2025 官方 Python 运行时可枚举打开工程/结果树并一键直出校验后载入；求解状态自动轮询，同一工程以稳定文件名原子刷新并追踪前后原始 SHA-256；独立网络语义指纹排除注释/格式噪声，数值未变时保留端口与手动工作区；已在真实打开的 CST 2025 `test.cst` 上完成 1001 点 S1P 首次直出、无副本重复刷新和配置保留验收 | 完成真实重复求解中“运行→完成→自动解锁”的动态验收，并扩展 HFSS 官方接口 | P2 |

## 数值验收标准

每个 golden case 必须记录输入文件哈希、端口状态、频段、拓扑、元件模型、算法版本和随机种子。

### Touchstone 输入语义基线

解析器现按 [Touchstone 2.0 官方规范](https://ibis.org/touchstone_ver2.0/touchstone_ver2_0.pdf)
区分二端口历史 `21_12` 顺序、显式 `12_21` 顺序和三端口以上的矩阵行顺序，并支持
Full/Lower/Upper 对称矩阵、跨行 `[Reference]` 与 DC 起始频点。此前把所有 N 端口数据
统一按二端口列顺序转置的隐患已修复；非互易三端口回归使用不同的 `Sij/Sji` 锁定方向，
最大复数误差为 0。Optenni 教程目录内现有 12 个 S3P（包括从 0 Hz 开始的 Ideal SP2T）
均可由权威核心直接解析。确定性产物见
`artifacts/benchmarks/touchstone-format-baseline.json`。

规范中的 `[Reference]` 仅允许每端口正实数阻抗；因此复数参考阻抗不会被伪装成
Touchstone 2.0 能力。未来若从 EM 求解器 API 引入复数参考环境，必须使用独立格式标注
波定义，并以 Kurokawa 功率波、共轭匹配和无源功率界限建立单独 golden 后才能进入产品。

### 单端口权威入口

当前产品 `/api/tuning/optimize` 与兼容 `/api/tune/single` 均调用
`engine.tuning_service.run_tuning_single`；后者只转换旧响应字段并明确返回
`deprecated_endpoint / authoritative_endpoint / authoritative_solver`，不再复制中心频点搜索和
带内重评分。`/api/optimize`、`/api/multipass`、`/api/joint-optimize` 及旧 tune 路由仍为外部
脚本兼容保留，但在 OpenAPI 标记 deprecated，Web API 客户端已移除这些调用。

实测核心的 `MeasuredSearchConfig.allowed_topology_codes`、`allowed_topology_codes_by_port` 和
`include_zero_component` 在理想种子、深层拓扑发现和最终联合排名三处统一生效；逐端口配置
优先于旧的全局配置，改变约束也会使检查点的理想候选缓存失效。统一 API 接受每端口
`allowed_topology_codes`，代码以 DUT 向外的顺序表达（例如 `SL`、`PC`、`PLSC`），`0` 表示
裸 DUT；空列表、非法语法及超过该端口元件上限的代码返回 HTTP 400。Web 已提供常用工程
预设和自定义白名单，并在搜索诊断中显示实际约束。回归证明端口 1 仅允许 `SL`、端口 2
仅允许 `PC` 时，核心与产品 service 的所有候选均严格服从各自约束且不会混入裸 DUT。

零元件请求也不再进入 legacy direct evaluation。`max_components=0` 由同一 measured physical
core 在全部带内频点计算裸 DUT，结果标记为 `rfmatch_core_physical_bare_dut`、
`bare_dut_core_baseline=true`，且不会误称为实测元件搜索。三频点解析回归逐点验证
`η=1-|S11|²`、零元件损耗和功率守恒误差 `<1e-12`。单端口 service 同时统一拒绝负数、
超过六元件、空/反向频带和越界端口，避免非法请求落入另一套求解器。

实测目录不再硬性要求同时存在 L 和 C。核心从非空 catalog 推导可用类型集合，并在所有
拓扑阶段排除无法采购的器件种类；仅含一个实测电感的目录配合 `Series-L` 白名单已通过
产品 service 回归，所有结果均为 `rfmatch_core`、`available_component_kinds=[L]` 且无
legacy fallback。统一 API 的零元件请求则在 `component_library=None` 下通过，Web 也不再
因 Component Series 为空阻止裸 DUT 运行。

该契约现已扩展到多端口全矩阵联合搜索：仅电感目录仍保留 DUT 耦合、定向隔离目标、
逐端口元件上限和功率守恒，并且所有候选只含电感；全端口 `max_components=0` 时即使
`component_library=None` 也由同一核心返回唯一裸 DUT 候选，标记
`rfmatch_core_physical_bare_dut_joint`。统一 API 与 Web 对固定 LC 搜索只要求至少一个
实测系列，tuner/switch 等确实需要双类型目录的模式仍保持更严格的校验。

### 端口与频段优先级契约

统一调谐请求的每个端口现支持 `port_weight`，并为 `bands_mhz` 提供等长
`band_weights`；默认值均为 `1.0`，因此旧项目与既有 golden 的评分完全不变。核心实际使用的
有效权重为二者乘积，并在 dB 域乘到该频段相对目标的 margin 上，再进入既有的带间与端口间
最差/平均聚合。零权重可明确排除某个频段，但所有启用频段不能同时为零；NaN、无穷、负值、
超过 100 或数组长度不一致均在 API 边界拒绝。

该权重同时进入单端口连续损耗感知拓扑种子和最终实测 S2P evaluator，多端口则进入完整耦合
矩阵评价。若实测核心不可用，带自定义权重的请求不会静默回落到不理解该契约的旧优化器。
候选诊断记录原始与有效权重，Web、项目 JSON、HTML 和 PDF 均可追溯。确定性基准用两组互有
取舍的候选锁定决策变化：默认权重选择 A，将第二端口提高到 3 倍后选择 B；产物见
`artifacts/benchmarks/priority-weight-scoring-baseline.json`，生成命令为
`python scripts/run_priority_weight_benchmark.py`。

### 六元件渐进拓扑搜索基线

产品现在支持每端口最多 6 个真实 S2P 元件。1–4 元件继续使用已对齐 Optenni 案例的
固定语法；5–6 元件使用逐深度 series/shunt 交替拓扑束搜索，再进行离散料号吸附和完整
节点功率求解。`scripts/run_six_component_search_benchmark.py` 在两颗电感、两颗电容、
六个位置上精确枚举 8192 个物理候选。高质量 beam=128 基线对 exhaustive 全局 top-10
的精确料号和拓扑召回率均为 100%，最佳得分差为 0 dB，最大功率误差为 0。两阶段
离散策略先为全部拓扑评估一个最近料号种子，再只展开前 32 个拓扑的完整邻域；父层
连续值暖启动又将子拓扑精修限制为冷启动轮次的四分之一。理想评估由 22181 降至 6821，
物理评估由 9031 降至 2047，峰值跟踪内存约 11.5 MiB；产物见
`artifacts/benchmarks/six-component-search-baseline.json`。

该小目录中 progressive 约 23 秒、exhaustive 约 35 秒；单个案例的计时不能外推为所有规模都
更快。它的价值在于真实器件目录扩大后可以用 beam、两阶段离散邻域、懒加载、超时和协作取消限制
工作量；beam=128 是校准/高质量模式，不是默认快速模式。任何“无遗漏”声明只适用于该
固定枚举域，不能外推到任意器件库和任意拓扑。

真实目录预算曲线使用官方 Optimization Settings DUT、49 个 Coilcraft 0402HP 电感和
322 个 Murata GQM18 电容、5 个频点、每端口最多 6 元件。2 秒预算返回可追溯裸 DUT；
8 秒截断时已得到 6 元件 `-1.3630 dB` 部分解；20 秒预算实际约 10.1 秒完成并提升到
`-1.1441 dB`。质量随预算单调不降，完整运行仅加载 46/371 个 S2P 模型，所有结果功率
闭合误差为 0。输入和目录指纹见
`artifacts/benchmarks/real-catalog-six-component-budget.json`。

同一 DUT 与 371 个真实元件现增加检查点续算对照。8 秒运行在 591 次物理评估后返回
`-1.3630 dB` 部分解；追加 12 秒时复用 591 个精确缓存条目，仅约 4.10 秒便完成到
`-1.1441 dB`。独立 20 秒冷启动约 10.15 秒，最终拓扑、得分、835 次唯一物理评估、
46 个加载模型和功率守恒完全一致；追加阶段耗时为冷启动的约 40.4%。基准命令会在得分、
拓扑、唯一评估数、质量单调性或功率守恒不一致，或追加阶段不再节省显著时间时失败。
产物见 `artifacts/benchmarks/real-catalog-continuation-checkpoint.json`，SHA-256 为
`DBC39555A3905E107B6D16FD5BEC8469F4A01D0FC2503C32C5651BF8A6267548`。

产品 UI 对截断方案显示 anytime 状态、停止阶段及理想/物理评估数，并可调用
`/api/tuning/continue` 增加总预算。续跑按网络签名合并旧/新候选，因此最佳分数单调不降。
单端口和多端口联合实测 S2P 路径现保留会话内优化器检查点，复用已加载模型、理想拓扑前沿、
逐端口子问题与精确
物理评估缓存；回归要求“截断后续跑”的累计唯一物理评估数等于相同设置冷启动完整搜索，
且最佳结果一致。不支持该状态的兼容路径仍明确返回 `deterministic_rerun_merge`，项目快照
也不伪装成可恢复的内存检查点。

普通单端口和多端口优化现与 tuner、switch、传输线共用后台 job。实测核心逐端口上报
理想拓扑、离散 S2P 展开、全矩阵联合排名和精确精修进度；取消信号进入同一个物理求解器，
并保留取消前已完成的功率守恒候选。API 回归同时验证任务提交不阻塞、终态进度以及部分
结果不会因 `cancelled` 状态被清空。追加预算也通过 `/api/tuning/continue/jobs` 使用同一
后台契约，非阻塞回归要求提交耗时显著短于模拟的续算运行时间。

第一阶段默认门槛：

### Optenni 4.3 实机导出基线（2026-07-15）

已从正在运行的 `Optenni Lab Optimization Settings` 工程采集原始 DUT、PCSL
匹配网络 S2P，以及 531 点原生文本曲线。核心把导出的网络嵌入原始 DUT 后，
相对 Optenni 曲线的 S11 RMS 误差为 `1.04e-5 dB`，最大误差为
`7.82e-5 dB`。该案例已固化为自动测试；文件哈希、目标频段、UI 摘要及
Optenni 4.3 导出注意事项记录在
`benchmarks/optenni_exports/optimization_settings_pcsl_manifest.json`。

### 传输线解析物理基线

`scripts/run_transmission_line_benchmark.py` 用闭式公式独立校验 50 Ω 无损线相位、
频率相关有损线的传递功率、75 Ω 四分之一波阻抗变换，以及 45° 开路/短路 Stub
的输入导纳。三频点有损案例同时验证 DUT 吸收功率、器件损耗和反射功率闭合；当前
最大功率平衡误差为 `0`。可复现产物见
`artifacts/benchmarks/transmission-line-physical-baseline.json`。这是本项目自身的解析
认证，并不冒充 Optenni 跨软件 golden；后续获得 Optenni 传输线工程导出后再增加
逐频交叉验证。该产物还要求自动综合从 100 Ω 负载恢复 50 Ω 系统的理论
`70.7107 Ω / 90°` 四分之一波变换，并限制评估次数，防止数值正确但搜索能力回退。

PCB 几何基线见 `artifacts/benchmarks/microstrip-geometry-baseline.json`。它在 1、
2.45、10 GHz 与独立 scikit-rf/Qucs Hammerstad–Jensen + Kirschning–Jansen 参考值
交叉验证，同时固化 50 Ω 线宽反解、铜/介质损耗、制造约束、Dk/线宽灵敏度和
损耗感知自动综合。详细适用边界见 `docs/PCB_MICROSTRIP_MODEL.md`。

导入式布局基线见 `artifacts/benchmarks/layout-block-physical-baseline.json`。它以
0.7 dB 匹配互易 S2P 验证 DUT/连接器两侧的级联顺序、传递功率、布局损耗和功率闭合；
API 回归另验证安全 SNP 目录加载、完整频带覆盖、被动性策略及 SHA-256 回传。该基线是
内部物理防回退，不代表已完成 Optenni 或板级 EM 的跨软件认证；输入边界见
`docs/LAYOUT_BLOCKS.md`。

该基线现在不仅能复算已知网络，也能从原始 DUT 自动找回方案。连续综合使用工程中
相同的通用损耗假设（电感 Q=30 @ 1 GHz、电容 ESR=0.4 Ω），在 6 种标准双元件
拓扑中将 PCSL 排名第一，得到 `5.9547 nH / 0.48348 pF`；相对导出网络的精确
`5.91510 nH / 0.483989 pF`，两者误差均小于 1%，并且同一目标函数得分不差于
Optenni 已保存候选。确定性搜索共 4444 次评估，已固化为自动回归。

为使有损综合可用于交互搜索，单端口通用 L/C 网络采用向量化阻抗递推和反向电压/
电流传播来分离 DUT 吸收与器件损耗；它与通用节点 Y 求解逐点一致到 `1e-12`，而
该 PCSL 的 1080 次物理评估从约 `23.5 s` 降至 `0.26 s`。

同一实机工程的第二组原生 golden 覆盖 `1.0–1.1 / 1.7–2.0 /
2.9–3.1 / 4.8–5.0 GHz` 四个频段和六元件 `PLSCSLPCSLPC` 网络。依据完整
531 点 S11 曲线反演 UI 舍入前的元件值后，Python 有损物理求解相对 Optenni 的
RMS 误差为 `6.73e-6 dB`、最大误差为 `4.85e-5 dB`。该导出还明确验证了
Optenni 的 “ave eff.” 是 dB 域算术平均：第二频段为 `-1.4 dB`，而线性算术
平均换算会得到 `-1.29 dB`。核心固定/跨状态良率与原生导入复算现统一采用这一
口径；文件哈希、拓扑、显示值及反演值记录在
`benchmarks/optenni_exports/optimization_settings_4bands_default_best_manifest.json`。

产品的固定单端口流程现先用该损耗感知核心在完整目标频段综合理想 L/C 网络，
再把预测拓扑排到实测元件库搜索的首位。理想种子的拓扑、连续值、得分、评估次数和
功率守恒误差会保存在候选诊断及 HTML/PDF 报告中，并与最终真实 BOM 明确分开；这使
限时搜索更早覆盖高价值拓扑，同时不把通用 Q/ESR 模型误报成实际器件性能。该先验的
电感 Q、Q 参考频率、电感 ESR 和电容 ESR 已可通过 API/UI 配置；默认保持 Q=30@1 GHz、
C ESR=0.4 Ω，而 Radiation Efficiency 官方教程基准显式采用 Q=50@1 GHz、C ESR=0.3 Ω。

固定单端口的最终选型已改为完整频段上的实测 S2P 物理搜索：理想连续值只负责缩小
候选，所有入选料号再通过节点电路求解器计算 DUT 吸收、器件损耗、总效率和功率
守恒后排名。文件、ZIP 与 SQLite/动态适配器统一成懒加载模型，只有分层搜索实际
到达的料号才展开 S2P。仓库 Murata SQLite 实测包含约 5888 个料号，基线搜索只
加载 65 个模型、完成 151 次物理评估，约 3.2 秒返回 PCSL 且功率误差为 0。
产品物理路径现已覆盖 1–4 个元件，包括标准双元件、Pi、T 与交替四元件梯形族；
同一 5888 条 SQLite 库在四元件上限下实测仅加载 76 个模型、完成 317 次物理评估，
约 4.26 秒返回两个候选且功率误差为 0。超过四元件时仍会明确记录兼容路径回退原因。

搜索质量现有独立 exhaustive 校准器，而不再只依赖“找到了一个好结果”。在 Optenni
optimization-settings 原始 DUT 的 17 个确定性频点、6 个电感和 6 个电容小目录上，
精确枚举 313 个完整 S2P 物理候选。分层搜索对 exhaustive 全局 top-10 的精确料号
召回率和拓扑召回率均为 `100%`，最佳得分差为 `0 dB`；当前版本完成 63 次物理评估，
比 exhaustive 的 313 次少 `79.9%`；在环境指纹
`a83c101a138…` 的本机实测约 `2.91 s` 对 `6.26 s`，峰值 tracemalloc
约 `0.57 MiB` 对 `1.39 MiB`。关键修正是保留已经付费计算的精修邻域候选，并让每轮
坐标精修围绕固定基点展开，避免候选覆盖依赖坐标顺序。可复现产物见
`artifacts/benchmarks/optenni-single-port-search-recall.json`；命令会在召回率低于
95% 时失败。

上述小目录 exhaustive 证据现增加统一产品入口的完整采购系列认证。官方 Optimization
Settings DUT 在不限制拓扑的 Balanced 请求中使用 49 个 Coilcraft 0402HP 电感和 64 个
AVX ACCU-P 0402 电容（0.05–9.1 pF，覆盖 Optenni 的 0.484 pF 目标）。产品在 `3.68 s`
内以 103 次物理评估、43 个懒加载模型把 `PCSL` 排第 1，并选择 `C0402SEr45` 0.45 pF +
`04HP5N6` 5.6 nH；两值相对 Optenni 反演理想值偏差为 `7.02% / 5.33%`。真实 S2P 采购
网络的最低/平均效率与 Optenni 理想网络仅差 `+0.02769 / +0.02751 dB`，最大功率闭合
误差为 0。最初使用不覆盖目标值的 GQM18 目录会得到 PLSC；基准因此同时固化目录值域，
明确“目录不可实现”与“搜索召回失败”不能混为一谈。产物为
`artifacts/benchmarks/optenni-product-optimization-settings.json`。

原生文本导入器现可直接读取 Optenni 4.3 的制表符格式、`%` 元数据、
`Frequency [GHz]`、无 `[dB]` 后缀的 `S11`，并依据 Y 轴元数据把总效率
从 dB 转为线性值。

同一工程的 100 次 circuit-evaluation 容差结果也已导出。原生文件包含 531
个频点、标称曲线，以及 S11/total-efficiency 各 100 个重名样本列。专用解析器按
列位置保留全部样本。1.7–2.5 GHz 内若只要求最低效率 ≥ `-1.0 dB`，通过
`54/100`；Optenni 同时要求 dB 域算术平均效率 ≥ `-0.7 dB`（等价于线性效率
的几何平均），联合通过数为 `0/100`，
与界面显示的 `Yield: 0%` 完全一致。Python 核心、产品 API 和 UI 已加入向后兼容
的平均效率门槛，真实文件回归测试固定上述口径。

匹配设置的 Tolerances 页确认通用电感、电容均为 `±2%`。通过 S2P 的 ABCD
矩阵提取出精确标称值 `5.915099856 nH / 0.483989482 pF`，再联合拟合每条 S11
与效率曲线，可将 100 组样本重建到 RMS 中位数 `1.59e-5 dB`、最大值
`1.79e-5 dB`。L/C 样本都落在 0.98–1.02 内，标准差接近均匀分布理论值，
KS 检验也不拒绝均匀分布。因此产品默认制造公差采样已对齐为有界均匀分布，仍保留
截断正态作为可选统计模型；完整反演可由
`python scripts/analyze_optenni_tolerance.py` 重复执行。

跨状态良率不再把各状态分别抽样后平均：每一次制造样本只抽取一组共享固定器件
偏差，并用同一组偏差重算全部 switch/tuner 状态与全部配置频段；只有每个配置都同时
满足最低效率、dB 域平均效率和回损门槛才计为通过。结果同时返回 joint yield、每配置
individual yield、Wilson 区间和最差样本，Web 与自包含 HTML 报告均展示这些口径。

环境良率模型现进一步覆盖 L/C 系统工艺偏置、批次相关性和温度漂移：系统偏置先按
器件类型一致地移动制造分布中心；相关器件偏差通过 Gaussian copula
生成，同时严格保留用户选择的有界 uniform 或截断 normal 边缘分布；每块样机只抽取
一个温度，按 L/C 各自 tempco 对全部端口和全部 switch/tuner 状态施加一致漂移。默认
偏置/相关性为 0 且温度关闭，因此旧基线的随机序列和结果保持不变。API、Web、项目文件、
HTML 与 PDF 都记录 L/C 偏置、相关系数、温区、参考温度和 L/C tempco。

可复现环境基准继续使用已有 Optenni 原生证据的 PCSL DUT、精确 `5.915099856 nH /
0.483989482 pF` 网络和 81 个带内点。200 样本独立 ±2% uniform 控制组仍复现联合
良率 `0%`；在明确标记为“工程假设、不是 Optenni 原生导出”的 `ρ=0.7`、`-40–85 °C`、
L/C tempco=`100/-30 ppm/°C` 场景下，良率仍为 `0%`，但 P5 裕量相对控制组变化
`+0.00368 dB`；独立的 `L +1% / C -1%` 系统偏置场景仍为 `0%` 良率，P5 裕量变化
`-0.00619 dB`。这也说明环境和工艺偏置不应被简单当成固定惩罚，而必须通过完整网络逐样本重算。
产物见 `artifacts/benchmarks/optenni-environmental-yield.json`。

- 同一固定电路逐频点复算：复数 S 参数绝对误差 ≤ `1e-3`。
- total efficiency 线性绝对误差 ≤ `1e-3`。
- 被动无源网络功率守恒误差 ≤ `1e-6`；真实测量模型按数值容差单独记录。
- 搜索算法不声称“无遗漏”，改用小规模 exhaustive 的 top-k recall；当前单端口实测
  S2P 校准为 100%，P0 门槛保持 ≥ 95%。
- 相同输入、配置和随机种子必须产生相同候选排序。
- 性能基准同时记录墙钟时间、评估次数、`physical evaluations / end-to-end wall second`、
  候选覆盖率、峰值 Python 分配、CPU/逻辑核、OS、Python/NumPy 和稳定环境指纹。
  跨机器只比较质量与工作量；只有环境指纹一致时才直接比较墙钟。统一清单见
  `artifacts/benchmarks/search-performance-baseline.json`。

## 当前真实器件基线

产品的 Component Series 选择链路现已闭合。后端保留完整可发现目录，同时维持校准过的
0402HP/GQM18 默认目录；本机 Optenni 库当前识别 4,394 个电感、12,086 个电容和 94 个
文件夹级系列。UI 可按厂商、系列与已知封装字段搜索，选择值以 `L:: / C::` 类型化 ID
进入统一 tuning 请求。核心实际接收过滤后的只读目录，而非仅记录界面状态；空选或缺少
任一器件类型会返回可操作诊断。每个候选、项目快照、HTML/PDF 报告均记录所选系列、
有效 L/C 数量和目录身份 SHA-256；改变数据目录会使旧运行时检查点失效。

参数级筛选也已进入同一产品闭环。用户可在选族后约束厂商、封装、最大容差、电压代码和
介质类型，预览 API
返回逐类型有效数量、按值排除数、未知字段排除数及元数据来源分布。SQLite 元件的逐料号
容差/封装/厂商标记为 `database`，料号解析所得电压代码和介质标记为
`part_number_inferred`；直接扫描 Optenni 文件夹时只把目录可验证信息标为
`catalog_path`，不透明 `index.dat` 不被当作权威来源，无法验证的容差保持 `unknown`。
默认策略为保留并标记未知项以兼容既有搜索，严格采购策略可改为排除。相同配置和最终
目录指纹写入候选诊断、项目快照与 HTML/PDF 报告。

采购交付现增加聚合 BOM CSV 和实测替代料分析。CSV 保留数量、端口/位置、连接方式、
厂商/系列/封装/容差/电压代码/介质及元数据来源。点击候选中的实测料号会在当前端口频带
比较同类型目录候选的完整二端口 S 矩阵，以 RMS 差异为主、标称值偏差为小权重并列项，
同时公开最大差异、物理比较数量、模型失败和预选截断；该排名是电气相似度证据，不替代
厂商额定值与供应链验证。

官方 `3_antennas.s3p` 以 15 个确定性采样点运行，器件来自本机 Optenni 库中的 Coilcraft 0402HP（41 个标准值）与 Murata GQM18（去重后 51 个值）。当前自动综合结果：

- 目标得分 `-3.6736 dB`，三端口得分分别为 `-1.9771 / -3.2099 / -3.7508 dB`。
- 物理评估 1851 次、理想种子评估 5903 次，墙钟时间约 35.5 秒。
- 最大功率平衡误差为 `0`（当前浮点输出），平均器件损耗为入射功率的 `1.74%`。
- 每个最终器件均记录厂商料号、标称值、连接位置和 S2P SHA-256；完整可复现实验见 `artifacts/benchmarks/optenni-measured-baseline.json`。

该成绩是工程基线，不等同于已经达到 Optenni 的全局搜索质量。下一阶段应以 Optenni 对同一输入、相同器件库和相同约束导出的固定电路/曲线作为 golden，量化得分差、逐频点误差和 top-k 拓扑召回率。

同一官方 `3_antennas.s3p` 现增加了可精确枚举的耦合搜索校准：三个端口各允许
0–1 个元件，目录使用 Optenni 自带的 `4.7/39 nH` Coilcraft 0402HP 与
`1/2 pF` Murata GQM18 实测 S2P，共 729 个联合物理候选。每端口 shortlist
宽度 6/7 时精确料号 top-10 recall 分别只有 70%/80%；宽度 8 达到 `100%`，
拓扑 recall `100%`、最佳得分差 `0 dB`。当前分层搜索完成 539 次物理评估、约
11.69 秒和 4.12 MiB 峰值跟踪内存，对比 exhaustive 的 729 次、14.81 秒和
5.00 MiB。该结果把产品默认 `per_port_keep=8` 从经验参数提升为有官方输入支撑的
最低达标宽度；产物见 `artifacts/benchmarks/optenni-multiport-search-recall.json`。
完整 0–2 元件/大目录仍需继续分层校准。

更大端口数的确定性校准使用 Optenni 官方
`5 - Antenna arrays/4_element_dipole_array.s4p`：4 个端口各允许空网络，或把
`4.7/39 nH` Coilcraft 0402HP 与 `1/2 pF` Murata GQM18 以串/并联放置，
共 `9^4 = 6,561` 个联合物理候选。每端口 shortlist 宽度 4 已达到精确料号和
拓扑 top-10 recall `100%`、最佳得分差 `0 dB`；分层搜索只做 298 次物理评估，
对比穷举的 6,561 次，在同一环境中约 `3.08 s` 对 `70.48 s`，峰值 Python
跟踪内存约 `1.88 MiB` 对 `39.53 MiB`。环境指纹为
`a83c101a138f4861d0a8c3c0cc232b3ea51f1403b2dc52630281b4525dd36fab`，产物见
`artifacts/benchmarks/optenni-four-port-search-recall.json`。该证据只证明此小目录、
每端口最多 1 元件的搜索召回与缩放性，不代表完整采购目录的全局最优保证。

2026-07-15 已从正在运行的 Optenni Lab 4.3 定位官方工程，并由只读 `.opr` XML 解析器
自动提取（而非人工抄写）
`multiantenna_project.opr` 保存首选方案：P1=`SCPL`（1.0 pF GQM1885 +
2.0 nH 0402HP）、P2=`PCSL`（1.0 pF + 5.6 nH）、P3=`PCSL`
（3.0 pF + 5.6 nH）。Optenni 三个频段显示的 min/average 总效率分别为
`-1.7/-1.1`、`-2.2/-1.4`、`-2.1/-1.6 dB`；核心用完全相同的 DUT 与六个
实测 S2P 复算后，所有值相对一位小数 UI 的误差均不超过 `0.049 dB`。
在 1.24 GHz 驱动 P3 时，反射功率和辐射功率相对 UI 分别只差 `0.011%` 和
`0.049%`，总功率严格为 1。Optenni 与核心对 component-loss/coupling 的分类边界
约有 `0.5%` 差异，但两项合计在 `0.2%` UI 舍入容差内一致，因此报告明确区分
“总效率已对齐”和“损耗分类口径仍不同”。机器可复现产物见
`artifacts/benchmarks/optenni-multiport-live-golden.json`，生成命令为
`python scripts/run_optenni_multiport_live_golden.py`。解析器同时认证工程为 Optenni 4.3、
`hasResults=1`、内嵌 1001 点三端口数据、101 个候选（100 个匹配解）以及首选索引 1；
独立清单见 `benchmarks/optenni_exports/multiantenna_project_opr_manifest.json`，生成命令为
`python scripts/extract_optenni_opr_manifest.py`。OPR 没有保存权威结果曲线，因此清单不会
把核心复算冒充成 Optenni 逐频导出。

在数值重放之外，`optenni-saved-winner-discovery.json` 进一步验证自动发现能力：搜索只获得
同一 DUT、三个频段以及 saved BOM 所属的 2 个电感/2 个电容实测模型，不注入参考 placements。
固定 `SCPL / PCSL / PCSL` 拓扑时，77 次物理评估即把精确六料号网络排在第 1；完全不限制
拓扑、`per_port_keep=8` 时，675 次物理评估仍同时把参考拓扑和精确料号排在第 1，得分差
为 `0 dB`。单独诊断确认宽度 7 在该 saved-winner 案例也能召回，但产品继续采用宽度 8，
因为另一项 729 候选 exhaustive 基准证明 8 才能保持完整 top-10 recall。这是四料号参考网格
上的搜索证据，不外推为完整目录全局最优证明。

完整参考模型集合进一步暴露并关闭了独立端口 shortlist 的强耦合缺口。旧精修即使固定正确
拓扑并加载全部 145 个模型，仍因逐元件贪心更新而比 saved winner 差 `1.13 dB`。核心现在
可在完整矩阵上联合优化多端口理想值，并对每个两元件端口执行块坐标 beam：beam 会保留
暂时变差的端口级移动，跨端口迭代后再比较；邻域按“不同标称值 × 同值实测型号”构造，
避免同值料号挤占值域。41 个电感 + 104 个电容模型的正式基准完成 3368 次物理评估、只加载
54 个模型，Optenni 精确 BOM 保留在 rank 17，同时找到拓扑和值相同、仅 P2 电容实测型号
不同且核心目标高 `0.065 dB` 的方案。该 thorough 路径只在所有端口明确拓扑、每端口不超过
两元件且预算至少 60 秒时启用；产物为 `optenni-saved-winner-full-catalog-discovery.json`。

同一案例也已从产品统一 `run_tuning_joint` 服务入口端到端重放，而非直接调用核心：目录适配
保留 41L+104C 及同标称值实测型号的身份和容差，修正矩阵顺序后的单进程搜索在 `92.29 s`
内完成 4173 次物理评估、懒加载 55 个模型，无兼容回退或预算截断，并把 Optenni 精确 BOM 排在第 17；最优
替代方案高 `0.065 dB`，功率闭合误差为 0。产品按请求边界精确插值，离线核心基准取最近
实测频点，因此 saved BOM 的产品分数为 `-2.198916 dB`，与核心基准相差 `0.000687 dB`，
两者不可混用为同一采样口径。产物为 `optenni-product-saved-winner-constrained.json`。

完全不提供拓扑白名单时，独立端口 top-8 会把参考拓扑在 P0/P2 以单端口诊断排名
41/31 提前删除；简单扩大 shortlist 或按独立料值拼接拓扑均不能解决。自动超深度路径现在
为每端口保留最多 13 种 0–2 元件拓扑代表，在完整矩阵上筛选组合，保留端口×拓扑多样性，
再围绕耦合最优拓扑搜索所有单端口邻居，并对未用满深度的增长邻居做强优化。无提示正式
基准用 24545 次廉价理想评估和 6295 次实测物理评估，在 `173.41 s` 内把正确拓扑排第 1、
精确 BOM 排第 17，并复现高 `0.065 dB` 的替代型号。产品仅在最多三端口、每端口最多两
元件且预算至少 150 秒时启用；120 秒交互档位不宣称完整目录自动拓扑召回。产物为
`optenni-saved-winner-full-catalog-automatic.json`。

同一无提示路径已通过统一产品 `run_tuning_joint` 入口复验：完整目录转换、服务预算选择、
结果对象映射和诊断均参与执行。分阶段计数显示原路径主要成本为 2196 次联合排名和 3705 次
端口块精修；自动路径将精修 beam 从 8 校准为 4，同时保留受约束路径的 beam 8，最终在
`137.27/150 s` 内以 4601 次物理评估、69 个懒加载模型无截断完成；正确拓扑仍为第 1、
精确 BOM 提升到第 15，产品采样口径下最优替代高
`0.065135 dB` 且功率闭合误差为 0。产物为
`optenni-product-saved-winner-automatic.json`，它是 `Automatic topology deep` UI/API 档位
的端到端证据，而不仅是核心内部能力证明。

产品多端口 `/tuning` 现优先调用同一个全矩阵 `rfmatch_core` 实测 S2P 求解器，
支持逐端口 0–4 元件限制，并把完整耦合、DUT 吸收、器件损耗、隔离、曲线与 BOM
转成统一结果；旧 `JointMultiPortOptimizer` 只在目录或请求不受支持时作为带明确原因
的兼容回退。UI、HTML 与 PDF 会把召回率标成“参考校准而非当前目录证明”，同时显示
上述 saved-winner 数值 golden 的适用范围。

产品 `timeout_seconds` 现为真正的合作式软截止，而不再只是一个未执行的请求字段。
搜索会先建立功率守恒的无网络基线，按预算选择理想种子档位，优先完成所有端口后再做
全耦合联合排名；截止时保留已完成候选并返回 `search_truncated`、终止阶段和搜索档位。
官方三端口、原始扫描 49 个电感与 322 个电容、每端口最多 2 个元件的实测中，旧路径
请求 10 秒实际运行 `34.34 s`；新路径在 `10.19 s` 返回 10 个候选，三个端口均有
网络，最佳得分 `-4.361 dB`，对比 34 秒完整搜索 `-4.032 dB`，功率误差为 0。
产物见 `artifacts/benchmarks/optenni-product-joint-time-budget.json`。

多端口检查点另有完整真实目录对照，使用同一官方三天线 DUT、49 个 0402HP 电感与
322 个 GQM18 电容，三个端口各允许 0–1 个元件。5 秒初始运行在 377 次物理评估后于
joint ranking 截断，得分 `-4.5302 dB`；追加预算复用全部逐端口子问题和精确缓存，约
3.02 秒完成到 `-4.4569 dB`。独立 20 秒冷启动约 8.15 秒，最终三端口拓扑、得分、
577 次唯一物理评估、16 个加载模型和功率守恒完全一致，增量耗时约为冷启动的 37.1%。
产物见 `artifacts/benchmarks/real-catalog-joint-continuation-checkpoint.json`，SHA-256 为
`4A05B58BEC604FD2A85A8E2EA31304509C1DCFB42A370D84B95D8A6A6C908995`。

0–2 元件/端口的完整真实目录缺口也已关闭。为使严格完成性对照可在门禁环境复现，该基准
对每个端口频段使用 2 个确定性采样点、beam=8。8 秒部分运行完成 518 次物理评估并得到
`-4.3692 dB`；追加阶段约 2.68 秒完成到 `-4.13537 dB`。独立 28 秒配置冷启动约
10.44 秒，最终三端口拓扑、得分、722 次唯一物理评估、55 个加载模型及功率守恒完全一致，
增量耗时约为冷启动的 25.6%。产物见
`artifacts/benchmarks/real-catalog-joint-two-component-continuation.json`，SHA-256 为
`790804F5A0AF2ACCE5BFB16BC415A144F09004B72F0C429B3D67B646BD68FC6D`。这证明检查点完整性，
不把 2 点/频段的优化质量外推为高密度频率扫描的全局最优证明。

隔离约束验证：对端口 1 驱动、端口 2 接收的 `S21` 在 2.50–2.69 GHz 设置 `≤ -20 dB`。无约束理想方案最差为 `-9.11 dB`；约束优化后最差为 `-20.35 dB` 并通过，完整结果见 `artifacts/benchmarks/optenni-ideal-isolation-baseline.json`。命令格式示例：

```powershell
python scripts/run_optenni_baseline.py --case multiantenna-different-bands `
  --isolation-target 1:2:2.5e9:2.69e9:-20
```

产品端到端验证使用官方 `3_antennas.s3p`、三个启用端口、Murata SQLite 实测器件和 `S21 ≤ -20 dB`：优化在约 26.8 秒内返回 10 个候选，最佳方案 `S21=-23.68 dB`，界面显示逐目标 Worst/Average/PASS。产品搜索已从每个两元件拓扑约 4.5 万次盲目穷举改为有界粗搜与局部精修，并将局部拓扑集合与核心/Optenni 统一。

## 多阻抗配置基线

官方 `free_space.s1p / cover.s1p / cover_w_spacer.s1p` 使用教程规定的 `2.400–2.483 GHz` 与 `2.500–2.690 GHz` 双频段。核心为三个环境保持完全相同的有序网络和料号，并以 dB 域的真实最差场景/加权平均场景评分：

- 理想 LC 自动综合最差效率为 `-0.9044 dB`，5945 次评估，约 5.2 秒；结果见 `artifacts/benchmarks/optenni-multi-scenario-ideal-baseline.json`。
- Coilcraft 0402CS / Murata GJM15 实测 S2P 自动综合得到 `GJM1552C1H5R6DB01 5.6 pF` 串联、`04CS4N3 4.3 nH` 并联（候选顺序为 DUT 向外），与教程电路完全一致；最差效率 `-0.9983 dB`，而 Optenni 教程显示约 `-1.0 dB`。
- 实测基准完成 1967 次理想种子和 1910 次物理评估，最大功率平衡误差为 `0`，平均器件损耗 `1.40%`；结果见 `artifacts/benchmarks/optenni-multi-scenario-measured-baseline.json`。
- 完整产品基准加载 43 个 Coilcraft 0402CS 与 882 个 Murata GJM15 文件模型，不注入参考拓扑。
  45 秒搜索先完成 2352 次理想排名，让全部 16 个二元件拓扑各取得物理粗筛，再深搜 3 个
  优先拓扑并执行 140 次完整目录邻域精修。20 个候选最终在 82 个独立频点上复算重排：教程
  `Series-C / Shunt-L` 拓扑第 1、5.6 pF / 4.3 nH 参考值网络第 2；最佳 5.3 pF / 3.9 nH
  最差效率为 `-0.99605 dB`，相对实测核心基线仅 `+0.00229 dB`，功率误差为 0。搜索约
  45.08 秒、密集验证约 4.23 秒，产物见
  `artifacts/benchmarks/optenni-product-multi-scenario.json`。
- 产品端现与主调谐流程共用 Quick / Balanced / Thorough / Automatic topology deep 命名档位；命名档位会锁定时间、beam 和频带采样点，手动改参数则明确标记为 Custom。结果同时公开物理评估次数、实际构建的器件模型数、时间预算占用和是否截断。器件 S2P 模型按本次频率网格缓存，不再为每个场景候选重复构建。

该结果证明共享网络语义、S2P 物理模型、频段最差效率和教程参考电路已在名义值层面对齐。仍需 Optenni 导出的逐频 CSV 才能把图表级接近升级为逐点 golden 误差认证。

## 可调电容 MDIF 基线

官方 `Variable_capacitor_Tutorial_antenna.s1p` 与
`Variable_capacitor_tutorial.mdif` 已接入独立核心、产品 API 和 Web UI。
MDIF 解析支持多状态元数据、RI/MA/DB 数据格式和频率单位换算；每个配置在同一
实测固定网络下独立选择状态，并统一计算 total efficiency、器件损耗与功率守恒。

- 固定网络按 DUT 向外为串联 `GJM1555C1H2R8WB01 2.8 pF`、串联 `04CS15N 15 nH`。
- Set 1（704–746 + 1920–2170 MHz）自动选择 `8 pF`，得分 `-1.2788 dB`。
- Set 2（791–862 + 1920–2170 MHz）自动选择 `2 pF`，得分 `-1.6336 dB`。
- Set 3（880–960 + 1920–2170 MHz）自动选择 `1 pF`，得分 `-1.7328 dB`。
- balanced 总评分 `-1.6406 dB`，平均 total efficiency `85.0%`，平均器件损耗
  `2.41%`，最大功率平衡误差为 `0`。
- 基准产物记录输入与器件 SHA-256，见
  `artifacts/benchmarks/optenni-variable-capacitor-baseline.json`；浏览器端到端已验证，
  控制台无错误。

自动综合模式在每一次理想网络评分中动态选择各 configuration 的
最佳 MDIF 状态，再将连续值吸附到完整 0402CS/GJM15 实测目录，最后只对排名靠前的
候选执行全频物理复核。官方案例的确定性结果为：

- `2423` 次快速联合理想评估、`7` 次 tuner-state 预计算、`16` 次完整实测物理复核；
- 自动召回串联 `2.8 pF + 15 nH` 以及 `8 / 2 / 1 pF` 状态映射；
- 精确评分 `-1.6405996731 dB`，与固定官方电路复算完全一致；
- 独立核心基准约 `22.34 s`；冷启动产品 HTTP 全链路约 `22.82 s`，最大功率平衡误差 `0`；
- 产品适配层把 361 个 DUT 点严格裁剪为 46 个活跃频段点，粗搜每个 configuration 最多使用 18 个确定性频点，最终候选仍在全部活跃频点与全部 MDIF 状态上复核；
- 可复现产物见 `artifacts/benchmarks/optenni-variable-capacitor-synthesis-baseline.json`。

产品 API/UI 已提供自动综合开关，并返回理想评估数、精确物理评估数、加载模型数和
最终固定网络；结果页直接显示 DUT 向外的连接顺序、实测料号、标称值和搜索统计。
后台 job API 与 UI 轮询已实现 `queued → tuner_state_precompute → ideal_search →
measured_shortlist → exact_physical → complete` 阶段进度；真实 HTTP 验证在约 `26.97 s`
完成并返回官方解。中途取消在 `ideal_search` 阶段进入 `cancelled`，结果为空，证明取消
已传递到核心而非仅停止界面等待。

## 多路开关 MDIF 基线

官方 `10.6 Impedance tuning using a switch` 案例已完成教程 PDF 视觉核验、N 端口
MDIF 解析和确定性物理重放。解析器现在保留 `000`、`all off`、`RFC-RF1` 等分类
状态，不再把状态强制转换为浮点数，并支持换行存储的任意 N 端口 ACDATA 记录；原有
二端口 variable-C 接口和测试保持兼容。

- SP3T 教程最佳拓扑 #1（第 12 页）：输入依次串联 `15 nH / 2.0 pF`，三个 throw
  分别串联 `1.0 nH / 2.6 pF / 1.2 pF`。这套方案此前未进入基线，现已补齐。
- SP3T 教程简化拓扑 #4：输入串联 `13 nH`，三个 throw 分别串联
  `2.3 / 1.2 / 0.8 pF`，Set 1/2/3 使用 `100 / 010 / 001`。
- SP2T 优化拓扑 #5：输入串联 `13 nH`，两个 throw 分别串联 `1.2 / 0.8 pF`；
  Set 1 同时闭合两路、等效 `2.0 pF`，Set 2/3 分别闭合 RF1/RF2。
- 求解保留 throw 之间的完整 S 参数耦合：先用天线终止 RFC，再为各 throw 加串联
  阻抗，最后在 Y 参数域按等电压、总电流相加合并支路。
- 两套拓扑的三配置全频重放约 `0.25 s`；SP2T 六个频段中心点已建立
  `1e-9 dB` 容差的黄金测试。
- 教程 PDF SHA-256、视觉核验页码、三类设计角色、输入 SHA-256、每配置状态、频段
  最差/平均回损与中心频点回损见
  `artifacts/benchmarks/optenni-switch-tuning-baseline.json`。

这一阶段证明“真实多端口开关读取 → 状态连接 → 共享支路网络 → 逐频 S11”数值链路已经
闭合。

开关自动综合的第一阶段也已完成：每次候选评分都在完整耦合模型上比较允许的全部状态，
并在连续对数域联合优化所有 throw 元件和共享输入元件。状态约束既可固定为教程 SP3T 的
`100/010/001`，也可像 SP2T 一样留空，让三组配置自由选择四种状态。

- SP3T 自动枚举 `2^3=8` 种支路 C/L 组合，选中 CCC，并得到约
  `12.84 nH + 2.40/1.28/0.88 pF`，状态为 `100/010/001`。
- SP2T 自动枚举 `2^2=4` 种支路组合，选中 CC，并得到约
  `12.45 nH + 1.32/0.91 pF`；自动状态映射为
  `all on / RFC-RF1 / RFC-RF2`，与教程拓扑 #5 一致。
- 搜索只使用 46 个真实活跃频点，预计算 SP3T 8 状态、SP2T 4 状态；加入完整波功率
  重建后的两套自动综合基准合计约 `13.97 s`，见
  `artifacts/benchmarks/optenni-switch-tuning-synthesis-baseline.json`。
- 产品 `mode=switch` 已接入统一后台 job、阶段进度与协作式取消。真实 HTTP SP2T 全链路
  约 `2.07 s`；SP3T 在 `switch_topology_search` 阶段取消后状态为 `cancelled`、结果为空。
- Web 的 Switch States 模式已提供 MDIF 路径、多频段配置、每配置可选状态约束，以及
  throw/input 网络结果卡片。
- 完整 RFC/throw 功率波已重建：输入 accepted power 被分解为 DUT absorbed power 与
  switch loss，并逐频计算守恒误差。官方 ideal MDIF 存在约 `1e-4` 量级的微小非无源
  数值增益，系统保留带符号的模型功率并单独报告 non-passivity，而不是裁剪后掩盖问题。

共享输入 Synthesis block 和真实料号复核也已完成：

- 输入块覆盖有序的 0–2 元件 `series/shunt × L/C`，共 `1 + 4 + 16 = 21` 种拓扑；
  两级 topology beam 先筛 throw 类型，再与输入拓扑交叉粗搜并对领先组合多起点精修。
- 通用节点 Y 求解器支持 throw 和输入位置的真实二端口 S2P；ESR、自谐振、fixture
  非对称、开关耦合及每个器件的实功率损耗同时参与评分。
- 教程简化 #5 的实测复核精确召回 `GJM15 1.2 pF / 0.8 pF + 0402CS 13 nH`，
  54 个物理候选，平均匹配网络损耗 `3.56%`，见
  `artifacts/benchmarks/optenni-switch-tuning-measured-baseline.json`。
- 完整 0–2 元件实测综合完成 2799 次理想评估、198 次完整物理复核并加载 27 个 S2P
  模型，约 `27.62 s`。最佳两元件输入块为 `shunt 0.3 pF → series 9 nH`，两条
  switch 支路为 `0.9 / 1.4 pF`，得分 `-1.6361 dB`；功率误差 `8.0e-13`。
- 产品保留按输入复杂度分组的最佳方案：0/1/2 个输入元件分别约为
  `-5.0202 / -1.7416 / -1.6361 dB`，让用户在最简 BOM 与最佳性能之间选择；完整产物见
  `artifacts/benchmarks/optenni-switch-full-network-measured-baseline.json`。
- API 和 Web 现在为这些独立物理解标注 `best_performance / simplest_bom /
  performance_complexity_compromise`，显示相对最佳解的 dB 差值，并把官方教程页码与 PDF
  哈希作为“仅供参考、非当前请求证明”的校准元数据透出，避免把单一冠军误当成唯一产品答案。
- 真实产品 HTTP 后台任务启用 measured refinement 与 0–2 元件搜索后约 `28.18 s` 完成，
  返回 2799 次理想评估、198 次物理复核、27 个加载模型、三档复杂度方案和
  `7.98e-13` 最大功率误差；产品端使用当前元件目录的最优容差后缀料号，数值与独立
  benchmark 一致。
- 三档复杂度候选现已提升为独立 `TuningResult`：每档重新执行对应开关状态、实测元件
  与完整波功率计算，拥有独立评分、效率、损耗、网络和带内曲线。排名表与复杂度表均可
  切换当前方案，选择状态经 `/api/tuning/select` 同步到会话，并随项目候选列表和
  `selected_index` 保存恢复；Switch 曲线使用已保存的全物理带内数据，不经过普通 LC
  sweep 近似。
- full-network benchmark 现保存 Set 1/2/3 各自的逐频 `S11`、total/accepted
  efficiency、switch loss 与 matching-network loss（分别 30/33/35 个带内频点）。
  `validate_optenni_switch_export.py` 可读取 Optenni 常见 GHz/MHz/Hz、S11 dB 和
  total-efficiency 线性/百分比/dB 列，插值到相同频点后报告最大与 RMS 误差；默认门槛为
  `0.05 dB` 与 `0.005` 绝对效率。当前仍缺的是 Optenni 端实际导出的三组曲线文件。

开关案例剩余的关键工作是用 Optenni 导出的逐频 CSV 做跨软件误差认证。

## 阶段计划

### 阶段 A：可信基线

完成官方单端口、宽带和辐射效率案例；建立 golden 导出校验、自动测试和统一 benchmark 报告。

### 阶段 B：核心收敛

以 adapter 让 Web 产品调用 `packages/rfmatch-core` 的网络/evaluator；新旧引擎双跑一致后才删除重复实现。

### 阶段 C：专业匹配能力

完成多端口、多环境、容差良率、真实离散元件和可取消并行搜索。

### 阶段 D：产品闭环

增加项目保存、版本化报告、候选并排比较、任务进度/取消、异常诊断和中文专业术语统一。

### 阶段 E：高级 RF 工作流

在已完成的参数化传输线/Stub 手动评估与自动综合基础上，扩展几何/介质模型和
布局块导入，再推进 EM/VNA 接口和阵列能力。
