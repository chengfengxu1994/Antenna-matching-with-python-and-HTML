# PCB 微带几何模型与适用边界

## 数值链路

产品的 Transmission Line 模式可以只搜索抽象的 `Z0 + 电长度`，也可以启用
`Convert and score as manufacturable microstrip`。启用后，每次候选评估均执行：

1. 在制造线宽上下限内反解目标特性阻抗对应的实际线宽；
2. 根据有效介电常数把目标电长度转换为物理长度；
3. 在每个优化频点重新计算频散后的特性阻抗与传播常数；
4. 分别计算铜导体损耗和介质损耗；
5. 把微带作为物理二端口/Stub 写入节点图，重新计算 S 参数、总效率和功率守恒。
6. 良率分析中分别抽取线宽、物理线长、基板厚度和相对介电常数，再从几何层重算
   阻抗、频散、相位与损耗；同一制造样本在所有候选状态间共享。

因此，PCB 模式不是在理想传输线结果旁边附加一个线宽标签，而是用实际几何模型
参与候选排序。损耗感知目标可能主动选择比无损理论值略短的走线，以换取更高总效率。

## 采用的模型

- 准静态特性阻抗、有效介电常数和有限铜厚修正：Hammerstad–Jensen，
  *Accurate Models for Microstrip Computer-Aided Design*，IEEE MTT-S 1980，
  [DOI 10.1109/MWSYM.1980.1124303](https://doi.org/10.1109/MWSYM.1980.1124303)。
- 频率相关有效介电常数与阻抗：Kirschning–Jansen，
  *Accurate model for effective dielectric constant of microstrip with validity up to millimetre-wave frequencies*，
  [DOI 10.1049/el:19820186](https://doi.org/10.1049/el:19820186)。
- 导体损耗：Wheeler 增量电感/趋肤效应方法，并使用 RMS 铜粗糙度修正。
- 实现交叉验证：[scikit-rf MLine 官方模型](https://scikit-rf.readthedocs.io/en/latest/api/media/generated/skrf.media.mline.MLine.html)
  与 [Qucs 技术文档](https://qucs.github.io/docs/technical/technical.pdf)。

核心只依赖 NumPy；scikit-rf 用作独立 golden 生成来源，不是运行依赖。

## 当前包含

- 均匀、各向同性、单层介质上的微带；
- 有限铜厚；
- Kirschning–Jansen 微带频散；
- 铜电阻率、趋肤深度和 RMS 表面粗糙度；
- 介质损耗角正切；
- 直通线、开路 Stub 和短路 Stub；
- 线宽限制、几何反解、密集扫频和逐频功率守恒。
- 可复现的线宽、线长、板厚、介电常数制造容差；`width_tolerance_pct` 对齐
  Optenni `msWidthToler` 的物理线宽语义，而不是误作电长度公差。

## 当前不包含

- 阻焊层、覆盖膜或多层混合介质；
- 玻纤编织和介电各向异性；
- 共面侧地、有限地平面或接地过孔栅栏；
- 弯折、T 接、焊盘、连接器 launch、过孔和参考面转换；
- 开路端延伸、端部辐射和一般不连续结构；
- 铜箔 Huray/球模型；
- 由板厂工艺或实测提取的宽带复数 Dk/Df。

涉及上述结构时，应导入 EM/VNA 生成的 S2P 布局块，或在后续全波插件中验证。
FR-4 的 Dk/Df 随频率、树脂含量和玻纤结构变化；默认 `Er=4.5, tanδ=0.02`
仅是工程示例，正式设计必须使用板厂 stackup 数据或测试提取值。

## 回归证据

`artifacts/benchmarks/microstrip-geometry-baseline.json` 固化：

- 1、2.45、10 GHz 的 scikit-rf/Qucs 独立参考值；
- 2.45 GHz、FR-4 示例的 50 Ω 线宽与 90° 物理长度；
- 铜损、介质损耗、传递功率和功率闭合；
- 100 Ω 负载的损耗感知自动综合；
- ±10% 线宽和 Dk 4.3/4.7 的方向性灵敏度。

Optenni UWB 教程工程的机器提取证据固化于
`benchmarks/optenni_exports/uwb_microstrip_tolerance_settings.json`：工程启用传输线综合、
包含基板，且 `msWidthToler=20%`。该 OPR 只证明设置口径，不包含权威逐频容差包络，
因此当前不宣称容差曲线数值已与 Optenni 完全一致。

运行：

```powershell
python scripts/run_microstrip_geometry_benchmark.py
```
