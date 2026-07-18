# 料号级环境元数据

S2P 文件描述某一测试条件下的射频行为，不能仅凭 25 °C 模型可靠推导温度系数或生产
偏置。RF Match Studio 因此使用独立、可哈希、精确料号匹配的 JSON sidecar。默认文件名为
`data/Murata/component_environment.json`，也可以在“元件数据源”中指定其他路径。

```json
{
  "schema_version": 1,
  "source": {
    "name": "实验室温箱测试",
    "document": "LAB-RF-2026-014",
    "sha256": "可选的64位来源文件SHA-256",
    "evidence_level": "laboratory_measurement"
  },
  "components": [
    {
      "part_number": "精确厂商料号",
      "tempco_ppm_per_c": 125.0,
      "systematic_bias_pct": -0.4
    }
  ]
}
```

证据等级只允许：

- `manufacturer_datasheet`：厂商数据表或厂商特性数据；必须填写来源文档。
- `laboratory_measurement`：实验室温箱或批次统计结果；必须填写来源文档。
- `engineering_assumption`：明确的工程假设，不得描述成厂商实测事实。

校验规则：料号不区分大小写但必须唯一；每条记录至少提供 tempco 或系统偏置之一；偏置必须
大于 `-100%`；所有数值必须有限；来源 SHA-256 如提供必须是 64 位十六进制。

计算优先级为“精确料号 sidecar > 良率面板中的 L/C 全局回退”。料号级值会随实测 S2P
模型进入固定网络、Switch 与可调器件多状态 Monte Carlo。项目快照同时保存 sidecar 的
SHA-256；恢复工程时哈希不一致会被明确报告，避免悄悄使用不同环境假设。

仓库中的 `component_environment.example.json` 仅为结构示例，料号和数值均不是可用于生产
设计的器件证据。
