# Runtime data

- `Murata/`：供应商 S2P、阻抗数据和 SQLite 元件数据库。
- `snp/`：待优化的 DUT Touchstone 文件及配套效率数据。

可选的料号级温漂/工艺数据放在 `Murata/component_environment.json`，格式见
`docs/COMPONENT_ENVIRONMENT_METADATA.md`。示例文件只演示结构，不能作为真实器件参数使用。

这两个目录默认不进入 Git。需要共享可复现用例时，只把脱敏的小型样本放入专门的测试 fixture。
