# Tests

- `api/`：不依赖真实 DUT 的 HTTP 请求契约和状态回归，可进入持续集成。
- `engine/`：算法单元测试以及依赖真实数据的集成用例。没有对应 SNP 时，只运行不依赖外部数据的测试。
- `benchmarks/`：耗时性能基准，不作为普通单元测试自动执行。
- `manual/`：Optenni 对比、绘图、数据库检查和端到端人工烟测脚本。
- `packages/rfmatch-core/tests/`：独立核心包自身的数值契约，不迁入这里，以保持包可独立测试和发布。

测试统一从仓库根目录执行，并将 `apps/api` 加入 `PYTHONPATH`。
