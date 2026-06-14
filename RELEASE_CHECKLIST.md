# KairoRAG 发布检查清单

- [ ] 运行 `python -m pytest`。
- [ ] 运行 `python -m ruff check .`。
- [ ] 执行 secret scan，确认没有真实 API key。
- [ ] 运行 `kairo doctor --repo`。
- [ ] 运行 `kairo doctor --release`。
- [ ] 运行 `kairo eval --suite all --fake-providers`。
- [ ] 运行 `kairo eval --compare baseline.json current.json --output-dir data/eval_compare`。
- [ ] 执行 `docker build` 或 Docker Compose smoke test。
- [ ] 确认 README 已更新。
- [ ] 确认 legacy 主路径未恢复。
- [ ] 确认 live tests 默认跳过。
- [ ] 确认 `.env` 未提交。
- [ ] 确认 trace / audit / eval output 未提交。
- [ ] 确认版本号和发布说明正确。
