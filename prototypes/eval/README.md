# Scheduler Evaluation Explorer

这是一个无构建步骤的本地评测结果查看器。服务递归扫描
`experiments/11_scheduler_evaluation/runs/mnms-preliminary` 下含有
`config.yaml`、`results.jsonl` 和 `summary.json` 的完成 run，并从每条完成请求重新计算质量、Resource 和时延的均值与 p95。

启动：

```bash
uv run python prototypes/eval/server.py --host 127.0.0.1 --port 8766
```

也可以用 `--runs-root /path/to/runs` 指定其他数据目录。浏览器打开
`http://127.0.0.1:8766/`。页面支持：

- 最新 run / 全部历史 run 切换，以及 scheduler、请求速率、配置、seed 筛选；
- 汇总表排序和当前表格 CSV 导出；
- 固定请求速率的 scheduler 横向透视、scheduler × 请求速率指标矩阵；
- 当前筛选命中的 run 明细，包含完成和失败计数。

接口：

- `GET /api/data?mode=latest|history&scheduler=sdts&rate=2&config=<options.configs[].id>&seed=7`（config 值使用 URL 编码）
- `GET /api/data.csv`（接受同样筛选参数）
- `GET /api/health`

`config` 是保留配置字段生成的 canonical JSON 身份（页面下拉框显示可读标签），因此刷新后不会因为新增配置而改变既有身份。配置身份保留 dataset、profiling database、scoring context、snapshot 和窗口等字段，同时移除 seed、rate 和输出目录等运行元数据。默认 latest 按 scheduler/config/rate/seed 选择最新时间戳，避免 sdts 多轮重复结果污染统计。Resource 是每请求 `node_timeline[*].gpu_memory_mib` 的求和；失败请求不参与三项指标；缺失值显示为 `—`，真实的 0 会保留。

测试：

```bash
uv run pytest prototypes/eval/test_eval_server.py
```
