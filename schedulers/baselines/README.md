# Baseline Scheduler 评估入口

此目录只保存最终可评估的 Scheduler 策略：两种启发式、两种学习方法、两种
LLM 优化方法，各一个独立 `.py` 文件。方法的训练、搜索、提示词和候选选择过程
放在 [`../../experiments/baseline_methods/`](../../experiments/baseline_methods/)，
评估输出写入 `runs/`。具体方法及文件名确定后再添加源码，不使用空策略占位。

每个文件提供 `propose(view)`。`view.candidate_dags` 含三条可重复的候选 DAG；策略根据 `view.system_state` 与 profiling evidence 选择其中一条，再给该 DAG 的每个节点分配 Configuration 和 Compatible Device。返回 `SchedulerOutput(dag, assignments, path_index)`。分配映射格式为：

```python
{
    node.node_id: {
        "configuration_id": "snapshot 中该工具的配置 ID",
        "device_id": "与该配置兼容的设备 ID",
    }
    for node in dag.nodes
}
```

这是接口形状示意，不是一份可直接运行的策略。策略只能用 view 中的三条 DAG、system state 和 profiling evidence 做决策；可信评估器负责校验、计时、模拟和评分。旧 assignments-only 策略默认选择第 0 条 DAG。`scripts/eval.py` 每次调用 `propose` 时都会重新执行该源码文件；文件顶层的模型加载或远程调用也进入 Scheduler Computation Time。评估时应固定数据、split、快照及评分配置，只替换 baseline 文件、版本和输出路径：

```bash
uv run python scripts/eval.py \
  --dataset data/planner-candidates.example.jsonl \
  --profiling-database docs/examples/profiling-database.fake.json \
  --scheduler-source schedulers/baselines/example_method.py \
  --scheduler-version 10 --split test \
  --output runs/example_method-test.jsonl
```

`example_method.py` 是待实现的策略文件名；正式比较时需替换为选定的六个策略及
同一组实际数据和快照文件。所有方法的开发
与选择只使用 train/validation；锁定六个最终文件后再运行 test 比较。除 Composite
Score 外，应分别报告质量、latency、Resource、失败数和三条路径的选择分布。

现有 `schedulers/naive_*.py` 仍为演化实验的三个根 Candidate，`schedulers/oracle_source.py` 为穷举参考。目录及未来动态 DAG 的接口边界见 [`../../docs/evaluation-pipeline.md`](../../docs/evaluation-pipeline.md)。
