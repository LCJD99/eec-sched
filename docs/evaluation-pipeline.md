# Evaluation 实验主链路与重构边界

## 目标链路

```text
任务 / workload
  → Tool Planner（一次 LLM 调用生成 3 条候选 Tool-Call DAG）
  → 离线 JSONL（所有 baseline 复用同一组 Planner 结果）
  → Scheduler（3 条 DAG + 系统状态 + Profiling Database Snapshot）
  → 具体 Tool-Call DAG + 每节点 Configuration / Compatible Device
  → Evaluator（可信的离线回放、校验与统计）
  → latency / Resource / throughput 等指标
```

Planner 只定义工具、依赖和三条可选 DAG，不决定 Configuration 或设备；三条 DAG 可以相同。每个任务的 Planner 结果只生成并保存一次。Scheduler 从中选一条，并决定所选 DAG 的 Configuration 与设备。Evaluator 不接受 Scheduler 自报的耗时、资源消耗或指标；所有 baseline 与演化产生的 Scheduler Candidate 都应经过同一输入、同一 Profiling Database Snapshot 和同一可信评估规则。

## 独立离线 Tool Planner 输入输出接口

Tool Planner 不绑定 MnMS 或任何具体数据集。实现位于
`experiments/09_offline_tool_planning/run.py`，通用批处理逻辑位于
`src/eec_sched/offline_tool_planner.py`。每个任务只调用一次 LLM；不做修复重试。

输入由三个文件和模型配置组成：

- workload JSONL 每行严格符合 [`tool-planner-input.schema.json`](schemas/tool-planner-input.schema.json)，内容是 `{"id": "...", "request": "..."}`；`id` 也可为 JSON 数字并会规范为字符串。解析器拒绝额外字段，以免参考答案或数据集专属字段进入 prompt。
- tool catalog JSON 符合 [`tool-planner-catalog.schema.json`](schemas/tool-planner-catalog.schema.json)，外层是 `{"tools": [...]}`。每个工具提供 `tool_id`、`description`、`inputs`、`outputs`；端口提供 `modality`（`audio`、`image`、`text`），输入端口还可提供 `required`，默认值为 `true`。
- system prompt 可编辑，使用文本文件，也支持包含字符串或 `{"system_prompt": "..."}` 的 JSON 文件。它要求模型只返回 `{"dags": [...]}`，且 `dags` 必须恰好有三项；三项都要完成同一个 request，并在不牺牲任务相关性的前提下尽可能采用不同的合理工具路径，允许确实没有有用替代路径时重复。

可从 [`config.yaml`](../experiments/09_offline_tool_planning/config.yaml)、[`requests.example.jsonl`](../experiments/09_offline_tool_planning/requests.example.jsonl)、[`tool_catalog.example.json`](../experiments/09_offline_tool_planning/tool_catalog.example.json) 和 [`system_prompt.txt`](../experiments/09_offline_tool_planning/system_prompt.txt) 开始配置。

模型配置直接从仓库的 `configs/model/*.yaml` 选择，默认是
[`deepseekv4.yaml`](../configs/model/deepseekv4.yaml)。文件使用 `base_url`、`token`、
`model` 字段；`token` 可以是显式值，也可以写成 `${oc.env:变量名}` 引用环境变量。
运行命令无需分别传模型参数。token 不写入 manifest、失败文件或 stdout。

批处理 harness 先检查原始响应是 JSON 对象且顶层 `dags` 数组长度为三，再使用
`tool-call-dag.schema.json`、`ToolCallPlanCandidates` 和
`validate_plan_candidates()` 校验。这里先做严格形状检查，所以旧的
`plans_from_dict()` 单 DAG 自动复制兼容逻辑不会被新入口接受。任意一条 DAG 无效时，
该任务写入失败 JSONL，批处理继续处理下一条；模型/传输错误同样按任务记录并继续，
只有全局输入、配置或 token 错误直接以非零状态退出。

成功输出是 Scheduler 可消费的 canonical JSONL，每行只有
`trace_id`、`task_input`、`system_state`、`dags` 四个外层字段：

```json
{
  "trace_id": "task-001",
  "task_input": {"request": "..."},
  "system_state": {},
  "dags": [{"nodes": [], "final_outputs": []}, {"nodes": [], "final_outputs": []}, {"nodes": [], "final_outputs": []}]
}
```

实际 DAG 仍须满足正式 DAG schema 的非空要求。失败 sidecar 每行记录 `id`、`stage`、
`reason`；manifest 只记录模型名、去敏 endpoint、prompt 版本、输入与工具目录文件名、
任务计数和成功/失败计数，不保存模型原文或 token。模块始终提供自然语言 `request` 文本输入；
对 request 中带常见扩展名的显式图片/音频路径或 URL，会按出现顺序提供
`image_0`/`audio_0` 输入，并在 task input 中保留原始引用。它不会打开、下载或读取媒体，
校验只使用模态 placeholder；没有显式可识别引用时，不会虚构 image/audio 输入。

批量运行示例：

```bash
uv run python experiments/09_offline_tool_planning/run.py \
  input_path=data/requests.jsonl \
  catalog_path=data/tool-catalog.json
```

默认使用 `deepseekv4.yaml`；改用其他共享模型配置时加 `model=local_qwen` 等
Hydra 配置组选项。`limit=2` 或 `item_id=task-001` 可用于 smoke run；输入和成功输出均保留原始顺序。
`output_dir` 留空时入口自动创建带 UTC 时间戳的新 run 目录；显式指定的目录若已有任一
产物也会被拒绝，避免重跑覆盖已有模型调用结果。

## 现有代码实际走向

| 阶段 | 当前实现 | 当前边界 |
| --- | --- | --- |
| Tool Planner | `experiments/09_offline_tool_planning/run.py` 通过 `offline_tool_planner.py` 对每条 request 一次调用生成并校验三条 DAG；`planning.py` 提供候选 DAG 校验 | 主实验入口读取离线结果，不在每次 baseline 运行时调用 LLM；批量生成脚本已经提供，并将成功、失败和 manifest 分开落盘 |
| Workload | `workflow.py` 读取包含 `dags`、`task_input`、`system_state` 的 JSONL；`EvaluationTrace` 保存三条候选，并按 4:3:3 切分 | 默认 `data/mnms-ground-truth-dags.jsonl` 仍来自 MnMS 参考计划；旧单 DAG 行的复制仅是既有评估数据导入兼容行为，新 Planner 入口会在 harness 前拒绝单 DAG 响应 |
| Profiling | `profiling/snapshot.py` 读取并校验固定 Profiling Database Snapshot；`configs/evidence/` 指定快照 | `SchedulerView.snapshot_evidence` 暴露本 Trace 相关工具、设备及有向传输证据 |
| Scheduler | `candidate.py` 的 `SchedulerView` 暴露 `candidate_dags` / `dags`、`system_state` 与 profiling evidence；`evolution/engine.py` 调用 `propose(view)` | 新策略可返回 `SchedulerOutput(dag, assignments, path_index)`；现有 `schedulers/*.py` 仍只返回分配映射，因此默认选第 0 条 |
| Evaluator | `evaluation/evaluator.py` 校验选中 DAG 属于三条候选、校验配置和设备并计时；`simulator.py` 只模拟所选 DAG；`scoring.py` 计算指标 | 单 Trace 离线模拟，输出 makespan、含 Scheduler Computation Time 的 latency，以及各节点 GPU memory MiB 之和；没有多请求 throughput |
| 实验入口 | `app/run_evolution.py` 用 Hydra 组合离线 DAG 数据集、快照、Candidate 和最终评估；`scripts/eval.py` 可对任意一个 `propose(view)` 源码在指定 split 上评估 | baseline 可用同一输入文件和评估入口比较；在线 Planner 调用仍独立于评估运行 |

`schedulers/naive_fastest.py`、`naive_quality.py`、`naive_resource.py` 是现有根 Candidate。`schedulers/oracle_source.py` 是穷举参考策略，计算成本较高，不应当混同于普通 baseline。`docs/adr/0001-compose-experiments-with-hydra.md` 已规定实验选择由 Hydra composition root 完成；后续增加 Planner、Scheduler 和 workload 配置组时应沿用这一原则。

## 三候选 DAG 数据格式与 Scheduler 接口

每条 JSONL 记录是一项任务。其 canonical 格式见 [`tool-call-plan-candidates.schema.json`](schemas/tool-call-plan-candidates.schema.json) 与 [`planner-candidates.example.jsonl`](../data/planner-candidates.example.jsonl)：

```json
{
  "trace_id": "task-001",
  "task_input": {"text": "..."},
  "system_state": {"devices": {"edge": {"queue_depth": 1}}},
  "dags": [
    {"nodes": [], "final_outputs": []},
    {"nodes": [], "final_outputs": []},
    {"nodes": [], "final_outputs": []}
  ]
}
```

上面仅示意外层字段；每条 DAG 必须符合 [`tool-call-dag.schema.json`](schemas/tool-call-dag.schema.json)，因此实际 `nodes` 和 `final_outputs` 不能为空。`dags` 必须恰好三条，允许重复。`trace_id`、`task_input`、`system_state` 可随离线任务保存；Planner 生成流程的模型、prompt、工具目录和输入计数记录在 manifest 中。新 schema 可离线验证，Planner harness 还会调用 `validate_plan_candidates()` 做工具、端口及无环校验。旧单 DAG JSONL 会被评估导入层兼容读入并复制为三条相同路径，但这条兼容规则不适用于新的原始模型响应。

六种对照方法按两种启发式、两种学习方法、两种 LLM 优化方法规划。方法的训练、搜索和生成过程放在 [`../experiments/baseline_methods/`](../experiments/baseline_methods/)；最终可评估的单文件策略放在 [`../schedulers/baselines/`](../schedulers/baselines/)，提供 `propose(view)`。`view.candidate_dags` 是三条只读 DAG，`view.system_state` 是当前任务附带的只读状态，`view.snapshot_evidence` 是 profiling evidence。返回 `SchedulerOutput`，例如：

```python
from eec_sched import SchedulerOutput

def propose(view):
    index = 0  # 根据 view.system_state 和 profiling evidence 决定 0、1 或 2
    dag = view.candidate_dags[index]
    assignments = {
        node.node_id: {"configuration_id": "...", "device_id": "..."}
        for node in dag.nodes
    }
    return SchedulerOutput(dag, assignments, path_index=index)
```

示例中的配置 ID 和设备 ID 必须替换为 snapshot 中兼容的实际值。Evaluator 校验 `path_index` 与选中 DAG 一致、节点分配只覆盖所选 DAG，并只模拟该路径。重复 DAG 仍可用显式 `path_index` 区分。旧只返回 assignments 的 baseline 继续运行，默认第 0 条路径；这种策略没有比较三条路径。`SchedulerOutput.resource_configuration` 当前只是保留字段，执行器尚未消费它，不应把它当成已实现的资源控制。

离线比较示例：

```bash
uv run python scripts/eval.py \
  --dataset data/planner-candidates.example.jsonl \
  --split all \
  --scheduler-source schedulers/naive_fastest.py \
  --scheduler-version 1 \
  --output runs/naive-fastest-three-dag.jsonl
```

评估 JSONL 包含三条 `candidate_dags`、`selected_path_index`、选中的 `dag`、单独的 `assignments`、状态和指标。对多个 baseline 只替换 `--scheduler-source`、版本和输出路径，保持数据文件、split、快照与评分配置一致。

## 指标及公平比较口径

- **Latency**：现有 `EvaluationReport.latency` = 模拟 makespan + 实测 Scheduler Computation Time；同时保留两者明细。所有 baseline 使用同一计时边界。
- **Resource**：现有 `Resource` 是所选节点 `gpu_memory_mib` 的加总，单位 MiB，不能直接称为货币成本、峰值显存或能耗。若要报告 resource cost，须另外声明设备占用时长、计价规则和聚合方式。
- **Throughput**：当前单 DAG / 单 Trace 结果不足以推断吞吐量。需要固定任务到达序列、并发度、初始系统状态、设备及链路排队规则；再报告完成任务数除以观测窗口，以及失败率和尾延迟。
- **Accuracy**：当前 `Composite Score` 默认给质量、时延、Resource 各 1/3 权重。第一阶段可以在同一实验配置中设 `accuracy_weight: 0`，但评估器仍会查询质量 profile；若希望完全取消质量证据依赖，需要另作契约修改。原始指标应始终单独报告，不只给 Composite Score。

## 建议实施顺序

1. **完成离线 Planner 生成工序**：按 `experiments/09_offline_tool_planning/run.py` 的通用接口，为每项 request 只调用一次 LLM，生成并逐条校验三候选 DAG，写入共享 JSONL；保存模型、prompt 版本、输入文件名及失败原因。任何具体数据集（包括 MnMS）都通过 `id`/`request` JSONL 接入，参考答案不能作为 Planner 输入。
2. **迁移可比较的 baseline**：新策略使用三候选接口；现有根 Candidate 暂保留兼容路径，并明确它们始终选第 0 条。比较结果需单独报告路径选择分布和各原始指标。
3. **完善状态与资源语义**：定义 `system_state` 的标准字段、资源配置的可用选项及可信校验，必要时让状态参与模拟；保持 profiling snapshot 与状态证据可复放。
4. **多请求离线评估**：增加固定到达序列与跨任务设备 / 链路队列，定义 throughput、尾延迟及真正的 resource cost；此阶段再确定任务正确性指标。

三候选 DAG 的数据和选路接口已经接入；第 1、3、4 步仍待实现，当前单任务分数不能当成多请求吞吐量结果。
