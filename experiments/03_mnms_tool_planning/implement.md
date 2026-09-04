# 实验 03 实现方案

## 结论

现有代码可以覆盖大部分主链路，不需要重新实现一套 Planner 或 DAG 类型。仓库级
Tool-Call DAG JSON Schema 已固定在 `docs/schemas/tool-call-dag.schema.json`。实验后续
只需新增固定 revision 的 MnMS 数据加载、Query 到请求输入清单的适配、运行产物与
一层可测试的实验入口。

## 已检查、可直接复用的模块

| 现有位置 | 直接用途 | 结论 |
| --- | --- | --- |
| `src/eec_sched/mnms_tools.py::mnms_tool_specs()` | 提供当前项目支持的 28 个工具及端口模态 | 直接复用，禁止在实验中复制工具表 |
| `src/eec_sched/domain.py::ToolCallPlan` | Scheduler 使用的 DAG Python 类型 | 作为最终内存对象 |
| `src/eec_sched/openai_compatible.py::OpenAICompatiblePlannerClient` | OpenAI-compatible HTTP 调用和 Planner 接口 | 复用调用方式；实验层补充严格 schema、记录和修复循环 |
| `src/eec_sched/openai_compatible.py::plan_from_dict()` | JSON DAG 转 `ToolCallPlan` | 直接复用 |
| `src/eec_sched/planning.py::validate_plan()` | 检查工具、端口、模态、环和 final output | 作为主要确定性校验 |
| `docs/schemas/tool-call-dag.schema.json` | 固定 `ToolCallPlan` 的 JSON 形状 | 作为唯一正式 DAG JSON Schema，实验不复制 |
| `src/eec_sched/evaluation/evaluator.py::evaluate_scheduler_instance()` | 证明 `ToolCallPlan` 能进入 Scheduler/可信评估路径 | 用合成 profiling snapshot 做兼容检查，不把分数当实验结果 |
| `docs/examples/profiling-database.fake.json` | 覆盖全部当前工具的三设备开发快照 | 只用于 Scheduler 输入兼容检查 |

不直接复用 `external/mnms/mnms/run_plan_agent.py`。它依赖 AutoGen，输出 MnMS 自己的
plan 格式，使用官方 33 工具语义，也没有生成本项目的 `ToolCallPlan`。把它包一层会
同时维护两套格式，反而增加转换和校验风险。

## 已发现的缺口

1. `OpenAICompatiblePlannerClient` 要求调用者先提供 `PlanningRequest.inputs`，而 MnMS
   数据行只直接给出自然语言 Query 和参考 plan。
2. 当前 validator 用运行时对象识别模态：文本是 `str`、图片是 `PIL.Image`、音频是
   `Path`。本实验不下载真实媒体，因此需要仅用于规划校验的 typed placeholder。
3. Hugging Face 数据集仓库的多个 JSON 文件列不完全一致；当前 Dataset Viewer 也
   显示整库自动推断失败。因此实现不能依赖“扫描整个仓库再猜 schema”。

## 建议目录

```text
experiments/03_mnms_tool_planning/
├── prd.md
├── implement.md
├── run.py                   # 唯一 CLI 入口
└── tests/                   # 只放该实验的本地小 fixture（不得含参考 plan 泄漏到 prompt）
```

运行结果默认写到被 gitignore 的目录：

```text
experiments/03_mnms_tool_planning/runs/<run-id>/
├── manifest.json
├── dags.jsonl
└── failures.jsonl
```

## 唯一实验入口

建议 `run.py` 对外只提供一个主流程：

```python
run_experiment(config) -> RunSummary
```

CLI 和测试都走这个接口。内部可以拆函数，但不新增暂时没有第二种实现的公共抽象。

主流程按下面顺序执行：

1. 读取显式指定的 MnMS revision 和 split 对应的单个 JSON 数据文件。
2. 按 `id` 或稳定顺序选择样本。
3. 只取 `id` 和 `user_request` 进入规划输入；参考 plan 字段留在 loader 外部。
4. 从 Query 构造请求输入清单。
5. 调用 Agent 生成 JSON DAG。
6. 依次使用 `docs/schemas/tool-call-dag.schema.json` 做 JSON Schema 校验，再执行
   `plan_from_dict()` 和 `validate_plan()`。
7. 用开发 profiling snapshot 和确定性合法 assignment 调用可信评估入口，确认 DAG
   确实能作为 Scheduler 输入。此步的 Composite Score 不保存为研究结论。
8. 只将全部通过的记录追加到 `dags.jsonl`；其余写入 `failures.jsonl`。

## MnMS 数据加载

默认读取：

```text
repo_id: zixianma/mnms
revision: e9cb0112a0a88788fc7559e27cdf40646f30ad3f
file/split: test_human_verified_filtered.json
```

revision 必须是配置项并写入 manifest，不能默默使用浮动的 `main`。实现优先通过
`huggingface_hub` 下载这个明确文件并复用其本地 cache；解析器只接受预期的 JSON
对象列表或 JSONL。若上游文件格式改变，直接报出 `dataset_load` 失败。

测试使用极小本地 fixture，不访问网络。另设一个显式的集成测试或 smoke command
验证官方文件，避免普通单元测试受网络影响。

## Query 输入适配

适配不能读取参考 plan。v1 使用下面的确定性规则：

- 始终暴露 `query` 文本输入，其值是完整 `user_request`；
- 仅从 `user_request` 本身识别带常见扩展名的媒体引用；
- 图片引用暴露为 `image_0`、`image_1`，音频引用暴露为 `audio_0`、`audio_1`；
- 原始文件名保存在 request input manifest 中；
- 给 `validate_plan()` 时，图片使用内存中的最小 PIL placeholder，音频使用 Path
  placeholder，文本使用真实字符串；这些 placeholder 只证明模态，不伪装成真实媒体；
- Query 没有足够信息支撑合法输入时，记录 `unsupported_input`，不得读取参考 plan
  来补答案。

这一步保证 Scheduler 获得正确依赖图，但不承诺 DAG 已可直接执行真实媒体。真实
工具执行明确不属于实验 03。

## Agent 调用与修复

Agent 的 system instruction 固定要求：

- 只输出 schema 允许的 JSON；
- 只能使用提供的工具 ID 和端口；
- 输入来源只能是 request input 或已生成节点的 output；
- 不输出 Configuration、Device 或解释文字；
- node ID 稳定且在当前 DAG 内唯一；
- 至少声明一个 final output。

首个版本允许最多 2 次修复。修复请求只带确定性校验错误，例如
`unknown_tool` 或 `modality_mismatch`，不带参考 plan。重试耗尽后写失败记录。

保留现有 OpenAI-compatible 传输方式，不引入 OpenAI SDK。API key 只从指定环境变量
读取；日志保存 endpoint 的非敏感标识、模型名、attempt 数、耗时和 prompt version，
不保存 Authorization header。

## 正式 DAG Schema 与 Scheduler 兼容

`docs/schemas/tool-call-dag.schema.json` 是仓库中唯一正式的 Tool-Call DAG JSON
Schema，使用 JSON Schema Draft 2020-12，只描述现有 `ToolCallPlan` 的 JSON 投影：

- 顶层仅允许 `nodes`、`final_outputs`；
- node 仅允许 `node_id`、`tool_id`、`inputs`；
- input source 仅允许 `kind`、`name`，`kind=node` 时要求 `port`；
- final output 仅允许 `node_id`、`port`；
- 所有对象使用 `additionalProperties: false`。

Schema 有意不重复实现工具存在性、端口模态、引用目标、节点 ID 唯一性和无环性；
这些跨记录规则继续由 `validate_plan()` 负责。通过两层校验后再调用
`plan_from_dict()` 得到 Scheduler 已使用的 `ToolCallPlan`。实验目录不得复制或修改
一份私有 schema，也不要创建第二套 DAG dataclass。

Scheduler 输入兼容检查使用假 profiling snapshot：为每个节点确定性选择该工具第一
个合法 Configuration 和兼容设备，再通过 `evaluate_scheduler_instance()` 的公开 seam
运行一次。此检查只记录 passed/failed；因为 profiles 是 synthetic，产生的 latency、
resource、accuracy 或 Composite Score 都不得写成真实实验指标。

## 运行方式（coding 完成后的目标）

```bash
export OPENAI_API_KEY="..."

uv run python experiments/03_mnms_tool_planning/run.py \
  --sample-id 123 \
  --base-url https://api.openai.com/v1 \
  --model <model-name> \
  --api-key-env OPENAI_API_KEY
```

批量 smoke run：

```bash
uv run python experiments/03_mnms_tool_planning/run.py \
  --limit 10 \
  --base-url <openai-compatible-base-url> \
  --model <model-name> \
  --api-key-env <key-env-name>
```

## Coding 顺序

1. 先为 `docs/schemas/tool-call-dag.schema.json` 写 schema/round-trip 单元测试，覆盖
   request source、node source、缺失 node port 和未知字段。
2. 写固定 revision 的 dataset loader，用本地 fixture 测试字段隔离。
3. 写 Query 输入适配及 text/image/audio 模态测试。
4. 写 mocked transport 测试：合法 DAG、一次修复成功、重试耗尽。
5. 串起 `run_experiment()` 和 CLI，验证成功/失败文件严格分离。
6. 加 Scheduler 公开 seam 的兼容测试。
7. 最后显式运行一个官方 MnMS 样本加真实 LLM 的 smoke test，并检查产物中没有
   参考 plan 和密钥。

## 验证命令

后续实现至少运行：

```bash
uv run pytest -q -s
uv run python -m compileall -q src tests experiments/03_mnms_tool_planning
git diff --check
```

真实联网 smoke test 单独执行并报告样本 ID、模型、结果路径和三项 validation 状态，
不让它成为默认离线测试的前置条件。

## 暂不处理

- 与 MnMS 参考 DAG 的图匹配评分；
- 真实媒体下载和工具执行；
- 支持当前目录之外的 5 个 MnMS 官方工具；
- 批量并发、限流恢复和长期 checkpoint；
- 在实验目录维护第二份 DAG schema。
