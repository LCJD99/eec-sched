# 实验 03：用 LLM Planner 生成真实 MnMS Query 的 Tool-Call DAG

## 一句话说明

从官方 MnMS 数据集中读取真实 `user_request`，让一个 LLM Agent 只根据 Query
和本项目支持的工具目录生成 Tool-Call DAG；只有符合本项目 DAG 约定、通过确定性
校验且能够被 Scheduler 接收的结果，才写入成功结果文件。

## 需要你优先确认的理解

本实验验证的是下面这条链路：

```text
真实 MnMS user_request
        ↓
LLM Tool Planner（只选工具和数据依赖）
        ↓
ToolCallPlan DAG（不含 Configuration、Device 或调度结果）
        ↓
确定性校验
        ↓
Scheduler 可接收的输入
```

关键边界：

- Agent 是 Planner，不是 Scheduler。它只能选择工具、节点输入来源和最终输出。
- Agent 不选择模型 Configuration，不选择 `device | edge | cloud`，也不决定执行顺序。
- MnMS 的 `plan_str`、`code_str` 和 `alt_plans_str` 是参考答案，只可用于后续评测，
  不得进入 Agent 的上下文，也不得参与生成输入的预处理。
- 本实验生成 DAG，不执行 DAG 中的真实模型或外部工具。
- 本实验不评价 Scheduler 的优劣；它只证明生成结果可以进入现有 Scheduler 接口。

如果以上边界正确，后续 coding 就按 `implement.md` 执行。

## 数据范围

默认数据源为官方 Hugging Face 数据集 `zixianma/mnms`，默认使用
`test_human_verified_filtered`。该集合是官方说明中的人工验证、过滤并可执行的
882 条计划对应的 Query，适合先作为真实 workload 来源。

实现时必须固定并记录：

- 数据集名称；
- 数据集 revision；
- split；
- 样本 `id`；
- 原始 `user_request`；
- 运行时实际使用的 Planner 模型与 prompt 版本。

数据加载失败时必须明确失败，不允许自动换成内置样例或合成 Query。

官方来源：

- [MnMS Dataset](https://huggingface.co/datasets/zixianma/mnms)
- [MnMS Repository](https://github.com/RAIVNLab/mnms)

## 输入

实验入口至少支持：

- 通过样本 ID 运行一条真实 Query；
- 按数据集中的稳定顺序运行前 N 条 Query；
- 指定 LLM 的 OpenAI-compatible base URL 和模型名；
- 通过环境变量读取 API key，命令行和结果文件中都不出现密钥；
- 指定输出目录。

Planner 看到的内容只能包括：

- 当前样本的 `user_request`；
- 从 Query 本身识别出的可用请求输入，例如 Query 文本、图片文件引用或音频文件引用；
- `mnms_tool_specs()` 暴露的工具 ID、说明和输入/输出端口；
- 上一次确定性校验产生的简短错误（仅在有限修复重试时提供）。

## 成功输出

每个成功样本写成一个 JSONL record。record 至少包含：

```json
{
  "schema_version": "1.0.0",
  "sample": {
    "dataset": "zixianma/mnms",
    "revision": "<pinned revision>",
    "split": "test_human_verified_filtered",
    "id": 123,
    "user_request": "..."
  },
  "planner": {
    "model": "...",
    "prompt_version": "v1",
    "attempts": 1,
    "latency_ms": 0.0
  },
  "request_inputs": [
    {"name": "query", "modality": "text", "value": "..."}
  ],
  "dag": {
    "nodes": [
      {
        "node_id": "node_0",
        "tool_id": "text_summarization",
        "inputs": {
          "text": {"kind": "request", "name": "query"}
        }
      }
    ],
    "final_outputs": [
      {"node_id": "node_0", "port": "text"}
    ]
  },
  "validation": {
    "dag_schema": "passed",
    "tool_plan": "passed",
    "scheduler_input": "passed"
  }
}
```

其中 `dag` 是 Scheduler 输入的 JSON 投影，能够通过现有 `plan_from_dict()` 转成
`ToolCallPlan`。成功文件中的每一条记录都必须满足：

- 节点 ID 唯一；
- 工具存在于本项目的 MnMS 工具目录；
- 必需输入齐全；
- request/node 输入来源合法；
- 上下游端口模态一致；
- 图无环；
- 至少有一个合法 final output；
- 不含 Configuration、Device、调度时间或评分字段；
- 转换后的 `ToolCallPlan` 可以交给现有 Scheduler/可信评估入口。

## 失败输出

下载失败、LLM 调用失败、JSON 解析失败、重试耗尽或 DAG 校验失败的样本写入独立
的 `failures.jsonl`。失败记录保留样本 ID、阶段、简短原因和尝试次数，但不得混入
成功 DAG 文件。这样下游可以把成功文件整体当作 Scheduler 输入集合，不需要再次
筛选。

## 验收标准

1. 一条指定 ID 的 Query 确实来自固定 revision 的官方 MnMS 数据文件。
2. 测试能够证明 Agent 请求中没有 `plan_str`、`code_str` 或 `alt_plans_str`。
3. Agent 使用的是 `mnms_tool_specs()` 的实际工具目录，而不是手抄的第二份目录。
4. 至少一条真实 Query 经真实 LLM 调用生成多节点或单节点 DAG，并进入成功文件。
5. 成功 DAG 同时通过 JSON Schema、`validate_plan()` 和 Scheduler 输入兼容检查。
6. 无效工具、缺失端口、模态错误、环和非法 final output 都不会进入成功文件。
7. 同一次运行留下数据 revision、模型、prompt 版本、耗时和失败证据。
8. API key 不出现在 prompt 日志、结果文件或异常文本中。

## 非目标

- 不执行 MnMS 工具，不下载图片、音频或模型权重；
- 不比较生成 DAG 与参考 DAG 的准确率；
- 不实现 Scheduler 的 Configuration/Device 决策；
- 不修改 Evolution Loop；
- 不把 MnMS 官方 33 个工具强行全部加入本项目。本实验只允许当前项目已支持的
  28 个工具，遇到无法用该目录表达的 Query 时应记录为失败或不支持。
