# Evolution Memory 与 Reflection Context 研究

## 1. 结论

eec-sched 不应该把“Memory”实现成一个不断追加文字的长 prompt，也不应该把以前所有 `EvolutionGraph` 合并成一个跨运行大图。更合适的设计是：

1. **当前运行的 `EvolutionGraph` 是工作记忆。** 它完整保存本次 Evolution Loop 中的 Candidate、父代关系和同一个 Evolution Trace Set 上的可信评估，负责本次父代选择和谱系比较。
2. **跨运行 Memory 是可检索的长期证据。** 它保存旧运行中可重放、可核验、可追溯的信息，但不加入当前 `Parent Candidate Pool`，也不直接改变可信评估规则。
3. **Reflection Context 是临时证据包。** 它不是 Memory 本身，而是 Context 管理模块针对“这一次 mutation / crossover 要解释什么”生成的、受预算约束的只读投影。
4. **原始历史与可学习结论必须分开。** 可信评估事实不可变；Reflection 写出的原因和经验只是带证据引用的假设，不能反过来覆盖原始事实。
5. **压缩的目标不是最短，而是保留判别力。** Reflection 至少应同时看到一个失败或回退案例、一个相似成功案例和一个边界案例，才能判断问题来自 DAG 结构、Configuration、设备分配、跨设备传输还是 Scheduler 自身耗时。

可以把完整链路概括为：

```text
完整可信事实 -> 确定性证据卡片 -> 对比选择与历史检索 -> Reflection Packet
              \-> Replay Trace Coreset     \-> 可证伪的 Strategy Hypothesis
```

这里的关键不是“把更多 Trace 塞给模型”，而是保持下面这条关系完整：

```text
优化目标 -> 当前策略 -> 哪些决策发生了变化 -> 可信结果如何变化
        -> 哪个相似案例没有发生同样结果 -> 哪个假设值得下一轮验证
```

《[Learning Beyond Gradients](https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/learning-beyond-gradient.en.md)》把 tests、logs、replays、trial summaries 和 memory 都视为 coding agent 的反馈，并明确指出只吸收历史而不压缩历史会增加系统耦合复杂度。GEPA 也不是只给 Reflection 一个分数：它把执行轨迹中的输入、输出和 reasoning 与数值分数、编译错误或 rubric 反馈一起交给 Reflection，从而进行显式的诊断和隐式 credit assignment（[GEPA 原论文，第 3 节](https://arxiv.org/pdf/2507.19457)）。这两点直接支持本研究的方向。

本文是一份**建议设计**，不表示以下 Memory、检索、压缩和持续学习能力已经实现。

## 2. 当前实现：已经有什么，缺少什么

### 2.1 已有的完整事实

当前 [`TraceEvaluation`](../src/eec_sched/candidate.py) 并非只有 score。它已经能够保留：

- `EvaluationTrace`：`trace_id`、task input 和 Tool-Call DAG；
- `SchedulerView`：只读 DAG、Scoring Context、Snapshot digest 和与该 Trace 相关的 Profiling Database Snapshot evidence；
- 可信归一化后的完整 `SchedulerProposal`；
- `scored`、`rejected` 或 `failed` 状态和原因；
- Scheduler Computation Time；
- [`EvaluationReport`](../src/eec_sched/evaluation/models.py) 中逐节点模拟结果、跨设备传输、quality、latency、execution / communication Resource Score 以及最终 utility。

因此当前问题不是“评估器没有产生历史”，而是“模型上下文没有把历史整理成容易归因的证据”。

### 2.2 当前 Reflection Input

当前 [`ReflectionInput`](../src/eec_sched/evolution/models.py) 只有：

- operator：`mutation` 或 `crossover`；
- 涉及 Candidate 的 Scheduler Strategy Description；
- 由 root、lineage 或 crossover 规则选出的若干完整 `TraceEvaluation`。

当前选择规则有价值：root 看最好和最差 Trace；descendant mutation 看相对直接父代的最大进步和最大回退；crossover 看两个父代各自最占优的 Trace。但它仍有四个限制：

1. `TraceEvaluation` 太深，里面的 DAG、Snapshot evidence 和完整报告会让 prompt 体积随 Trace 复杂度增长；
2. 没有明确告诉 Reflection “为何选择这个 Trace”和“它与哪个证据形成对比”；
3. score 没有被拆成 quality、latency、Resource Score、跨设备传输和 Scheduler Computation Time 的变化；
4. 没有跨运行经验、被证伪的旧假设、Replay Trace 或 Final Evaluation 防泄漏策略。

当前 Concise Projection 只留下 `trace_id`、status、score 和 reason，适合报告，不足以反思；直接传完整 `TraceEvaluation` 又太冗杂。建议在两者中间增加专门的证据表示。

### 2.3 当前不能提前宣称的能力

- `EvolutionGraph` 只覆盖一次 Evolution Loop；它不是长期 Memory；
- Reflection Agent 没有跨运行查询能力；
- 当前没有 Replay Trace Coreset 或 retention 指标；
- 当前没有验证、合并、淘汰 Strategy Insight 的生命周期；
- Final Evaluation 结果不会且不应回写当前 Evolution Loop。

## 3. 三种历史不要混成一个对象

### 3.1 Execution Evidence Ledger：事实层

这是不可变、可追溯的实验事实账本。它回答“当时到底执行了什么”。它可以保存大对象或内容寻址后的 artifact，但默认不进入模型 prompt。

事实层的内容必须来自可信控制路径。模型生成的根因说明、Reflection Advice 和策略总结可以一同归档，但必须标成 annotation，不能伪装成 evaluator fact。

### 3.2 Replay Trace Coreset：保留性层

这是从旧 Evolution Trace 中挑出的一个小型、版本化集合，用来检查新 Candidate 是否破坏旧能力。它不是所有历史的随机样本，而应覆盖：

- 已知失败类型；
- 曾经发生过明显回退的 Trace；
- 代表性 DAG / tool / transfer 模式；
- 稀有但后果严重的反例；
- 新策略容易过拟合的边界案例。

持续学习研究把 replay 看作同时保持 stability 和 plasticity 的常见方法；CLEAR 明确用旧经验 replay 改善 stability，同时保留对新经验的 plasticity（[NeurIPS 2019 论文页面](https://papers.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html)）。这不能直接证明同一种采样算法适合 eec-sched，但支持“新 Trace 与历史 Replay 必须同时出现”的结构。

### 3.3 Strategy Insight Catalog：认识层

这是跨运行保存的可读策略知识，回答“在什么条件下，哪一种调度倾向可能有效或有风险”。每条 Insight 至少包含：

- 适用条件，例如 Snapshot / Scoring Context 版本、tool 组合、DAG 结构和设备环境；
- 现象与可能根因；
- 建议改变的决策因素；
- 预测的 quality、latency 或 Resource Score 变化；
- 支持证据 ID 和反对证据 ID；
- 状态：`proposed`、`supported`、`contradicted` 或 `retired`；
- 最近一次验证所在的 run。

Insight 不是事实缓存。一个 Reflection 模型写出的漂亮结论，只有在后续完整 Candidate Evaluation 中符合预期，才能从 `proposed` 进入 `supported`；出现反例时应补充 scope 或转为 `contradicted`，而不是删除反例。

[Reflexion](https://arxiv.org/pdf/2303.11366) 将完整 trajectory 视为短期记忆，将从 trajectory 和 reward 生成的 verbal reflection 视为长期记忆，并把长期记忆限制在很小的条目数以控制上下文。它的 ablation 还表明，只有最近的原始 trajectory 不如再经过一次 verbal reflection。这个结果支持“原始事实与压缩 insight 分层”，但不意味着模型总结可以替代本项目的可信事实。

## 4. `EvolutionGraph` 与跨运行 Memory 的边界

| 维度 | 当前运行 `EvolutionGraph` | 跨运行 Memory |
|---|---|---|
| 时间范围 | 一次 Evolution Loop | 多次已经结束的 run |
| 主要职责 | 当前父代选择、谱系比较、最终 Candidate 选择 | 经验检索、Replay、回归风险和历史假设 |
| 证据完整性 | 当前固定 Evolution Trace Set 上的全部 Candidate 证据 | 经版本化、去重和筛选的历史证据及其原始引用 |
| score 可比性 | 同一个 Trace 顺序、Snapshot 和评分契约下可直接比较 | 默认不可直接比较；先检查 Trace、Snapshot 和评分版本兼容性 |
| 是否属于 Parent Candidate Pool | 是 | 否 |
| 是否允许跨运行 parent edge | 否 | Memory 只能记录来源关系，不能偷偷改变图的 parent 语义 |
| 生命周期 | run 内增长，run 结束后封存 | 跨 run 更新，但需要容量和淘汰策略 |

这个边界必须保留已接受的 [ADR 0001](adr/0001-evolve-schedulers-through-an-evolution-graph.md)：每次运行建立新的图，当前 `Parent Candidate Pool` 只包含本次图中已经评估的 Candidate。

如果以后想用旧 Candidate 作为新运行起点，应把它定义为**应用层 root 初始化**，在新图中仍然是 parent list 为空的新 root；跨运行来源用单独的 provenance 记录，而不是把旧 Scheduler Version 放进 `Parent Scheduler Versions`。这是一项新的设计决策，不能由 Memory 模块暗中完成。

## 5. 原始执行历史应保存什么

建议每个 Candidate × Trace 形成一个不可变 `ExecutionEvidence`。以下字段是逻辑 schema，不要求按同名 Python dataclass 实现。

### 5.1 可重现身份

- `evidence_id` 和 schema version；
- run ID、round、operator、Candidate Scheduler Version、Parent Scheduler Versions；
- Candidate source hash 和 Scheduler Strategy Description hash；
- Trace ID、Trace cohort role（new / replay / development / final）；
- Snapshot digest、Scoring Context version、evaluator version；
- 随机 seed、agent / prompt 配置版本和运行时间；
- 原始 artifact 的 content hash。

没有这些身份，两个相同 score 可能来自不同评分公式、不同设备证据或不同 Trace，跨运行比较会得到错误结论。

### 5.2 输入与结构，而不是无差别 payload

需要保存原始 `EvaluationTrace` 以便重放，但检索索引只提取与调度相关的结构特征：

- DAG fingerprint、节点数、边数、宽度、最长依赖链；
- tool multiset、fork / join、ready-node 并行度；
- 潜在跨设备边和输出大小区间；
- Scoring Context；
- 相关 Configuration / Compatible Device 的 profile ID 与版本。

默认不要把完整 `task_input`、自然语言 payload 或全部 `snapshot_evidence` 注入 Reflection。它们可能很大，也可能包含与调度策略无关或敏感的信息。确实影响调度的 Input Bucket 应作为结构化字段出现；原始 payload 留在受权限控制的 artifact 中。

### 5.3 Candidate 行为

- 可信归一化后的每个节点 Configuration / Compatible Device assignment；
- 与比较对象相比，哪些 assignment 改变；
- 产生跨设备传输的依赖边；
- Scheduler Computation Time；
- 对 `rejected` / `failed` 案例，在受限 artifact 中保留原始 reason，并生成确定性归一化的 reason code。

Candidate 抛出的异常文本属于**不可信数据**：它可能很长，也可能意外包含 prompt-like 内容。进入 Reflection Packet 前只能作为数据字段处理，必须清洗控制字符、限制长度并明确标注来源；默认优先给 reason code 和可信评估器生成的结构化诊断，不把原始异常直接拼进系统指令。

当前 Scheduler 是一次返回完整 Proposal 的程序，并没有可观察的逐步 reasoning。因此文档和 Memory 只能把 assignment 与后果称为**行为证据**，不能把它虚构成 Candidate 的“思考过程”。

异常文本也不能直接当作可信指令。Candidate exception、validation message 或外部日志可能包含任意字符串、超长 payload，甚至类似 prompt 指令的内容。原文可以作为隔离 artifact 保存；进入 EvidenceCard 前必须由可信代码完成类型标注、reason code 归一化、长度上限、转义和敏感字段清洗。Reflection 看到的异常文本必须被清楚包在“被观察数据”字段中，不能与系统指令拼接成同一权限层。

### 5.4 可信后果

- status 与 score contribution；
- quality lower bound；
- simulated makespan、latency proxy；
- execution 与 communication Resource Score；
- normalized performance 和 utility；
- 逐节点 start / finish 与关键路径投影；
- transfer 的 source / destination device、latency 与 Resource Score；
- validation errors 或 Candidate exception；
- 与直接父代、另一个 crossover parent 或选定 baseline 的逐指标 delta。

失败状态不能只按 score 排序。当前契约中 `rejected` / `failed` 固定贡献零，而合法 Proposal 的 utility 可能为负；如果简单选择“最低分 Trace”，有可能漏掉真实运行失败。

### 5.5 进化语义

- 当轮 Reflection Packet ID；
- Reflection 产生的 Strategy Hypothesis / Advice；
- Coding Agent 声明的 Strategy Description；
- 新 Candidate 相对父代的 source diff hash；
- 假设预期改变的证据和实际变化；
- 接受、反驳或未决的判断及其证据引用。

这部分让历史不只回答“Candidate 4 得了 0.8”，还能回答“Candidate 4 为了减少 cloud-edge transfer 做了什么；在哪些 Trace 上符合预测；在哪些 Trace 上引入了回退”。

## 6. Reflection 真正应该看到什么

Reflection 不应看到全部账本，也不应只看到 score。建议每次构造一个 `ReflectionPacket`，固定包含以下六部分。

### 6.1 目标与权限

- 当前评分公式的通俗说明和各指标方向；
- Candidate 只能输出 assignment，可信评估器负责 validation、Canonical Evaluation Order、simulation、timing 和 score；
- `rejected` / `failed` 的零分语义；
- 当前 operator 和本轮要解释的问题。

这一段防止 Reflection 建议修改不属于 Scheduler 的 evaluator 权力。

### 6.2 当前策略与比较对象

- 一个 mutation parent，或两个 crossover parents 的 Scheduler Strategy Description；
- direct-parent lineage 的简短变化；
- 上一轮可证伪假设及其预测；
- 不包含 Scheduler source code，继续遵守当前 Reflection Input Policy；源码仍只给 Coding Agent。

### 6.3 对比证据卡片

完整 `TraceEvaluation` 先被确定性投影成 `EvidenceCard`：

```text
证据身份：evidence_id、run、Candidate、Trace、Snapshot / evaluator 版本
场景签名：DAG 结构、tool 组合、Scoring Context、相关 profile 版本
决策差异：相对比较对象改变的 assignments 与新增/消失的 transfers
结果分解：status、reason、quality、latency、Resource Score、utility 及 delta
机制线索：关键路径、主要 transfer、Scheduler Computation Time
选择理由：regression / matched-success / boundary / unique-failure / replay
原始引用：可以定位到不可变 artifact，但默认不展开
```

“确定性投影”很重要：数值、delta、关键路径和 reason code 应由可信代码计算，不能让 LLM 从长日志中自由抄写。

### 6.4 从长期 Memory 检索到的 Insight

只给出与当前 Snapshot、Scoring Context、DAG / tool 模式和 operator 相关的少量 Insight。每条 Insight 必须同时显示：

- scope；
- 状态和最近验证时间；
- 支持证据与反对证据数量；
- 至少一个可访问的 evidence ID；
- 已知回归风险。

只做 embedding top-k 不够。检索应先用结构化兼容条件过滤，再在兼容集合中计算语义相关性和行为相似度。

### 6.5 未覆盖信息清单

Packet 应明确给出本次被省略的证据数量、涉及哪些 cluster，以及省略原因。这样 Reflection 至少知道自己看到的是样本而非全部历史，避免把局部模式说成普遍规律。

### 6.6 结构化输出契约

Reflection 不应直接跳到“改代码”。建议先输出：

- `phenomenon`：观察到的现象；
- `hypothesized_cause`：可能原因；
- `supporting_evidence_ids`；
- `contradicting_evidence_ids`；
- `scope_conditions`；
- `proposed_strategy_change`；
- `predicted_metric_changes`；
- `regression_risks`；
- `discriminating_traces`：下一次评估最能证伪该假设的 Trace 类型。

Coding Agent 再把这个 Strategy Hypothesis 与完整父代源码结合，产生新的完整 Candidate。这样 Reflection 的质量能够单独检查，也能在下一轮被证伪。

## 7. 如何压缩 Trace 而不丢关键反例

### 7.1 先投影，再选择，最后写自然语言

推荐顺序是：

1. **确定性投影**：从 DAG、Proposal 和 `EvaluationReport` 计算 EvidenceCard；
2. **去重与聚类**：按行为和结果 signature 合并重复案例；
3. **分层选样**：保证失败、成功、边界和 replay 覆盖；
4. **预算装箱**：在 token / card 上限内装入 Packet；
5. **可选语言摘要**：只解释已经选定的事实，不重新计算事实。

如果先让 LLM 总结所有长 Trace，再从摘要中检索，早期的错误压缩会永久丢失反例。

### 7.2 使用调度行为 signature 去重

建议 signature 至少组合：

```text
(DAG 结构簇, tool 组合, assignment 模式, transfer 模式,
 status / reason code, metric-delta 方向, snapshot / scoring version)
```

完全相同的 Evidence 只保留一个 active representative，并记录出现次数和其他 evidence IDs。不能只按 task 文本 embedding 去重：两个文本不同的 Trace 可能产生相同调度行为；两个文本相似的 Trace 也可能因为 DAG 或 profile 不同而具有不同调度意义。

### 7.3 采用“覆盖优先”的对比选样

不要把所有 Evidence 混在一起计算一个加权相关度，然后简单 top-k。推荐先满足不可互相替代的槽位：

1. **最大回退**：相对直接父代或对照 Candidate 的核心负 delta；
2. **matched success**：DAG / tool / Scoring Context 相似，但 assignment 或结果不同的成功案例；
3. **边界案例**：一个很小决策变化就使 status、关键路径或主要指标方向翻转；
4. **唯一失败**：每种 reason code 或罕见故障模式至少保留代表；
5. **历史 Replay 反例**：与当前建议可能冲突的旧案例；
6. **多样性补位**：再覆盖未出现的 DAG / assignment / transfer cluster。

ExpeL 的经验抽取明确使用同一任务的 failed / successful trajectory 对，并另外从不同任务的成功轨迹中提取共同 good practices；推理时再按任务相似度取 top-k 成功经验（[ExpeL 原论文，第 4.2 节](https://arxiv.org/pdf/2308.10144)）。这支持“失败要和相似成功对照”，而不是孤立地总结最低分样本。

### 7.4 给罕见反例保护名额

纯 reservoir sampling 能近似已经见过的数据频率，但小概率模式可能从有限 buffer 中消失。GSS 的原论文把固定大小 replay buffer 选择解释为约束缩减，并用多样性避免重复样本占满容量，同时也指出普通 reservoir sampling 可能漏掉 minor modes（[Gradient-based Sample Selection](https://arxiv.org/pdf/1903.08671)）。

eec-sched 没有可直接使用的神经网络梯度，因此不能照搬 GSS 的 gradient similarity；本文的**项目内推论**是用 Trace Score Vector、assignment delta、DAG / transfer signature 和 failure code 作为行为特征。淘汰时应遵守：

- 不删除某个 failure signature 的最后一个代表；
- 不删除某条仍在 `supported` Insight 中被引用的唯一证据；
- 同一高频 cluster 优先合并重复项；
- 稀有但高回归损失的案例拥有保护名额；
- 被删除的 active card 仍可保留原始 archive 引用。

### 7.5 保留反证，不把冲突“总结掉”

如果两条 Insight 冲突，应保留各自的 scope 与证据，而不是合成“视情况而定”。例如：

```text
Insight A：宽 DAG 且 edge-cloud transfer 大时，集中到 edge 降低 latency。
反例 B：同类 DAG 但 edge execution profile 较慢时，集中到 edge 反而增加 makespan。
```

Reflection 应看到这个冲突，并把“edge profile 相对速度”作为待验证条件。反证是收窄策略适用范围的资源，不是噪声。

FunSearch 在 program database 中按多输入 score signature 聚类并保持程序多样性，而不是只保留一个全局最好程序（[Nature 原论文](https://www.nature.com/articles/s41586-023-06924-6)）。这里不能直接照搬其 island algorithm，但它支持“在多 Trace 行为上保持不同强项”的原则，和当前按 Trace Score Vector 做 Pareto / cosine 比较的思路一致。

### 7.6 一个具体的覆盖优先选择算法

建议把选择策略实现成确定性的 `Contrastive Coverage Selection`，输入是当前 graph 证据、允许读取的历史 Memory、比较对象和 Packet budget，输出是 EvidenceCard 列表与 omitted manifest：

```text
1. Hard gate
   - 排除 final / audit_only；
   - 排除 trusted evaluator fault；
   - 跨运行证据先按 Snapshot、Scoring Context、evaluator version 检查兼容性；
   - 不兼容证据只可作为带 stale 标记的定性提醒，不参与数值 delta。
   - 原始 task payload 不进入检索语料；Candidate 异常先清洗、截断并标成 untrusted。

2. Project
   - 用可信代码把完整 Evidence 投影成 EvidenceCard；
   - 计算 assignment、transfer、关键路径和逐指标 delta；
   - 归一化 status / reason code。

3. Deduplicate and cluster
   - content hash 完全相同的卡片合并；
   - 按 DAG、assignment、transfer、failure 和 metric-delta signature 分 cluster；
   - 每个代表保留 occurrence count 和全部 evidence IDs。

4. Fill required slots in order
   - 每个未覆盖 failure signature 的一个代表；
   - 相对比较对象回退最大的卡片；
   - 与该回退最相似、但结果成功的 matched card；
   - 一个结果或瓶颈发生翻转的 boundary card；
   - 一个会挑战当前 Strategy Hypothesis 的 historical replay card。

5. Diversity fill
   - 对尚未覆盖的 signature cluster，每轮选择能增加最多新结构/行为覆盖的卡片；
   - 覆盖增量相同时，依次按回退严重度、与当前 subject 的相关性、较新验证时间排序；
   - 不用 Candidate Score 作为唯一 tie-breaker。

6. Pack and report
   - 不拆断一张卡片，也不删除 evidence ID；
   - 超预算时先减少同 cluster 重复代表，再减少普通成功案例；
   - protected failure / counterexample 仍超预算时，返回显式 budget error 或要求更大预算，
     不能静默截断；
   - manifest 记录各 cluster 的入选数、遗漏数和选择理由。
```

这里的“最相似”先比较结构化特征：相同 Snapshot / Scoring Context、相近 DAG signature、相同 tool 组合和相似 profile 条件；embedding 只作为最后的语义补充。这样选出的 matched success 才能帮助区分调度决策差异，而不是只在任务文字上看起来相似。

## 8. Memory 的写入、验证、压缩和淘汰

### 8.1 写入

- 当前 run 内，完整事实先进入 `EvolutionGraph`；
- 每轮 Context 管理模块直接读取 active graph，不必把未完成 run 反复写入长期 Memory；
- run 结束后，把 graph 与运行配置封存为 `SealedRunEvidence`；
- 通过 schema、角色和 hash 校验后，一次性提交长期 Memory；
- Final Evaluation 可以进入审计 archive，但必须标记 `audit_only`，不得被 Reflection 查询。

这样可以防止中途失败的 run 留下一半写入的长期知识，也保持 `EvolutionGraph` 是当前运行事实的唯一权威。

### 8.2 Insight 验证

每条假设都记录预测，例如：

```text
在“宽 DAG + 大输出跨设备边”范围内，减少 device switch；
预期 communication Resource Score 下降，quality 不下降，latency 不明显回退。
```

新 Candidate 跑完**完整固定 Evolution Trace Set**后，再对照预测：

- 主要预测满足且没有受保护回归：`proposed -> supported`；
- 出现明确反例：保持 `proposed` 并缩小 scope，或进入 `contradicted`；
- evaluator / Snapshot 版本不兼容：标记 stale，不做真假判断；
- 仅在 Reflection minibatch 上变好：仍然是 `proposed`，不能晋级。

GEPA 也把用于产生反馈的 minibatch 与用于候选选择的较大 `Dpareto` 分开，候选先在 minibatch 改进后才进入完整选择集评估（[GEPA Algorithm 1](https://arxiv.org/pdf/2507.19457)）。本项目当前不是同一算法，但“局部反思证据不能替代完整候选评估”的原则适用。

### 8.3 周期性压缩

压缩不是删除原始证据，而是维护 active view：

- 合并重复 EvidenceCard，保留 occurrence count 和全部原始引用；
- 把多个局部 Insight 提炼为一个有明确 scope 的较一般 Insight；
- 保留合并前 Insight IDs，确保可以追踪推理来源；
- 将过期 Snapshot / evaluator 下的 Insight 移出默认检索，但不改写历史；
- 对长期没有支持、只有反证的 Insight 进入 `retired`；
- 定期用 Replay Trace 重新验证仍然 active 的 Insight。

AlphaEvolve 的 prompt sampler 会从 program database 取多个过去方案，并渲染程序执行结果和 evaluator scores；其高层结构也明确包含 past trials、ideas、quality scores 和 other feedback（[AlphaEvolve 白皮书，第 2.2 节](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)）。这支持保留“方案 + 执行结果 + 反馈”的联系，但不意味着应该把数据库全文放进一次 prompt。

## 9. New / Replay / Final 数据隔离

### 9.1 是否必须有“新 Trace 分布”

**Memory 能力不要求先出现新的 Trace 分布；但要把过程称为有实质意义的持续学习，时间上必须持续出现新的可检验信息。** 这里的“新信息”不等于必须先证明数据分布发生漂移，它可以是：

- 从同一总体分布新采到的 Trace；
- 新 tool / DAG 结构或新的输入区间；
- 新设备、Configuration、Profiling Database Snapshot 或 Scoring Context；
- 新 Candidate 在旧 Replay Trace 上暴露出的新行为、失败或回归；
- 新的可信环境反馈或人工约束。

因此要区分三种情况：

1. **固定 Trace、固定 Snapshot，但持续产生新 Candidate Evaluation。** 系统仍可积累 Evidence、验证 Insight 并改进 Scheduler；更准确的名称是“带持久 Memory 的迭代程序搜索”或“固定经验上的持续优化”。
2. **时间上持续到达新 Trace 或其他新环境证据，同时 replay 旧 Trace。** 系统既要吸收新规律，又要避免旧能力退化，符合本文所讨论的持续学习场景；新 Trace 即使来自同一分布也成立。
3. **没有任何新输入、行为结果或反馈，只把同一段历史重复交给 Reflection。** 这不产生新的可证伪证据，最多是重新压缩或重新解释 Memory，不能算新的学习。

也就是说，持续学习需要的是**顺序到达的新证据**，不是强制要求每批 Trace 来自不同概率分布。分布变化是需要应对的一种情形，而不是定义 Memory 的前置条件。

### 9.2 三类 Trace 角色

持续学习场景下，建议每个时间窗口明确三种 Trace 角色：

| Trace 角色 | 用途 | 可进入 Reflection | 可用于更新 Candidate | 是否进入长期学习 Memory |
|---|---|---:|---:|---:|
| New Trace Batch | 学习新到达经验，衡量 plasticity | 是 | 是 | run 封存后可以 |
| Replay Trace Set | 检查旧能力，衡量 retention | 是 | 是；因此属于开发数据 | 可以，且应版本化 |
| Final Evaluation Trace Set | 最终泛化与 Oracle 报告 | 否 | 否 | 只进入 `audit_only` archive |

在第 (t) 次运行中，可以固定：

\[
\mathcal T_{E,t}=\mathcal T_{new,t}\cup\mathcal T_{replay,t}
\]

所有 Candidate 都在同一个固定 \(\mathcal T_{E,t}\) 上评估，以保持 run 内 Trace Score Vector 可比。New 和 Replay 应保留 cohort 标签，从而单独报告：

\[
\text{Plasticity}_t
=J_{new,t}(C_{t+1})-J_{new,t}(C_t)
\]

\[
\text{Retention}_t
=J_{replay,t}(C_{t+1})-J_{replay,t}(C_t)
\]

这两个诊断指标不自动改变当前 Candidate Score 的算术平均定义；如果以后要改变 new / replay 权重或设置保留性门槛，需要单独修改评分契约。

必须执行以下隔离规则：

1. Memory query 在存储层就排除 `final` / `audit_only`，不能只靠 prompt 提醒模型别看；
2. Final Trace 的 ID、原始内容、score、Oracle 和由其产生的摘要都不能用于 Reflection、Coding、父代选择或 prompt 调整；
3. 如果根据 Final Evaluation 改了 Candidate、Memory、prompt 或检索规则，这批 Trace 已变成 development data；必须更换新的 untouched Final Evaluation Trace Set；
4. 旧 Final Trace 只有在报告冻结、明确退役并准备了新的最终留出集后，才可以在未来 run 被重新标成 development / replay；角色变更要有审计记录；
5. Replay Trace 本身不是 final holdout。它被反复使用后可能被 Candidate 针对性优化，需要轮换 coreset 和保留真正未见 Trace。

## 10. 推荐的深模块 Interface

不建议把去重器、向量库、clusterer、summarizer、token counter 和淘汰策略都暴露给 `EvolutionLoop`。它们应当藏在一个深模块后面，调用方只需要理解两个操作：

```python
class ContextMemory:
    def prepare(
        self,
        subject: ReflectionSubject,
        active_graph: EvolutionGraph,
        budget: ContextBudget,
    ) -> ReflectionPacket:
        """为本轮反思生成有证据清单、无 final 泄漏的只读上下文。"""

    def commit(self, sealed_run: SealedRunEvidence) -> MemoryCommitReceipt:
        """原子地封存一个完成的 run，并产生新的 Memory Version。"""
```

这只是推荐的逻辑 interface；在只有一种实现前，不需要为了形式再创建许多 Adapter。一个内存实现和一个持久化实现确实需要替换时，这个 seam 才成为真实的 Adapter seam。

### 10.1 `prepare` 隐藏的复杂度

- 校验 Snapshot / evaluator / Scoring Context 兼容性；
- 从 active graph 建立 lineage / crossover 对比；
- 确定性生成 EvidenceCard；
- 查询 historical evidence 与 Insight；
- 去重、分 cluster、保护罕见失败；
- 满足 contrastive slots；
- 按预算装箱并生成 omitted manifest；
- 硬性排除 Final Evaluation；
- 返回不可变 `ReflectionPacket`。

### 10.2 `commit` 隐藏的复杂度

- 检查 run 是否完成和 graph / evaluation 版本是否一致；
- 校验 Trace role 与 Final 隔离；
- content-addressed artifact 去重；
- 更新 Replay Trace Coreset；
- 更新 Insight 支持 / 反证关系；
- 执行 active-memory 容量和保护策略；
- 原子生成新 Memory Version 和可审计 receipt。

### 10.3 关键类型

```text
ReflectionSubject
  operator, selected parents, direct lineage, current hypothesis, run config

ContextBudget
  maximum cards, maximum tokens, required evidence slots

ReflectionPacket
  contract, strategies, evidence cards, retrieved insights,
  omitted manifest, evidence manifest, memory version

SealedRunEvidence
  run config, graph, trace roles, agent artifacts, final audit record

MemoryCommitReceipt
  old/new memory version, accepted/rejected artifacts, coreset changes
```

`ReflectionPacket` 应包含 Memory Version 和每个 EvidenceCard 的来源。相同 `subject + active graph + memory version + budget + selection-policy version` 应产生可重现的 packet；若为了探索引入随机采样，必须记录 seed 和候选集合。

## 11. 推荐数据流

```mermaid
flowchart TD
    traces["New Traces + Replay Trace Set"] --> evaluator["Trusted Evaluator"]
    candidate["Scheduler Candidate"] --> evaluator
    evaluator --> traceeval["完整 TraceEvaluation"]
    traceeval --> graph["当前 run 的 EvolutionGraph"]

    graph --> prepare["ContextMemory.prepare"]
    memory["跨运行 Memory Version"] --> prepare
    prepare --> cards["确定性 Evidence Cards"]
    cards --> select["对比选择、去重、反例保护、预算装箱"]
    select --> packet["ReflectionPacket + Evidence Manifest"]
    packet --> reflection["Reflection Agent"]
    reflection --> hypothesis["可证伪 Strategy Hypothesis"]
    hypothesis --> coding["Coding Agent + Parent Source"]
    coding --> candidate

    graph --> seal["run 结束后 SealedRunEvidence"]
    seal --> commit["ContextMemory.commit"]
    commit --> memory

    graph --> final["独立 Final Evaluation"]
    final --> audit["Audit-only Archive"]
    audit -. "禁止进入 prepare" .-> prepare
```

## 12. 必须保持的不变量

1. **可信事实不可改写。** Memory summary 和 Insight 不能覆盖 Proposal、status、reason、report 或 score。
2. **证据可追溯。** Packet 中每个数值结论和 Insight 都能回到 evidence ID、run、Trace、Candidate、Snapshot 和 evaluator version。
3. **Reflection 不看源码。** Context 管理不改变当前 Reflection / Coding 的职责分离；Coding Agent 只拿选中 parents 的完整 source。
4. **Memory 不进入 Parent Candidate Pool。** 当前 graph 的父代选择只使用当前 run 已评估 Candidate。
5. **跨运行 score 默认不可比。** 只有 Trace、顺序、Snapshot、Scoring Context 和 evaluator contract 兼容时才允许做数值 delta。
6. **Final Evaluation 不可学习。** Final / Oracle 信息在 storage query 层被排除。
7. **系统错误不是 Candidate 经验。** trusted evaluator fault 应中止运行或进入 system incident，不得伪装为 Candidate `failed` Evidence。
8. **失败独立于 score 选择。** `rejected` / `failed` 必须按 status 和 reason 进入证据选择，不能依赖最低 score。
9. **摘要是派生物。** 删除或重建摘要不能破坏原始 Evidence Ledger。
10. **Insight 带反证。** active Insight 至少能显示支持和反对证据；不能只保存成功故事。
11. **异常文本按不可信数据处理。** 进入 Reflection 前必须归一化、清洗、转义和截断，不能获得 prompt 指令权限。
12. **Context 有硬预算。** 超预算时按明确优先级省略，并在 manifest 中报告，不允许无提示截断。
13. **Memory 更新有版本。** 一次 run commit 要么全部成功，要么完全不生效。

## 13. 主要失败模式与防护

| 失败模式 | 具体后果 | 防护 |
|---|---|---|
| 只传 Candidate Score | 不知道是 quality、latency、Resource Score、传输还是程序耗时造成变化 | 指标分解 + assignment delta + 关键路径 / transfer 证据 |
| 直接传完整系统 Trace | 大量 task payload、重复 profile 和无关节点稀释注意力 | 原始 archive + 确定性 EvidenceCard + budget |
| 只选最低分 | 漏掉 `failed` / `rejected`，也缺少可归因的成功对照 | status 分层 + matched success + boundary case |
| 只检索相似成功 | 形成确认偏误，看不到当前策略的回归边界 | 必选反证和历史 Replay slot |
| LLM 摘要被当成事实 | hallucination 逐轮固化，最终污染长期 Memory | facts / annotations 分层，Insight 需实测验证 |
| 热门 cluster 占满 buffer | 稀有失败和小概率分布消失 | 去重、cluster quota、最后一个失败代表保护 |
| 旧 Memory 已过期 | Snapshot、profile 或评分公式变化后仍套用旧规则 | version compatibility filter + stale 状态 |
| 跨 run 直接比较 score | 不同 Trace 或 evaluator 下的数值被误认为改进 | provenance gate；不兼容时只做定性参考 |
| Replay 逐渐成为公开测试集 | Candidate 针对 coreset 写特例，retention 虚高 | coreset 轮换、隐藏 final、未见 Trace 报告 |
| Final 泄漏 | 最终评估变成训练反馈，报告失去可信度 | 独立 namespace 和 query ACL，不依赖 prompt 自律 |
| Memory 无限增长 | Context 复杂度和维护成本持续上升 | active view 压缩、Insight 合并、archive 分层 |
| 过度压缩 | 一个反例被概括掉，策略适用范围被错误放大 | evidence manifest、protected counterexamples、可展开原始 artifact |
| 混淆相关和因果 | assignment 与 score 同时变化就被断言为根因 | 输出“假设 + 反证 + 预测”，用后续 discriminating traces 验证 |
| task input 泄露 | prompt 携带敏感或与调度无关的业务内容 | payload 默认不进卡片，受权限 artifact，结构特征检索 |

## 14. 建议的最小落地顺序

虽然本文不实现代码，但如果后续进入实施，建议按信息价值而不是存储技术分阶段：

1. **先做 EvidenceCard。** 用现有 `TraceEvaluation` 确定性产生 score decomposition、assignment delta、status / reason 和 provenance，先解决“只有 score 或完整 Trace”两极问题。
2. **再做 ReflectionPacket 与结构化 Hypothesis。** 保留当前 root / lineage / crossover 语义，加入 matched success、选择理由、反证和 manifest。
3. **再做跨运行 Evidence Ledger。** 先只读检索，不改变父代选择和评分。
4. **再做 Replay Trace Coreset。** 分开报告 New / Replay 的 plasticity 与 retention，保持 Final 隔离。
5. **最后做 Insight 生命周期和自动压缩。** 没有 evidence provenance、反证和回放验证前，不应自动把模型总结晋级为长期知识。

这条顺序的理由是：先保证 Reflection 得到的证据正确、紧凑、可追溯，再扩大时间范围。否则跨运行 Memory 只会更快地积累低质量摘要。

## 15. 研究依据与适用边界

- [Learning Beyond Gradients 官方文章](https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/learning-beyond-gradient.en.md)：支持 Heuristic System 应连接 policy、feedback、experiment records、replays/tests、memory 和 update，并同时吸收与压缩历史。
- [GEPA 原论文](https://arxiv.org/pdf/2507.19457)与[官方实现](https://github.com/gepa-ai/gepa)：支持从 trajectory 提取 inputs、outputs、reasoning 和 textual evaluator feedback，而不只使用 scalar reward；也支持 minibatch 反思与较大选择集评估分离。
- [Reflexion 原论文](https://arxiv.org/pdf/2303.11366)：支持把完整 trajectory 当短期记忆，把 distilled verbal reflection 当长期记忆，并限制长期上下文容量。
- [ExpeL 原论文](https://arxiv.org/pdf/2308.10144)：支持对比同一任务的成功与失败轨迹、从跨任务成功中提炼共同 Insight，以及检索相似成功经验。
- [FunSearch 原论文](https://www.nature.com/articles/s41586-023-06924-6)：支持按多输入 score signature 保持程序群体多样性，避免只围绕全局最好结果搜索。
- [AlphaEvolve 白皮书](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)：支持 program database、past trials / ideas、程序执行结果、scores 和其他 feedback 共同形成代码改进上下文。
- [Experience Replay for Continual Learning](https://papers.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html)：支持同时考虑旧知识 stability 与新知识 plasticity。
- [Gradient-based Sample Selection](https://arxiv.org/pdf/1903.08671)：支持有限 replay buffer 需要多样性和约束覆盖，且随机频率采样可能漏掉 minor modes。

这些工作来自 prompt evolution、language agents、程序搜索和神经网络 continual learning，不共享同一个形式化问题。本文明确属于结合 eec-sched 权限和数据结构后的**设计推论**，尤其是 EvidenceCard schema、行为 signature、罕见失败保护、`ContextMemory` interface 以及 New / Replay cohort 组织；它们需要在本项目的真实 Trace 上通过消融实验验证。
