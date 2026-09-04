# PSOCM SOTA：一手文献核查

> 核查日期：2026-08-23。本文只使用论文主页、正式 proceedings、作者论文或 arXiv 等一手来源。结论用于约束 PSOCM 的 SOTA 分类和 gap 表述，不代表穷尽式系统综述。

## 结论先行

建议把四宫格的两个轴写成：

- **Workload**：结构与系统交互相对固定的传统/非 Agentic Workflow vs. Agentic Workflow；
- **Policy construction**：在人工规定的优化变量、策略结构或控制机制内求解 vs. 由 LLM 在程序空间中生成和迭代可执行策略。

第二个轴不宜简称为“预定义搜索空间 vs. 开放搜索空间”。FunSearch、EoH、ReEvo、Vulcan (2025) 也都需要人定义接口、程序骨架或 evaluator；真正的差异是它们是否让 LLM **改写决策逻辑本身**。

当前最稳妥的四宫格锚点如下：

| | 人工规定的优化/策略结构 | LLM 生成、迭代可执行策略程序 |
|---|---|---|
| 传统或非 Agentic Workflow | Vulcan (NSDI 2024) | FunSearch、EoH、ReEvo、ReVEL；系统方向还必须讨论 Vulcan (2025 preprint) 与 SchedCP |
| Agentic Workflow | Murakkab (OSDI 2026)、LLM-as-Scheduler (ACL 2026) | **Ours 的目标位置，但只能作有范围限定的 gap，不能直接宣称 first** |

可以防守的观察是：

> 在本轮核查的一手工作中，Agentic Workflow 系统仍在预先规定的优化或路由结构内作决策；LLM program evolution 则已能生成数学、组合优化乃至系统资源管理的可执行策略，但尚未看到一项工作同时面向 Agentic Workflow 的端边云跨层调度，并从细粒度系统 Trace 中保留“Workflow 条件—节点 Configuration/Device 决策—quality/latency/resource/failure 后果”的关系，以此迭代可复用的完整 Scheduler 策略。

这是一条基于已核查集合的 **scoped observation**，不是系统综述意义上的空白证明。

## 1. 传统/非 Agentic Workflow × 人工规定的求解结构

### Vulcan：Automatic Query Planning for Live ML Analytics

- 准确信息：Yiwen Zhang et al., **“Vulcan: Automatic Query Planning for Live ML Analytics,” NSDI 2024**, pp. 1385–1402。 [USENIX 论文主页](https://www.usenix.org/conference/nsdi24/presentation/zhang-yiwen)；[论文 PDF](https://www.usenix.org/system/files/nsdi24-zhang-yiwen.pdf)
- 它为 live ML query 联合选择 pipeline、operator placement 和 query configuration，并在 edge hierarchy 中优化 accuracy、latency 和 resource consumption。
- 它不是可执行 Scheduler 程序进化。Pipeline 构造使用作者设计的 filter 指标，placement 使用复用与剪枝，configuration 使用 Bayesian Optimization；系统在这些人工定义的变量和算法结构中产生 query plan。
- 因而它适合作为“跨层 ML workflow planning 已经存在，但决策逻辑仍由系统设计者规定”的代表。它并非 Agentic Workflow，也不能用来支撑“Agentic Workflow 缺少跨层可见性”这一事实；后者应由 Murakkab 支撑。

## 2. Agentic Workflow × 人工规定的求解结构

### Murakkab

- 准确信息：Gohar Irfan Chaudhry et al., **“Murakkab: Resource-Efficient Agentic Workflow Orchestration in Cloud Platforms,” OSDI 2026**, pp. 567–587。它已经是正式 OSDI 2026 论文，不应只标成 2025 arXiv。 [USENIX 论文主页](https://www.usenix.org/conference/osdi26/presentation/chaudhry)；[论文 PDF](https://www.usenix.org/system/files/osdi26-chaudhry.pdf)
- Murakkab 用 declarative workflow abstraction 解耦 workflow specification 与 execution configuration；profile-guided optimizer 和 adaptive runtime 联合选择 workflow knobs、model/tool、hardware、parallelism、provisioning 与 routing。其 workload 包含 video Q/A、code generation 和 math Q/A。
- 其 optimizer 明确采用 **MILP**，在 SLO、demand、capacity 和 resource budget 约束下优化 energy、cost 或 accuracy；运行时按 epoch 重算，并由 auto-scaler 响应短时负载变化。
- 因而它是最重要的正面对照：它已经解决 Agentic Workflow 的 cross-layer visibility 和资源优化，不能把这些本身写成本文独有贡献。差异应落在：Murakkab 的调度逻辑/变量由 MILP 与 runtime 机制预先规定，不让 LLM 基于 Trace 改写可执行 Scheduler 程序。

### LLM-as-Scheduler

- 准确信息：Dawei Xiang et al., **“LLM-as-Scheduler: Agentic Workflow Dynamic Scheduling,” ACL 2026 Long Papers**, pp. 12752–12763。 [ACL Anthology](https://aclanthology.org/2026.acl-long.581/)
- 它按 query 动态选择/截断 agentic workflow，采用 lightweight gate 加 LLM scheduler 的两阶段 cascade，在 accuracy、latency 与 token cost 之间权衡。
- 这说明“LLM 参与 Agentic Workflow scheduling”也不能作为 novelty。它没有进化一个可复用的 Scheduler 程序，控制动作主要是 routing/workflow selection，因此仍归入预定义控制机制一侧。

## 3. 非 Agentic / 明确优化问题 × 可执行程序演化

### FunSearch

- 准确信息：Bernardino Romera-Paredes et al., **“Mathematical discoveries from program search with large language models,” Nature 625, 468–475 (2024)**；论文于 2023-12 在线发表。因此表格若按卷期年份应写 **Nature 2024**，也可注明 online 2023。 [Nature 论文](https://www.nature.com/articles/s41586-023-06924-6)
- FunSearch 将预训练 LLM、程序数据库和 evaluator 组成 evolutionary loop，生成并评分可执行函数，应用于 cap set、bin packing 和 job scheduling。
- 它已经证明 LLM 可在程序空间搜索；但通常依赖用户提供的 problem specification、efficient evaluator、程序 skeleton，并集中演化关键函数。Prompt 主要采样高分程序，而不是解释异构系统执行 Trace。

### EoH

- 准确信息：Fei Liu et al., **“Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model,” ICML 2024**, PMLR 235:32201–32223。 [PMLR 论文主页](https://proceedings.mlr.press/v235/liu24bs.html)
- EoH 在 evolutionary search 中共同演化自然语言 heuristic “thoughts”和可执行代码，实验覆盖 online bin packing、TSP 和 flow-shop scheduling。
- 它直接支撑“人工 heuristic 设计成本高”和“LLM 可以突破人工枚举的 heuristic component space”，但其反馈仍来自定义明确的 benchmark objective，不面向 Agentic Workflow 的多层系统 Trace。

### ReEvo

- 准确信息：Haoran Ye et al., **“ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution,” NeurIPS 2024**。 [NeurIPS 论文主页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4ced59d480e07d290b6f29fc8798f195-Abstract-Conference.html)；[论文 PDF](https://papers.neurips.cc/paper_files/paper/2024/file/4ced59d480e07d290b6f29fc8798f195-Paper-Conference.pdf)
- ReEvo 将 individual 编码成 heuristic code，以 pairwise relative fitness 形成 short-term reflection，再将经验累积为 long-term reflection，用于 crossover 和 elite mutation；作者明确将长期反思描述为避免 context memory blowup 的方法。
- 因此不能声称“现有 LLM evolution 只看单一总分”“不会压缩上下文”或“不会利用反馈修改逻辑”。可区分之处是：ReEvo 比较 heuristic code 与 meta-objective，而本文希望把 Workflow 条件、assignment 改变、失败状态和分解的系统后果组成有调度语义的 Trace evidence。

### ReVEL

- 当前状态：Cuong Van Duc et al., **“ReVEL: Multi-Turn Reflective LLM-Guided Heuristic Evolution via Structured Performance Feedback,” arXiv:2604.04940 (2026)**。本轮没有找到正式接收记录；公开稿还出现 ACL ARR/anonymous submission 版本，因此只能标 **arXiv 2026 / submission**，不能写成已接收 venue。 [arXiv 论文主页](https://arxiv.org/abs/2604.04940)
- ReVEL 用 performance-profile vector 对候选 heuristic 作 similarity/diversity grouping，并在组内进行多轮、累积反馈驱动的 reflection。
- 它是对“有效上下文”宽泛 novelty 的直接威胁。本文不能只声称“选择有价值的反馈”“structured performance feedback”或“让 LM 读懂 performance profile”。必须说明本文处理的是异构系统 Trace 的因果/对比语义，而 ReVEL 处理的是组合优化候选群体的 performance-profile grouping。

## 4. 必须讨论的近邻与 gap 威胁

### Vulcan：Instance-Optimal Systems Heuristics Through LLM-Driven Search

- 当前状态：Rohit Dwivedula et al., **“Vulcan: Instance-Optimal Systems Heuristics Through LLM-Driven Search,” arXiv:2512.25065 (2025 preprint)**。注意它与 NSDI 2024 的同名系统无关。 [arXiv 论文主页](https://arxiv.org/abs/2512.25065)
- 它把 resource-management policy 表示为 Value/Rank 型函数，在接口和 scaffold 内对 LLM 生成的 executable code 做 evolutionary search；案例包含 cache eviction 与 memory tiering，也讨论 scheduler 可表示为 runnable-task ranking。
- 这使“LLM evolution 只解决数学问题”“系统策略仍只能人工编写”以及“首次在系统中演化可执行策略”都不可成立。本文只能强调 Agentic Workflow 的跨节点语义—系统耦合，以及 Trace context 的特定表示。

### SchedCP / Towards Agentic OS

- 当前状态：**“Towards Agentic OS: An LLM Agent Framework for Linux Schedulers,” 2025 preprint / NeurIPS 2025 ML for Systems workshop listing**。 [NeurIPS 页面](https://neurips.cc/virtual/2025/129092)；[作者/Workshop PDF](https://mlforsystems.org/assets/papers/neurips2025/paper32.pdf)
- SchedCP 提供 Workload Analysis Engine、evolving Scheduler Policy Repository 和 Execution Verifier；sched-agent 使用 Observation、Planning、Execution、Learning agents，从 profiling/tracing 和部署反馈中生成、验证和改进 executable eBPF scheduler policy。
- 它不是 Agentic Workflow scheduling，而是用 agent 优化 Linux scheduler；但它已经覆盖“LLM + system traces/profiles + executable scheduler code + feedback refinement”。所以本文的独特性不能只建立在这些关键词的组合上，而必须落实到 Agentic Workflow 的 decision/outcome representation、受信评估边界和可验证的信息选择机制。

### GEPA

- 准确信息：Lakshya A. Agrawal et al., **“GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning,” ICLR 2026**。 [ICLR 论文主页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0e9e708b6f48e14fd0ac29e167413f76-Abstract-Conference.html)
- GEPA 从 compound AI system trajectory（reasoning、tool calls、tool outputs）中进行自然语言 reflection，提出并验证 prompt updates，并通过 Pareto frontier 合并经验。
- 它不生成端边云 Scheduler 代码，但已经占据“从 Agent/compound-AI trajectory 中提取语言反馈并演化系统组件”的大范围表述。本文应把对象限定为可执行 Scheduler policy，把 evidence 限定为受信 evaluator 产生的调度决策—系统后果关系，并以 GEPA 作为强 context/evolution baseline 或至少 related work。

## 5. 对 PSOCM 表述的直接修正

### 可以保留

- 传统 heuristic、RL 和优化方法通常需要开发者规定策略表示、变量、模型结构或训练机制，带来人工设计、建模、调参与训练成本。
- Agentic Workflow 使工具 Configuration 的语义质量、DAG 依赖、设备执行和通信代价发生耦合，需要统一、可信、可分解的反馈模型。
- Trace 的价值不只是 scalar fitness；本文可以研究如何保留调度条件、决策与后果的关系，并在上下文预算下选出对策略修改有判别力的证据。

### 必须收窄或删除

- 删除“现有 LLM evolution 只提供总分”：ReEvo、ReVEL、GEPA 已提供 reflection、structured/grouped feedback 或完整 trajectory。
- 删除“现有系统方法不能根据反馈修改逻辑”：Vulcan (2025) 与 SchedCP 已经生成/改进系统策略代码。
- 不把 cross-layer visibility、declarative Agentic Workflow abstraction 本身列为独有贡献：Murakkab 已明确提出并实现。
- 不把“LLM 用于 Agentic Workflow scheduling”列为独有贡献：LLM-as-Scheduler 已出现。
- 不使用“首个/首次”表述，除非后续完成可复现的系统综述式检索。

### 推荐的 L2 / C2 连接句

> 现有 LLM heuristic evolution 已能利用代码、fitness、performance profile 或 agent trajectory 形成反思，但这些反馈抽象主要服务于定义明确的组合优化、prompt optimization 或单一系统 policy。Agentic Workflow 的跨层调度 Trace 同时包含 DAG 条件、节点级 Configuration/Device 决策、合法性与失败状态，以及 quality、latency、communication 和 resource 后果；如何在上下文预算内保留这些关系并驱动完整 Scheduler 策略的可靠修改，仍是本文需要解决的核心挑战。

## 6. 写作与实验含义

- SOTA 表中把两篇 Vulcan 写全标题，避免混淆。
- 论文年份写成：FunSearch **Nature 2024 (online 2023)**；Murakkab **OSDI 2026**；ReVEL **arXiv 2026/submission**。
- M2 baseline 至少应覆盖：scalar-fitness evolution、ReEvo-style pairwise reflection、ReVEL-style performance-profile grouping，以及 GEPA-style full-trajectory reflection；否则无法证明本文的 Trace representation/selection 是必要设计。
- 系统 comparison 应至少讨论 Murakkab、LLM-as-Scheduler、Vulcan (2025) 和 SchedCP；它们分别封住“Agentic scheduling”“LLM scheduler”“system policy evolution”“scheduler code + traces”四个过宽 claim。
