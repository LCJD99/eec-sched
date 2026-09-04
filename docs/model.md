# 把 eec-sched 建模成一个“类似强化学习”的系统

## 1. 结论

这个系统不适合硬塞进一个单层强化学习模型。更贴合现状的表达是两个嵌套层次：

1. **Trace 内层：一次性调度决策。** 一个 Scheduler Candidate 看到一个只读 `SchedulerView`，一次返回整个 Tool-Call DAG 的 Configuration 和 Compatible Device 分配，然后由可信评估器统一模拟和打分。它更像“一步结束的上下文决策”（可类比 contextual bandit），而不是逐节点控制。
2. **Evolution 外层：多轮代码改进。** Evolution Loop 根据已有 Candidate 的逐 Trace 结果选择父代，让 Reflection Agent 总结证据，让 Coding Agent 产生一份新的完整 Scheduler 源码，再交给同一个可信评估器。这个过程像有限步长的强化学习，但真正被更新的是程序代码，而不是神经网络参数。

第二层对应《[Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/)》所说的 Heuristic Learning：共享“状态—动作—反馈—更新”闭环，但让 coding agent 根据 reward、测试、日志、回放等显式证据修改软件结构，而不是做反向传播。文章的[公开 artifact 仓库](https://github.com/Trinkle23897/learning-beyond-gradients)也把策略程序、实验记录和复现材料放在一起。本文借用这个思路；eec-sched 的 JSONL Experience Memory 是显式跨运行证据，不代表模型权重训练、自动回归或在线持续学习。

从已有研究范式看，这个外层也更接近 [FunSearch 原论文](https://www.nature.com/articles/s41586-023-06924-6)和 [AlphaEvolve 白皮书](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)中的“LLM 生成程序 + 自动评估器 + 候选群体/程序库”闭环，而不是 policy-gradient 训练。这里引用它们只用于说明方法类别，不表示 eec-sched 已经实现了这些系统的全部搜索、并行评估或程序数据库能力。

标准强化学习用 Agent、Environment、State、Action、Reward、Transition、Episode 和 Return 描述交互；这里沿用这套词汇，含义以 [Sutton 与 Barto《Reinforcement Learning: An Introduction》第 2 版第 2–3 章](http://incompleteideas.net/book/RLbook2020.pdf)为准。

## 2. 两层闭环

```mermaid
flowchart LR
    roots["3 个根 Scheduler Candidates"] --> eval["在固定 Evolution Trace Set 上可信评估"]
    eval --> graph["Evolution Graph：源码、父代、策略说明、逐 Trace 证据"]
    graph --> select["选择 mutation / crossover 和父代"]
    select --> reflect["Reflection Agent 读取选中的证据"]
    reflect --> code["Coding Agent 生成新的完整 Scheduler 源码"]
    code --> eval
    graph -->|"固定轮数结束，按 Candidate Score 选最高者"| final["独立 Final Evaluation Trace Set"]
    oracle["Oracle Reference，仅最终评估可用"] --> final
    final --> report["Candidate / Oracle 对比报告；不再回写外循环"]
```

这张图描述的是当前 [`EvolutionLoop`](../src/eec_sched/evolution/engine.py) 的控制边界。Hydra 入口 [`app/run_evolution.py`](../src/eec_sched/app/run_evolution.py) 会组合 Diagnosis、Coding、Memory、行为特征、父代选择、可信评估器和 Oracle 后启动完整循环。

## 3. Trace 内层：一步结束的结构化决策

### 3.1 概念对应

| 强化学习概念 | eec-sched 中的对应物 | 场景内的准确含义 |
|---|---|---|
| Environment | 固定的 Profiling Database Snapshot、一个 Evaluation Trace 和可信评估器 | Environment 持有测量证据、校验、计时、Canonical Evaluation Order、仿真和打分权力。Candidate 不能修改这些规则。 |
| State | `EvaluationTrace + ProfilingDatabaseSnapshot +` 可信评估状态 | 这是环境用来决定结果的完整事实，不等于 Candidate 实际能看到的内容。 |
| Observation | 一个 Trace 对应的只读 `SchedulerView` | 包含 Tool-Call DAG、打分上下文、相关 Configuration / Compatible Device / Transfer Profile 证据；不暴露可变 snapshot 句柄或评估器状态。当前 `task_input` 被保留在 Trace 中，但不直接放进 `SchedulerView`。 |
| Policy | 一个带严格 Scheduler Version 的 `SchedulerCandidate.propose(view)` 源码 | Policy 是可执行程序，不是神经网络参数。相同 Candidate 会在固定 Evolution Trace Set 的每个 Trace 上调用一次。 |
| Action | 完整 `SchedulerProposal` | 一次给 DAG 的**每个节点**选择一个 Configuration 和 Compatible Device。这是一个组合式联合动作，不是逐节点动作序列。 |
| Transition | 可信评估器校验 Proposal、测量 Scheduler Computation Time、按固定顺序仿真，再生成 `TraceEvaluation` | Candidate 只提议分配；它不能自行报告时间、选择执行顺序、计算 reward 或决定自己是否有效。 |
| Reward | `TraceEvaluation.score_contribution` | `scored` 时取可信 utility；`rejected` 或 `failed` 时取 `0`，同时保留原因。 |
| Terminal | 得到 `scored`、`rejected` 或 `failed` 的 Trace 结果 | 一个 Proposal 被评估后，这个 Trace episode 立即结束，没有下一次节点级动作。 |
| Episode | 一次 `(SchedulerView, SchedulerProposal, TraceEvaluation)` | 一个 Trace 就是一个一步 episode。相同 DAG 的两次 Planning occurrence 仍然是两个 Trace。 |
| Return | 该 Trace 的 reward | 因为只有一步，折扣因子不影响结果。多个 Trace 的目标由外层聚合。 |

这组对应直接来自当前 [`candidate.py`](../src/eec_sched/candidate.py) 的 `SchedulerView`、`SchedulerCandidate`、`EvaluationTrace` 和 `TraceEvaluation`，以及 [`evaluate_scheduler_instance`](../src/eec_sched/evaluation/evaluator.py) 的权限划分。

### 3.2 形式化

固定一个 Profiling Database Snapshot \(D\)。对第 \(i\) 个 Trace：

\[
x_i=(\text{task input}_i,\;\text{DAG}_i), \qquad
o_i=\operatorname{View}(D,x_i)
\]

Scheduler Version 为 \(v\) 的程序策略 \(\pi_v\) 一次产生完整联合动作：

\[
a_i=\pi_v(o_i)
=\{n\mapsto(c_n,d_n)\mid n\in\text{DAG}_i\}
\]

其中 \(c_n\) 是节点 \(n\) 的 Configuration，\(d_n\) 是 Compatible Device。可信评估器 \(E_D\) 决定状态、报告和分数：

\[
(z_i,m_i,u_i)=E_D(x_i,a_i),
\qquad z_i\in\{\texttt{scored},\texttt{rejected},\texttt{failed}\}
\]

当前实现中的 Trace reward 是：

\[
r_i(v)=
\begin{cases}
u_i,& z_i=\texttt{scored}\\
0,& z_i\in\{\texttt{rejected},\texttt{failed}\}
\end{cases}
\]

在固定 Evolution Trace Set \(\mathcal T_E\) 上，Candidate Score 是完整 Trace 集上的算术平均：

\[
J_E(v)=\frac{1}{|\mathcal T_E|}\sum_{i\in\mathcal T_E}r_i(v)
\]

因此，内层最准确的称呼是**一步、结构化动作的决策问题**。**Tool-Call DAG Node 不是 RL time step**：把每个 DAG 节点写成一步会改变真实权限，当前 Candidate 不能决定 Canonical Evaluation Order，也不会在分配一个节点后看到新的环境状态。

### 3.3 当前 reward 到底奖励什么

可信评估器先计算：

- 最终输出所依赖节点的 Quality Profile 下界乘积 \(q\)；
- `Scheduler Computation Time + simulated makespan` 得到的 latency proxy \(L\)；
- 执行与跨设备传输 Resource Score 之和 \(E\)。

然后使用当前 [`evaluation/scoring.py`](../src/eec_sched/evaluation/scoring.py) 中的连续 utility：

\[
\widehat q=
\begin{cases}
0,&q_{min}=1\\
\dfrac{q-q_{min}}{1-q_{min}},&q_{min}<1
\end{cases},
\qquad
p=\gamma_{score}\widehat q
 +(1-\gamma_{score})\frac{L_{max}-L}{L_{max}},
\qquad
u=\frac{p}{E+\epsilon}
\]

这里代码里的 `gamma` 是质量与时延的权重，本文记为 \(\gamma_{score}\)；它**不是**强化学习里的时间折扣因子。`accuracy_feasible` 和 `latency_feasible` 目前只是诊断字段，不是硬门槛。

还有一个需要保留的现状事实：有效 Proposal 的 utility 可能为负，而 `rejected` / `failed` 的贡献固定为零。因此在某些数据下，无效 Candidate 可能比负分但有效的 Candidate 更占优。这是把现有分数直接当 RL reward 时必须显式承认的语义；本文不暗中修改它。如果以后要加入失败惩罚或 reward 截断，应当作为单独的评分契约决策。

## 4. Evolution 外层：有限轮次的代码搜索

### 4.1 Agent、状态、动作和环境

外层不能把 Coding Agent 单独当成一个看见全部状态的 RL Agent。当前控制被拆成了固定的 Evolution Loop、Reflection Agent 和 Coding Agent：

| 强化学习概念 | 外层对应物 | 当前逻辑 |
|---|---|---|
| Agent | 整个 Evolution Module | 包括固定的 operator / parent selection、Reflection Agent 和 Coding Agent。单独任何一部分都没有完整的决策信息和权限。 |
| Environment | 可信评估器、固定 Snapshot、固定 Evolution Trace Set | 对任意新 Candidate 返回不可由模型伪造的逐 Trace 结果与 Candidate Score。 |
| State \(s_t\) | 当前 Evolution Graph、已评估 Candidate、剩余轮数及本轮固定配置 | Graph 中保留所有已评估 Candidate 的源码、父代、策略说明和证据。严格的随机过程还要把随机选择和模型生成的不确定性看作 transition distribution。 |
| Observation | 传给两个模型角色的受限上下文 | Reflection Agent 只看相关策略说明与选中的完整 Trace 证据，不看源码；Coding Agent 看 Reflection Advice 和一个或两个父代的完整源码。Observation 小于完整 state。 |
| Policy / update rule | operator selection、parent selection、reflection 和 code generation 的组合规则 | 这套规则决定“下一份程序从哪里来”，但当前没有对这套外层规则做梯度训练。 |
| Action \(a_t\) | 产生一个新的完整 Scheduler Candidate | 从系统整体看，动作包括选中的 mutation / crossover、父代，以及生成的新源码和策略说明；其中前两项由固定规则决定，源码由模型提出。 |
| Transition | 校验版本与父代关系，在全部 Evolution Trace 上可信评估，然后把 Candidate 和 Evaluation 追加到 Graph | Graph 只增长；较差、`rejected` 或 `failed` 的 Candidate 也保留，不会回滚环境。 |
| Terminal | 完成配置的固定 rounds | Candidate 失败只产生零分证据，不终止外循环；可信评估器自身故障、重复版本或错误父代等契约错误会中止运行。 |

### 4.2 外层目标和一个自洽的 reward 定义

当前代码没有名为“outer reward”的字段。它直接在固定轮数结束后，从整个 Graph 选择 Candidate Score 最高者：

\[
v^*=\arg\max_{v\in\mathcal G_T}J_E(v)
\]

如果需要把这个过程写成标准有限时域 return，可定义一个**派生量**，而不是声称代码已经实现了新 reward：

\[
B_t=\max_{v\in\mathcal G_t}J_E(v),
\qquad
R_t^{outer}=B_{t+1}-B_t
\]

取外层折扣 \(\delta=1\)，则：

\[
\sum_{t=0}^{T-1}R_t^{outer}=B_T-B_0
\]

因此最大化外层累计 reward 与当前“固定轮数后保留最高 Candidate Score”目标一致。这个定义不会因为生成许多一般水平的 Candidate 而虚增 return，也不会把 Final Evaluation 的信息泄漏回搜索过程。它只是推荐的数学表达；当前实现真正存储和使用的仍然是逐 Trace 结果、Trace Score Vector 和 Candidate Score。

### 4.3 Exploration 与 exploitation

当前外层的探索不是 epsilon-greedy，也没有 learned value function。Hydra 可在两组策略之间切换：

- 每轮按配置概率选择 Scheduler Mutation 或 Scheduler Crossover；
- `search=legacy`：mutation 从 Trace Score Vector 的 Pareto frontier 抽取父代，crossover 从高分 Candidate 中选择 Trace Score Vector 余弦相似度最低的一对；
- `search=psocm`：mutation 保留行为特征空间中的稀疏候选，crossover 选择 Behavior Feature Vector 互补的一对；
- Coding Agent 根据父代源码和 Reflection Advice 生成新程序，提供更开放的程序空间探索；
- 固定轮数后只按 Candidate Score 选最高者，这是最终 exploitation 规则，和父代选择规则是两件事。

这些规则由 [`EvolutionLoop`](../src/eec_sched/evolution/engine.py) 和可替换的 [`strategies.py`](../src/eec_sched/evolution/strategies.py) 给出。它们更接近带显式谱系的黑盒程序搜索，而不是 policy-gradient、Q-learning 或 actor-critic 算法。

## 5. Episode、transition 与 credit assignment

### 5.1 两种 episode 不要混用

| 层次 | episode 起点 | 一个 transition | episode 终点 |
|---|---|---|---|
| Trace 内层 | Candidate 收到一个 `SchedulerView` | 返回完整 Proposal，可信评估器生成一个 `TraceEvaluation` | 立即以 `scored` / `rejected` / `failed` 结束 |
| Evolution 外层 | 三个根 Candidate 已在 Evolution Trace Set 上完成初始评估 | 选择父代、生成一份新源码、评估并追加一个 Graph 节点 | 固定 rounds 完成，选出最高 Candidate Score 的 Candidate |

Final Evaluation 是外层 episode 结束后的留出评估，不是下一步 transition，也不应该产生下一轮代码更新。

Oracle Reference 也**不是 critic**：它不在 Evolution Loop 中估计 \(V(s)\) 或 \(Q(s,a)\)，不为 Candidate 生成 advantage，也不参与父代选择。它只在选出最终 Candidate 后提供同分数定义下的参考结果。

### 5.2 当前系统如何分配“功劳”

内层只有一个联合动作，所以没有跨时间的 temporal credit assignment。一个 Trace 的 scalar score 评价整份 Proposal；每个节点的 Configuration、Compatible Device、执行和传输明细只是可诊断证据，并不是独立的 per-node reward。

外层的 credit assignment 是显式、基于案例且不经过反向传播的：

- 根 Candidate 没有父代，Root Reflection 选它最高分和最低分的 Trace；
- descendant mutation 把 Candidate 与每个直接父代对齐比较，选最大进步和最大退步的 Trace；
- crossover 选两个父代各自相对占优最大的 Trace；
- Reflection Agent 把这些证据变成文字建议，Coding Agent 再把建议落实为完整代码；
- 每份新代码随后独立跑完整 Evolution Trace Set，真实效果不能由模型自己声明。

这相当于把“哪类 Trace 变好或变坏”作为局部 credit signal。它能指导下一次代码改动，但当前没有 \(V(s)\)、\(Q(s,a)\)、TD bootstrap、advantage estimator 或 replay-buffer sampling，也没有把信用继续分配到某一行源码或某一次模型 token 生成。

## 6. 训练信号与最终评估必须隔离

| 边界 | Evolution / 类训练阶段 | Final Evaluation / 最终报告阶段 |
|---|---|---|
| Trace | 固定 Evolution Trace Set；每个 Candidate 都跑完整集合 | 独立 Final Evaluation Trace Set；当前构造器至少强制两边 `trace_id` 不重合 |
| 用途 | 产生逐 Trace 反馈、选择父代、生成新 Candidate、选择最终 Candidate | 只评估已经选出的一个 Candidate |
| Oracle Reference | 禁止使用 | 必须由可信依赖提供，并与 Candidate 使用同一分数定义比较 |
| 是否回写学习 | 可以进入 Reflection / Coding 的后续上下文 | 不可以；否则这批 Trace 就不再是最终留出集 |
| Snapshot | 当前实现两阶段使用同一个 Profiling Database Snapshot | 因而当前只检验对新 Trace 的泛化，不检验设备或 Profile 分布变化下的泛化 |

这里的“训练”是类比：模型权重没有在本仓库内更新，发生的是 Candidate 源码搜索和选择。如果根据 Final Evaluation 结果改 prompt、改源码、调 rounds 或重选 Candidate，那么这次 Final Evaluation 已经变成开发/验证反馈；要保持可信最终结果，必须再换一套未使用过的 Final Evaluation Trace Set。

## 7. 当前已经具备什么，尚未具备什么

### 已实现的闭环部件

- 严格版本化、带父代和策略说明的 Scheduler Candidate；
- 每次运行从三个无父代根 Candidate 开始的 Evolution Graph；
- 固定 Evolution Trace Set 上的逐 Trace 可信评估和算术平均 Candidate Score；
- mutation / crossover、父代选择、受限 Reflection 上下文和完整源码生成接口；
- `scored`、`rejected`、`failed` 三种 Candidate 结果以及零分贡献规则；
- 固定轮数终止、最高 Candidate Score 选择；
- 独立 Final Evaluation Trace Set 和仅最终可用的 Oracle Reference。

### 不应被文档提前宣称的能力

- **不是 Deep RL 训练。** 没有梯度、策略网络、critic 或模型权重更新。
- **不是逐节点在线控制。** Candidate 一次返回完整分配，执行顺序由可信评估器决定。
- **不是模型权重层面的 Continual Learning。** Evolution Graph 仍属于单次 Evolution Loop；`memory=jsonl` 可以跨运行保留显式 Experience Memory，但不会训练或更新模型权重。
- **不是在线自治系统。** Hydra 入口可以组合并运行完整离线实验，但 Trace、根 Candidate、可信 evaluator 和 Oracle 都来自显式配置，不会由模型自行替换。
- **不是对真实设备的在线试错。** 当前 reward 来自固定 Profiling Database Snapshot 上的确定性仿真和可信计时，而不是在 device / edge / cloud 上执行工具后得到的在线环境 reward。

## 8. 推荐采用的统一说法

> eec-sched 把 Scheduler 源码视为可演化的程序策略。对单个 Trace，Scheduler 在只读视图上做一次完整联合分配，可信评估器负责校验、仿真、计时和 reward；对一次 Evolution Loop，系统在固定 Trace 集上反复生成和评估新程序，用显式逐 Trace 证据完成无梯度的代码级 credit assignment，并在固定轮数后选择最高 Candidate Score 的程序。独立 Final Evaluation 只负责报告泛化结果和 Oracle 差距，不参与程序更新。

这是一种**类似强化学习的、带可信环境反馈的程序演化**。它与 Heuristic Learning 的核心思路一致，但当前落地边界是“单次、有限轮次、固定评估集上的 Scheduler 代码搜索”，还不是完整的长期 Heuristic System。
