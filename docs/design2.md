# Design 2：基于 LLM 进化算法的调度策略生成

## 2. 研究目标与总体闭环

Design 2 搜索可执行的 Scheduler Candidate，而不是直接搜索某条工作流的一次性节点分配。每个 Candidate 是完整的 `propose(view)` 策略代码：输入只读 Scheduler View，输出该固定工具调用 DAG 每个节点的一个 Configuration 与 Compatible Device。可信评估器负责校验提案、模拟执行和计算分数；候选代码不能决定执行顺序、报告自己的时延或修改评分规则。LLM 负责根据诊断证据生成有语义结构的新代码，进化模块负责选择父代、维护行为多样性和记忆，最终候选仍由独立评估决定。

设固定 Evolution Trace Set 为 $\mathcal D_{\rm evo}=\{\tau_1,\ldots,\tau_n\}$。候选 $p$ 在 Trace $i$ 上的可信综合分数为 $s_i(p)$；被拒绝或执行失败的 Trace 记为零，同时保留原因。Candidate Score 为

$$F(p)=\frac{1}{n}\sum_{i=1}^{n}s_i(p).$$

每次进化从当前轮次已经评估的全部 Candidate 中选择父代，调用 Design 1 的 Bottleneck Diagnosis 或对照实验中的反思模块产生修改依据，LLM 输出新策略，可信评估器立即评分，再更新 Evolution Graph、Repertoire 和 Experience Memory。Evolution Graph 保存完整血缘，Repertoire 负责行为覆盖，二者不能混用。最终按 $F$ 选出候选，并仅在互不重叠的 Final Evaluation Trace Set 上比较可信结果与 Oracle Reference；Oracle 不参与进化轮次。

## 2.1 调度策略的行为特征空间

### 2.1.1 从可信 Trace 到行为向量

策略表现和策略行为需要分开表示。$F(p)$ 用于最终排序；Behavior Feature Vector $\phi(p)$ 描述策略如何做选择。基于现有评估字段，可定义

$$
\phi(p)=\big(\overline Q,\ \overline{T/L},\ \overline{R/M},\
\rho_{\rm device},\rho_{\rm edge},\rho_{\rm cloud},\
\rho_{\rm cross},\ \sigma_s,\ \rho_{\rm fail}\big),
$$

其中前三项为跨 scored Trace 的质量、时延和资源统计量；$\rho_{\rm device/edge/cloud}$ 是节点放置比例；$\rho_{\rm cross}$ 是跨设备依赖边比例；$\sigma_s$ 是跨 Trace 分数离散度；$\rho_{\rm fail}$ 是 rejected/failed 比例。所有量来自 Candidate Evaluation Evidence。为了避免毫秒、MiB 和比例在距离计算中互相压倒，使用在实验开始前冻结的参考范围进行截断与归一化；缺失或失败结果使用独立的缺失标记，不能简单当作“低时延、低资源”。吞吐量和请求间公平性只在多请求负载实验中追加为行为维度，单条 DAG 的模拟结果不足以推断这两项。

### 2.1.2 质量—多样性 Repertoire

将归一化特征空间按事先声明的分箱边界划为单元 $\mathcal B$。每个单元保存 Candidate Score 最高的代表及其血缘版本，覆盖率为

$$C(\mathcal R)=\frac{|\{b\in\mathcal B: \mathcal R_b\neq\varnothing\}|}{|\mathcal B|}.$$

候选进入空单元增加覆盖；在已有单元内仅当 $F$ 更高时替换代表。单元边界、维度与失败处理在一轮实验内固定，避免随着种群变化重定义“新颖”。对高维空间可使用预定义行为原型或稀疏网格，避免网格体积指数增长。完整 Evolution Graph 始终保留所有已评估候选供血缘追踪和父代选择；Repertoire 不删除图节点。

## 2.2 基于蒙特卡罗树的父代选择

### 2.2.1 搜索树与动作

将一次繁殖机会建成深度有限的搜索树。根节点表示当前种群；第一层选择算子 $o\in\{\text{mutation},\text{crossover}\}$；第二层选择第一个父代 $p_1$；若为 crossover，第三层选择与 $p_1$ 不同的 $p_2$。叶节点对应可执行动作 $a=(o,p_1)$ 或 $a=(o,p_1,p_2)$。父代集合取当前 Evolution Graph 中所有已评估 Candidate，不用单代窗口，也不把 Repertoire 误当作唯一父代池。每加入新 Candidate，树可增量扩展新动作，但既有动作的统计量保持可追踪。

采用“选择—扩展—生成—回传”迭代。选择阶段对已有子节点使用 UCB：

$$
\operatorname{UCB}(a)=\overline r_a+c\sqrt{\frac{\ln(1+N_{\rm parent})}{1+N_a}}+\eta\,\operatorname{prior}(a),
$$

其中 $N_a$ 是动作访问次数，$\overline r_a$ 是平均后代回报，先验由父代质量、行为距离及 Design 1 的诊断可操作性计算。未访问动作优先尝试；Candidate 数量增长时用渐进扩展控制分支数，例如允许的子动作数 $k(N)=\lceil\kappa N^\alpha\rceil$，$0<\alpha<1$。扩展阶段对 mutation 选择行为稀疏区域或具有可检验瓶颈的候选，对 crossover 优先选择高质量且行为互补的两位父代。

一次叶动作调用 LLM 生成一个新 Candidate，并用同一 Evolution Trace Set 评估。令父代参考值 $F_{\rm ref}$ 为单亲分数或双亲较高分，定义有界回报

$$
r=\operatorname{clip}_{[0,1]}\!\left(
\alpha_1\,g(F(p')-F_{\rm ref})+
\alpha_2\,\nu(p')+
\alpha_3\,\Delta C(p')+
\alpha_4\,\operatorname{stab}(p')
\right),\quad \sum_j\alpha_j=1.
$$

$g$ 将分数差按实验前固定尺度映射到 $[0,1]$；$\nu$ 是到最近已有行为向量的归一化距离；$\Delta C$ 表示新单元覆盖；$\operatorname{stab}$ 同时考虑失败率和跨 Trace 分数波动。拒绝或失败的候选给零回报并保留失败原因。回报沿叶到根回传以更新访问次数和均值。这样父代价值来自它们**产生后代的历史效果**，而非父代自己的当前分数；高分但连续产生重复后代的组合会逐渐失去优势。每轮至少执行一次真实生成和评估，不能用未运行的 LLM 猜测结果冒充 MCTS rollout。

树统计只服务于父代选择，不能代替可信 Candidate Score 或 Final Evaluation。对于三个初始根 Candidate 与少量轮次，可先以 Pareto/互补性先验确保每类动作被访问；预算增加后逐步让经验回报主导。实现时用统一的有状态繁殖选择器在新 Candidate 评估后调用 `update(action, reward)`，使算子和父代组合共享树回报。

### 2.2.2 一个可复核的选择示例

设 A 的 Candidate Score 最高且经常将节点放在云端，B 在边缘放置较多，C 的分数较低但会随网络证据改变放置。树的质量先验会让 A 较早被选中，覆盖与行为距离会保留 B、C 的探索机会。若 A+B 交叉得到一个在固定 Trace 集上同时降低时延与资源、进入新行为单元且无失败的后代，`(crossover,A,B)` 及其祖先统计获得较高回报。若 A 的连续变异仅重复已有行为或导致失败，其平均繁殖回报下降；树随后会更多尝试 B、C 或其他新父代。这个例子说明选择逻辑，实际收益须由可信评估数值计算。

## 2.3 面向特征空间覆盖的语义进化算子

### 2.3.1 目标区域引导的语义变异

在 Repertoire 中选择一个空单元或低质量单元 $b^*$。给每个单元计算目标优先级 $J(b)=\lambda_1\mathbf 1[b\text{ 为空}]+\lambda_2(1-F_b)+\lambda_3 P(b\text{ 可到达}\mid\text{历史变异})$；空单元的 $F_b$ 按零处理，权重在实验前固定。然后计算当前父代行为向量与该区域中心的差 $d=\operatorname{center}(b^*)-\phi(p)$。特征差本身不直接转成代码；先与 Design 1 的 Bottleneck Diagnosis 合并成可检验的修改约束。例如目标为“降低云端占比但保持质量”，诊断指出“非关键路径的云端放置增加双向传输”，则生成请求明确给出应修改的节点选择条件、允许牺牲的指标范围、相关 Trace 证据和父代完整源码。

LLM 输出完整的新 `propose(view)` 代码和 Scheduler Strategy Description；程序先进行语法检查，再由可信评估器校验每条 Trace 的接口与提案并测量。变异记录一个 Parent Scheduler Version。若目标单元未被到达，也记录实际 $\phi(p')$ 与失败原因，供树搜索和记忆更新。目标区域仅引导探索，不强制接受低质量结果。

### 2.3.2 互补行为驱动的策略交叉

对双亲 $p_1,p_2$，先找到各自具有最大相对优势的完整 Trace，并比较配置选择、设备放置、传输和质量损失。形成可组合机制表：例如父代一在高带宽状态下的云端加速规则，父代二在弱网络条件下的边缘保底规则。LLM 同时获得两份完整源码、Crossover Reflection、各自优势 Trace 和冲突条件，要求输出一份具有明确分支条件的完整策略；不指定某一父代为主代码。

交叉后要检查：每个 DAG 节点均被分配；设备与 Configuration 兼容；条件分支覆盖边界输入；源代码仍只通过 Scheduler View 读取证据。若无法同时保留两个优势，可信评估会显示退化，记忆记录具体退化 Trace，而不是只保存一个平均分。新 Candidate 记录两个 Parent Scheduler Versions。

## 2.4 长期跨代经验记忆与防退化

每轮保存结构化 Experience：`operator`、父代版本、目标行为单元、诊断证据引用、修改意图、源码摘要、子代版本、各 Trace 分数差、行为位移、失败/拒绝原因，以及快照和实验配置摘要。由此分别归纳成功模式、失败模式和仅在特定输入/状态下有效的条件模式。经验检索以工具类型、瓶颈类型、行为目标、工作流形状及快照兼容性过滤，再按证据强度与相似度排序；LLM 只能把检索结果当作生成线索，不能据此跳过可信评估。

记忆至少在一个 Evolution Loop 的所有代之间持续。跨运行复用时只迁移**经验摘要**和经过验证的规则，不把旧运行的 Candidate 自动加入新运行的 Parent Candidate Pool；每次新运行仍以三个不同的根 Candidate 开始，并在本轮固定 Trace 集上重新评估。快照摘要不同、质量指标变更或实验输入分布明显偏移的经验标记为待复核，不能直接作为性能保证。

防退化采用两道机制。第一，生成前从记忆检索“相似改动曾使哪些 Trace 回退”，把不可破坏的行为条件写入提示；第二，生成后在固定 Evolution Trace Set 上做逐 Trace 比较，若平均分未提升但填补高价值空单元，可保留为探索代表；若同时没有质量收益、覆盖收益且出现显著失败，则只保留在完整 Evolution Graph 与失败记忆中，不作为 Repertoire 优胜代表。最终结果仍按 Candidate Score 选择，任何覆盖奖励都不能改变最终可信评分。

