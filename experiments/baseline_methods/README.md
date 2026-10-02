# Baseline 方法开发

这里保存六种对照方法的**产生过程**；最终参与统一评估的 Scheduler 源码放在
[`../../schedulers/baselines/`](../../schedulers/baselines/)。现有
`experiments/04_baseline/` 是演化实验的 Hydra 配置，不是这六种方法的开发目录。

规划三组、每组两种方法。确定具体方法后再建立对应子目录，不预先指定算法或模型：

```text
experiments/baseline_methods/
├── README.md
├── heuristic/<method_name>/
├── learning/<method_name>/
└── llm_optimization/<method_name>/

schedulers/baselines/
├── README.md
├── <heuristic_method_1>.py
├── <heuristic_method_2>.py
├── <learning_method_1>.py
├── <learning_method_2>.py
├── <llm_method_1>.py
└── <llm_method_2>.py
```

`<method_name>` 使用可辨识的方法名，同名 `.py` 是它唯一的评估入口。方法目录可放
训练、搜索或提示词迭代脚本、配置和方法说明；不要求三类方法拥有相同内部结构。
每种方法的说明至少写清输入数据、使用的 split、Profiling Database Snapshot、
优化目标与预算、随机种子（如适用）、选定最终策略的依据，以及生成
`schedulers/baselines/<method_name>.py` 的命令。生成记录和较大的中间产物保存在
`runs/` 或 `outputs/`，不要让评估入口依赖某次临时运行目录。

启发式方法在开发目录中记录规则和参数选择；学习方法在这里训练及选择模型；
LLM 优化方法在这里保留 prompt、模型配置、候选源码和选择过程。三类方法最终都
交付独立的 `propose(view)` 源码，由同一个 `scripts/eval.py` 加载。学习方法在
评估时使用的参数应固化在最终 `.py` 中，不读取训练目录里的临时权重；如果 LLM 方法在 `propose` 中
继续调用模型，须把该调用视为评估时的 Scheduler 计算，并记录模型与请求配置。

所有方法使用相同的离线三候选 DAG 数据、split、Profiling Database Snapshot 和
评分配置。当前 `ToolCallPlanDataset.split()` 按输入顺序切成 4:3:3；开发时可用
train 拟合、validation 选择，test 留给锁定策略后的比较。不要把 test 的评估结果
反馈到训练、提示词修改或策略筛选中。评估命令和接口见
[`../../schedulers/baselines/README.md`](../../schedulers/baselines/README.md)。

若某方法接入现有 Hydra 实验入口，组合配置仍遵循
[`../../docs/adr/0001-compose-experiments-with-hydra.md`](../../docs/adr/0001-compose-experiments-with-hydra.md)；
本目录本身不引入另一套配置机制。
