# 实验记录
第一次完整跑通整个流程

## 流程
完整执行的流程是

1. 执行进化
```bash
uv run scripts/run_evolution.py --config runs/001_first_workflow/config.yaml
```

2. 使用可视化工具分析
```bash
uv run prototypes/evolution_visualizer/server.py --result_path runs/001_first_workflow/xxx.json
```

## 结果分析

从目前这个实验数据看出进化的效果并不好，最优结果仍是初始的版本。可能的原因有如下：

1. qwen3-8b 模型本身的能力不够强，无法在进化中产生更优的结果。
2. 进化的 evaluator 存在问题，没有完整评估从端侧的流程（但是这个不是主要原因）
3. 单节点的数据太多，并不需要复杂的策略就能找到较好的策略
