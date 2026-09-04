# PSOCM 方法论

## 结构组织： 
1. Problem 
2. SoTA & Limitations 
3. Opportunity 
4. Challenges 
5. Model 
6. Contributions 
    
## Problem 部分 
请明确给出： 
- 研究问题是什么
- Input 是什么
- Output 是什么 
- Significance 是什么 
  
要求： 
- 先把任务定义清楚，再说明问题的重要性 
- Problem definition 要尽量写成一句可作为论文主问题陈述的话 
- 研究问题重要性是向读者表示为什么你研究这个问题值得我们花时间去研究，也值得发表

## SoTA & Limitations 部分 

**这个部分尤为重要**

主要回答三个问题
- 前人已经在这个问题或相似问题上做了什么工作
- 已有工作如何分类？一般使用四宫格的方式整理，从两个维度整理现有工作（如下表）

|     | A1               | A2        |
| --- | ---------------- | --------- |
| B1  | A1-B1 和我们的工作最不相关 | A2-B1 类工作 |
| B2  | A1-B2 类工作        | Ours      |

- 每一类的工作有什么局限性Limitation？按照四宫格分类后，对每类工作的局限性进行总结

这一部分是重点。请不要只是罗列相关工作，而是要从两个维度对现有工作进行系统划分： 
1. 场景维度（Scenario Dimension） B1 B2
2. 技术维度（Technical Dimension） A1 A2

对每个维度，请进一步将现有工作划分为 2 类，并满足以下要求： 
- 这些类别必须尽可能“完备覆盖”现有工作，而不是只挑对我有利的几类 
- 类别之间应尽量相互独立、边界清晰 
- 分类标准必须统一，不能混用多个标准 
- 每一类都需要简要概括其代表性思路、适用条件和局限 
- 最终要明确指出：我的工作位于“哪一个场景类别 × 哪一个技术类别”的交叉点上 
- 要突出这个交叉点为什么尚未被充分研究，从而体现论文的唯一性 
  
在上述二维划分基础上，请提炼出 2 个最核心的 limitations，记为： 
- L1：场景维度上的 limitation 即 A1-B1 和 A2-B1 的 limitation
- L2：技术维度上的 limitation 即 A1-B2 的 limitation
  
要求： 
- L1 必须来源于场景维度的分类结果 
- L2 必须来源于技术维度的分类结果 
- limitation 必须是“结构性的共性问题”，而不是零散缺点 
- limitation 要能够自然导向后续的 opportunity 和 model design 
  
特别说明： 
- 场景维度用于回答“现有工作主要在哪些应用/系统场景下研究这个问题” 
- 技术维度用于回答“现有工作主要用哪些方法范式来解决这个问题” 
- 我的论文的新颖性，需要通过“场景维度 × 技术维度”的组合来明确定位 

  
## Opportunity 部分 

这一部分本质上是论文的 motivation。请不要默认 opportunity 一定来自 preliminary experiments。 相反，请先判断：这篇论文最适合哪一种 motivation pattern，再据此提炼 opportunities。 

可选的 motivation patterns 包括但不限于： 
1. Observation-driven: 来自 preliminary experiments / case studies / empirical findings 
2. Assumption-breakdown driven: 来自现有方法关键假设在新场景下失效 
3. Capability-shift driven: 来自新模型/新硬件/新平台/新接口能力的出现 
4. New-observability driven: 来自关键因素从不可观测变为可观测 
5. New-controllability driven: 来自系统出现新的 control knobs 或优化自由度 
6. Objective-gap driven: 来自现有 surrogate objective 与真实目标之间的不一致 
7. Slack-exploitation driven: 来自系统中未被利用的 slack / buffer / idle resources / tolerance 
8. Representation-mismatch driven: 来自现有表示方式不适合问题本质 
9. Structure-exploitation driven: 来自问题中潜在但未被利用的结构（如相关性、层次性、稀疏性、因果性、依赖性） 

请基于前面的 limitations，给出： 
- O1 -> L1 
- O2 -> L2 
   
需要注意：
- Opportunity 不是方法设计，不要提前写 model/module/framework 
- Opportunity 的作用是提供“新支点”，说明为什么这个问题值得以新的方式被研究 
- 不同论文可采用不同 motivation pattern，不要机械套用某一种 
- 选择的 motivation pattern 必须与论文内容最匹配，而不是形式上最漂亮 
- **opportunity 最终的呈现还是两段话，而不是分点的表现形式**
  
## Challenges 部分

请进一步从 opportunities 中提炼出 2 个核心 challenges，记为： 
- C1 -> O1 
- C2 -> O2 
  
要求： 
- challenge 不能只是重复 limitation 
- challenge 必须是“把 opportunity 落地时遇到的难题” 
- challenge 要写成明确的研究问题，如“如何建模……”“如何统一……”“如何优化……” 
  
## Model 部分 
请针对 challenges 给出对应的模型设计，记为： 
- M1 -> C1 
- M2 -> C2 
  
要求： 
- 每个模块都必须有明确目标，说明它是为了解决哪个 challenge 
- 说明每个模块的核心思想，有什么insight支撑了我们的设计（或者说这样设计背后的逻辑道理是什么） - 说明这些模块如何组成完整框架 
- 当两个Model无法解决上述两个挑战时，可以设计第三个model，但是也是针对两个challenges设计的。 

## F. Contributions 部分
请将贡献总结为三类： 
- Conceptually 
- Technically 
- Experimentally 
  
要求： 
- Conceptually：强调新问题、新场景组合、新视角或新范式 
- Technically：强调新模型、新机制、新系统设计 
- Experimentally：强调实验验证、效果提升、适用范围或 benchmark 贡献
