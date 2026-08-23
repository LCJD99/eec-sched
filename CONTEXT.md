# End-Edge-Cloud Scheduling

This context defines the fixed evidence and choices used to schedule tool-call DAGs across a three-device system.

## Language

**Profiling Database Snapshot**:
An immutable, versioned set of tool, Configuration, execution, and transfer evidence used as one replayable evaluator input.
_Avoid_: Live profiling database, benchmark results

**Configuration**:
A finite, opaque, prevalidated joint choice of every quality- or execution-relevant knob for one tool.
_Avoid_: Parameter combination, preset, Cartesian-product choice

**Quality Profile**:
Device-independent evidence for one tool Configuration's raw semantic metric and conservative normalized quality lower bound.
_Avoid_: Accuracy score, device profile

**Resource Score**:
An unqualified scalar for resource consumption; lower values indicate a less resource-intensive execution or transfer.
_Avoid_: Cost score, normalized rating

**Execution Profile**:
Measured warm latency, incremental execution Resource Score, and representative output size for one compatible Configuration-device pair.
_Avoid_: Latency profile, hardware benchmark

**Transfer Profile**:
Measured propagation, bandwidth, setup Resource Score, and per-byte Resource Score evidence for one directed cross-device path.
_Avoid_: Network estimate, undirected link

**Compatible Device**:
A device on which a Configuration has a complete Execution Profile; explicit incompatibility is distinct from an unfinished measurement.
_Avoid_: Available device, missing profile

**Input Bucket**:
The fixed representative input class under which every Execution Profile in a snapshot is measured.
_Avoid_: Dataset split, dynamic input size

**Scheduler Candidate**:
An identifiable version of untrusted Scheduler source code in the Evolution Graph. It retains its Parent Scheduler Versions, textual Scheduler Strategy Description, and detailed Candidate Evaluation Evidence; when evaluated, it receives one Scheduler View and returns only one Configuration and Compatible Device for every node of that fixed Tool-Call DAG.
_Avoid_: Policy parameters, scheduling plan, self-modifying evaluator

**Parent Scheduler Versions**:
The list of Scheduler Versions directly used to produce one Scheduler Candidate. A mutation normally records one parent and a crossover records two.
_Avoid_: Parent node ID, single mandatory parent

**Evolution Graph**:
The DAG of all Scheduler Candidates in one Evolution Loop, connected by their Parent Scheduler Versions. Every Evolution Loop creates a new graph that does not reuse an earlier graph's Candidates for parent selection. A fresh graph starts with three distinct root Candidates whose parent lists are empty; all three receive an initial Candidate Evaluation before evolution begins. Each graph node is a Scheduler Candidate; the unqualified term Node remains reserved for a Tool-Call DAG node.
_Avoid_: Candidate node, Tool-Call DAG, bounded parent archive

**Scheduler Strategy Description**:
A textual explanation of the scheduling strategy embodied by one Scheduler Candidate's complete executable source code.
_Avoid_: Reflection Advice, executable Scheduler code

**Candidate Evaluation Evidence**:
The retained per-Trace Scheduler Proposals and trusted evaluation results for one Scheduler Candidate, including detailed metrics and every Tool-Call DAG node's Configuration and Compatible Device assignment.
_Avoid_: Profiling Database Snapshot, Candidate-reported score

**Scheduler Version**:
A strict unique identifier for one Scheduler Candidate. Requests and results carry this identifier; the trusted adapter resolves it to the Candidate's code without embedding that code in the request.
_Avoid_: Non-unique Scheduler name, source code payload

**Scheduler View**:
The complete read-only scheduling information exposed to a Scheduler Candidate for exactly one Trace: the Tool-Call DAG, scoring context, Configuration and Compatible Device evidence, and directed Transfer Profile evidence from the active Profiling Database Snapshot. The trusted adapter constructs it; the Scheduler Candidate cannot access the whole request, the snapshot handle, or evaluator state.
_Avoid_: Mutable evaluator state, profiling database handle, whole evaluation request

**Scheduler Proposal**:
The Scheduler Candidate's complete assignment of one Configuration and Compatible Device to every node in its Scheduler View. The trusted evaluator, rather than the Candidate, validates it and determines elapsed time, execution order, and score.
_Avoid_: Candidate-reported timing, Candidate-controlled execution order, score claim

**Scheduler Computation Time**:
The wall-clock time the trusted evaluator measures while a Scheduler Candidate computes one Scheduler Proposal. It is stored as its own result field and, for a scored proposal, is included in the total latency used for the final score.
_Avoid_: Candidate-reported time, simulated tool execution time

**Canonical Evaluation Order**:
The trusted evaluator's deterministic order for simulating a Tool-Call DAG: dependency order first, then stable node-ID order when multiple nodes are ready. A Scheduler Candidate cannot change this order.
_Avoid_: Scheduler-controlled ordering, runtime-dependent ordering

**Oracle Reference**:
The trusted reference score calculated only during final evaluation, using the same score definition as a Scheduler Candidate for comparison.
_Avoid_: Evolution-loop signal, real execution result, scoring formula

**Candidate Score**:
The simple arithmetic mean over every Trace in one fixed Trace set, used to compare Scheduler Candidates during evolution. A `failed` or `rejected` Trace contributes zero while retaining its failure or rejection reason.
_Avoid_: Oracle score, weighted score, per-Trace score

**Evaluation Status**:
One of `scored` for a valid proposal with a score, `rejected` for invalid input or assignments, or `failed` when the Scheduler Candidate cannot run to completion. A trusted-evaluator fault is reported as a system error, not as a Scheduler Candidate status.
_Avoid_: Feasible, infeasible, detailed failure taxonomy
**Evolution Module**:
The module that compares Scheduler Candidates using every per-Trace score, selects relevant full Trace contexts for model context, and asks a model for a proposed Scheduler Candidate change. It does not calculate scores or change trusted evaluation rules.
_Avoid_: Scheduler Candidate, trusted evaluator, automatic evaluator modification

**Model Context**:
The selected full Trace contexts supplied by the Evolution Module to a model, including the DAG, Scheduler Candidate choices, evaluation result, and score.
_Avoid_: All benchmark data, mutable evaluator input

**Trace Selection Strategy**:
A replaceable rule used by the Evolution Module to select full Trace contexts from all per-Trace scores for Model Context; selecting the best and worst scores is one possible rule.
_Avoid_: Fixed best-and-worst rule, score calculation

**Parent Selection Strategy**:
A rule that selects one or more Scheduler Candidates from the Parent Candidate Pool to produce a new Candidate. Mutation and crossover may use different Parent Selection Strategies; neither determines which Candidate is retained as the final winner.
_Avoid_: Final Candidate selection, Trace Selection Strategy

**Mutation Parent Selection Strategy**:
The Parent Selection Strategy that forms the Pareto frontier of the Parent Candidate Pool using Trace Score Vectors, then uniformly samples one Candidate as the mutation parent.
_Avoid_: Crossover parent selection, Candidate Score winner selection

**Crossover Parent Selection Strategy**:
The Parent Selection Strategy that takes up to five Scheduler Candidates with the highest Candidate Scores, then selects the pair whose Trace Score Vectors have the lowest cosine similarity. When fewer than five Candidates exist, it uses all available Candidates.
_Avoid_: Uniform Pareto sampling, arbitrary Candidate pair

**Evolution Operator Selection Strategy**:
A probabilistic rule that selects mutation or crossover for each Evolution Loop round using configurable probabilities.
_Avoid_: Fixed operator sequence, stagnation-triggered crossover

**Scheduler Mutation**:
The evolution operation that selects one Scheduler Candidate through the Mutation Parent Selection Strategy, forms Lineage Reflection against that Candidate's direct parents, and asks a Coding Agent to modify the selected Candidate's complete source code. The resulting Candidate records the selected Candidate as its one direct parent.
_Avoid_: Two-parent crossover, comparison with an independently sampled Candidate

**Scheduler Crossover**:
The evolution operation that selects two Scheduler Candidates through the Crossover Parent Selection Strategy, forms Crossover Reflection, and gives both complete parent source codes to a Coding Agent without designating either as the primary codebase. The Coding Agent synthesizes one new complete Scheduler Candidate, which records both selected Candidates as its direct parents.
_Avoid_: Scheduler Mutation, arbitrary two-Candidate merge

**Parent Candidate Pool**:
Every Scheduler Candidate evaluated so far in the active Evolution Loop, regardless of whether its Trace evaluations are scored, rejected, or failed. Parent selection compares the complete pool without a generation-only window, status-based eligibility filter, or capacity limit.
_Avoid_: Current generation, bounded parent archive, cross-run candidate archive

**Trace Score Vector**:
The ordered vector of one Scheduler Candidate's scalar per-Trace score contributions over the fixed Evolution Trace Set. Candidates may be compared by Pareto dominance across these Trace dimensions for parent selection, while final-winner selection continues to use Candidate Score.
_Avoid_: Multi-objective evaluator output, unordered Trace scores, Candidate Score

**Lineage Reflection**:
Model Context evidence formed by comparing a Scheduler Candidate with each of its direct parents over aligned Trace Score Vectors. For every parent, it includes the full Trace context for the Candidate's greatest improvement and greatest regression relative to that parent.
_Avoid_: Comparison with an arbitrary Candidate, aggregate-score-only feedback

**Root Reflection**:
The initial mutation evidence for a root Scheduler Candidate with no direct parent. It includes that Candidate's highest-scoring and lowest-scoring full Trace contexts; descendants use Lineage Reflection instead.
_Avoid_: Invented parent comparison, descendant reflection

**Crossover Reflection**:
Model Context evidence formed by comparing two crossover parents over aligned Trace Score Vectors. It includes the full Trace context where each parent has its greatest score advantage over the other.
_Avoid_: Each parent's absolute best Trace, all Trace contexts, Lineage Reflection

**Reflection Advice**:
The textual Scheduler change guidance produced from selected Candidate Evaluation Evidence. A Coding Agent uses it with parent Scheduler source code to produce a new complete Scheduler Candidate and Scheduler Strategy Description.
_Avoid_: Code patch, Scheduler Strategy Description, trusted evaluation result

**Reflection Input Policy**:
A replaceable rule for constructing Reflection Agent input. The current policy supplies the relevant Scheduler Strategy Descriptions and selected full Trace evidence, but excludes Scheduler source code; experiments may replace the policy without changing Candidate Evaluation or Coding Agent authority.
_Avoid_: Fixed prompt implementation, Coding Agent input, source-code execution

**Evolution Loop**:
A configured fixed-number sequence in which every round probabilistically selects mutation or crossover, produces one new Scheduler Candidate, evaluates it, and adds it to the active Evolution Graph before comparing Candidate Scores.
_Avoid_: Open-ended agent loop, evaluator mutation

**Candidate Evaluation**:
The trusted Evaluation run for every Scheduler Candidate during the Evolution Loop. It produces per-Trace scores and a Candidate Score without calculating an Oracle Reference.
_Avoid_: Final Evaluation, Oracle search

**Final Evaluation**:
The evaluation performed after the Evolution Loop for the active Evolution Graph's highest-Candidate-Score Scheduler Candidate, using the independent Final Evaluation Trace Set. It additionally calculates an Oracle Reference and reports their comparison.
_Avoid_: Per-round candidate evaluation, Oracle inside evolution

**Evolution Trace Set**:
The fixed Trace set used for every Candidate Evaluation in one Evolution Loop.
_Avoid_: Final Evaluation Trace Set, selected Model Context

**Final Evaluation Trace Set**:
The independent Trace set used only in Final Evaluation for the selected Scheduler Candidate and its Oracle Reference comparison.
_Avoid_: Evolution Trace Set, evolution input
