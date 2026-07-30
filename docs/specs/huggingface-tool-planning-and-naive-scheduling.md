# Hugging Face Tool Planning and Naive Scheduling Framework

## Problem Statement

The project needs an executable foundation for composing heterogeneous Hugging Face models as tools before research begins on the core scheduler. Today the repository contains isolated model experiments, but there is no common tool contract, no type-checked composition model, no LLM-driven planner, no device-aware profiling interface, no accuracy/latency objective, and no executor that can run a planned graph.

A user should be able to provide a natural-language request plus image/text inputs and explicit accuracy/latency requirements. The system should plan a tool-call DAG, select one profiled configuration for each tool with a transparent naive heuristic, and execute the graph using real Hugging Face models. The foundation must keep the future scheduler replaceable without coupling planning, tool execution, or profiling to the naive algorithm.

## Solution

Build a small end-to-end framework around a registry of Hugging Face tools. Each tool exposes lightweight metadata through a `ToolSpec` and performs inference through a separate `ToolRunner`. Tools communicate only through named ports whose modality is `image` or `text`; internally, those values are represented uniformly as PIL images and Python strings.

An LLM planner receives the request and tool catalog and returns a tool-level DAG. A deterministic validator rejects unknown tools, missing ports, modality mismatches, cycles, and invalid final outputs, with bounded LLM repair attempts. The planner never chooses model variants or execution parameters.

Each tool exposes a finite set of legal, opaque configurations covering dimensions such as Model Variant, Input Fidelity, and Approximation. Offline accuracy and latency profiling supplies the measurements needed by scheduling. Accuracy is normalized per tool relative to a declared floor and the best measured reference configuration. DAG accuracy is estimated by multiplying node qualities. Warm, device-specific p95 tool latency is summed because the first executor is single-device and serial.

Requests explicitly provide minimum accuracy `A_m`, maximum latency `L_m`, and preference `gamma`. These are used as hard feasibility constraints and in a normalized accuracy/latency objective. The naive scheduler uses measured per-node Pareto frontiers and a greedy downgrade/local-improvement heuristic rather than exhaustive search. The executor then prepares the selected real models, runs the validated DAG, and reports the plan, selected configurations, predictions, results, and execution observations.

The first milestone proves the architecture with three or four representative real Hugging Face tools and at least one multi-node DAG. The concrete production tool and model catalogs will be supplied separately.

## User Stories

1. As a framework user, I want to submit a natural-language request with image and/or text inputs, so that the system can compose models on my behalf.
2. As a framework user, I want to provide a minimum acceptable accuracy, so that selected configurations do not intentionally fall below my quality requirement.
3. As a framework user, I want to provide a maximum acceptable latency, so that the scheduler can respect my response-time requirement.
4. As a framework user, I want to provide an accuracy-versus-latency preference, so that feasible plans reflect my priorities.
5. As a framework user, I want the system to return the final image or text result, so that I can consume the planned workflow output.
6. As a framework user, I want to see the planned DAG, so that I can understand which tools were selected and how data flows between them.
7. As a framework user, I want to see the selected configuration for every node, so that a run is explainable.
8. As a framework user, I want predicted DAG accuracy, latency, and utility reported, so that I can inspect the scheduler decision.
9. As a tool author, I want to register a tool through lightweight metadata, so that planning and scheduling do not load its model.
10. As a tool author, I want to define named input and output ports, so that tools with multiple inputs of the same modality remain unambiguous.
11. As a tool author, I want ports to use only the `image` and `text` modalities, so that the first composition model stays intentionally small.
12. As a tool author, I want model-specific preprocessing and postprocessing isolated in a runner, so that downstream tools receive uniform values.
13. As a tool author, I want to expose multiple legal configurations for one tool, so that different model variants and approximation levels can be scheduled.
14. As a tool author, I want configurations to remain opaque to the scheduler, so that the framework does not encode unreliable assumptions about model names or fidelity labels.
15. As a tool author, I want to supply a tool-specific evaluation dataset, metric, metric direction, and floor, so that quality has a meaningful definition.
16. As a tool author, I want a tool to remain executable even before accuracy profiling is available, so that adapter development can proceed independently.
17. As a profiling operator, I want to benchmark warm tool execution on the current device, so that scheduling uses relevant latency measurements.
18. As a profiling operator, I want latency measurements to include preprocessing, inference, postprocessing, and required data movement, so that they represent node execution cost.
19. As a profiling operator, I want model download and loading measured separately from warm execution, so that setup cost is not confused with the first scheduler SLA.
20. As a profiling operator, I want latency profiles to record the model, configuration, device, default input bucket, p50, p95, and sample count, so that scheduler lookups are well-defined.
21. As a profiling operator, I want accuracy profiles to record the model name, configuration parameters, and raw metric, so that normalized quality can be derived without excessive experiment metadata.
22. As a profiling operator, I want missing current-device latency profiles reported explicitly, so that the scheduler does not guess or borrow incompatible measurements.
23. As a planner developer, I want the LLM to plan only tool categories and connections, so that scheduling retains control over models and configurations.
24. As a planner developer, I want the LLM backend behind a provider-neutral interface, so that another compatible service can be substituted later.
25. As a planner developer, I want an initial OpenAI-compatible client and a deterministic fake client, so that real planning and reliable tests are both possible.
26. As a planner developer, I want structured planner output validated deterministically, so that malformed or unsafe graphs are never executed.
27. As a planner developer, I want bounded repair attempts after validation failures, so that recoverable LLM mistakes can be corrected without infinite loops.
28. As a scheduler researcher, I want the naive scheduler isolated behind a replaceable interface, so that later scheduler contributions can be compared without rewriting the framework.
29. As a scheduler researcher, I want dominated configurations removed using measured Pareto frontiers, so that the baseline does not choose a strictly worse option.
30. As a scheduler researcher, I want a deterministic greedy heuristic, so that baseline decisions are reproducible and explainable.
31. As a scheduler researcher, I want the heuristic to start from the highest-quality plan and trade quality for latency, so that hard accuracy constraints remain visible throughout the search.
32. As a scheduler researcher, I want the heuristic to distinguish proven infeasibility from failure to find a feasible plan, so that the limitations of greedy search are reported honestly.
33. As an executor developer, I want independent DAG branches executed serially on one device in the first version, so that observed execution matches the scheduler's additive latency model.
34. As an executor developer, I want selected models prepared before timed execution, so that warm latency predictions and observations are comparable.
35. As an executor developer, I want a node failure to fail the DAG with structured context, so that the system does not silently violate accuracy constraints by changing configuration.
36. As a test author, I want fake LLM, tool runner, and profile dependencies injectable at the public request-execution seam, so that the complete workflow can be tested without network or GPU access.
37. As a maintainer, I want three or four representative real-model adapters and a multi-node smoke workflow, so that the abstraction is proven against actual Hugging Face behavior.
38. As a maintainer, I want tool planning, scheduling, profiling, and execution to have separate responsibilities, so that future work can evolve each area independently.

## Implementation Decisions

- The first deliverable is a working vertical slice using three or four representative real Hugging Face tools and at least one multi-node DAG. It is not necessary to adapt every eventual tool or model in this issue.
- The only connectable modalities are `image` and `text`. A tool declares named input and output ports, and an edge connects one concrete output port to one concrete input port with the same modality.
- A tool with multiple values of the same modality distinguishes them by port name. Composite signatures are maps of named ports rather than ordered modality lists.
- DAG-internal image values are PIL Image objects and text values are Python strings. Tool runners may use tensors, arrays, paths, or processor-specific values internally but must not expose them to downstream tools.
- Structured native model outputs such as bounding boxes are not a third connectable modality in this version. A tool may expose an annotated image or textual description and retain structured raw output as non-connectable execution metadata.
- Tool metadata and runtime inference are separated. `ToolSpec` contains the tool identifier, description, named ports, legal configurations, and profile lookup information. `ToolRunner` handles model acquisition, preprocessing, inference, postprocessing, and normalized outputs.
- A registry associates each `ToolSpec` with a runner factory. Planner, validator, scheduler, and their tests operate without importing model runtimes or loading model weights.
- Every tool exposes a finite, predeclared collection of legal configurations. A configuration has a stable identifier and includes Model Variant, Input Fidelity, Approximation, plus optional opaque model-specific parameters. Arbitrary cartesian products or runtime mutation of individual parameters are not supported.
- The scheduler treats configuration contents as opaque. Configuration order, names, or apparent model size do not imply quality or latency; decisions use measured profiles only.
- Each accuracy-capable tool supplies an evaluation dataset or loader, a raw metric function, whether higher or lower is better, a floor metric, and a reference-selection rule. The default reference is the configuration with the best measured raw metric.
- After correcting metric direction where necessary, node quality is normalized as `q(c) = (score(c) - score(floor)) / (score(reference) - score(floor))`, bounded to `[0, 1]`. The reference has quality 1 and the floor has quality 0. Raw metrics remain the persisted source of truth; normalized quality is derived.
- A tool without the required accuracy evaluation/profile can still run but cannot participate in a request that requires accuracy-aware scheduling. The system reports `accuracy_profile_required` rather than fabricating quality.
- DAG accuracy is the first-order estimate `A = product(q_v)` across scheduled nodes. This is explicitly a compositional proxy, not a claim of measured end-to-end task accuracy.
- Latency profiling measures warm end-to-end tool latency, including preprocessing, inference, postprocessing, and necessary data movement, but excluding download and model loading.
- Latency profiles are device-specific and record at least model, configuration, device identifier, default input bucket, p50, p95, and sampling count. Accuracy profiles record model name, configuration parameters, and raw metric.
- The first scheduler always queries each tool's `default` input bucket. Input-bucket propagation and runtime bucket selection are not implemented.
- Missing latency data for the current device makes that configuration unschedulable. If a node has no usable configuration, scheduling returns `profiling_required`; normal request execution does not launch implicit profiling.
- Planner input consists of the natural-language request, initial named image/text inputs, and the available tool catalog. Scheduling constraints are structured request fields rather than values inferred from prose.
- Planner output is a structured tool-level DAG containing tool nodes, input bindings, port-to-port edges, and designated final outputs. It contains no model variant or execution configuration choices.
- The planner uses a provider-neutral LLM client. The first real adapter targets an OpenAI-compatible structured-output API, while a deterministic fake client supports tests.
- A deterministic validator runs before scheduling. It checks tool existence, required input bindings, port existence, modality equality, acyclicity, and valid final outputs. Invalid planner output is returned to the LLM for a bounded number of repair attempts; an unrepairable plan fails without execution.
- Requests provide `A_m` in `[0,1]`, `L_m > 0`, and `gamma` in `(0,1)`. `A >= A_m` and `L <= L_m` are hard feasibility constraints.
- Because first-version execution is serial on one device, predicted DAG latency is `L = sum(latency_v)` using warm p95 measurements from the default bucket.
- The normalized utility among feasible plans is `P(A,L) = gamma * (A - A_m) / (1 - A_m) + (1 - gamma) * (L_m - L) / L_m`. If `A_m = 1`, the accuracy-surplus term is defined as zero and only plans with `A = 1` are feasible.
- The naive scheduler is heuristic, not exhaustive. It first builds a measured Pareto frontier for each node. A configuration is removed when another has no lower quality and no higher latency, with at least one strict improvement.
- The initial plan selects the highest-quality configuration at every node. While latency exceeds `L_m`, the heuristic considers profile-defined lower-quality/lower-latency moves that preserve `A >= A_m` and selects the move maximizing latency saved per logarithmic DAG-quality loss: `delta_L / -delta_log_A`.
- Once a feasible plan is found, deterministic single-node local search accepts configuration changes that preserve both hard constraints and improve `P`, stopping when no improving move remains. Tie-breaking must be stable so identical inputs and profiles produce identical plans.
- Cheap bounds are checked before greedy search: if the sum of the fastest node configurations exceeds `L_m`, latency infeasibility is proven; if the product of the highest node qualities is below `A_m`, accuracy infeasibility is proven.
- If those bounds do not prove infeasibility but greedy search finds no feasible joint configuration, the result is `no_feasible_solution_found`, not `infeasible`.
- The executor runs nodes in topological order, serially, on one device. Logical independence in the DAG does not imply first-version parallel execution.
- The executor prepares all configurations selected by the scheduling plan before timing DAG execution. Setup latency, predicted warm execution latency, and observed execution latency are recorded separately; `L_m` applies to warm execution in this version.
- Node failure is fail-fast. The executor returns the failed node and configuration, completed-node observations, and structured error information. It does not automatically select a fallback configuration or resume with a different plan.
- The concrete production tool list, model list, configuration sets, datasets, and metrics will be supplied after the common framework is established.

## Testing Decisions

- The primary and highest test seam is the public request-execution application interface. A test supplies a request plus injected fake LLM client, fake tool runners, and in-memory profiles, then observes only externally visible planning, scheduling, execution, and error behavior.
- Good tests assert stable external contracts: accepted inputs, validated DAGs, chosen configurations, predicted `A`, `L`, and `P`, final image/text outputs, and structured failure status. Tests must not assert internal call ordering or private helper implementation unless order is part of the serial-execution contract.
- The principal end-to-end test covers a multi-node request from natural language through planner output, deterministic validation, naive configuration selection, topological execution, and final output.
- Planner behavior is tested at the same seam with fake LLM responses for a valid DAG, an invalid DAG followed by a successful repair, and exhaustion of repair attempts.
- Port validation is tested through requests whose plans contain unknown tools, missing required inputs, unknown ports, image/text mismatches, cycles, or invalid final outputs. None may reach a runner.
- Scheduler behavior is tested through controlled profile fixtures that expose dominated configurations, a necessary greedy downgrade, hard accuracy rejection, hard latency rejection, successful local utility improvement, proven infeasibility, and `no_feasible_solution_found`.
- Profiling availability is tested by omitting current-device latency and accuracy records and asserting `profiling_required` or `accuracy_profile_required` without inference execution.
- Executor behavior is tested with fake runners that emit image/text values and with a failing runner, asserting serial dependency propagation and fail-fast structured errors.
- Narrow mathematical tests may supplement the application seam where exact edge cases are otherwise obscure: quality normalization (including metric direction and `A_m = 1`), DAG quality multiplication, Pareto dominance, utility calculation, and deterministic tie-breaking.
- Real Hugging Face integration tests are limited smoke tests for the representative adapters and at least one multi-node DAG. They validate the PIL/string boundary and real preprocessing/postprocessing, but are separated from fast deterministic tests because they may require model downloads or accelerator access.
- There is no existing test suite or architectural testing prior art in the repository. The new public application seam establishes the project convention; isolated scripts in the existing external examples are reference material for model behavior, not the test seam.
- Python commands and tests use the repository-prescribed `uv run python` environment.

## Out of Scope

- The research contribution or final optimization algorithm for the scheduler.
- Exhaustive configuration search or claims that the naive greedy heuristic is complete or globally optimal.
- Adapting the full production tool and model catalogs before those lists are supplied.
- More connectable data modalities such as bounding boxes, masks, embeddings, audio, tensors, or arbitrary JSON.
- Automatic schema conversion between tools.
- Parallel branch execution, multi-device placement, end/edge/cloud placement, batching, or resource-contention modeling.
- Incorporating model download, cold start, cache residency, model switching, or memory pressure into the scheduling objective.
- Runtime rescheduling, preemption, speculative execution, or automatic fallback after a node failure.
- Per-request or propagated input buckets; the first version uses one default latency bucket per tool.
- Cross-device latency inference or borrowing profiles from another device.
- Inferring hard scheduling constraints from natural-language phrases.
- Calibrated end-to-end accuracy models, tool-pair interaction corrections, or proof that normalized losses have identical user-perceived meaning across tasks.
- Rich experiment provenance beyond the agreed minimal accuracy and latency profile fields.
- Hard real-time enforcement or detailed semantics for observed runtime SLA overruns.
- A user interface or production service deployment unless separately requested.

## Further Notes

- The repository currently contains a minimal entry point and standalone Hugging Face experiments for tasks such as captioning, summarization, depth estimation, super-resolution, and segmentation. They demonstrate candidate model behavior but do not yet form a reusable framework.
- No project domain glossary or applicable ADR currently exists. This spec establishes the initial vocabulary: Tool, ToolSpec, ToolRunner, Configuration, Profile, Tool-Call DAG, Planner, Scheduling Plan, Naive Scheduler, and Executor.
- The multiplicative DAG accuracy model and greedy scheduler are deliberate baseline approximations. Their limitations should be visible in APIs and experimental reporting so later scheduler work has a trustworthy comparison point.
- A successful first milestone is a reproducible request that plans and executes a real multi-node DAG, with valid modality connections, profiled configuration choices, explicit constraints, and inspectable scheduling output.
