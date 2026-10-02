# Multi-request evaluator: first replay contract

## Scope

One workload scenario fixes an observation window `T`, a Profiling Database
Snapshot, and an ordered set of requests. Each request contains an arrival time
in `[0, T)`, a unique Trace ID, its input, and three Planner candidate DAGs.
The same materialized arrival sequence is reused for every Scheduler Candidate
compared in that scenario. Five request rates may produce five separate
scenarios; this contract does not combine their scores.

`eec_sched.workload` can generate exponential inter-arrival gaps from a fixed
seed and save the realized times in an arrival manifest. The manifest refers
to Trace templates by ID, so the Planner dataset must also be frozen when
replaying it across baselines.

The evaluator owns the simulation clock, resource state, proposal validation,
and event log. A Scheduler Candidate is invoked exactly once when each request
arrives. It chooses one candidate DAG and a Configuration and Compatible Device
for every node. The chosen assignments do not reserve a resource until the
corresponding operation is ready. The candidate receives a read-only view of
the current state and no future arrivals.

## Time and resource rules

The simulation is deterministic and non-preemptive. Each device executes at
most one node at a time; each directed transfer path transmits at most one
data block at a time. Ready operations wait in FIFO queues, with stable IDs
breaking equal-time ties. Distinct directed paths have independent queues and
capacity. Execution time is the chosen Execution Profile's warm latency p95.
Transfer time uses the current simulator's propagation delay plus bytes divided
by directed-path bandwidth, including request ingress and final-output return.

An operation becomes ready only when its dependencies are complete. At one
timestamp, completions and newly ready work are settled before arrivals. Same
timestamp arrivals are admitted in Trace-ID order; each subsequent Scheduler
invocation sees decisions already accepted at that timestamp. Rejected or
failed proposals consume no device or link time and do not stop the replay.

`SchedulerView.system_state` is derived from the evaluator, not copied from a
Trace. It exposes simulation time and, for each device and directed link,
whether it is busy, the ready queue depth, the running operation's remaining
time, and estimated committed work. Committed work includes accepted but
not-yet-ready operations on that resource. These are estimates from the frozen
profiles, not knowledge of future requests.

Scheduler computation time is measured with a real clock and reported
separately. It does not advance simulated time in this first contract.

## Outputs

Every arrived request has a record with arrival, decision outcome, selected
path, assignments, node and transfer timeline, completion time, response
latency, Scheduler computation time, and rejection or failure reason. Response
latency is `completion_time - arrival_time`, including queue waits and final
output transfer. Operations use the pair `(trace_id, node_id)` as identity.

The scenario reports completed requests by `T`, the number of admitted
unfinished requests at `T`, throughput as completions by `T` divided by `T`
in seconds, latency quantiles, the Snapshot digest, and the Scheduler Version.
The evaluator continues until all accepted
requests finish so their response latencies remain visible. Rejected and failed
requests are counted separately and excluded from completion latency quantiles.

No multi-request Candidate Score or cross-rate aggregate is defined here.
The existing single-request evaluator and its score remain available for
compatibility. Load-dependent scoring is a later design decision.
