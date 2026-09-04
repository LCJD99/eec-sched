# Information-theoretic quality composition across candidate tool DAGs

## Recommendation

Use **conditional semantic-information retention (CSIR)**: define one task-relevant
target random variable `Y` for all semantically equivalent candidate DAGs, measure
the fraction of task-relevant mutual information retained across each transition
of a canonical DAG frontier, and multiply those conditional factors. This preserves
the useful shape of the current `product(q_v)` proxy while replacing its unsupported
node-independence assumption with a conditional chain-rule construction.

For task family `tau`, query/context `Q`, initial inputs `X`, and final output `O_D`
of candidate DAG `D`, the ideal score is

```text
J_k = I(Y; S_k | Q)
q_k = J_k / J_(k-1)
R(D, c) = product_k q_k
        = I(Y; O_D | Q) / I(Y; X | Q)
```

where `c` is the vector of node configurations and `S_k` is defined below. This
requires `I(Y; X | Q) > 0`; a task not identifiable from its declared inputs is
outside the schedulable domain. If an intermediate `J_k` is zero, all subsequent
retention is zero and ratios after that point need not be evaluated. `R` is a
dimensionless **information-retention score**, not accuracy. Map it to an
end-to-end success probability with a held-out calibrator before applying an
accuracy SLA.

## Mathematical assumptions

1. **Common semantics.** All candidates for a query share the same `Y`, input
   distribution, and final evaluation protocol. For classification, `Y` can be
   the class; for VQA it can be the normalized answer; for generation it must be
   a declared finite set of task-sufficient semantic attributes (objects,
   relations, claims, or constraints). Equal output modality alone is not enough.
2. **Explicit information boundary.** Given `Q` and its declared inputs, a node
   is a stochastic channel whose randomness is independent of `Y`. Retrieval,
   sensors, caches, or other external evidence must appear as explicit node
   inputs. Then the data-processing inequality applies: local processing cannot
   increase information about `Y`. See the authoritative MIT information-theory
   notes on mutual information, conditional independence, and strong data
   processing inequalities ([Polyanskiy and Wu, Chapter 2](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/)).
3. **Canonical frontier.** Fix a deterministic topological order. Let `S_0 = X`.
   At step `k`, replace the consumed live inputs whose final use has occurred by
   node output `Z_k`, retaining every value still needed by an unexecuted node;
   call the resulting live semantic frontier `S_k`. Copying a value at a fork is
   lossless. The final projection leaves `S_m = O_D`.
4. **Conditional, not marginal, factors.** Define

   ```text
   J_k = I(Y; S_k | Q)
   q_k = J_k / J_(k-1) in [0, 1].
   ```

   The ratios telescope, so the product is invariant to how total loss is
   attributed among nodes, even though individual conditional factors can depend
   on the chosen canonical order. Equivalently, define additive log loss
   `ell_k = -log q_k`; then `-log R = sum_k ell_k`, which is convenient for the
   scheduler. At forks and joins, the
   live frontier conditions on sibling information, preventing redundant evidence
   from being counted twice. Marginal per-branch scores do not have this property:
   multivariate sources can contain redundant and synergistic information
   ([Williams and Beer, 2010](https://arxiv.org/abs/1004.2515)).

This formulation instantiates the information-bottleneck principle: preserve
information relevant to `Y`, rather than all bits of a high-dimensional image or
text ([Tishby, Pereira, and Bialek, 2000](https://arxiv.org/abs/physics/0004057)).

### Structural SDPI/percolation bound (useful, but not predicted accuracy)

There is a second, topology-explicit construction that should be reported beside
CSIR. For each node channel `K_v`, its KL strong data-processing coefficient is

```text
eta_v = sup_(U -> parents(v) -> Z_v) I(U; Z_v) / I(U; parents(v)).
```

For a Bayesian network, Polyanskiy and Wu bound end-to-end information
contraction by the probability that a source-to-sink path survives independent
site percolation with node-open probabilities `eta_v`
([Polyanskiy and Wu, 2017, Theorem 5](https://arxiv.org/abs/1508.06025)). For a
serial chain this is exactly `product_v eta_v`; for two disjoint parallel paths
with path retentions `r_1` and `r_2`, it is `1 - (1-r_1)(1-r_2)`. Thus the
percolation expression extends the product without double-counting overlapping
paths.

**Crucial interpretation:** this theorem supplies an **upper bound on a channel
contraction coefficient**, not an equality, an accuracy estimate, or a guarantee
that a higher bound means a better realized workflow. The supremum defining
`eta_v` is also worst-case over input auxiliaries and is generally much harder to
estimate than task-conditioned retention. Therefore use percolation as a
structural bound/feature and sanity-check baseline; calibrate it on held-out
end-to-end executions before any scheduling decision calls it predicted quality.

## Offline profiling protocol

1. **Declare the task abstraction.** Freeze `tau`, `Y`, query/input distribution,
   and semantic-equivalence rules. Scores from different `Y` definitions are not
   comparable.
2. **Use three disjoint splits.** Use one split to fit semantic probes, one to fit
   the factor model and end-to-end calibrator, and one untouched split for final
   reporting. Run every candidate/configuration on paired examples and retain
   intermediate frontiers.
3. **Estimate conditional entropy with a probabilistic semantic probe.** Fit a
   frozen, task-specific `p_phi(y | S, Q)` and estimate
   `C_phi(S) = mean[-log p_phi(y_i | S_i, q_i)]`. Use a shared multimodal adapter
   and the same target space, training budget, and validation rule across frontier
   types. Log score is strictly proper, so a truthful conditional distribution is
   optimal ([Gneiting and Raftery, 2007](https://doi.org/10.1198/016214506000001437)).
   Check calibration separately; calibration is a joint property of predictions
   and outcomes ([Gneiting, Balabdaoui, and Raftery, 2007](https://doi.org/10.1111/j.1467-9868.2007.00587.x)).
4. **Estimate paired retention.** Use the probe cross-entropy to form the
   practical lower-bound proxy
   `J_hat_k = H_hat(Y | Q) - C_phi(S_k)` and
   `q_hat_k = J_hat_k / J_hat_(k-1)`. Fit a hierarchical conditional
   model keyed by task family, tool, configuration, input bucket, upstream quality
   bucket, fan-in, and join/fork context. This model supplies reusable runtime
   factors while retaining conditional-dependence features.
5. **Quantify uncertainty.** Bootstrap paired examples by query. Store the mean,
   lower confidence bound of `q_k`, sample count, calibration diagnostics, and
   out-of-distribution range. A significantly estimated `q_k > 1` violates the
   assumed information boundary or exposes probe error; report it rather than
   silently treating it as a quality gain. The scheduler may cap its point estimate
   at one only after recording that diagnostic.
6. **Calibrate to the actual endpoint.** On complete held-out DAG runs, fit a
   monotone map `g_tau(log R, topology_features)` to the common binary/graded
   end-to-end success event. Validate with log loss/Brier score and reliability
   plots. This step is what turns retention into predicted accuracy.

The existing five-example profile protocol in
[`mnms-profile-evaluation-options.md`](../specs/mnms-profile-evaluation-options.md)
is adequate only for a smoke test, not for fitting entropy probes, conditional
interaction terms, or calibration. Choose the sample size from held-out learning
curves and a predeclared confidence-interval width. Direct neural mutual-information
estimators such as MINE are possible
([Belghazi et al., 2018](https://proceedings.mlr.press/v80/belghazi18a.html)), and
InfoNCE gives a contrastive lower bound
([van den Oord, Li, and Vinyals, 2018](https://arxiv.org/abs/1807.03748)), but they
should be diagnostics rather than the primary estimator here: any
distribution-free high-confidence MI lower bound from `N` samples is limited to
`O(log N)` ([McAllester and Stratos, 2020](https://proceedings.mlr.press/v108/mcallester20a.html)).

## Runtime score

For every candidate `D` and configuration vector `c`:

1. construct its canonical frontier sequence and the profile feature vector for
   every transition;
2. predict a conservative factor `q_k^L` (the stored lower confidence bound);
3. compute `R_LCB(D,c) = product_k q_k^L`;
4. compute `A_hat_LCB = g_tau(log R_LCB, topology_features)`;
5. reject unsupported/OOD transitions instead of borrowing an unconditional
   factor; then optimize latency, resource use, and `A_hat_LCB` among supported
   candidates.

Optionally compute the SDPI site-percolation bound from profiled channel
coefficients and pass it to `g_tau` as an additional topology feature. Do not use
the uncalibrated bound directly as `A_hat`.

The scheduler may still use additive log loss:
`-log R_LCB = sum_k -log q_k^L`. Thus the existing greedy notion of
"latency saved per logarithmic DAG-quality loss" remains available, but its loss
now has an information-theoretic and graph-conditional meaning.

## Required baselines and ablations

- current product of independently normalized tool metrics;
- product of marginal node-success probabilities (calibrated but without
  frontier conditioning);
- CSIR without join/fork/upstream-context features;
- full CSIR with conditional factors and end-to-end calibration;
- uncalibrated and calibrated SDPI site-percolation structural scores;
- a direct topology-aware end-to-end predictor with no per-node decomposition;
- measured end-to-end quality for every executed candidate (evaluation oracle,
  not a deployable scheduler).

Report ranking correlation/regret, SLA violation rate, calibration error and log
loss, not only average accuracy. Ablate forks/joins separately to show whether
conditional factors actually correct redundancy and synergy.

## Limitations and falsification conditions

- CSIR compares alternative DAGs only within one declared task target `Y`; it is
  not a universal scale across captioning, detection, summarization, and image
  generation.
- A semantic probe measures the chosen task abstraction and inherits its blind
  spots. If no defensible finite target or probabilistic evaluator exists, mark
  the workflow unsupported for accuracy-aware scheduling.
- Exact frontier factors are DAG- and context-dependent. The reusable
  hierarchical profile is an approximation and needs OOD rejection; a single
  scalar per tool/configuration is not sufficient.
- Mutual-information preservation does not by itself guarantee user utility or
  correctness. The common end-to-end evaluator and held-out calibration remain
  mandatory.
- Conditional profiles can be data-hungry, especially at joins. Poorly estimated
  interactions may make full CSIR worse than a direct end-to-end predictor; that
  comparison is a falsification test, not an optional benchmark.
- Hidden external information, adaptive prompts, distribution shift, or a probe
  that performs materially differently across modalities breaks the stated
  interpretation. These cases must be modeled explicitly or excluded.
