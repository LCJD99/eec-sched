# ADR 0001: Compose experiment seams with Hydra

Status: accepted

## Context

The paper design requires repeatable replacement and ablation of search,
diagnosis, memory, model, evaluation, evidence, and runtime choices. The former
application mixed configuration parsing, evolution control flow, trusted
evaluation, and model calls. It also duplicated public contracts across several
top-level modules.

A Hydra configuration group is an experimental choice, but it is not
necessarily a Python package. Creating one directory for every small adapter
would add navigation cost without hiding meaningful complexity.

## Decision

Hydra is the sole composition mechanism for experiment runs. The default
configuration follows the PSOCM direction with performance behavior features,
feature-diverse mutation, complementary crossover, a feature repertoire, and
run-local memory. Baseline and ablation variants remain explicit choices.

Python source is grouped by implementation depth:

- `profiling` owns measurement campaigns, scorers, profiles, and immutable
  evidence snapshots.
- `evaluation` owns trusted simulation, validation, and scoring.
- `diagnosis` owns empty, one-shot, and tool-using diagnosis adapters.
- `evolution` owns the loop, search strategies, memory adapters, and contracts.
- Small alternatives that share the same contract remain in one file, such as
  `evolution/strategies.py` and `evolution/memory.py`.

The Hydra composition root is the only layer allowed to select concrete
adapters. Domain modules depend on protocols or stable data contracts instead
of Hydra objects.

## Consequences

Experiments can be reproduced from resolved configuration and components can be
replaced without editing orchestration code. Empty implementations provide
real ablation paths rather than conditional behavior hidden inside the loop.
Configuration layout and source layout deliberately do not match one-to-one;
new directories should be introduced only when an implementation is deep
enough to justify an independent namespace.
