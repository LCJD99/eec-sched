"""Generation seams for mutation and crossover adapters."""

from __future__ import annotations

from typing import Protocol, Sequence

from .models import EvolutionDiagnosisContext, SchedulerCandidate, SchedulerCandidateDraft


class EvolutionOperator(Protocol):
    def generate(
        self,
        parents: tuple[SchedulerCandidate, ...],
        diagnosis: EvolutionDiagnosisContext,
        *,
        advice: str = "",
    ) -> SchedulerCandidateDraft: ...


class MutationOperator(Protocol):
    def generate(
        self, parent: SchedulerCandidate, diagnosis: EvolutionDiagnosisContext, *, advice: str = ""
    ) -> SchedulerCandidateDraft: ...


class CrossoverOperator(Protocol):
    def generate(
        self, parents: tuple[SchedulerCandidate, SchedulerCandidate], diagnosis: EvolutionDiagnosisContext, *, advice: str = ""
    ) -> SchedulerCandidateDraft: ...
