"""Stable interface for turning trusted evidence into evolution guidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class DiagnosisResult:
    """A reproducible bottleneck conclusion consumed by an evolution operator."""

    advice: str
    bottlenecks: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    confidence: float | None = None


class Diagnosis(Protocol):
    """Diagnose one bounded view of trusted Candidate Evaluation Evidence."""

    def diagnose(self, evidence: Mapping[str, object]) -> DiagnosisResult: ...
