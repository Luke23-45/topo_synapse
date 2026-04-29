from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from synapse.utils.io import ensure_dir, save_json


@dataclass
class TestCase:
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ExperimentReport:
    experiment_id: str
    experiment_name: str
    formal_reference: str
    claim: str
    cases: list[TestCase] = field(default_factory=list)
    duration_seconds: float = 0.0
    status: str = "PENDING"

    @property
    def passed_cases(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def total_cases(self) -> int:
        return len(self.cases)

    def add_case(self, case: TestCase) -> None:
        self.cases.append(case)

    def finalize(self) -> None:
        self.status = "PASS" if self.passed_cases == self.total_cases else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed_cases"] = self.passed_cases
        payload["total_cases"] = self.total_cases
        return payload


class ExperimentTimer:
    def __enter__(self) -> "ExperimentTimer":
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = perf_counter() - self._start


def save_report_json(report: ExperimentReport, path: str | Path) -> None:
    save_json(path, report.to_dict())


def print_report(report: ExperimentReport) -> None:
    print(f"[{report.experiment_id}] {report.experiment_name}: {report.status}")
    print(f"Passed {report.passed_cases}/{report.total_cases} cases in {report.duration_seconds:.2f}s")


@dataclass
class RunCapsule:
    root: Path
    artifacts: Path
    metrics: Path
    logs: Path


def create_run_capsule(output_dir: str | Path, name: str) -> RunCapsule:
    root = ensure_dir(Path(output_dir) / name)
    artifacts = ensure_dir(root / "artifacts")
    metrics = ensure_dir(root / "metrics")
    logs = ensure_dir(root / "logs")
    return RunCapsule(root=root, artifacts=artifacts, metrics=metrics, logs=logs)
