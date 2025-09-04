"""Core domain model for volunteer scheduling system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta


@dataclass(frozen=True)
class ShiftRequirement:
    """A specific worker requirement for a shift.

    Represents one type of worker needed for a shift, such as "AV Tech" or "Security".
    A single shift can have multiple requirements for different types of workers.
    """

    required_skills: set[str]
    min_workers: int
    max_workers: int
    description: str = ""


@dataclass
class SchedulingConstraints:
    """Constraints that govern the scheduling algorithm behavior.

    Pure constraint object with no domain-specific aggregation - just the rules
    needed for the scheduling algorithm to operate.
    """

    min_workers_per_shift: int
    max_workers_per_shift: int
    min_shifts_per_worker: int
    max_shifts_per_worker: int
    min_transition_time: timedelta = field(
        default_factory=lambda: timedelta(minutes=30),
    )
    fairness_enabled: bool = True
    prefer_consecutive_shifts: bool = False


@dataclass(frozen=True)
class Shift:
    """Represents a work shift with time, location, and worker requirements.

    A shift represents a single event that may require multiple types of workers.
    Each requirement specifies different skills and worker counts needed.
    """

    id: str
    start_time: datetime
    end_time: datetime
    location: str
    requirements: list[ShiftRequirement] = field(default_factory=list, hash=False)

    @property
    def total_min_workers(self) -> int:
        """Total minimum workers needed across all requirements."""
        return sum(req.min_workers for req in self.requirements)

    @property
    def total_max_workers(self) -> int:
        """Total maximum workers allowed across all requirements."""
        return sum(req.max_workers for req in self.requirements)


@dataclass(frozen=True)
class Worker:
    """Represents a worker with their skills and capabilities."""

    id: str
    name: str
    skills: set[str] = field(default_factory=set, hash=False)


@dataclass(frozen=True)
class WorkerPreference:
    """Represents a worker's preference for a specific shift."""

    worker: Worker
    shift: Shift
    preference_level: int  # Higher values indicate stronger preference


@dataclass(frozen=True)
class Assignment:
    """Represents an assignment of a worker to a specific requirement within a shift."""

    worker: Worker
    shift: Shift
    requirement: ShiftRequirement
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


@dataclass
class AssignmentResult:
    """Result of attempting to assign a worker to a shift."""

    success: bool
    worker_id: str
    shift_id: str
    error_reason: str | None = None
