"""Core domain model for volunteer scheduling system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta


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
    # Minimum time between shifts for the same worker to allow for travel/transition
    min_transition_time: timedelta
    fairness_enabled: bool
    prefer_consecutive_shifts: bool


@dataclass(frozen=True)
class Shift:
    """Represents a work shift with time, location, and worker requirements.

    A shift represents a single, homogeneous type of work requiring workers
    with specific skills and capacity constraints.
    """

    id: str
    start_time: datetime
    end_time: datetime
    location: str
    required_skills: set[str] = field(hash=False)
    min_workers: int
    max_workers: int
    role_description: str


@dataclass(frozen=True)
class Worker:
    """Represents a worker with their skills and capabilities."""

    id: str
    name: str
    skills: set[str] = field(hash=False)


@dataclass(frozen=True)
class WorkerPreference:
    """Represents a worker's preference for a specific shift."""

    worker: Worker
    shift: Shift
    preference_level: int  # Lower values indicate stronger preference


@dataclass(frozen=True)
class Assignment:
    """Represents an assignment of a worker to a shift."""

    worker: Worker
    shift: Shift
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
