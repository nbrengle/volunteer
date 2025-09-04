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


@dataclass
class ConferenceConfig:
    """Configuration parameters for a conference.

    Defines the constraints and parameters that govern how workers
    can be assigned to shifts within a conference.
    """

    start_time: datetime
    duration_days: int
    min_workers_per_shift: int
    max_workers_per_shift: int
    min_shifts_per_worker: int
    max_shifts_per_worker: int


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
    """Represents a conference worker with their skills and capabilities."""

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


@dataclass
class Conference:
    """Represents a conference with workers and shifts to be scheduled."""

    id: str
    name: str
    config: ConferenceConfig
    workers: list[Worker] = field(default_factory=list)
    shifts: list[Shift] = field(default_factory=list)
    preferences: list[WorkerPreference] = field(default_factory=list)

    def add_worker(self, worker: Worker) -> None:
        """Add a worker to the conference."""
        self.workers.append(worker)

    def add_shift(self, shift: Shift) -> None:
        """Add a shift to the conference."""
        self.shifts.append(shift)

    def add_preference(self, preference: WorkerPreference) -> None:
        """Add a worker preference for a shift."""
        self.preferences.append(preference)

    def get_worker(self, worker_id: str) -> Worker | None:
        """Get worker by ID."""
        for worker in self.workers:
            if worker.id == worker_id:
                return worker
        return None

    def get_shift(self, shift_id: str) -> Shift | None:
        """Get shift by ID."""
        for shift in self.shifts:
            if shift.id == shift_id:
                return shift
        return None

    def get_preferences_for_worker(self, worker: Worker) -> list[WorkerPreference]:
        """Get all preferences for a specific worker."""
        return [
            preference for preference in self.preferences if preference.worker == worker
        ]

    def get_preferences_for_shift(self, shift: Shift) -> list[WorkerPreference]:
        """Get all worker preferences for a specific shift."""
        return [
            preference for preference in self.preferences if preference.shift == shift
        ]

    def shifts_overlap(self, shift1: Shift, shift2: Shift) -> bool:
        """Check if two shifts have overlapping time periods."""
        return (
            shift1.start_time < shift2.end_time and shift2.start_time < shift1.end_time
        )
