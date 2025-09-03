"""Core domain model for volunteer scheduling system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


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
    """Represents a work shift at a conference with time, location, and capacity."""

    id: str
    start_time: datetime
    end_time: datetime
    location: str
    max_workers: int


@dataclass(frozen=True)
class Worker:
    """Represents a conference worker."""

    id: str
    name: str


@dataclass(frozen=True)
class Assignment:
    """Represents an assignment of a worker to a shift."""

    worker: Worker
    shift: Shift
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
    assignments: list[Assignment] = field(default_factory=list)

    def add_worker(self, worker: Worker) -> None:
        """Add a worker to the conference."""
        self.workers.append(worker)

    def add_shift(self, shift: Shift) -> None:
        """Add a shift to the conference."""
        self.shifts.append(shift)

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

    def get_assignments_for_worker(self, worker: Worker) -> list[Assignment]:
        """Get all assignments for a specific worker."""
        return [
            assignment for assignment in self.assignments if assignment.worker == worker
        ]

    def get_assignments_for_shift(self, shift: Shift) -> list[Assignment]:
        """Get all assignments for a specific shift."""
        return [
            assignment for assignment in self.assignments if assignment.shift == shift
        ]

    def is_worker_assigned_to_shift(self, worker: Worker, shift: Shift) -> bool:
        """Check if worker is assigned to the specified shift."""
        return any(
            assignment.worker == worker and assignment.shift == shift
            for assignment in self.assignments
        )

    def is_worker_at_shift_capacity(self, worker: Worker) -> bool:
        """Check if worker is at maximum shift capacity."""
        worker_assignments = self.get_assignments_for_worker(worker)
        return len(worker_assignments) >= self.config.max_shifts_per_worker

    def is_shift_at_capacity(self, shift: Shift) -> bool:
        """Check if shift is at maximum worker capacity."""
        shift_assignments = self.get_assignments_for_shift(shift)
        return len(shift_assignments) >= shift.max_workers

    def assign_worker(self, worker: Worker, shift: Shift) -> Assignment:
        """Assign a worker to a shift.

        Args:
            worker: Worker to assign
            shift: Shift to assign worker to

        Returns:
            Assignment object representing the assignment

        Raises:
            ValueError: If assignment constraints are violated
        """
        # Check constraints
        if self.is_worker_at_shift_capacity(worker):
            worker_assignments = self.get_assignments_for_worker(worker)
            max_shifts = self.config.max_shifts_per_worker
            current_shifts = len(worker_assignments)
            msg = (
                f"Worker '{worker.id}' is at capacity "
                f"({current_shifts}/{max_shifts} shifts)"
            )
            raise ValueError(msg)

        if self.is_shift_at_capacity(shift):
            shift_assignments = self.get_assignments_for_shift(shift)
            msg = (
                f"Shift '{shift.id}' is full "
                f"({len(shift_assignments)}/{shift.max_workers} workers)"
            )
            raise ValueError(msg)

        if self.is_worker_assigned_to_shift(worker, shift):
            msg = f"Worker '{worker.id}' is already assigned to shift '{shift.id}'"
            raise ValueError(msg)

        # Create and add the assignment
        assignment = Assignment(worker=worker, shift=shift)
        self.assignments.append(assignment)
        return assignment

    def unassign_worker_from_shift(self, worker: Worker, shift: Shift) -> None:
        """Remove a worker from a shift.

        Args:
            worker: Worker to unassign
            shift: Shift to unassign worker from

        Raises:
            ValueError: If worker is not assigned to shift
        """
        for assignment in self.assignments:
            if assignment.worker == worker and assignment.shift == shift:
                self.assignments.remove(assignment)
                return

        msg = f"Worker '{worker.id}' is not assigned to shift '{shift.id}'"
        raise ValueError(msg)
