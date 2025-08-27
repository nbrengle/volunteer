"""Core domain classes for conference worker scheduling system."""

import bisect
import contextlib
import random
import secrets
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConferenceConfig:
    """Configuration parameters for a conference.

    All parameters are required - each conference has unique requirements.
    """

    start_time: datetime
    duration_days: int
    shifts_per_day: int
    min_workers_per_shift: int
    max_workers_per_shift: int
    min_shifts_per_worker: int
    max_shifts_per_worker: int
    max_preferences_per_worker: int

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.duration_days <= 0:
            msg = "Duration days must be positive"
            raise ValueError(msg)
        if self.shifts_per_day <= 0:
            msg = "Shifts per day must be positive"
            raise ValueError(msg)
        if self.min_workers_per_shift <= 0:
            msg = "Minimum workers per shift must be positive"
            raise ValueError(msg)
        if self.max_workers_per_shift < self.min_workers_per_shift:
            msg = "Maximum workers per shift must be >= minimum"
            raise ValueError(msg)
        if self.min_shifts_per_worker <= 0:
            msg = "Minimum shifts per worker must be positive"
            raise ValueError(msg)
        if self.max_shifts_per_worker < self.min_shifts_per_worker:
            msg = "Maximum shifts per worker must be >= minimum"
            raise ValueError(msg)
        if self.max_preferences_per_worker <= 0:
            msg = "Maximum preferences per worker must be positive"
            raise ValueError(msg)


@dataclass
class Shift:
    """Represents a work shift at a conference with time, location, and capacity."""

    id: str
    start_time: datetime
    end_time: datetime
    location: str
    max_workers: int
    assigned_worker_ids: set[str] = field(default_factory=set)

    @property
    def available_spots(self) -> int:
        """Number of worker spots still available for this shift."""
        return self.max_workers - len(self.assigned_worker_ids)

    @property
    def is_full(self) -> bool:
        """Whether this shift has reached its maximum worker capacity."""
        return len(self.assigned_worker_ids) >= self.max_workers

    def add_worker(self, worker_id: str) -> None:
        """Add a worker to this shift.

        Raises:
            ValueError: If shift is already full or worker already assigned.

        """
        if self.is_full:
            msg = (
                f"Shift {self.id} is already full "
                f"({len(self.assigned_worker_ids)}/{self.max_workers})"
            )
            raise ValueError(msg)
        if worker_id in self.assigned_worker_ids:
            msg = f"Worker {worker_id} is already assigned to shift {self.id}"
            raise ValueError(msg)
        self.assigned_worker_ids.add(worker_id)

    def remove_worker(self, worker_id: str) -> None:
        """Remove a worker from this shift.

        Raises:
            ValueError: If worker is not assigned to this shift.

        """
        if worker_id not in self.assigned_worker_ids:
            msg = f"Worker {worker_id} is not assigned to shift {self.id}"
            raise ValueError(msg)
        self.assigned_worker_ids.remove(worker_id)


@dataclass
class Worker:
    """Represents a conference worker with shift preferences and assignments.

    Capacity constraints are now managed at the Conference level rather than
    being individual worker constraints.
    """

    id: str
    name: str
    shift_preferences: list[str] = field(default_factory=list)  # Ordered by preference
    assigned_shift_ids: set[str] = field(default_factory=set)

    def add_preference(self, shift_id: str) -> None:
        """Add a shift preference if not already present."""
        if shift_id not in self.shift_preferences:
            self.shift_preferences.append(shift_id)

    def set_preferences(self, shift_ids: list[str]) -> None:
        """Set worker preferences to the provided list."""
        self.shift_preferences = shift_ids.copy()

    def get_preference_score(self, shift_id: str) -> int:
        """Get preference score for a shift (higher = more preferred)."""
        try:
            # Higher score for higher preference (index 0 gets highest score)
            return len(self.shift_preferences) - self.shift_preferences.index(shift_id)
        except ValueError:
            return 0

    def get_preference_rank(self, shift_id: str) -> int | None:
        """Get the rank of a shift preference (1-based). None if not in preferences."""
        try:
            return self.shift_preferences.index(shift_id) + 1
        except ValueError:
            return None

    def add_shift_assignment(self, shift_id: str) -> None:
        """Add a shift assignment.

        Raises:
            ValueError: If shift is already assigned to this worker.

        """
        if shift_id in self.assigned_shift_ids:
            msg = f"Shift {shift_id} is already assigned to worker {self.id}"
            raise ValueError(msg)
        self.assigned_shift_ids.add(shift_id)

    def remove_shift_assignment(self, shift_id: str) -> None:
        """Remove a shift assignment.

        Raises:
            ValueError: If shift is not assigned to this worker.

        """
        if shift_id not in self.assigned_shift_ids:
            msg = f"Shift {shift_id} is not assigned to worker {self.id}"
            raise ValueError(msg)
        self.assigned_shift_ids.remove(shift_id)

    def is_assigned_to_shift(self, shift_id: str) -> bool:
        """Check if worker is assigned to the specified shift."""
        return shift_id in self.assigned_shift_ids


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

    def __post_init__(self) -> None:
        """Initialize lookup dictionaries after dataclass creation."""
        self._shifts_by_id = {shift.id: shift for shift in self.shifts}
        self._workers_by_id = {worker.id: worker for worker in self.workers}

    def add_worker(self, worker: Worker) -> None:
        """Add a worker to the conference."""
        self.workers.append(worker)
        self._workers_by_id[worker.id] = worker

    def is_worker_at_shift_capacity(self, worker: Worker) -> bool:
        """Check if worker has reached the conference's maximum shifts per worker."""
        return len(worker.assigned_shift_ids) >= self.config.max_shifts_per_worker

    def get_worker_available_shifts(self, worker: Worker) -> int:
        """Get number of additional shifts this worker can take."""
        return self.config.max_shifts_per_worker - len(worker.assigned_shift_ids)

    def can_worker_add_preference(self, worker: Worker) -> bool:
        """Check if worker can add more preferences within conference limits."""
        return len(worker.shift_preferences) < self.config.max_preferences_per_worker

    def set_worker_preferences(self, worker: Worker, shift_ids: list[str]) -> None:
        """Set worker preferences, respecting conference limit."""
        max_prefs = self.config.max_preferences_per_worker
        worker.set_preferences(shift_ids[:max_prefs])

    def add_shift(self, shift: Shift) -> None:
        """Add a shift to the conference."""
        self.shifts.append(shift)
        self._shifts_by_id[shift.id] = shift

    def get_shift(self, shift_id: str) -> Shift | None:
        """Get shift by ID, returns None if not found."""
        return self._shifts_by_id.get(shift_id)

    def get_worker(self, worker_id: str) -> Worker | None:
        """Get worker by ID, returns None if not found."""
        return self._workers_by_id.get(worker_id)

    def get_workers_with_availability(self) -> list[Worker]:
        """Get workers who can still be assigned to more shifts."""
        return [
            worker
            for worker in self.workers
            if not self.is_worker_at_shift_capacity(worker)
        ]

    def get_available_shifts(self) -> list[Shift]:
        """Get shifts that still have open spots."""
        return [shift for shift in self.shifts if not shift.is_full]

    def assign_worker_to_shift(self, worker_id: str, shift_id: str) -> bool:
        """Assign a worker to a shift. Returns False if assignment fails."""
        result = self.try_assign_worker_to_shift(worker_id, shift_id)
        return result.success

    def try_assign_worker_to_shift(
        self,
        worker_id: str,
        shift_id: str,
    ) -> AssignmentResult:
        """Attempt to assign a worker to a shift with detailed error information."""
        # Validate inputs exist
        validation_result = self._validate_assignment_inputs(worker_id, shift_id)
        if isinstance(validation_result, AssignmentResult):
            return validation_result

        # validation_result is now guaranteed to be tuple[Worker, Shift]
        worker, shift = validation_result

        # Check constraints and conflicts
        constraint_error = self._check_assignment_constraints(worker, shift, shift_id)
        if constraint_error:
            return constraint_error

        # Attempt the assignment
        try:
            worker.add_shift_assignment(shift_id)
            shift.add_worker(worker_id)
            return AssignmentResult(
                success=True,
                worker_id=worker_id,
                shift_id=shift_id,
                error_reason=None,
            )
        except ValueError as e:
            # If either assignment fails, ensure clean rollback
            with contextlib.suppress(ValueError):
                worker.remove_shift_assignment(shift_id)
            with contextlib.suppress(ValueError):
                shift.remove_worker(worker_id)
            return AssignmentResult(
                success=False,
                worker_id=worker_id,
                shift_id=shift_id,
                error_reason=str(e),
            )

    def _validate_assignment_inputs(
        self,
        worker_id: str,
        shift_id: str,
    ) -> tuple[Worker, Shift] | AssignmentResult:
        """Validate that worker and shift exist, returning them or an error."""
        worker = self.get_worker(worker_id)
        shift = self.get_shift(shift_id)

        if not worker:
            return AssignmentResult(
                success=False,
                worker_id=worker_id,
                shift_id=shift_id,
                error_reason=f"Worker '{worker_id}' not found",
            )

        if not shift:
            return AssignmentResult(
                success=False,
                worker_id=worker_id,
                shift_id=shift_id,
                error_reason=f"Shift '{shift_id}' not found",
            )

        return (worker, shift)

    def _check_assignment_constraints(
        self,
        worker: Worker,
        shift: Shift,
        shift_id: str,
    ) -> AssignmentResult | None:
        """Check capacity, duplicate, and time conflict constraints."""
        worker_id = worker.id

        if self.is_worker_at_shift_capacity(worker):
            max_shifts = self.config.max_shifts_per_worker
            current_shifts = len(worker.assigned_shift_ids)
            return AssignmentResult(
                success=False,
                worker_id=worker_id,
                shift_id=shift_id,
                error_reason=(
                    f"Worker '{worker_id}' is at capacity "
                    f"({current_shifts}/{max_shifts} shifts)"
                ),
            )

        if shift.is_full:
            return AssignmentResult(
                success=False,
                worker_id=worker_id,
                shift_id=shift_id,
                error_reason=(
                    f"Shift '{shift_id}' is full ({shift.max_workers} workers)"
                ),
            )

        if worker.is_assigned_to_shift(shift_id):
            return AssignmentResult(
                success=False,
                worker_id=worker_id,
                shift_id=shift_id,
                error_reason=(
                    f"Worker '{worker_id}' is already assigned to shift '{shift_id}'"
                ),
            )

        if self._has_time_conflict(worker, shift):
            conflicting_shifts = [
                s.id
                for s_id in worker.assigned_shift_ids
                if (s := self.get_shift(s_id)) and self._shifts_overlap(s, shift)
            ]
            return AssignmentResult(
                success=False,
                worker_id=worker_id,
                shift_id=shift_id,
                error_reason=(
                    f"Shift '{shift_id}' ({shift.start_time}-{shift.end_time}) "
                    f"conflicts with assigned shifts: {conflicting_shifts}"
                ),
            )

        return None

    def unassign_worker_from_shift(self, worker_id: str, shift_id: str) -> bool:
        """Remove a worker from a shift. Returns False if not assigned."""
        worker = self.get_worker(worker_id)
        shift = self.get_shift(shift_id)

        if not worker or not shift:
            return False

        try:
            worker.remove_shift_assignment(shift_id)
            shift.remove_worker(worker_id)
        except ValueError:
            return False
        else:
            return True

    def _has_time_conflict(self, worker: Worker, new_shift: Shift) -> bool:
        """Check if assigning worker to new_shift would create time conflicts."""
        for assigned_shift_id in worker.assigned_shift_ids:
            assigned_shift = self.get_shift(assigned_shift_id)
            if assigned_shift and self._shifts_overlap(assigned_shift, new_shift):
                return True
        return False

    def _shifts_overlap(self, shift1: Shift, shift2: Shift) -> bool:
        """Check if two shifts have overlapping time periods."""
        return (
            shift1.start_time < shift2.end_time and shift2.start_time < shift1.end_time
        )


@dataclass
class CapacityValidation:
    """Results of checking if there's enough worker capacity for all shifts."""

    total_shift_slots: int
    total_worker_capacity: int
    is_feasible: bool
    shortage: int


@dataclass
class AllocationResult:
    """Results of the shift allocation process."""

    success: bool
    assignments: dict[str, list[str]]
    capacity_check: CapacityValidation
    error: str | None = None


@dataclass
class AllocationStats:
    """Statistics about the current shift allocation state."""

    total_workers: int
    workers_with_assignments: int
    total_worker_slots: int
    filled_worker_slots: int
    worker_utilization_rate: float
    total_shift_slots: int
    filled_shift_slots: int
    shift_utilization_rate: float
    total_shifts: int
    fully_staffed_shifts: int
    all_shifts_staffed: bool
    total_assignments: int
    preference_satisfied_assignments: int
    preference_satisfaction_rate: float
    preference_fulfillment: dict[int, int]


class ShiftScheduler:
    """Schedules workers to shifts based on preferences and constraints.

    Uses cryptographically secure random number generation for fair,
    unpredictable scheduling decisions in production environments.
    """

    def __init__(
        self,
        conference: Conference,
        rng: random.Random | secrets.SystemRandom | None = None,
    ) -> None:
        """Initialize scheduler with conference.

        Args:
            conference: The conference to schedule
            rng: Optional random number generator for dependency injection.
                 Defaults to cryptographically secure SystemRandom for production.

        """
        self.conference = conference
        # Use provided RNG or default to cryptographically secure random
        self.rng = rng if rng is not None else secrets.SystemRandom()

        # Performance optimization: maintain sorted available workers
        self._available_workers: list[Worker] = []
        self._worker_capacity_remaining: dict[str, int] = {}
        # Cache sorted shifts per worker for faster time conflict detection
        self._worker_sorted_shifts: dict[str, list[tuple[datetime, datetime, str]]] = {}
        self._initialize_worker_tracking()

    def _initialize_worker_tracking(self) -> None:
        """Initialize worker availability tracking for performance optimization."""
        self._worker_capacity_remaining = {
            worker.id: self.conference.config.max_shifts_per_worker
            for worker in self.conference.workers
        }
        # Sort workers by ID for consistent ordering
        self._available_workers = sorted(self.conference.workers, key=lambda w: w.id)
        # Initialize empty sorted shift caches
        self._worker_sorted_shifts = {
            worker.id: [] for worker in self.conference.workers
        }

    def _update_worker_capacity(self, worker_id: str) -> None:
        """Update worker capacity tracking after assignment."""
        if worker_id in self._worker_capacity_remaining:
            self._worker_capacity_remaining[worker_id] -= 1
            if self._worker_capacity_remaining[worker_id] <= 0:
                # Remove worker from available list
                worker = self.conference.get_worker(worker_id)
                if worker is not None and worker in self._available_workers:
                    self._available_workers.remove(worker)

    def _add_shift_to_worker_cache(self, worker_id: str, shift: Shift) -> None:
        """Add a shift to worker's sorted shift cache for fast conflict detection."""
        if worker_id in self._worker_sorted_shifts:
            shift_tuple = (shift.start_time, shift.end_time, shift.id)
            shifts = self._worker_sorted_shifts[worker_id]
            # Insert in sorted order by start time
            bisect.insort(shifts, shift_tuple)

    def _has_time_conflict_optimized(self, worker_id: str, new_shift: Shift) -> bool:
        """Optimized time conflict detection using sorted shift cache."""
        if worker_id not in self._worker_sorted_shifts:
            return False

        shifts = self._worker_sorted_shifts[worker_id]
        if not shifts:
            return False

        # Use binary search to find potentially overlapping shifts
        new_start, new_end = new_shift.start_time, new_shift.end_time

        # Find shifts that start before new_shift ends
        for start_time, end_time, _ in shifts:
            # Two shifts overlap if one starts before the other ends
            if start_time < new_end and new_start < end_time:
                return True
            # If this shift starts after new_shift ends, no more overlaps possible
            if start_time >= new_end:
                break
        return False

    def _get_available_workers_for_shift(self, shift: Shift) -> list[Worker]:
        """Get workers available for a specific shift with optimized filtering."""
        available = []
        for worker in self._available_workers:
            # Quick capacity check first
            if self._worker_capacity_remaining[worker.id] <= 0:
                continue
            # Already assigned to this shift check
            if worker.is_assigned_to_shift(shift.id):
                continue
            # Use optimized time conflict check
            if self._has_time_conflict_optimized(worker.id, shift):
                continue
            available.append(worker)
        return available

    def validate_capacity(self) -> CapacityValidation:
        """Check if there's enough worker capacity to fill all shifts."""
        total_shift_slots = sum(shift.max_workers for shift in self.conference.shifts)
        total_worker_capacity = (
            len(self.conference.workers) * self.conference.config.max_shifts_per_worker
        )

        return CapacityValidation(
            total_shift_slots=total_shift_slots,
            total_worker_capacity=total_worker_capacity,
            is_feasible=total_worker_capacity >= total_shift_slots,
            shortage=max(0, total_shift_slots - total_worker_capacity),
        )

    def allocate_shifts(self) -> AllocationResult:
        """Allocate workers to shifts ensuring ALL shifts are fully staffed.

        Uses two-phase approach: preference-based assignment first,
        then fill remaining slots.
        """
        capacity_check = self.validate_capacity()
        if not capacity_check.is_feasible:
            return AllocationResult(
                success=False,
                error=(
                    f"Cannot allocate all shifts - need {capacity_check.shortage} more "
                    f"worker slots. Current: {capacity_check.total_worker_capacity}, "
                    f"required: {capacity_check.total_shift_slots} slots."
                ),
                assignments={},
                capacity_check=capacity_check,
            )

        # Phase 1: Preference-based assignment with fair tie-breaking
        self._assign_by_preferences()

        # Phase 2: Fill remaining unfilled shifts
        self._fill_remaining_shifts()

        # Return final assignments
        assignments = {
            worker.id: list(worker.assigned_shift_ids)
            for worker in self.conference.workers
            if worker.assigned_shift_ids
        }

        return AllocationResult(
            success=True,
            assignments=assignments,
            capacity_check=capacity_check,
        )

    def _create_preference_tuples(self) -> list[tuple[Worker, Shift, int]]:
        """Create 3-tuples (worker, shift, score) for all valid preferences."""
        preference_tuples: list[tuple[Worker, Shift, int]] = []

        for worker in self.conference.workers:
            for shift_id in worker.shift_preferences:
                shift = self.conference.get_shift(shift_id)
                if shift is None:
                    continue
                score = worker.get_preference_score(shift_id)
                if score > 0:
                    preference_tuples.append((worker, shift, score))

        return preference_tuples

    def _sort_and_shuffle_preferences(
        self,
        preference_tuples: list[tuple[Worker, Shift, int]],
    ) -> None:
        """Sort by score and shuffle within score groups for fairness."""
        # Sort by score (highest first)
        preference_tuples.sort(key=lambda x: -x[2])

        # Group by score and shuffle within each score level
        current_score = None
        score_group_start = 0
        for i, (_, _, score) in enumerate(preference_tuples):
            if score != current_score:
                if current_score is not None:
                    # Shuffle the previous score group
                    score_group = preference_tuples[score_group_start:i]
                    self.rng.shuffle(score_group)
                    preference_tuples[score_group_start:i] = score_group
                current_score = score
                score_group_start = i

        # Shuffle the last score group
        if preference_tuples:
            score_group = preference_tuples[score_group_start:]
            self.rng.shuffle(score_group)
            preference_tuples[score_group_start:] = score_group

    def _can_assign_worker_to_shift(self, worker: Worker, shift: Shift) -> bool:
        """Check if worker can be assigned to shift based on all constraints."""
        return not (
            shift.is_full
            or self._worker_capacity_remaining[worker.id] <= 0
            or worker.is_assigned_to_shift(shift.id)
            or self._has_time_conflict_optimized(worker.id, shift)
        )

    def _process_preference_assignments(
        self,
        preference_tuples: list[tuple[Worker, Shift, int]],
    ) -> None:
        """Process assignments from sorted preference tuples."""
        for worker, shift, _score in preference_tuples:
            if not self._can_assign_worker_to_shift(worker, shift):
                continue

            # Use the existing assign method which handles all validation
            result = self.conference.try_assign_worker_to_shift(worker.id, shift.id)
            if result.success:
                # Update our tracking
                self._update_worker_capacity(worker.id)
                # Add to sorted shift cache for faster future conflict detection
                self._add_shift_to_worker_cache(worker.id, shift)

    def _assign_by_preferences(self) -> None:
        """Phase 1: Assign workers based on preferences with fair tie-breaking."""
        preference_tuples = self._create_preference_tuples()
        self._sort_and_shuffle_preferences(preference_tuples)
        self._process_preference_assignments(preference_tuples)

    def _fill_remaining_shifts(self) -> None:
        """Phase 2: Fill any remaining unfilled shifts with available workers."""
        unfilled_shifts = [
            shift for shift in self.conference.shifts if not shift.is_full
        ]

        for shift in unfilled_shifts:
            while not shift.is_full:
                # Use optimized worker availability check
                available_workers = self._get_available_workers_for_shift(shift)

                if not available_workers:
                    # No workers available for this shift
                    break

                # Assign the first available worker
                worker = available_workers[0]
                if self.conference.assign_worker_to_shift(worker.id, shift.id):
                    # Update our tracking
                    self._update_worker_capacity(worker.id)
                    # Add to sorted shift cache
                    self._add_shift_to_worker_cache(worker.id, shift)
                else:
                    # If assignment failed unexpectedly, break to avoid infinite loop
                    break

    def get_allocation_stats(self) -> AllocationStats:
        """Get statistics about the current allocation."""
        total_workers = len(self.conference.workers)
        workers_with_assignments = len(
            [w for w in self.conference.workers if w.assigned_shift_ids],
        )

        total_worker_slots = (
            len(self.conference.workers) * self.conference.config.max_shifts_per_worker
        )
        filled_worker_slots = sum(
            len(w.assigned_shift_ids) for w in self.conference.workers
        )

        total_shift_slots = sum(s.max_workers for s in self.conference.shifts)
        filled_shift_slots = sum(
            len(s.assigned_worker_ids) for s in self.conference.shifts
        )

        # Check if all shifts are fully staffed
        fully_staffed_shifts = sum(1 for s in self.conference.shifts if s.is_full)

        # Count preference fulfillment
        preference_fulfillment: dict[int, int] = {}
        total_assignments = 0
        preference_satisfied_assignments = 0

        for worker in self.conference.workers:
            for shift_id in worker.assigned_shift_ids:
                total_assignments += 1
                rank = worker.get_preference_rank(shift_id)
                if rank:
                    preference_fulfillment[rank] = (
                        preference_fulfillment.get(rank, 0) + 1
                    )
                    preference_satisfied_assignments += 1

        return AllocationStats(
            total_workers=total_workers,
            workers_with_assignments=workers_with_assignments,
            total_worker_slots=total_worker_slots,
            filled_worker_slots=filled_worker_slots,
            worker_utilization_rate=filled_worker_slots / total_worker_slots
            if total_worker_slots > 0
            else 0,
            total_shift_slots=total_shift_slots,
            filled_shift_slots=filled_shift_slots,
            shift_utilization_rate=filled_shift_slots / total_shift_slots
            if total_shift_slots > 0
            else 0,
            total_shifts=len(self.conference.shifts),
            fully_staffed_shifts=fully_staffed_shifts,
            all_shifts_staffed=fully_staffed_shifts == len(self.conference.shifts),
            total_assignments=total_assignments,
            preference_satisfied_assignments=preference_satisfied_assignments,
            preference_satisfaction_rate=preference_satisfied_assignments
            / total_assignments
            if total_assignments > 0
            else 0,
            preference_fulfillment=preference_fulfillment,
        )
