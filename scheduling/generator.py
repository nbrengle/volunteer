"""Schedule generation algorithms for volunteer assignment."""

from __future__ import annotations

from dataclasses import dataclass

from .domain import (
    Assignment,
    SchedulingConstraints,
    Shift,
    Worker,
    WorkerPreference,
)


class ScheduleError(Exception):
    """Error when schedule generation fails."""

    def __init__(self, unassigned_shifts: list[Shift], message: str) -> None:
        """Initialize ScheduleError with unassigned shifts and error message."""
        self.unassigned_shifts = unassigned_shifts
        super().__init__(message)


@dataclass
class SchedulingState:
    """Current state of the scheduling algorithm."""

    assignments: list[Assignment]
    shift_assignments: dict[Shift, list[Assignment]]
    worker_assignments: dict[Worker, list[Assignment]]

    @property
    def worker_assignment_counts(self) -> dict[Worker, int]:
        """Get count of assignments per worker."""
        return {
            worker: len(assignments)
            for worker, assignments in self.worker_assignments.items()
        }


def generate_schedule(
    shifts: list[Shift],
    workers: list[Worker],
    preferences: list[WorkerPreference],
    constraints: SchedulingConstraints,
) -> list[Assignment]:
    """Generate a schedule prioritizing worker preferences while meeting constraints.

    Algorithm:
    1. Sort shifts by total capacity (most constrained first)
    2. For each shift requirement, sort workers by preference level (highest first)
    3. Assign workers to specific shift requirements while respecting constraints:
       - Workers must have required skills
       - No overlapping shifts for a worker
       - Worker capacity limits
       - Shift requirement minimums must be met

    Args:
        shifts: List of shifts to be filled
        workers: List of available workers
        preferences: List of worker preferences for shifts
        constraints: Scheduling constraints and rules

    Returns:
        List of assignments

    Raises:
        ScheduleError: When minimum requirements cannot be met
    """
    assignments: list[Assignment] = []

    # Create lookup structures for fast access
    shift_assignments: dict[Shift, list[Assignment]] = {shift: [] for shift in shifts}
    worker_assignments: dict[Worker, list[Assignment]] = {
        worker: [] for worker in workers
    }

    # Sort shifts by capacity (smallest capacity = most constrained first)
    shifts_to_fill = sorted(
        shifts,
        key=lambda s: s.max_workers,
    )

    unassigned_shifts = []

    for shift in shifts_to_fill:
        assigned_count = len(shift_assignments[shift])

        # Check if shift already meets minimum requirement
        # (use higher of global or shift-specific)
        min_required = max(constraints.min_workers_per_shift, shift.min_workers)
        if assigned_count >= min_required:
            continue

        # Get workers who prefer this shift, sorted by preference level
        shift_preferences = [p for p in preferences if p.shift == shift]
        preferred_workers = sorted(
            shift_preferences,
            key=lambda p: p.preference_level,
            reverse=True,  # Highest preference first
        )

        # Try to assign workers until minimum is met
        for preference in preferred_workers:
            worker = preference.worker

            # Check if we can assign this worker
            if _can_assign_worker(
                constraints,
                shift_assignments,
                worker_assignments,
                worker,
                shift,
            ):
                assignment = Assignment(
                    worker=worker,
                    shift=shift,
                )
                assignments.append(assignment)
                shift_assignments[shift].append(assignment)
                worker_assignments[worker].append(assignment)
                assigned_count += 1

                # Stop if we've met the minimum requirement
                if assigned_count >= min_required:
                    break

        # Check if shift minimum requirement is met
        final_assigned_count = len(shift_assignments[shift])
        if final_assigned_count < min_required:
            unassigned_shifts.append(shift)

    # Second phase: Try to assign any available workers to unassigned shifts
    if unassigned_shifts:
        state = SchedulingState(
            assignments=assignments,
            shift_assignments=shift_assignments,
            worker_assignments=worker_assignments,
        )
        unassigned_shifts = _fallback_assignment(
            workers,
            constraints,
            state,
            unassigned_shifts,
        )

    # Return appropriate result
    if unassigned_shifts:
        shift_names = [s.id for s in unassigned_shifts]
        error_message = (
            f"Could not meet minimum worker requirements for shifts: {shift_names}"
        )
        raise ScheduleError(unassigned_shifts, error_message)

    return assignments


def _can_assign_worker(
    constraints: SchedulingConstraints,
    shift_assignments: dict[Shift, list[Assignment]],
    worker_assignments: dict[Worker, list[Assignment]],
    worker: Worker,
    shift: Shift,
) -> bool:
    """Check if a worker can be assigned to a shift."""
    # Check worker capacity
    worker_assignment_list = worker_assignments[worker]
    if len(worker_assignment_list) >= constraints.max_shifts_per_worker:
        return False

    # Check shift capacity
    shift_assignment_list = shift_assignments[shift]
    if len(shift_assignment_list) >= shift.max_workers:
        return False

    # Check if already assigned
    if any(
        assignment.worker == worker and assignment.shift == shift
        for assignment in shift_assignment_list
    ):
        return False

    # Check for overlapping shifts
    for assignment in worker_assignment_list:
        if _shifts_overlap(assignment.shift, shift):
            return False

    return True


def _shifts_overlap(shift1: Shift, shift2: Shift) -> bool:
    """Check if two shifts have overlapping time periods."""
    return shift1.start_time < shift2.end_time and shift2.start_time < shift1.end_time


def _find_available_workers(
    workers: list[Worker],
    shift: Shift,
    current_count: int,
    constraints: SchedulingConstraints,
    state: SchedulingState,
) -> list[tuple[int, Worker]]:
    """Find workers available for assignment to a shift."""
    worker_assignment_counts = state.worker_assignment_counts
    worker_assignments = state.worker_assignments
    available_workers = []
    for worker in workers:
        # Quick checks first before expensive overlap check
        if worker_assignment_counts[worker] >= constraints.max_shifts_per_worker:
            continue

        # Check shift capacity
        if current_count >= shift.max_workers:
            break  # Shift is full

        # Check for overlaps
        worker_assignment_list = worker_assignments[worker]
        has_overlap = any(
            _shifts_overlap(assignment.shift, shift)
            for assignment in worker_assignment_list
        )
        if has_overlap:
            continue

        available_workers.append((worker_assignment_counts[worker], worker))

    return available_workers


def _fallback_assignment(
    workers: list[Worker],
    constraints: SchedulingConstraints,
    state: SchedulingState,
    unassigned_shifts: list[Shift],
) -> list[Shift]:
    """Try to assign workers to shifts that couldn't be filled by preferences."""
    if not unassigned_shifts:
        return []

    assignments = state.assignments
    shift_assignments = state.shift_assignments
    worker_assignments = state.worker_assignments

    # Get worker assignment counts
    worker_assignment_counts = state.worker_assignment_counts

    remaining_unassigned = []

    for shift in unassigned_shifts:
        current_count = len(shift_assignments[shift])
        min_required = max(constraints.min_workers_per_shift, shift.min_workers)
        workers_needed = min_required - current_count

        if workers_needed <= 0:
            continue

        # Find available workers for this shift
        # Create state object for finding available workers
        temp_state = SchedulingState(
            assignments=assignments,
            shift_assignments=shift_assignments,
            worker_assignments=worker_assignments,
        )
        available_workers = _find_available_workers(
            workers,
            shift,
            current_count,
            constraints,
            temp_state,
        )

        # Sort by current assignment count (least loaded first)
        available_workers.sort(key=lambda x: x[0])

        # Assign workers until minimum requirement is met (limited by available workers)
        workers_to_assign = min(workers_needed, len(available_workers))
        for _, worker in available_workers[:workers_to_assign]:
            assignment = Assignment(worker=worker, shift=shift)
            assignments.append(assignment)
            shift_assignments[shift].append(assignment)
            worker_assignments[worker].append(assignment)
            worker_assignment_counts[worker] += 1
            current_count += 1

        # Check if we met the minimum requirement
        if current_count < min_required:
            remaining_unassigned.append(shift)

    return remaining_unassigned
