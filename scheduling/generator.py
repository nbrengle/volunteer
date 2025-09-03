"""Schedule generation algorithms for volunteer assignment."""

from __future__ import annotations

from dataclasses import dataclass

from .domain import Assignment, Conference, Shift, Worker


@dataclass
class Schedule:
    """A successfully generated schedule."""

    conference: Conference
    assignments: list[Assignment]


@dataclass
class ScheduleError:
    """Error when schedule generation fails."""

    unassigned_shifts: list[Shift]
    error_message: str


def generate_schedule(conference: Conference) -> Schedule | ScheduleError:
    """Generate a schedule prioritizing worker preferences while meeting constraints.

    Algorithm:
    1. Sort shifts by capacity (most constrained first)
    2. For each shift, sort workers by preference level (highest first)
    3. Assign workers to shifts while respecting constraints:
       - No overlapping shifts for a worker
       - Worker capacity limits
       - Shift minimum requirements must be met

    Args:
        conference: Conference with workers, shifts, and preferences

    Returns:
        Schedule on success, ScheduleError on failure
    """
    assignments: list[Assignment] = []

    # Create lookup structures for fast access
    shift_assignments: dict[Shift, list[Assignment]] = {
        shift: [] for shift in conference.shifts
    }
    worker_assignments: dict[Worker, list[Assignment]] = {
        worker: [] for worker in conference.workers
    }

    # Sort shifts by capacity (smallest capacity = most constrained first)
    shifts_to_fill = sorted(
        conference.shifts,
        key=lambda s: s.max_workers,
    )

    unassigned_shifts = []

    for shift in shifts_to_fill:
        assigned_count = len(shift_assignments[shift])

        # Check if shift already meets minimum requirement
        if assigned_count >= conference.config.min_workers_per_shift:
            continue

        # Get workers who prefer this shift, sorted by preference level
        shift_preferences = conference.get_preferences_for_shift(shift)
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
                conference,
                shift_assignments,
                worker_assignments,
                worker,
                shift,
            ):
                assignment = Assignment(worker=worker, shift=shift)
                assignments.append(assignment)
                shift_assignments[shift].append(assignment)
                worker_assignments[worker].append(assignment)
                assigned_count += 1

                # Stop if we've met the minimum requirement
                if assigned_count >= conference.config.min_workers_per_shift:
                    break

        # Check if shift minimum requirement is met
        final_assigned_count = len(shift_assignments[shift])
        if final_assigned_count < conference.config.min_workers_per_shift:
            unassigned_shifts.append(shift)

    # Second phase: Try to assign any available workers to unassigned shifts
    if unassigned_shifts:
        unassigned_shifts = _fallback_assignment(
            conference,
            assignments,
            shift_assignments,
            worker_assignments,
            unassigned_shifts,
        )

    # Return appropriate result
    if unassigned_shifts:
        shift_names = [s.id for s in unassigned_shifts]
        error_message = (
            f"Could not meet minimum worker requirements for shifts: {shift_names}"
        )
        return ScheduleError(
            unassigned_shifts=unassigned_shifts,
            error_message=error_message,
        )

    return Schedule(conference=conference, assignments=assignments)


def _get_assignments_for_shift(
    assignments: list[Assignment],
    shift: Shift,
) -> list[Assignment]:
    """Get assignments for a specific shift."""
    return [assignment for assignment in assignments if assignment.shift == shift]


def _get_assignments_for_worker(
    assignments: list[Assignment],
    worker: Worker,
) -> list[Assignment]:
    """Get assignments for a specific worker."""
    return [assignment for assignment in assignments if assignment.worker == worker]


def _can_assign_worker(
    conference: Conference,
    shift_assignments: dict[Shift, list[Assignment]],
    worker_assignments: dict[Worker, list[Assignment]],
    worker: Worker,
    shift: Shift,
) -> bool:
    """Check if a worker can be assigned to a shift."""
    # Check worker capacity
    worker_assignment_list = worker_assignments[worker]
    if len(worker_assignment_list) >= conference.config.max_shifts_per_worker:
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
        if conference.shifts_overlap(assignment.shift, shift):
            return False

    return True


def _fallback_assignment(
    conference: Conference,
    assignments: list[Assignment],
    shift_assignments: dict[Shift, list[Assignment]],
    worker_assignments: dict[Worker, list[Assignment]],
    unassigned_shifts: list[Shift],
) -> list[Shift]:
    """Try to assign workers to shifts that couldn't be filled by preferences."""
    if not unassigned_shifts:
        return []

    # Pre-compute worker assignment counts
    worker_assignment_counts = {
        worker: len(worker_assignments[worker]) for worker in conference.workers
    }

    remaining_unassigned = []

    for shift in unassigned_shifts:
        current_count = len(shift_assignments[shift])
        workers_needed = conference.config.min_workers_per_shift - current_count

        if workers_needed <= 0:
            continue

        # Find available workers for this shift
        available_workers = []
        for worker in conference.workers:
            # Quick checks first before expensive overlap check
            if (
                worker_assignment_counts[worker]
                >= conference.config.max_shifts_per_worker
            ):
                continue

            # Check shift capacity
            if current_count >= shift.max_workers:
                break  # Shift is full

            # Check for overlaps
            worker_assignment_list = worker_assignments[worker]
            has_overlap = any(
                conference.shifts_overlap(assignment.shift, shift)
                for assignment in worker_assignment_list
            )
            if has_overlap:
                continue

            available_workers.append((worker_assignment_counts[worker], worker))

        # Sort by current assignment count (least loaded first)
        available_workers.sort(key=lambda x: x[0])

        # Assign workers until minimum requirement is met
        for _, worker in available_workers[:workers_needed]:  # Only take what we need
            assignment = Assignment(worker=worker, shift=shift)
            assignments.append(assignment)
            shift_assignments[shift].append(assignment)
            worker_assignments[worker].append(assignment)
            worker_assignment_counts[worker] += 1
            current_count += 1

        # Check if we met the minimum requirement
        if current_count < conference.config.min_workers_per_shift:
            remaining_unassigned.append(shift)

    return remaining_unassigned
