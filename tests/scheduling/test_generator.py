"""Tests for schedule generation functionality."""

from datetime import UTC, datetime, timedelta

import pytest

from scheduling.domain import SchedulingConstraints, WorkerPreference
from scheduling.generator import (
    ScheduleError,
    _shifts_overlap,
    generate_schedule,
)

from .builders import ShiftBuilder, WorkerBuilder


def _create_default_constraints(
    min_workers_per_shift: int = 1,
    max_workers_per_shift: int = 3,
    min_shifts_per_worker: int = 1,
    max_shifts_per_worker: int = 5,
) -> SchedulingConstraints:
    """Create default scheduling constraints for tests."""
    return SchedulingConstraints(
        min_workers_per_shift=min_workers_per_shift,
        max_workers_per_shift=max_workers_per_shift,
        min_shifts_per_worker=min_shifts_per_worker,
        max_shifts_per_worker=max_shifts_per_worker,
        min_transition_time=timedelta(minutes=30),
        fairness_enabled=True,
        prefer_consecutive_shifts=False,
    )


class TestScheduleGeneration:
    """Test schedule generation with preferences and constraints."""

    def test_successful_schedule_generation(self) -> None:
        """Test generating a valid schedule with worker preferences."""
        # Create workers
        worker1 = WorkerBuilder().with_id("w1").build()
        worker2 = WorkerBuilder().with_id("w2").build()
        workers = [worker1, worker2]

        # Create shifts
        shift1 = ShiftBuilder().with_id("s1").with_duration_hours(9, 4).build()
        shift2 = ShiftBuilder().with_id("s2").with_duration_hours(14, 3).build()
        shifts = [shift1, shift2]

        # Add preferences
        pref1 = WorkerPreference(worker=worker1, shift=shift1, preference_level=5)
        pref2 = WorkerPreference(worker=worker2, shift=shift2, preference_level=4)
        preferences = [pref1, pref2]

        # Generate schedule with default constraints
        constraints = _create_default_constraints()
        assignments = generate_schedule(shifts, workers, preferences, constraints)

        # Should succeed
        expected_assignments = 2
        assert len(assignments) == expected_assignments

        # Verify assignments match preferences
        assignment_pairs = [(a.worker.id, a.shift.id) for a in assignments]
        assert ("w1", "s1") in assignment_pairs
        assert ("w2", "s2") in assignment_pairs


class TestOverlappingShiftValidation:
    """Test shift overlap detection."""

    def test_shifts_overlap_detection(self) -> None:
        """Test detection of overlapping shifts."""
        # Create overlapping shifts
        shift1 = (
            ShiftBuilder()
            .with_id("s1")
            .with_time_range(
                datetime(2024, 6, 1, 9, 0, tzinfo=UTC),
                datetime(2024, 6, 1, 13, 0, tzinfo=UTC),
            )
            .build()
        )

        shift2 = (
            ShiftBuilder()
            .with_id("s2")
            .with_time_range(
                datetime(2024, 6, 1, 11, 0, tzinfo=UTC),
                datetime(2024, 6, 1, 15, 0, tzinfo=UTC),
            )
            .build()
        )

        assert _shifts_overlap(shift1, shift2)
        assert _shifts_overlap(shift2, shift1)

    def test_shifts_no_overlap_detection(self) -> None:
        """Test detection of non-overlapping shifts."""
        # Create non-overlapping shifts
        shift1 = (
            ShiftBuilder()
            .with_id("s1")
            .with_time_range(
                datetime(2024, 6, 1, 9, 0, tzinfo=UTC),
                datetime(2024, 6, 1, 12, 0, tzinfo=UTC),
            )
            .build()
        )

        shift2 = (
            ShiftBuilder()
            .with_id("s2")
            .with_time_range(
                datetime(2024, 6, 1, 13, 0, tzinfo=UTC),
                datetime(2024, 6, 1, 16, 0, tzinfo=UTC),
            )
            .build()
        )

        assert not _shifts_overlap(shift1, shift2)
        assert not _shifts_overlap(shift2, shift1)

    def test_basic_scheduling_constraints(self) -> None:
        """Test creating basic scheduling constraints."""
        expected_min_workers = 2
        expected_max_workers = 4
        expected_min_shifts = 1
        expected_max_shifts = 3

        constraints = _create_default_constraints(
            min_workers_per_shift=expected_min_workers,
            max_workers_per_shift=expected_max_workers,
            min_shifts_per_worker=expected_min_shifts,
            max_shifts_per_worker=expected_max_shifts,
        )

        assert constraints.min_workers_per_shift == expected_min_workers
        assert constraints.max_workers_per_shift == expected_max_workers
        assert constraints.min_shifts_per_worker == expected_min_shifts
        assert constraints.max_shifts_per_worker == expected_max_shifts

    def test_schedule_with_no_preferences(self) -> None:
        """Test scheduling when workers have no preferences."""
        # Create workers
        worker1 = WorkerBuilder().with_id("w1").build()
        worker2 = WorkerBuilder().with_id("w2").build()
        workers = [worker1, worker2]

        # Create shifts
        shift1 = ShiftBuilder().with_id("s1").with_duration_hours(9, 4).build()
        shift2 = ShiftBuilder().with_id("s2").with_duration_hours(14, 3).build()
        shifts = [shift1, shift2]

        # No preferences - should use fallback assignment
        preferences: list[WorkerPreference] = []

        # Generate schedule
        constraints = _create_default_constraints()
        assignments = generate_schedule(shifts, workers, preferences, constraints)

        # Should succeed via fallback assignment
        expected_min_assignments = 2
        assert len(assignments) >= expected_min_assignments

    def test_schedule_impossible_constraints(self) -> None:
        """Test scheduling with impossible constraints."""
        # Create one worker
        worker = WorkerBuilder().with_id("w1").build()
        workers = [worker]

        # Create shift that needs more workers than available
        shift = ShiftBuilder().with_id("s1").with_duration_hours(9, 4).build()
        shifts = [shift]

        # No preferences
        preferences: list[WorkerPreference] = []

        # Impossible constraints - need 3 workers but only have 1
        constraints = _create_default_constraints(min_workers_per_shift=3)

        # Should raise ScheduleError
        with pytest.raises(
            ScheduleError,
            match="Could not meet minimum worker requirements",
        ):
            generate_schedule(shifts, workers, preferences, constraints)
