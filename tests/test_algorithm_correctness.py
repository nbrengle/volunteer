"""Tests for algorithm correctness - duplicate assignments and time conflicts."""

import random
from datetime import UTC, datetime, timedelta

from scheduling.scheduling_domain import (
    ConferenceConfig,
    Shift,
    ShiftScheduler,
)

from .constants import (
    DEFAULT_TEST_SEED,
    TEST_SHIFT_ID,
    TEST_WORKER_ID,
)
from .data_generator import ConferenceDataGenerator
from .test_builders import (
    ConferenceBuilder,
    ShiftBuilder,
    WorkerBuilder,
    create_overlapping_shifts_scenario,
    create_simple_test_scenario,
)


class TestDuplicateAssignmentPrevention:
    """Test suite to ensure workers cannot be assigned to same shift multiple times."""

    def test_worker_cannot_be_assigned_to_same_shift_twice(self) -> None:
        """Test that attempting to assign same worker to same shift twice fails."""
        conference, worker, shift = create_simple_test_scenario()

        # First assignment should succeed
        success1 = conference.assign_worker_to_shift(TEST_WORKER_ID, TEST_SHIFT_ID)
        assert success1, "First assignment should succeed"

        # Second assignment of same worker to same shift should fail
        success2 = conference.assign_worker_to_shift(TEST_WORKER_ID, TEST_SHIFT_ID)
        assert not success2, (
            "Second assignment of same worker to same shift should fail"
        )

        # Verify worker is only assigned once
        assert len(worker.assigned_shift_ids) == 1
        assert TEST_SHIFT_ID in worker.assigned_shift_ids

        # Verify shift has worker assigned only once
        assert len(shift.assigned_worker_ids) == 1
        assert TEST_WORKER_ID in shift.assigned_worker_ids

    def test_assign_worker_to_shift_prevents_duplicates(self) -> None:
        """Test that assign_worker_to_shift method prevents duplicate assignments."""
        conference = ConferenceBuilder().build()
        worker = WorkerBuilder().with_id("w1").build()
        shift = ShiftBuilder().with_id("s1").with_max_workers(3).build()

        conference.add_worker(worker)
        conference.add_shift(shift)

        # Assign worker to shift
        result1 = conference.assign_worker_to_shift("w1", "s1")
        assert result1, "Initial assignment should succeed"

        # Try to assign same worker to same shift again
        result2 = conference.assign_worker_to_shift("w1", "s1")
        assert not result2, "Duplicate assignment should fail"

        # Verify state is consistent
        assert worker.is_assigned_to_shift("s1")
        assert "w1" in shift.assigned_worker_ids
        assert len(worker.assigned_shift_ids) == 1
        assert len(shift.assigned_worker_ids) == 1

    def test_scheduler_algorithm_prevents_duplicate_assignments(self) -> None:
        """Test that the scheduling algorithm itself prevents duplicate assignments."""
        # Create a conference with overlapping worker preferences
        generator = ConferenceDataGenerator(seed=DEFAULT_TEST_SEED)

        config = ConferenceConfig(
            start_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
            duration_days=1,
            shifts_per_day=2,
            min_workers_per_shift=1,
            max_workers_per_shift=1,  # Only 1 worker per shift to force conflicts
            min_shifts_per_worker=1,
            max_shifts_per_worker=2,
            max_preferences_per_worker=5,
        )

        conference = generator.generate_conference(
            name="Duplicate Test",
            num_workers=2,
            num_shifts=2,
            config=config,
        )

        # Make both workers prefer the same shift to create potential duplicates
        min_workers = 2
        min_shifts = 1
        has_enough_workers = len(conference.workers) >= min_workers
        has_enough_shifts = len(conference.shifts) >= min_shifts
        if has_enough_workers and has_enough_shifts:
            shift_id = conference.shifts[0].id
            conference.workers[0].set_preferences([shift_id])
            conference.workers[1].set_preferences([shift_id])

        scheduler = ShiftScheduler(conference, rng=random.Random(DEFAULT_TEST_SEED))
        result = scheduler.allocate_shifts()

        assert result.success, "Allocation should succeed"

        # Verify no worker is assigned to the same shift multiple times
        for worker in conference.workers:
            unique_assignments = set(worker.assigned_shift_ids)
            assert len(unique_assignments) == len(worker.assigned_shift_ids), (
                f"Worker {worker.id} has duplicate shift assignments: "
                f"{worker.assigned_shift_ids}"
            )

        # Verify no shift has the same worker assigned multiple times
        for shift in conference.shifts:
            unique_workers = set(shift.assigned_worker_ids)
            assert len(unique_workers) == len(shift.assigned_worker_ids), (
                f"Shift {shift.id} has duplicate worker assignments: "
                f"{shift.assigned_worker_ids}"
            )


class TestTimeConflictDetection:
    """Test suite to ensure workers cannot be assigned to overlapping shifts."""

    def test_worker_cannot_have_overlapping_shifts(self) -> None:
        """Test that workers cannot be assigned to shifts with overlapping times."""
        conference, worker, shift1, shift2 = create_overlapping_shifts_scenario()

        # Assign worker to first shift
        result1 = conference.assign_worker_to_shift(TEST_WORKER_ID, "s1")
        assert result1, "First assignment should succeed"

        # Attempt to assign worker to overlapping shift should fail
        result2 = conference.assign_worker_to_shift(TEST_WORKER_ID, "s2")
        assert not result2, "Assignment to overlapping shift should fail"

        # Verify worker is only assigned to the first shift
        assert len(worker.assigned_shift_ids) == 1
        assert "s1" in worker.assigned_shift_ids
        assert "s2" not in worker.assigned_shift_ids

    def test_time_conflict_detection_prevents_assignment(self) -> None:
        """Test various time overlap scenarios are properly detected."""
        conference = (
            ConferenceBuilder().with_simple_config(max_shifts_per_worker=10).build()
        )
        worker = WorkerBuilder().with_id("w1").build()
        conference.add_worker(worker)

        # Base shift: 9 AM - 1 PM
        base_shift = Shift(
            id="base",
            start_time=datetime(2024, 6, 1, 9, 0, tzinfo=UTC),
            end_time=datetime(2024, 6, 1, 13, 0, tzinfo=UTC),
            location="Room A",
            max_workers=5,
        )
        conference.add_shift(base_shift)

        # Assign worker to base shift
        success = conference.assign_worker_to_shift("w1", "base")
        assert success, "Base assignment should succeed"

        # Test various overlapping scenarios
        overlap_scenarios = [
            # Complete overlap
            (
                "complete",
                datetime(2024, 6, 1, 9, 0, tzinfo=UTC),
                datetime(2024, 6, 1, 13, 0, tzinfo=UTC),
            ),
            # Start before, end during
            (
                "start_before",
                datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
                datetime(2024, 6, 1, 11, 0, tzinfo=UTC),
            ),
            # Start during, end after
            (
                "end_after",
                datetime(2024, 6, 1, 11, 0, tzinfo=UTC),
                datetime(2024, 6, 1, 15, 0, tzinfo=UTC),
            ),
            # Completely contains base shift
            (
                "contains",
                datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
                datetime(2024, 6, 1, 15, 0, tzinfo=UTC),
            ),
            # Base shift completely contains this one
            (
                "contained",
                datetime(2024, 6, 1, 10, 0, tzinfo=UTC),
                datetime(2024, 6, 1, 12, 0, tzinfo=UTC),
            ),
            # Touching at end (should be allowed - no overlap)
            (
                "touching_end",
                datetime(2024, 6, 1, 13, 0, tzinfo=UTC),
                datetime(2024, 6, 1, 17, 0, tzinfo=UTC),
            ),
            # Touching at start (should be allowed - no overlap)
            (
                "touching_start",
                datetime(2024, 6, 1, 5, 0, tzinfo=UTC),
                datetime(2024, 6, 1, 9, 0, tzinfo=UTC),
            ),
        ]

        for scenario_name, start_time, end_time in overlap_scenarios:
            shift = Shift(
                id=scenario_name,
                start_time=start_time,
                end_time=end_time,
                location=f"Room {scenario_name}",
                max_workers=5,
            )
            conference.add_shift(shift)

            result = conference.assign_worker_to_shift("w1", scenario_name)

            if scenario_name in ["touching_end", "touching_start"]:
                assert result, (
                    f"Non-overlapping shift '{scenario_name}' should be allowed"
                )
            else:
                assert not result, (
                    f"Overlapping shift '{scenario_name}' should be prevented"
                )

    def test_scheduler_respects_time_constraints(self) -> None:
        """Test that the scheduling algorithm respects time constraints."""
        generator = ConferenceDataGenerator(seed=DEFAULT_TEST_SEED)

        config = ConferenceConfig(
            start_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
            duration_days=1,
            shifts_per_day=4,  # Create multiple shifts in one day
            min_workers_per_shift=1,
            max_workers_per_shift=1,
            min_shifts_per_worker=1,
            max_shifts_per_worker=4,  # Allow workers to take multiple shifts
            max_preferences_per_worker=10,
        )

        # Generate a conference where shifts might overlap
        conference = generator.generate_conference(
            name="Time Conflict Test",
            num_workers=2,
            num_shifts=4,
            config=config,
        )

        # Manually create some overlapping shifts to test the scheduler
        overlapping_shift = Shift(
            id="overlap_test",
            start_time=conference.shifts[0].start_time
            + timedelta(hours=1),  # Start 1 hour after first shift starts
            end_time=conference.shifts[0].end_time
            + timedelta(hours=1),  # End 1 hour after first shift ends
            location="Overlap Room",
            max_workers=1,
        )
        conference.add_shift(overlapping_shift)

        scheduler = ShiftScheduler(conference, rng=random.Random(DEFAULT_TEST_SEED))
        result = scheduler.allocate_shifts()

        # The algorithm should succeed but respect time constraints
        assert result.success, "Allocation should succeed"

        # Verify no worker has overlapping shift assignments
        for worker in conference.workers:
            assigned_shifts = [
                shift
                for shift in conference.shifts
                if shift.id in worker.assigned_shift_ids
            ]

            # Check all pairs of assigned shifts for time overlap
            for i, shift1 in enumerate(assigned_shifts):
                for shift2 in assigned_shifts[i + 1 :]:
                    # Two shifts overlap if one starts before the other ends
                    overlap = (
                        shift1.start_time < shift2.end_time
                        and shift2.start_time < shift1.end_time
                    )
                    assert not overlap, (
                        f"Worker {worker.id} assigned to overlapping shifts: "
                        f"{shift1.id} ({shift1.start_time}-{shift1.end_time}) and "
                        f"{shift2.id} ({shift2.start_time}-{shift2.end_time})"
                    )
