"""Tests for schedule generation functionality."""

from datetime import UTC, datetime

import pytest

from scheduling.domain import ConferenceConfig, WorkerPreference
from scheduling.generator import Schedule, ScheduleError, generate_schedule

from .builders import ConferenceBuilder, ShiftBuilder, WorkerBuilder


class TestScheduleGeneration:
    """Test schedule generation with preferences and constraints."""

    def test_successful_schedule_generation(self) -> None:
        """Test generating a valid schedule with worker preferences."""
        conference = ConferenceBuilder().with_simple_config().build()

        # Create workers
        worker1 = WorkerBuilder().with_id("w1").build()
        worker2 = WorkerBuilder().with_id("w2").build()
        conference.add_worker(worker1)
        conference.add_worker(worker2)

        # Create shifts
        shift1 = ShiftBuilder().with_id("s1").with_duration_hours(9, 4).build()
        shift2 = ShiftBuilder().with_id("s2").with_duration_hours(14, 3).build()
        conference.add_shift(shift1)
        conference.add_shift(shift2)

        # Add preferences
        pref1 = WorkerPreference(worker=worker1, shift=shift1, preference_level=5)
        pref2 = WorkerPreference(worker=worker2, shift=shift2, preference_level=4)
        conference.add_preference(pref1)
        conference.add_preference(pref2)

        # Generate schedule
        result = generate_schedule(conference)

        # Should succeed
        assert isinstance(result, Schedule)
        expected_assignments = 2
        assert len(result.assignments) == expected_assignments
        assert result.conference is conference

        # Verify assignments match preferences
        assignment_pairs = [(a.worker.id, a.shift.id) for a in result.assignments]
        assert ("w1", "s1") in assignment_pairs
        assert ("w2", "s2") in assignment_pairs

    def test_schedule_with_overlapping_shifts_prevented(self) -> None:
        """Test that workers cannot be assigned overlapping shifts."""
        conference = ConferenceBuilder().with_simple_config().build()

        worker = WorkerBuilder().with_id("w1").build()
        conference.add_worker(worker)

        # Create overlapping shifts
        shift1 = ShiftBuilder().with_id("s1").with_duration_hours(9, 4).build()  # 9-13
        shift2 = (
            ShiftBuilder().with_id("s2").with_duration_hours(12, 3).build()
        )  # 12-15 (overlaps)
        conference.add_shift(shift1)
        conference.add_shift(shift2)

        # Worker prefers both shifts
        pref1 = WorkerPreference(worker=worker, shift=shift1, preference_level=5)
        pref2 = WorkerPreference(worker=worker, shift=shift2, preference_level=4)
        conference.add_preference(pref1)
        conference.add_preference(pref2)

        # Generate schedule
        result = generate_schedule(conference)

        # Should fail because we can't meet minimum requirements for both shifts
        assert isinstance(result, ScheduleError)
        assert len(result.unassigned_shifts) == 1
        assert "Could not meet minimum worker requirements" in result.error_message

    def test_schedule_respects_preference_priority(self) -> None:
        """Test that higher preference levels get priority."""
        conference = ConferenceBuilder().with_simple_config().build()

        # Create workers
        worker1 = WorkerBuilder().with_id("w1").build()
        worker2 = WorkerBuilder().with_id("w2").build()
        conference.add_worker(worker1)
        conference.add_worker(worker2)

        # Create shift with max 1 worker
        shift = ShiftBuilder().with_id("s1").with_max_workers(1).build()
        conference.add_shift(shift)

        # Worker1 has higher preference
        pref1 = WorkerPreference(worker=worker1, shift=shift, preference_level=5)
        pref2 = WorkerPreference(worker=worker2, shift=shift, preference_level=3)
        conference.add_preference(pref1)
        conference.add_preference(pref2)

        # Generate schedule
        result = generate_schedule(conference)

        # Should succeed with worker1 assigned (higher preference)
        assert isinstance(result, Schedule)
        assert len(result.assignments) == 1
        assert result.assignments[0].worker.id == "w1"

    def test_schedule_fails_when_minimum_workers_not_met(self) -> None:
        """Test schedule generation fails when minimum workers cannot be assigned."""
        conference = (
            ConferenceBuilder()
            .with_simple_config(
                max_shifts_per_worker=1,
            )
            .build()
        )

        # Only one worker
        worker = WorkerBuilder().with_id("w1").build()
        conference.add_worker(worker)

        # Two shifts both requiring minimum workers
        shift1 = ShiftBuilder().with_id("s1").with_duration_hours(9, 4).build()
        shift2 = ShiftBuilder().with_id("s2").with_duration_hours(14, 3).build()
        conference.add_shift(shift1)
        conference.add_shift(shift2)

        # Worker can only prefer one shift due to capacity limit
        pref = WorkerPreference(worker=worker, shift=shift1, preference_level=5)
        conference.add_preference(pref)

        # Generate schedule
        result = generate_schedule(conference)

        # Should fail - can't assign worker to both shifts
        assert isinstance(result, ScheduleError)
        assert len(result.unassigned_shifts) == 1
        assert "Could not meet minimum worker requirements" in result.error_message

    def test_schedule_respects_worker_capacity_limits(self) -> None:
        """Test that workers cannot exceed their shift capacity."""
        conference = (
            ConferenceBuilder()
            .with_simple_config(
                max_shifts_per_worker=1,
            )
            .build()
        )

        worker = WorkerBuilder().with_id("w1").build()
        conference.add_worker(worker)

        # Create two non-overlapping shifts
        shift1 = ShiftBuilder().with_id("s1").with_duration_hours(9, 2).build()
        shift2 = ShiftBuilder().with_id("s2").with_duration_hours(14, 2).build()
        conference.add_shift(shift1)
        conference.add_shift(shift2)

        # Worker prefers both
        pref1 = WorkerPreference(worker=worker, shift=shift1, preference_level=5)
        pref2 = WorkerPreference(worker=worker, shift=shift2, preference_level=4)
        conference.add_preference(pref1)
        conference.add_preference(pref2)

        # Generate schedule
        result = generate_schedule(conference)

        # Should fail because worker can only work one shift
        assert isinstance(result, ScheduleError)
        assert len(result.unassigned_shifts) == 1

    def test_schedule_with_multiple_workers_per_shift(self) -> None:
        """Test schedule generation with multiple workers assigned to same shift."""
        config = ConferenceConfig(
            start_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
            duration_days=1,
            min_workers_per_shift=2,
            max_workers_per_shift=3,
            min_shifts_per_worker=1,
            max_shifts_per_worker=5,
        )
        conference = ConferenceBuilder().with_config(config).build()

        # Create workers
        worker1 = WorkerBuilder().with_id("w1").build()
        worker2 = WorkerBuilder().with_id("w2").build()
        worker3 = WorkerBuilder().with_id("w3").build()
        conference.add_worker(worker1)
        conference.add_worker(worker2)
        conference.add_worker(worker3)

        # Create shift that can hold multiple workers
        shift = ShiftBuilder().with_id("s1").with_max_workers(3).build()
        conference.add_shift(shift)

        # All workers prefer this shift
        pref1 = WorkerPreference(worker=worker1, shift=shift, preference_level=5)
        pref2 = WorkerPreference(worker=worker2, shift=shift, preference_level=4)
        pref3 = WorkerPreference(worker=worker3, shift=shift, preference_level=3)
        conference.add_preference(pref1)
        conference.add_preference(pref2)
        conference.add_preference(pref3)

        # Generate schedule
        result = generate_schedule(conference)

        # Should succeed with at least 2 workers assigned
        assert isinstance(result, Schedule)
        min_expected_assignments = 2
        assert len(result.assignments) >= min_expected_assignments

        # All assignments should be to the same shift
        for assignment in result.assignments:
            assert assignment.shift.id == "s1"

    def test_schedule_does_not_modify_conference(self) -> None:
        """Test that schedule generation does not modify the original conference."""
        conference = ConferenceBuilder().with_simple_config().build()

        worker = WorkerBuilder().with_id("w1").build()
        shift = ShiftBuilder().with_id("s1").build()
        conference.add_worker(worker)
        conference.add_shift(shift)

        pref = WorkerPreference(worker=worker, shift=shift, preference_level=5)
        conference.add_preference(pref)

        # Verify conference starts with no assignments
        assert len(conference.assignments) == 0

        # Generate schedule
        result = generate_schedule(conference)

        # Conference should still have no assignments
        assert len(conference.assignments) == 0

        # But result should have assignments
        assert isinstance(result, Schedule)
        assert len(result.assignments) == 1


class TestOverlappingShiftValidation:
    """Test shift overlap detection."""

    def test_shifts_overlap_detection(self) -> None:
        """Test detection of overlapping shifts."""
        conference = ConferenceBuilder().with_simple_config().build()

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
                datetime(2024, 6, 1, 12, 0, tzinfo=UTC),
                datetime(2024, 6, 1, 16, 0, tzinfo=UTC),
            )
            .build()
        )

        assert conference.shifts_overlap(shift1, shift2)
        assert conference.shifts_overlap(shift2, shift1)

    def test_shifts_no_overlap_detection(self) -> None:
        """Test detection of non-overlapping shifts."""
        conference = ConferenceBuilder().with_simple_config().build()

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

        assert not conference.shifts_overlap(shift1, shift2)
        assert not conference.shifts_overlap(shift2, shift1)

    def test_assignment_prevented_for_overlapping_shifts(self) -> None:
        """Test that assignment is prevented for overlapping shifts."""
        conference = ConferenceBuilder().with_simple_config().build()
        worker = WorkerBuilder().build()
        conference.add_worker(worker)

        # Create overlapping shifts
        shift1 = ShiftBuilder().with_id("s1").with_duration_hours(9, 4).build()
        shift2 = ShiftBuilder().with_id("s2").with_duration_hours(12, 3).build()
        conference.add_shift(shift1)
        conference.add_shift(shift2)

        # Assign to first shift
        conference.assign_worker(worker, shift1)

        # Assignment to overlapping shift should fail
        with pytest.raises(ValueError, match="overlapping shift"):
            conference.assign_worker(worker, shift2)
