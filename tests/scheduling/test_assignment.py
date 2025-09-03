"""Tests for worker-to-shift assignment functionality."""

import pytest

from .builders import ConferenceBuilder, ShiftBuilder, WorkerBuilder


class TestAssignment:
    """Test worker assignment to shifts."""

    def test_successful_assignment(self) -> None:
        """Test successful assignment of worker to shift."""
        conference = ConferenceBuilder().with_simple_config().build()
        worker = WorkerBuilder().build()
        shift = ShiftBuilder().build()

        conference.add_worker(worker)
        conference.add_shift(shift)

        assignment = conference.assign_worker(worker, shift)

        assert assignment.worker == worker
        assert assignment.shift == shift
        assert assignment.created_at is not None

        # Verify assignment state through conference
        assert conference.is_worker_assigned_to_shift(worker, shift)
        assert assignment in conference.assignments

    def test_shift_at_capacity(self) -> None:
        """Test assignment fails when shift is at max capacity."""
        conference = ConferenceBuilder().with_simple_config().build()
        # Create a shift with max 1 worker
        shift = ShiftBuilder().with_max_workers(1).build()
        worker1 = WorkerBuilder().with_id("w1").build()
        worker2 = WorkerBuilder().with_id("w2").build()

        conference.add_shift(shift)
        conference.add_worker(worker1)
        conference.add_worker(worker2)

        # First assignment should succeed
        assignment1 = conference.assign_worker(worker1, shift)
        assert assignment1.worker == worker1
        assert assignment1.shift == shift

        # Second assignment should fail
        with pytest.raises(ValueError, match="is full"):
            conference.assign_worker(worker2, shift)

    def test_worker_at_capacity(self) -> None:
        """Test assignment fails when worker is at max shifts."""
        # Create conference with max 1 shift per worker
        conference = (
            ConferenceBuilder().with_simple_config(max_shifts_per_worker=1).build()
        )

        worker = WorkerBuilder().build()
        shift1 = ShiftBuilder().with_id("s1").build()
        shift2 = ShiftBuilder().with_id("s2").build()

        conference.add_worker(worker)
        conference.add_shift(shift1)
        conference.add_shift(shift2)

        # First assignment should succeed
        assignment1 = conference.assign_worker(worker, shift1)
        assert assignment1.worker == worker
        assert assignment1.shift == shift1

        # Second assignment should fail
        with pytest.raises(ValueError, match="is at capacity"):
            conference.assign_worker(worker, shift2)

    def test_duplicate_assignment(self) -> None:
        """Test assignment fails when worker already assigned to shift."""
        conference = ConferenceBuilder().with_simple_config().build()
        worker = WorkerBuilder().build()
        shift = ShiftBuilder().build()

        conference.add_worker(worker)
        conference.add_shift(shift)

        # First assignment should succeed
        assignment1 = conference.assign_worker(worker, shift)
        assert assignment1.worker == worker
        assert assignment1.shift == shift

        # Second assignment should fail
        with pytest.raises(ValueError, match="is already assigned"):
            conference.assign_worker(worker, shift)

    def test_unassign_worker_from_shift(self) -> None:
        """Test removing worker from shift."""
        conference = ConferenceBuilder().with_simple_config().build()
        worker = WorkerBuilder().build()
        shift = ShiftBuilder().build()

        conference.add_worker(worker)
        conference.add_shift(shift)

        # Assign first
        assignment = conference.assign_worker(worker, shift)
        assert conference.is_worker_assigned_to_shift(worker, shift)
        assert assignment in conference.assignments

        # Then unassign
        conference.unassign_worker_from_shift(worker, shift)

        # Verify state is cleaned up
        assert not conference.is_worker_assigned_to_shift(worker, shift)
        assert assignment not in conference.assignments

    def test_unassign_worker_not_assigned(self) -> None:
        """Test unassigning worker not assigned raises error."""
        conference = ConferenceBuilder().with_simple_config().build()
        shift = ShiftBuilder().build()
        worker1 = WorkerBuilder().with_id("w1").build()

        conference.add_shift(shift)
        conference.add_worker(worker1)

        with pytest.raises(ValueError, match="is not assigned"):
            conference.unassign_worker_from_shift(worker1, shift)


class TestConferenceEntityLookup:
    """Test conference entity lookup methods."""

    def test_get_worker(self) -> None:
        """Test worker lookup by ID."""
        conference = ConferenceBuilder().with_simple_config().build()
        worker = WorkerBuilder().with_id("test_worker").build()
        conference.add_worker(worker)

        found_worker = conference.get_worker("test_worker")
        assert found_worker is worker

        not_found = conference.get_worker("nonexistent")
        assert not_found is None

    def test_get_shift(self) -> None:
        """Test shift lookup by ID."""
        conference = ConferenceBuilder().with_simple_config().build()
        shift = ShiftBuilder().with_id("test_shift").build()
        conference.add_shift(shift)

        found_shift = conference.get_shift("test_shift")
        assert found_shift is shift

        not_found = conference.get_shift("nonexistent")
        assert not_found is None
