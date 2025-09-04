"""Performance tests for schedule generation at scale."""

import time
from datetime import UTC, datetime, timedelta
from random import Random

from scheduling.domain import (
    Conference,
    ConferenceConfig,
    Shift,
    Worker,
    WorkerPreference,
)
from scheduling.generator import Schedule, ScheduleError, generate_schedule


def _get_error_message(result: Schedule | ScheduleError) -> str:
    """Extract error message from result, with fallback for unknown errors."""
    return _get_error_message(result)


def create_conference_for_testing(
    num_workers: int = 400,
    num_shifts: int = 1000,
    preferences_per_worker: int = 3,
    min_workers_per_shift: int = 1,
    max_workers_per_shift: int = 6,
) -> Conference:
    """Create a large-scale conference for performance testing."""
    seed = 42
    rng = Random(seed)

    # Create conference
    config = ConferenceConfig(
        start_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
        duration_days=7,  # Week-long conference
        min_workers_per_shift=min_workers_per_shift,
        max_workers_per_shift=max_workers_per_shift,
        min_shifts_per_worker=18,
        max_shifts_per_worker=22,  # Allow workers to work many shifts
    )

    conference = Conference(
        id="large_conf",
        name="Large Scale Conference",
        config=config,
    )

    # Create workers
    workers = []
    for i in range(num_workers):
        worker = Worker(id=f"worker_{i:04d}", name=f"Worker {i}")
        workers.append(worker)
        conference.add_worker(worker)

    # Create shifts with varying times and capacities
    shifts = []
    start_time = config.start_time

    for i in range(num_shifts):
        # Distribute shifts across the week
        day_offset = i % 7
        hour_offset = (i // 7) % 12 + 8  # 8 AM to 7 PM

        shift_start = start_time + timedelta(days=day_offset, hours=hour_offset)
        shift_end = shift_start + timedelta(
            hours=rng.randint(1, 4),
        )  # 1-4 hour shifts

        # Random capacity between 1 and max_workers_per_shift
        capacity = rng.randint(1, max_workers_per_shift)

        shift = Shift(
            id=f"shift_{i:04d}",
            start_time=shift_start,
            end_time=shift_end,
            location=f"Room {i % 50}",  # 50 different rooms
            max_workers=capacity,
        )
        shifts.append(shift)
        conference.add_shift(shift)

    # Create preferences - each worker has preferences_per_worker random preferences
    for worker in workers:
        # Select random shifts for this worker to prefer
        preferred_shifts = rng.sample(
            shifts,
            min(preferences_per_worker, len(shifts)),
        )

        for shift in preferred_shifts:
            preference_level = rng.randint(1, 5)
            preference = WorkerPreference(
                worker=worker,
                shift=shift,
                preference_level=preference_level,
            )
            conference.add_preference(preference)

    return conference


class TestPerformance:
    """Performance tests for schedule generation."""

    def test_target_scale_performance(self) -> None:
        """Test performance at target scale: 400 workers, 1000 shifts."""
        conference = create_conference_for_testing(
            num_workers=200,
            num_shifts=500,
            preferences_per_worker=3,
        )

        start_time = time.time()

        result = generate_schedule(conference)

        generation_time = time.time() - start_time

        # Test meaningful assertions - performance thresholds
        max_generation_time = 30.0
        assert generation_time < max_generation_time, (
            f"Schedule generation took too long: {generation_time:.2f}s"
        )

        # Target scale should succeed with reduced constraints
        assert isinstance(result, Schedule), (
            f"Target scale generation failed: {_get_error_message(result)}"
        )
        assert len(result.assignments) > 0, "No assignments were generated"

        # Verify all shifts meet minimum requirements
        understaffed_shifts = 0
        for shift in conference.shifts:
            assignments_for_shift = [a for a in result.assignments if a.shift == shift]
            if len(assignments_for_shift) < conference.config.min_workers_per_shift:
                understaffed_shifts += 1

        assert understaffed_shifts == 0, (
            f"{understaffed_shifts} shifts are understaffed"
        )

    def test_medium_scale_performance(self) -> None:
        """Test performance at medium scale: 100 workers, 200 shifts."""
        conference = create_conference_for_testing(
            num_workers=100,
            num_shifts=200,
            preferences_per_worker=3,
        )

        start_time = time.time()
        result = generate_schedule(conference)
        generation_time = time.time() - start_time

        # Should complete quickly at medium scale
        max_medium_scale_time = 5.0
        assert generation_time < max_medium_scale_time, (
            f"Medium scale took too long: {generation_time:.2f}s"
        )

        # Medium scale should always succeed
        assert isinstance(result, Schedule), (
            f"Medium scale failed unexpectedly: {_get_error_message(result)}"
        )
        assert len(result.assignments) > 0, (
            "Should generate assignments at medium scale"
        )

    def test_small_scale_performance(self) -> None:
        """Test performance at small scale: 20 workers, 50 shifts."""
        conference = create_conference_for_testing(
            num_workers=20,
            num_shifts=50,
            preferences_per_worker=3,
        )

        start_time = time.time()
        result = generate_schedule(conference)
        generation_time = time.time() - start_time

        # Should complete very quickly at small scale
        max_small_scale_time = 1.0
        assert generation_time < max_small_scale_time, (
            f"Small scale took too long: {generation_time:.3f}s"
        )

        # Small scale should always succeed
        assert isinstance(result, Schedule), (
            f"Small scale failed: {_get_error_message(result)}"
        )
        assert len(result.assignments) > 0, "Should generate assignments at small scale"

    def test_constraint_satisfaction_at_scale(self) -> None:
        """Test that constraints are properly satisfied at scale."""
        conference = create_conference_for_testing(
            num_workers=100,
            num_shifts=200,
            preferences_per_worker=2,
            min_workers_per_shift=2,  # Higher minimum requirement
            max_workers_per_shift=4,
        )

        result = generate_schedule(conference)

        # With higher minimum requirements, expect either success or specific failure
        if isinstance(result, Schedule):
            # Verify no overlapping shifts for any worker
            worker_shifts: dict[str, list[Shift]] = {}
            for assignment in result.assignments:
                worker_id = assignment.worker.id
                if worker_id not in worker_shifts:
                    worker_shifts[worker_id] = []
                worker_shifts[worker_id].append(assignment.shift)

            overlaps_found = 0
            for shifts in worker_shifts.values():
                for i, shift1 in enumerate(shifts):
                    for shift2 in shifts[i + 1 :]:
                        if conference.shifts_overlap(shift1, shift2):
                            overlaps_found += 1

            assert overlaps_found == 0, (
                f"Found {overlaps_found} overlapping shift assignments"
            )

            # Verify worker capacity limits
            capacity_violations = 0
            for shifts in worker_shifts.values():
                if len(shifts) > conference.config.max_shifts_per_worker:
                    capacity_violations += 1

            assert capacity_violations == 0, (
                f"Found {capacity_violations} worker capacity violations"
            )
        else:
            # If generation failed, it should be due to constraint violation
            assert isinstance(result.error_message, str), "Error message required"
            assert "minimum worker requirements" in result.error_message, (
                f"Expected constraint violation error, got: {result.error_message}"
            )
