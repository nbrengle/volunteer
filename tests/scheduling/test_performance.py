"""Performance tests for schedule generation at scale."""

import time
from datetime import UTC, datetime, timedelta
from random import Random

from scheduling.domain import (
    SchedulingConstraints,
    Shift,
    ShiftRequirement,
    Worker,
    WorkerPreference,
)
from scheduling.generator import Schedule, ScheduleError, generate_schedule


def _get_error_message(result: Schedule | ScheduleError) -> str:
    """Extract error message from result, with fallback for unknown errors."""
    return getattr(result, "error_message", "Unknown error")


def create_test_data(
    num_workers: int = 400,
    num_shifts: int = 1000,
    preferences_per_worker: int = 3,
    min_workers_per_shift: int = 1,
    max_workers_per_shift: int = 6,
) -> tuple[list[Worker], list[Shift], list[WorkerPreference], SchedulingConstraints]:
    """Create test data for performance testing."""
    seed = 42
    rng = Random(seed)

    # Create constraints
    constraints = SchedulingConstraints(
        min_workers_per_shift=min_workers_per_shift,
        max_workers_per_shift=max_workers_per_shift,
        min_shifts_per_worker=18,
        max_shifts_per_worker=22,  # Allow workers to work many shifts
    )

    # Create workers
    workers = []
    for i in range(num_workers):
        worker = Worker(id=f"worker_{i:04d}", name=f"Worker {i}")
        workers.append(worker)

    # Create shifts with varying times and capacities
    shifts = []
    start_time = datetime(2024, 6, 1, 8, 0, tzinfo=UTC)

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

        # Create a default requirement that mimics the old max_workers behavior
        requirement = ShiftRequirement(
            required_skills=set(),
            min_workers=1,
            max_workers=capacity,
            description="Performance test requirement",
        )
        shift = Shift(
            id=f"shift_{i:04d}",
            start_time=shift_start,
            end_time=shift_end,
            location=f"Room {i % 50}",  # 50 different rooms
            requirements=[requirement],
        )
        shifts.append(shift)

    # Create preferences - each worker has preferences_per_worker random preferences
    preferences = []
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
            preferences.append(preference)

    return workers, shifts, preferences, constraints


class TestPerformance:
    """Performance tests for schedule generation."""

    def test_target_scale_performance(self) -> None:
        """Test performance at target scale: 400 workers, 1000 shifts."""
        workers, shifts, preferences, constraints = create_test_data(
            num_workers=400,
            num_shifts=1000,
            preferences_per_worker=3,
        )

        start_time = time.time()

        result = generate_schedule(shifts, workers, preferences, constraints)

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
        for shift in shifts:
            assignments_for_shift = [a for a in result.assignments if a.shift == shift]
            if len(assignments_for_shift) < constraints.min_workers_per_shift:
                understaffed_shifts += 1

        assert understaffed_shifts == 0, (
            f"{understaffed_shifts} shifts are understaffed"
        )

    def test_medium_scale_performance(self) -> None:
        """Test performance at medium scale: 100 workers, 200 shifts."""
        workers, shifts, preferences, constraints = create_test_data(
            num_workers=100,
            num_shifts=200,
            preferences_per_worker=3,
        )

        start_time = time.time()
        result = generate_schedule(shifts, workers, preferences, constraints)
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
        workers, shifts, preferences, constraints = create_test_data(
            num_workers=20,
            num_shifts=50,
            preferences_per_worker=3,
        )

        start_time = time.time()
        result = generate_schedule(shifts, workers, preferences, constraints)
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

    def test_high_demand_scenario_performance(self) -> None:
        """Test performance with high shift-to-worker ratio scenario."""
        workers, shifts, preferences, constraints = create_test_data(
            num_workers=80,  # Sufficient workers to meet demands
            num_shifts=120,  # Many shifts to test scalability
            preferences_per_worker=4,
            min_workers_per_shift=1,  # Keep minimum achievable
            max_workers_per_shift=6,  # Use default max that works
        )

        start_time = time.time()
        result = generate_schedule(shifts, workers, preferences, constraints)
        generation_time = time.time() - start_time

        # Should complete within reasonable time for high-demand scenario
        max_time = 3.0
        assert generation_time < max_time, (
            f"High demand scenario took too long: {generation_time:.2f}s"
        )

        # High demand scenario should succeed with proper constraints
        assert isinstance(result, Schedule), (
            f"High demand scenario failed: {_get_error_message(result)}"
        )
        assert len(result.assignments) > 0, "Should have assignments"

        # Verify each shift meets minimum requirement
        for shift in shifts:
            shift_assignments = [a for a in result.assignments if a.shift == shift]
            assert len(shift_assignments) >= constraints.min_workers_per_shift, (
                f"Shift {shift.id} has {len(shift_assignments)}, "
                f"need {constraints.min_workers_per_shift}"
            )

    def test_full_target_scale_performance(self) -> None:
        """Test performance at full target scale: 400 workers, 1000 shifts."""
        workers, shifts, preferences, constraints = create_test_data(
            num_workers=400,
            num_shifts=1000,
            preferences_per_worker=3,
        )

        start_time = time.time()
        result = generate_schedule(shifts, workers, preferences, constraints)
        generation_time = time.time() - start_time

        # Performance expectations - allow up to 60s for full scale
        max_full_scale_time = 60.0
        assert generation_time < max_full_scale_time, (
            f"Full scale generation took too long: {generation_time:.2f}s"
        )

        # Target scale should always succeed - this is our performance target
        assert isinstance(result, Schedule), (
            f"Algorithm failed at target scale: {_get_error_message(result)}"
        )
        assert len(result.assignments) > 0, "No assignments were generated"

        # Calculate basic statistics for validation
        shifts_filled = len({a.shift for a in result.assignments})
        workers_assigned = len({a.worker for a in result.assignments})

        # Basic sanity checks
        assert shifts_filled > 0, "No shifts were filled"
        assert workers_assigned > 0, "No workers were assigned"
