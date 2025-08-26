"""Performance tests for the conference scheduling system."""

import random
import statistics
import time
from dataclasses import dataclass

import pytest

from scheduling.scheduling_domain import (
    AllocationStats,
    Conference,
    ConferenceConfig,
    ShiftScheduler,
)

from .constants import (
    DEFAULT_TEST_SEED,
    MIN_CAPACITY_UTILIZATION,
    MIN_PREFERENCE_RATE,
    MIN_SHIFTS_STRESS,
    MIN_WORKERS_STRESS,
)
from .data_generator import ConferenceDataGenerator

# Performance test constants
SMALL_TIME_LIMIT = 1.0  # seconds
MEDIUM_TIME_LIMIT = 2.0  # seconds
LARGE_TIME_LIMIT = 5.0  # seconds
STRESS_TIME_LIMIT = 30.0  # seconds


@dataclass
class ProfileResult:
    """Results of profiling a scheduling algorithm run."""

    scenario: str
    num_workers: int
    num_shifts: int
    total_shift_slots: int
    total_worker_capacity: int
    avg_execution_time: float
    min_execution_time: float
    max_execution_time: float
    std_dev: float
    success_rate: float
    allocation_stats: AllocationStats
    runs: int


class PerformanceProfiler:
    """Profiles the performance of the scheduling algorithm."""

    def __init__(self) -> None:
        """Initialize profiler with empty results list."""
        self.results: list[ProfileResult] = []

    def profile_allocation(
        self,
        conference: Conference,
        scenario_name: str,
        runs: int = 3,
    ) -> ProfileResult:
        """Profile the allocation algorithm performance."""
        times = []
        success_count = 0
        last_scheduler = None

        for run in range(runs):
            # Reset assignments for each run
            for worker in conference.workers:
                worker.assigned_shift_ids.clear()
            for shift in conference.shifts:
                shift.assigned_worker_ids.clear()

            seed = hash((42, run)) % (2**32)
            scheduler = ShiftScheduler(conference, rng=random.Random(seed))

            start_time = time.perf_counter()
            result = scheduler.allocate_shifts()
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            times.append(execution_time)

            if result.success:
                success_count += 1

            # Keep track of the last scheduler for stats
            last_scheduler = scheduler

        # Calculate statistics
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0

        # Get allocation stats from the last profiled run
        if last_scheduler:
            allocation_stats = last_scheduler.get_allocation_stats()
        else:
            # Fallback empty stats if no scheduler ran
            allocation_stats = AllocationStats(
                total_workers=len(conference.workers),
                workers_with_assignments=0,
                total_worker_slots=sum(w.max_shifts for w in conference.workers),
                filled_worker_slots=0,
                worker_utilization_rate=0.0,
                total_shift_slots=sum(s.max_workers for s in conference.shifts),
                filled_shift_slots=0,
                shift_utilization_rate=0.0,
                total_shifts=len(conference.shifts),
                fully_staffed_shifts=0,
                all_shifts_staffed=False,
                total_assignments=0,
                preference_satisfied_assignments=0,
                preference_satisfaction_rate=0.0,
                preference_fulfillment={},
            )

        profile_result = ProfileResult(
            scenario=scenario_name,
            num_workers=len(conference.workers),
            num_shifts=len(conference.shifts),
            total_shift_slots=sum(s.max_workers for s in conference.shifts),
            total_worker_capacity=sum(w.max_shifts for w in conference.workers),
            avg_execution_time=avg_time,
            min_execution_time=min_time,
            max_execution_time=max_time,
            std_dev=std_dev,
            success_rate=success_count / runs,
            allocation_stats=allocation_stats,
            runs=runs,
        )

        self.results.append(profile_result)
        return profile_result


@pytest.fixture
def data_generator() -> ConferenceDataGenerator:
    """Provide a seeded data generator for consistent test results."""
    return ConferenceDataGenerator(seed=DEFAULT_TEST_SEED)


@pytest.fixture
def profiler() -> PerformanceProfiler:
    """Provide a performance profiler instance."""
    return PerformanceProfiler()


class TestSchedulerPerformance:
    """Test the performance characteristics of the scheduling algorithm."""

    def test_small_conference_performance(
        self,
        data_generator: ConferenceDataGenerator,
        profiler: PerformanceProfiler,
    ) -> None:
        """Test performance with small dataset (20 workers, 50 shifts)."""
        conference = data_generator.generate_conference(
            name="Small Test",
            num_workers=20,
            num_shifts=50,
        )

        result = profiler.profile_allocation(conference, "Small Conference", runs=5)

        assert result.success_rate == 1.0, "All runs should succeed"
        assert result.avg_execution_time < SMALL_TIME_LIMIT, "Too slow"
        assert result.allocation_stats.all_shifts_staffed, (
            "All shifts should be staffed"
        )

    def test_medium_conference_performance(
        self,
        data_generator: ConferenceDataGenerator,
        profiler: PerformanceProfiler,
    ) -> None:
        """Test performance with medium dataset (100 workers, 500 shifts)."""
        conference = data_generator.generate_conference(
            name="Medium Test",
            num_workers=100,
            num_shifts=500,
        )

        result = profiler.profile_allocation(conference, "Medium Conference", runs=3)

        assert result.success_rate == 1.0, "All runs should succeed"
        assert result.avg_execution_time < MEDIUM_TIME_LIMIT, "Too slow"
        assert result.allocation_stats.all_shifts_staffed, (
            "All shifts should be staffed"
        )

    def test_large_conference_performance(
        self,
        data_generator: ConferenceDataGenerator,
        profiler: PerformanceProfiler,
        large_conference_config: ConferenceConfig,
    ) -> None:
        """Test performance with large dataset (300 workers, 2000 shifts)."""
        conference = data_generator.generate_conference(
            name="Large Test",
            num_workers=300,
            num_shifts=2000,
            config=large_conference_config,
        )

        result = profiler.profile_allocation(conference, "Large Conference", runs=3)

        assert result.success_rate == 1.0, "All runs should succeed"
        assert result.avg_execution_time < LARGE_TIME_LIMIT, "Too slow"
        assert result.allocation_stats.all_shifts_staffed, (
            "All shifts should be staffed"
        )

    @pytest.mark.slow
    def test_stress_test_performance(
        self,
        data_generator: ConferenceDataGenerator,
        profiler: PerformanceProfiler,
        stress_test_config: ConferenceConfig,
    ) -> None:
        """Test performance with stress dataset (400+ workers, 8000+ shifts)."""
        # Generate feasible stress test conference

        conference = data_generator.generate_conference(
            name="Stress Test",
            num_workers=500,  # Ensure enough capacity
            num_shifts=8000,
            config=stress_test_config,
        )

        # Validate feasibility first
        feasibility = data_generator.validate_feasibility(conference)
        assert feasibility.is_feasible, (
            f"Conference is not feasible - need {feasibility.shortage} more worker "
            f"slots. Total required: {feasibility.total_shift_slots}, "
            f"available: {feasibility.total_worker_capacity}"
        )

        result = profiler.profile_allocation(conference, "Stress Test", runs=3)

        assert result.success_rate == 1.0, "All runs should succeed"
        assert result.avg_execution_time < STRESS_TIME_LIMIT, "Too slow"
        assert result.allocation_stats.all_shifts_staffed, (
            "All shifts should be staffed"
        )

        # Additional stress test assertions
        assert result.num_workers >= MIN_WORKERS_STRESS, "Not enough workers"
        assert result.num_shifts >= MIN_SHIFTS_STRESS, "Not enough shifts"
        assert result.total_shift_slots >= MIN_SHIFTS_STRESS, "Not enough slots"

    def test_high_preference_satisfaction(
        self,
        data_generator: ConferenceDataGenerator,
        profiler: PerformanceProfiler,
        performance_config: ConferenceConfig,
    ) -> None:
        """Test that preference satisfaction remains reasonable under load."""
        conference = data_generator.generate_conference(
            name="Preference Test",
            num_workers=200,
            num_shifts=1000,
            config=performance_config,
        )

        result = profiler.profile_allocation(conference, "Preference Test", runs=3)

        assert result.success_rate == 1.0, "All runs should succeed"
        assert result.allocation_stats.all_shifts_staffed, (
            "All shifts should be staffed"
        )

        # Should achieve reasonable preference satisfaction
        pref_rate = result.allocation_stats.preference_satisfaction_rate
        assert pref_rate > MIN_PREFERENCE_RATE, (
            f"Should satisfy at least 10% of preferences, got {pref_rate:.2%}"
        )

    def test_scalability_linear(
        self,
        data_generator: ConferenceDataGenerator,
        profiler: PerformanceProfiler,
    ) -> None:
        """Test that performance scales reasonably with dataset size."""
        scenarios = [(50, 100), (100, 200), (200, 400), (400, 800)]

        results = []
        for workers, shifts in scenarios:
            conference = data_generator.generate_conference(
                name=f"Scalability Test {workers}w_{shifts}s",
                num_workers=workers,
                num_shifts=shifts,
            )
            result = profiler.profile_allocation(
                conference,
                f"{workers}w_{shifts}s",
                runs=2,
            )
            results.append((workers * shifts, result.avg_execution_time))

        # Check that time doesn't increase exponentially
        for i in range(1, len(results)):
            prev_ops, prev_time = results[i - 1]
            curr_ops, curr_time = results[i]

            ops_ratio = curr_ops / prev_ops
            time_ratio = curr_time / prev_time

            # Time should not increase more than quadratically with operations
            assert time_ratio <= ops_ratio**2, (
                f"Performance degradation too steep: "
                f"{ops_ratio:.1f}x ops -> {time_ratio:.1f}x time"
            )

    def test_fairness_consistency(
        self,
        data_generator: ConferenceDataGenerator,
        high_competition_config: ConferenceConfig,
    ) -> None:
        """Test that fair allocation is consistent across runs."""
        conference = data_generator.generate_conference(
            name="Fairness Test",
            num_workers=100,
            num_shifts=200,
            config=high_competition_config,
        )

        # Run multiple times with different seeds
        satisfaction_rates = []
        for seed in [100, 200, 300, 400, 500]:
            # Reset conference
            for worker in conference.workers:
                worker.assigned_shift_ids.clear()
            for shift in conference.shifts:
                shift.assigned_worker_ids.clear()

            scheduler = ShiftScheduler(conference, rng=random.Random(seed))
            result = scheduler.allocate_shifts()
            stats = scheduler.get_allocation_stats()

            assert result.success, "All runs should succeed"
            assert stats.all_shifts_staffed, "All shifts should be staffed"

            satisfaction_rates.append(stats.preference_satisfaction_rate)

        # Satisfaction rates should be consistent (low variance)
        std_dev = statistics.stdev(satisfaction_rates)
        mean_rate = statistics.mean(satisfaction_rates)

        # Standard deviation should be less than 10% of mean
        assert std_dev < mean_rate * 0.1, (
            f"Inconsistent fairness across runs: "
            f"mean={mean_rate:.3f}, std={std_dev:.3f}"
        )


class TestDataGenerator:
    """Test the synthetic data generation functionality."""

    def test_generate_feasible_conferences(
        self,
        data_generator: ConferenceDataGenerator,
    ) -> None:
        """Test that generated conferences are feasible for full allocation."""
        scenarios = [
            {"num_workers": 50, "num_shifts": 100},
            {"num_workers": 100, "num_shifts": 500},
            {"num_workers": 200, "num_shifts": 1000},
        ]

        for params in scenarios:
            conference = data_generator.generate_conference(
                name=(
                    f"Feasibility Test {params['num_workers']}w_{params['num_shifts']}s"
                ),
                num_workers=params["num_workers"],
                num_shifts=params["num_shifts"],
            )
            feasibility = data_generator.validate_feasibility(conference)

            assert feasibility.is_feasible, (
                f"Generated conference is not feasible - need "
                f"{feasibility.shortage} more worker slots for "
                f"{params['num_workers']} workers, {params['num_shifts']} shifts. "
                f"Capacity: {feasibility.capacity_utilization:.2f}"
            )
            assert feasibility.capacity_utilization > MIN_CAPACITY_UTILIZATION, (
                "Should have reasonable capacity utilization"
            )

    def test_preference_distributions(
        self,
        data_generator: ConferenceDataGenerator,
        default_config: ConferenceConfig,
    ) -> None:
        """Test different preference distribution patterns."""
        distributions = ["uniform", "clustered", "realistic"]

        for dist in distributions:
            # Create config with specific preference distribution
            dist_config = ConferenceConfig(
                start_time=default_config.start_time,
                duration_days=default_config.duration_days,
                shifts_per_day=default_config.shifts_per_day,
                min_workers_per_shift=default_config.min_workers_per_shift,
                max_workers_per_shift=default_config.max_workers_per_shift,
                min_shifts_per_worker=default_config.min_shifts_per_worker,
                max_shifts_per_worker=default_config.max_shifts_per_worker,
                max_preferences_per_worker=default_config.max_preferences_per_worker,
                preference_distribution=dist,
            )

            conference = data_generator.generate_conference(
                name=f"Distribution Test {dist}",
                num_workers=50,
                num_shifts=100,
                config=dist_config,
            )

            # All workers should have preferences
            for worker in conference.workers:
                assert len(worker.shift_preferences) > 0, (
                    f"Worker {worker.id} should have preferences "
                    f"with {dist} distribution"
                )

                # Preferences should be valid shift IDs
                shift_ids = {s.id for s in conference.shifts}
                for pref in worker.shift_preferences:
                    assert pref in shift_ids, f"Invalid preference {pref}"

    def test_stress_test_generation(
        self,
        data_generator: ConferenceDataGenerator,
    ) -> None:
        """Test that stress test conference generation works."""
        stress_conf = data_generator.generate_stress_test_conference()
        feasibility = data_generator.validate_feasibility(stress_conf)

        assert len(stress_conf.workers) >= MIN_WORKERS_STRESS, "Not enough workers"
        assert len(stress_conf.shifts) >= MIN_SHIFTS_STRESS, "Not enough shifts"
        assert feasibility.is_feasible, (
            f"Stress test is not feasible - need {feasibility.shortage} more "
            f"worker slots. {len(stress_conf.workers)} workers with "
            f"{feasibility.total_worker_capacity} total capacity cannot fill "
            f"{feasibility.total_shift_slots} shift slots"
        )


# Benchmark tests removed to avoid Any type usage
# Performance is already tested in TestSchedulerPerformance class
