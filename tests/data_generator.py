"""Synthetic data generation for conference scheduling system."""

import random
import secrets
import uuid
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from scheduling.scheduling_domain import Conference, ConferenceConfig, Shift, Worker

from .constants import (
    HOURS_PER_DAY,
    PERFORMANCE_TEST_SEED,
    STRESS_TEST_DURATION_DAYS,
    STRESS_TEST_MAX_PREFERENCES_PER_WORKER,
    STRESS_TEST_MAX_SHIFTS_PER_WORKER,
    STRESS_TEST_MIN_SHIFTS_PER_WORKER,
    STRESS_TEST_SHIFTS,
    STRESS_TEST_SHIFTS_PER_DAY,
    STRESS_TEST_WORKERS,
)


@dataclass
class FeasibilityResult:
    """Results of checking if a conference is feasible for full allocation."""

    num_workers: int
    num_shifts: int
    total_shift_slots: int
    total_worker_capacity: int
    is_feasible: bool
    capacity_utilization: float
    shortage: int
    excess_capacity: int


class ConferenceDataGenerator:
    """Generate synthetic conference data for testing and validation."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize generator with optional random seed for reproducible results."""
        # RNG for synthetic test data generation
        self.seed = seed
        self.rng = self._create_random_generator(seed)
        self.locations = [
            "Main Auditorium",
            "Conference Room A",
            "Conference Room B",
            "Exhibition Hall",
            "Networking Lounge",
            "Registration Desk",
            "Catering Area",
            "Workshop Space",
            "Breakout Room 1",
            "Breakout Room 2",
        ]

    def _create_random_generator(
        self,
        seed: int | None,
    ) -> random.Random | secrets.SystemRandom:
        """Create appropriate random generator based on seed presence.

        For deterministic testing with seed, uses standard random.
        For production without seed, uses cryptographically secure random.
        """
        if seed is not None:
            # Create new Random instance with seed for testing reproducibility
            # This is NOT for cryptographic purposes - only for deterministic test data
            generator = random.Random()
            generator.seed(seed)
            return generator
        # Use cryptographically secure random for production
        return secrets.SystemRandom()

    def generate_shifts(
        self,
        num_shifts: int,
        config: ConferenceConfig,
    ) -> list[Shift]:
        """Generate shifts spread across conference days using configuration."""
        shifts = []

        # Calculate shift duration based on conference length and shifts per day
        shift_duration_hours = HOURS_PER_DAY / config.shifts_per_day
        shift_duration = timedelta(hours=shift_duration_hours)

        for i in range(num_shifts):
            # Distribute shifts across conference days
            day_offset = (i // config.shifts_per_day) % config.duration_days
            shift_in_day = i % config.shifts_per_day

            start_time = config.start_time + timedelta(
                days=day_offset,
                hours=shift_in_day * shift_duration_hours,
            )
            end_time = start_time + shift_duration

            shift = Shift(
                id=f"shift_{i:04d}",
                start_time=start_time,
                end_time=end_time,
                location=self.rng.choice(self.locations),
                max_workers=self.rng.randint(
                    config.min_workers_per_shift,
                    config.max_workers_per_shift,
                ),
            )
            shifts.append(shift)

        return shifts

    def generate_workers(
        self,
        num_workers: int,
        config: ConferenceConfig,
    ) -> list[Worker]:
        """Generate workers with varying shift capacities using configuration."""
        workers = []

        for i in range(num_workers):
            worker = Worker(
                id=f"worker_{i:04d}",
                name=f"Worker {i + 1}",
                max_shifts=self.rng.randint(
                    config.min_shifts_per_worker,
                    config.max_shifts_per_worker,
                ),
                max_preferences=config.max_preferences_per_worker,
            )
            workers.append(worker)

        return workers

    def assign_preferences(
        self,
        workers: list[Worker],
        shifts: list[Shift],
        config: ConferenceConfig,
    ) -> None:
        """Assign shift preferences to workers based on configuration patterns."""
        shift_ids = [shift.id for shift in shifts]

        for worker in workers:
            if config.preference_distribution == "uniform":
                # Each worker has uniform random preferences
                num_prefs = min(worker.max_preferences, len(shift_ids))
                preferences = self.rng.sample(shift_ids, num_prefs)

            elif config.preference_distribution == "clustered":
                # Workers tend to prefer shifts in certain locations/times
                cluster_size = min(len(shift_ids) // 4, worker.max_preferences * 2)
                start_idx = self.rng.randint(0, max(0, len(shift_ids) - cluster_size))
                cluster_shifts = shift_ids[start_idx : start_idx + cluster_size]
                num_prefs = min(worker.max_preferences, len(cluster_shifts))
                preferences = self.rng.sample(cluster_shifts, num_prefs)

            elif config.preference_distribution == "realistic":
                # Mix of popular shifts and random preferences
                popular_shifts = shift_ids[
                    : len(shift_ids) // 10
                ]  # Top 10% are "popular"

                num_popular = min(self.rng.randint(1, 3), len(popular_shifts))
                num_random = min(
                    worker.max_preferences - num_popular,
                    len(shift_ids) - num_popular,
                )

                preferences = self.rng.sample(popular_shifts, num_popular)
                remaining_shifts = [s for s in shift_ids if s not in preferences]
                preferences.extend(self.rng.sample(remaining_shifts, num_random))

                # Shuffle to randomize preference order
                self.rng.shuffle(preferences)

            else:
                msg = (
                    f"Unknown preference distribution: {config.preference_distribution}"
                )
                raise ValueError(msg)

            worker.set_preferences(preferences)

    def generate_conference(
        self,
        name: str,
        num_workers: int,
        num_shifts: int,
        config: ConferenceConfig | None = None,
    ) -> Conference:
        """Generate a complete conference with workers, shifts, and preferences."""
        if config is None:
            # Default configuration for synthetic data generation
            config = ConferenceConfig(
                start_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
                duration_days=3,
                shifts_per_day=16,
                min_workers_per_shift=1,
                max_workers_per_shift=3,
                min_shifts_per_worker=6,
                max_shifts_per_worker=15,
                max_preferences_per_worker=10,
                preference_distribution="realistic",
            )

        # Generate shifts
        shifts = self.generate_shifts(num_shifts=num_shifts, config=config)

        # Generate workers
        workers = self.generate_workers(num_workers=num_workers, config=config)

        # Assign preferences
        self.assign_preferences(workers, shifts, config)

        # Create conference
        return Conference(
            id=str(uuid.uuid4()),
            name=name,
            config=config,
            workers=workers,
            shifts=shifts,
        )

    def generate_stress_test_conference(self) -> Conference:
        """Generate a large conference for stress testing guaranteed feasible."""
        stress_config = ConferenceConfig(
            start_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
            duration_days=STRESS_TEST_DURATION_DAYS,
            shifts_per_day=STRESS_TEST_SHIFTS_PER_DAY,
            min_workers_per_shift=1,
            max_workers_per_shift=3,
            min_shifts_per_worker=STRESS_TEST_MIN_SHIFTS_PER_WORKER,
            max_shifts_per_worker=STRESS_TEST_MAX_SHIFTS_PER_WORKER,
            max_preferences_per_worker=STRESS_TEST_MAX_PREFERENCES_PER_WORKER,
            preference_distribution="realistic",
        )
        conference = self.generate_conference(
            name="Stress Test Conference",
            num_workers=STRESS_TEST_WORKERS,
            num_shifts=STRESS_TEST_SHIFTS,
            config=stress_config,
        )

        # Ensure the conference is feasible by adjusting worker capacities if needed
        self._ensure_conference_feasibility(conference)
        return conference

    def _ensure_conference_feasibility(self, conference: Conference) -> None:
        """Adjust worker capacities to ensure conference is feasible."""
        feasibility = self.validate_feasibility(conference)

        if not feasibility.is_feasible:
            # We need more worker capacity - distribute additional shifts among workers
            shortage = feasibility.shortage
            workers = list(conference.workers)

            # Sort workers by current capacity to distribute load fairly
            workers.sort(key=lambda w: w.max_shifts)

            # Add capacity to workers, starting with those who have the least
            for _, worker in enumerate(workers):
                if shortage <= 0:
                    break

                # Add 1-2 shifts per worker to spread the load
                additional_shifts = min(shortage, 2)
                worker.max_shifts += additional_shifts
                shortage -= additional_shifts

    def validate_feasibility(
        self,
        conference: Conference,
    ) -> FeasibilityResult:
        """Check if the generated conference data is feasible for full assignment."""
        total_shift_slots = sum(shift.max_workers for shift in conference.shifts)
        total_worker_capacity = sum(worker.max_shifts for worker in conference.workers)

        return FeasibilityResult(
            num_workers=len(conference.workers),
            num_shifts=len(conference.shifts),
            total_shift_slots=total_shift_slots,
            total_worker_capacity=total_worker_capacity,
            is_feasible=total_worker_capacity >= total_shift_slots,
            capacity_utilization=total_shift_slots / total_worker_capacity
            if total_worker_capacity > 0
            else 0,
            shortage=max(0, total_shift_slots - total_worker_capacity),
            excess_capacity=max(0, total_worker_capacity - total_shift_slots),
        )


def generate_test_scenarios() -> Generator[tuple[str, Conference], None, None]:
    """Generate various test scenarios for performance testing."""
    generator = ConferenceDataGenerator(seed=PERFORMANCE_TEST_SEED)

    yield (
        "Small Conference",
        generator.generate_conference(
            name="Small Conference",
            num_workers=20,
            num_shifts=50,
        ),
    )

    yield (
        "Medium Conference",
        generator.generate_conference(
            name="Medium Conference",
            num_workers=100,
            num_shifts=500,
        ),
    )

    yield (
        "Large Conference",
        generator.generate_conference(
            name="Large Conference",
            num_workers=300,
            num_shifts=2000,
        ),
    )

    yield (
        "Stress Test",
        generator.generate_stress_test_conference(),
    )

    # High worker demand scenario
    high_demand_config = ConferenceConfig(
        start_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
        duration_days=3,
        shifts_per_day=16,
        min_workers_per_shift=1,
        max_workers_per_shift=3,
        min_shifts_per_worker=8,
        max_shifts_per_worker=15,
        max_preferences_per_worker=10,
        preference_distribution="realistic",
    )
    yield (
        "High Worker Demand",
        generator.generate_conference(
            name="High Worker Demand",
            num_workers=50,
            num_shifts=1000,
            config=high_demand_config,
        ),
    )

    # Many small shifts scenario
    small_shifts_config = ConferenceConfig(
        start_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
        duration_days=3,
        shifts_per_day=16,
        min_workers_per_shift=1,
        max_workers_per_shift=2,
        min_shifts_per_worker=6,
        max_shifts_per_worker=15,
        max_preferences_per_worker=10,
        preference_distribution="realistic",
    )
    yield (
        "Many Small Shifts",
        generator.generate_conference(
            name="Many Small Shifts",
            num_workers=200,
            num_shifts=5000,
            config=small_shifts_config,
        ),
    )


if __name__ == "__main__":
    # Example usage and basic validation
    generator = ConferenceDataGenerator(seed=PERFORMANCE_TEST_SEED)

    # Generate a test conference
    test_conf = generator.generate_conference(
        name="Test Conference",
        num_workers=50,
        num_shifts=200,
    )

    # Validate feasibility
    feasibility = generator.validate_feasibility(test_conf)

    # Generate stress test
    stress_conf = generator.generate_stress_test_conference()
    stress_feasibility = generator.validate_feasibility(stress_conf)
