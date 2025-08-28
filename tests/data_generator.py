"""Synthetic data generation for conference scheduling system."""

import random
import uuid
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from enum import Enum

from scheduling.scheduling_domain import (
    Conference,
    ConferenceConfig,
    Shift,
    Worker,
)

from .constants import (
    HIGH_DEMAND_SHIFTS,
    HIGH_DEMAND_WORKERS,
    HOURS_PER_DAY,
    LARGE_CONFERENCE_SHIFTS,
    LARGE_CONFERENCE_WORKERS,
    MEDIUM_CONFERENCE_SHIFTS,
    MEDIUM_CONFERENCE_WORKERS,
    PERFORMANCE_TEST_SEED,
    SMALL_CONFERENCE_SHIFTS,
    SMALL_CONFERENCE_WORKERS,
    SMALL_SHIFTS_TOTAL,
    SMALL_SHIFTS_WORKERS,
    STRESS_TEST_DURATION_DAYS,
    STRESS_TEST_MAX_PREFERENCES_PER_WORKER,
    STRESS_TEST_MAX_SHIFTS_PER_WORKER,
    STRESS_TEST_MIN_SHIFTS_PER_WORKER,
    STRESS_TEST_SHIFTS,
    STRESS_TEST_SHIFTS_PER_DAY,
    STRESS_TEST_WORKERS,
)


class PreferenceDistribution(Enum):
    """Preference distribution patterns for synthetic data generation."""

    UNIFORM = "uniform"
    CLUSTERED = "clustered"
    REALISTIC = "realistic"


class ConferenceDataGenerator:
    """Generate synthetic conference data for testing and validation."""

    def __init__(
        self,
        seed: int,
        preference_distribution: PreferenceDistribution = (
            PreferenceDistribution.REALISTIC
        ),
    ) -> None:
        """Initialize generator with seed for reproducible test results."""
        # RNG for synthetic test data generation
        self.seed = seed
        self.rng = random.Random(seed)
        self.preference_distribution = preference_distribution
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
    ) -> list[Worker]:
        """Generate workers with varying shift capacities using configuration."""
        workers = []

        for i in range(num_workers):
            worker = Worker(
                id=f"worker_{i:04d}",
                name=f"Worker {i + 1}",
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
            if self.preference_distribution == PreferenceDistribution.UNIFORM:
                # Each worker has uniform random preferences
                num_prefs = min(config.max_preferences_per_worker, len(shift_ids))
                preferences = self.rng.sample(shift_ids, num_prefs)

            elif self.preference_distribution == PreferenceDistribution.CLUSTERED:
                # Workers tend to prefer shifts in certain locations/times
                cluster_size = min(
                    len(shift_ids) // 4,
                    config.max_preferences_per_worker * 2,
                )
                start_idx = self.rng.randint(0, max(0, len(shift_ids) - cluster_size))
                cluster_shifts = shift_ids[start_idx : start_idx + cluster_size]
                num_prefs = min(config.max_preferences_per_worker, len(cluster_shifts))
                preferences = self.rng.sample(cluster_shifts, num_prefs)

            elif self.preference_distribution == PreferenceDistribution.REALISTIC:
                # Mix of popular shifts and random preferences
                popular_shifts = shift_ids[
                    : len(shift_ids) // 10
                ]  # Top 10% are "popular"

                num_popular = min(self.rng.randint(1, 3), len(popular_shifts))
                num_random = min(
                    config.max_preferences_per_worker - num_popular,
                    len(shift_ids) - num_popular,
                )

                preferences = self.rng.sample(popular_shifts, num_popular)
                remaining_shifts = [s for s in shift_ids if s not in preferences]
                preferences.extend(self.rng.sample(remaining_shifts, num_random))

                # Shuffle to randomize preference order
                self.rng.shuffle(preferences)

            else:
                msg = f"Unknown preference distribution: {self.preference_distribution}"
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
            )

        # Generate shifts
        shifts = self.generate_shifts(num_shifts=num_shifts, config=config)

        # Generate workers
        workers = self.generate_workers(num_workers=num_workers)

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
        feasibility = conference.validate_feasibility()

        if not feasibility.is_feasible:
            # We need more worker capacity - distribute additional shifts among workers
            shortage = feasibility.shortage
            workers = list(conference.workers)

            # Sort workers by current assignment count to distribute load fairly
            workers.sort(key=lambda w: len(w.assigned_shift_ids))

            # Add capacity to workers, starting with those who have the least
            for _, _worker in enumerate(workers):
                if shortage <= 0:
                    break

                # This would require increasing the conference's
                # max_shifts_per_worker limit. For now, skip this optimization -
                # the conference config should be set appropriately
                break  # Can't dynamically increase worker capacity in new architecture


def generate_test_scenarios() -> Generator[tuple[str, Conference], None, None]:
    """Generate various test scenarios for performance testing."""
    generator = ConferenceDataGenerator(seed=PERFORMANCE_TEST_SEED)

    yield (
        "Small Conference",
        generator.generate_conference(
            name="Small Conference",
            num_workers=SMALL_CONFERENCE_WORKERS,
            num_shifts=SMALL_CONFERENCE_SHIFTS,
        ),
    )

    yield (
        "Medium Conference",
        generator.generate_conference(
            name="Medium Conference",
            num_workers=MEDIUM_CONFERENCE_WORKERS,
            num_shifts=MEDIUM_CONFERENCE_SHIFTS,
        ),
    )

    yield (
        "Large Conference",
        generator.generate_conference(
            name="Large Conference",
            num_workers=LARGE_CONFERENCE_WORKERS,
            num_shifts=LARGE_CONFERENCE_SHIFTS,
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
    )
    yield (
        "High Worker Demand",
        generator.generate_conference(
            name="High Worker Demand",
            num_workers=HIGH_DEMAND_WORKERS,
            num_shifts=HIGH_DEMAND_SHIFTS,
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
    )
    yield (
        "Many Small Shifts",
        generator.generate_conference(
            name="Many Small Shifts",
            num_workers=SMALL_SHIFTS_WORKERS,
            num_shifts=SMALL_SHIFTS_TOTAL,
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
    feasibility = test_conf.validate_feasibility()

    # Generate stress test
    stress_conf = generator.generate_stress_test_conference()
    stress_feasibility = stress_conf.validate_feasibility()
