"""Test builder utilities to reduce redundancy in test setup."""

from datetime import UTC, datetime, timedelta

from scheduling.scheduling_domain import Conference, ConferenceConfig, Shift, Worker

from .constants import (
    TEST_CONFERENCE_ID,
    TEST_SHIFT_ID,
    TEST_WORKER_ID,
)


class ConferenceBuilder:
    """Builder for creating test conferences with sensible defaults."""

    def __init__(self) -> None:
        """Initialize conference builder with defaults."""
        self.conference_id = TEST_CONFERENCE_ID
        self.name = "Test Conference"
        self.config = ConferenceConfig(
            start_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
            duration_days=1,
            shifts_per_day=4,
            min_workers_per_shift=1,
            max_workers_per_shift=3,
            min_shifts_per_worker=1,
            max_shifts_per_worker=5,
            max_preferences_per_worker=10,
        )

    def with_id(self, conference_id: str) -> "ConferenceBuilder":
        """Set the conference ID."""
        self.conference_id = conference_id
        return self

    def with_name(self, name: str) -> "ConferenceBuilder":
        """Set the conference name."""
        self.name = name
        return self

    def with_config(self, config: ConferenceConfig) -> "ConferenceBuilder":
        """Set the conference config."""
        self.config = config
        return self

    def with_simple_config(
        self,
        max_shifts_per_worker: int = 5,
        max_workers_per_shift: int = 3,
        max_preferences_per_worker: int = 10,
    ) -> "ConferenceBuilder":
        """Set simple config parameters."""
        self.config = ConferenceConfig(
            start_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
            duration_days=1,
            shifts_per_day=4,
            min_workers_per_shift=1,
            max_workers_per_shift=max_workers_per_shift,
            min_shifts_per_worker=1,
            max_shifts_per_worker=max_shifts_per_worker,
            max_preferences_per_worker=max_preferences_per_worker,
        )
        return self

    def build(self) -> Conference:
        """Build the conference."""
        return Conference(
            id=self.conference_id,
            name=self.name,
            config=self.config,
        )


class WorkerBuilder:
    """Builder for creating test workers with sensible defaults."""

    def __init__(self) -> None:
        """Initialize worker builder with defaults."""
        self.worker_id = TEST_WORKER_ID
        self.name = "Test Worker"

    def with_id(self, worker_id: str) -> "WorkerBuilder":
        """Set the worker ID."""
        self.worker_id = worker_id
        return self

    def with_name(self, name: str) -> "WorkerBuilder":
        """Set the worker name."""
        self.name = name
        return self

    def build(self) -> Worker:
        """Build the worker."""
        return Worker(
            id=self.worker_id,
            name=self.name,
        )


class ShiftBuilder:
    """Builder for creating test shifts with sensible defaults."""

    def __init__(self) -> None:
        """Initialize shift builder with defaults."""
        self.shift_id = TEST_SHIFT_ID
        self.start_time = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)
        self.end_time = datetime(2024, 6, 1, 17, 0, tzinfo=UTC)
        self.location = "Test Room"
        self.max_workers = 3

    def with_id(self, shift_id: str) -> "ShiftBuilder":
        """Set the shift ID."""
        self.shift_id = shift_id
        return self

    def with_time_range(self, start: datetime, end: datetime) -> "ShiftBuilder":
        """Set the time range."""
        self.start_time = start
        self.end_time = end
        return self

    def with_duration_hours(
        self,
        start_hour: int,
        duration_hours: int,
    ) -> "ShiftBuilder":
        """Set duration in hours from a start hour."""
        self.start_time = datetime(2024, 6, 1, start_hour, 0, tzinfo=UTC)
        self.end_time = self.start_time + timedelta(hours=duration_hours)
        return self

    def with_location(self, location: str) -> "ShiftBuilder":
        """Set the location."""
        self.location = location
        return self

    def with_max_workers(self, max_workers: int) -> "ShiftBuilder":
        """Set the maximum workers."""
        self.max_workers = max_workers
        return self

    def build(self) -> Shift:
        """Build the shift."""
        return Shift(
            id=self.shift_id,
            start_time=self.start_time,
            end_time=self.end_time,
            location=self.location,
            max_workers=self.max_workers,
        )


def create_simple_test_scenario() -> tuple[Conference, Worker, Shift]:
    """Create a simple test scenario with one conference, worker, and shift."""
    conference = (
        ConferenceBuilder()
        .with_simple_config(max_shifts_per_worker=1, max_workers_per_shift=2)
        .build()
    )

    worker = WorkerBuilder().build()
    shift = ShiftBuilder().with_max_workers(2).build()

    conference.add_worker(worker)
    conference.add_shift(shift)

    return conference, worker, shift


def create_overlapping_shifts_scenario() -> tuple[Conference, Worker, Shift, Shift]:
    """Create a scenario with overlapping shifts for time conflict testing."""
    conference = ConferenceBuilder().with_simple_config(max_shifts_per_worker=5).build()

    worker = WorkerBuilder().build()

    # First shift: 9 AM - 1 PM
    shift1 = ShiftBuilder().with_id("s1").with_duration_hours(9, 4).build()

    # Second shift: 12 PM - 4 PM (overlaps with first)
    shift2 = (
        ShiftBuilder()
        .with_id("s2")
        .with_duration_hours(12, 4)
        .with_location("Room B")
        .build()
    )

    conference.add_worker(worker)
    conference.add_shift(shift1)
    conference.add_shift(shift2)

    return conference, worker, shift1, shift2
