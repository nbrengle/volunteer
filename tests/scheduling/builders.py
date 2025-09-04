"""Test builder utilities to reduce redundancy in test setup."""

from datetime import UTC, datetime, timedelta

from scheduling.domain import (
    Shift,
    Worker,
)

# Test identifiers
TEST_WORKER_ID = "worker_001"
TEST_SHIFT_ID = "shift_001"


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
            skills=set(),
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
            required_skills=set(),
            min_workers=1,
            max_workers=self.max_workers,
            role_description="",
        )
