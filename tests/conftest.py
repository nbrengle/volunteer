"""Pytest fixtures for scheduling tests."""

from datetime import UTC, datetime

import pytest

from scheduling.scheduling_domain import ConferenceConfig


@pytest.fixture
def default_config() -> ConferenceConfig:
    """Create a default feasible configuration for testing."""
    return ConferenceConfig(
        start_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
        duration_days=3,
        shifts_per_day=16,
        min_workers_per_shift=1,
        max_workers_per_shift=3,
        min_shifts_per_worker=6,
        max_shifts_per_worker=15,
        max_preferences_per_worker=10,
    )


@pytest.fixture
def stress_test_config() -> ConferenceConfig:
    """Create a configuration for stress testing with high capacity."""
    return ConferenceConfig(
        start_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
        duration_days=14,
        shifts_per_day=16,
        min_workers_per_shift=1,
        max_workers_per_shift=3,
        min_shifts_per_worker=30,
        max_shifts_per_worker=50,
        max_preferences_per_worker=20,
    )


@pytest.fixture
def small_event_config() -> ConferenceConfig:
    """Create a configuration for small events with fewer shifts per worker."""
    return ConferenceConfig(
        start_time=datetime(2024, 6, 1, 9, 0, tzinfo=UTC),
        duration_days=1,
        shifts_per_day=8,
        min_workers_per_shift=2,
        max_workers_per_shift=4,
        min_shifts_per_worker=2,
        max_shifts_per_worker=4,
        max_preferences_per_worker=6,
    )


@pytest.fixture
def large_conference_config() -> ConferenceConfig:
    """Create a configuration for large conferences with sufficient capacity."""
    return ConferenceConfig(
        start_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
        duration_days=5,
        shifts_per_day=12,
        min_workers_per_shift=1,
        max_workers_per_shift=4,
        min_shifts_per_worker=15,
        max_shifts_per_worker=25,
        max_preferences_per_worker=15,
    )


@pytest.fixture
def high_competition_config() -> ConferenceConfig:
    """Create a configuration with clustered preferences for testing fairness."""
    return ConferenceConfig(
        start_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
        duration_days=2,
        shifts_per_day=12,
        min_workers_per_shift=1,
        max_workers_per_shift=3,
        min_shifts_per_worker=8,
        max_shifts_per_worker=12,
        max_preferences_per_worker=8,
    )


@pytest.fixture
def feasible_config() -> ConferenceConfig:
    """Create a configuration guaranteed to be feasible for typical test sizes."""
    return ConferenceConfig(
        start_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
        duration_days=3,
        shifts_per_day=16,
        min_workers_per_shift=1,
        max_workers_per_shift=3,
        min_shifts_per_worker=8,
        max_shifts_per_worker=18,
        max_preferences_per_worker=10,
    )


@pytest.fixture
def performance_config() -> ConferenceConfig:
    """Create a configuration suitable for performance testing."""
    return ConferenceConfig(
        start_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
        duration_days=3,
        shifts_per_day=16,
        min_workers_per_shift=1,
        max_workers_per_shift=3,
        min_shifts_per_worker=12,
        max_shifts_per_worker=25,
        max_preferences_per_worker=10,
    )
