"""Constants used throughout the scheduling system."""

# Time constants
HOURS_PER_DAY = 24

# Stress test configuration
STRESS_TEST_WORKERS = 400
STRESS_TEST_SHIFTS = 8000
STRESS_TEST_DURATION_DAYS = 14
STRESS_TEST_SHIFTS_PER_DAY = 16
STRESS_TEST_MIN_SHIFTS_PER_WORKER = 30
STRESS_TEST_MAX_SHIFTS_PER_WORKER = 50
STRESS_TEST_MAX_PREFERENCES_PER_WORKER = 20

# Test identifiers
TEST_WORKER_ID = "worker_001"
TEST_SHIFT_ID = "shift_001"
TEST_CONFERENCE_ID = "test-conf"

# Performance test thresholds
MIN_WORKERS_STRESS = 400
MIN_SHIFTS_STRESS = 8000
MIN_PREFERENCE_RATE = 0.1
MIN_CAPACITY_UTILIZATION = (
    0.2  # Reasonable utilization considering system reliability needs
)

# Test data generation - configurable seeds for deterministic testing
DEFAULT_TEST_SEED = 12345  # Primary seed for test reproducibility
PERFORMANCE_TEST_SEED = 67890  # Seed for performance benchmarks
STRESS_TEST_SEED = 111213  # Seed for stress testing scenarios

# Performance test scenario configurations
SMALL_CONFERENCE_WORKERS = 20
SMALL_CONFERENCE_SHIFTS = 50
MEDIUM_CONFERENCE_WORKERS = 100
MEDIUM_CONFERENCE_SHIFTS = 500
LARGE_CONFERENCE_WORKERS = 300
LARGE_CONFERENCE_SHIFTS = 2000
HIGH_DEMAND_WORKERS = 50
HIGH_DEMAND_SHIFTS = 1000
SMALL_SHIFTS_WORKERS = 200
SMALL_SHIFTS_TOTAL = 5000
