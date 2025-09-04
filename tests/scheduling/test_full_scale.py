"""Full scale performance test for 400 workers, 1000 shifts."""

import sys
import time
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scheduling.generator import Schedule, generate_schedule
from tests.scheduling.test_performance import create_conference_for_testing


def test_full_target_scale_performance() -> None:
    """Test performance at full target scale: 400 workers, 1000 shifts."""
    conference = create_conference_for_testing(
        num_workers=400,
        num_shifts=1000,
        preferences_per_worker=3,
    )

    start_time = time.time()

    result = generate_schedule(conference)

    generation_time = time.time() - start_time

    # Performance expectations - allow up to 60s for full scale
    max_full_scale_time = 60.0
    assert generation_time < max_full_scale_time, (
        f"Full scale generation took too long: {generation_time:.2f}s"
    )

    # Test meaningful assertions - expect either success or specific failure
    if isinstance(result, Schedule):
        assert len(result.assignments) > 0, "No assignments were generated"

        # Calculate basic statistics for validation
        shifts_filled = len({a.shift for a in result.assignments})
        workers_assigned = len({a.worker for a in result.assignments})

        # Basic sanity checks
        assert shifts_filled > 0, "No shifts were filled"
        assert workers_assigned > 0, "No workers were assigned"
    else:
        # If generation failed, ensure we know why
        assert isinstance(
            result.error_message,
            str,
        ), "Error message should be provided"
        assert len(result.unassigned_shifts) > 0, (
            "Should have unassigned shifts if generation failed"
        )


if __name__ == "__main__":
    test_full_target_scale_performance()
