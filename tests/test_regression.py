"""Pytest wrapper for regression tests."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "exampleApplication").exists(),
    reason="exampleApplication test data not available"
)
def test_regression():
    """Run the regression test suite."""
    test_dir = Path(__file__).parent
    script = test_dir / "check_regression.py"

    result = subprocess.run(
        [sys.executable, str(script), "--rtol=1e-6", "--atol=1e-7"],
        capture_output=True,
        text=True,
        cwd=test_dir.parent,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, f"Regression tests failed:\n{result.stdout}\n{result.stderr}"


def test_import():
    """Test that the package can be imported."""
    import diploshic
    from diploshic import fvTools, msTools, numba_stats

    # Basic sanity check - functions exist
    assert hasattr(fvTools, 'readFaArm')
    assert hasattr(msTools, 'msOutToHaplotypeArrayIn')
    assert hasattr(numba_stats, 'compute_r2_matrix')
