import pytest

slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="This test is slow. Set --runslow flag to run."
)
