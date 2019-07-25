import pytest
import os

slow = pytest.mark.skipif(
    os.getenv('RUN_SLOW', 'False') == 'False',
    reason="This test is slow."
)
