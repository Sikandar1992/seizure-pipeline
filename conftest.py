"""
conftest.py -- pytest configuration for the seizure_pipeline test suite.

Registers the --runslow CLI option and auto-skips tests marked
@pytest.mark.slow unless it is passed. These hooks (pytest_addoption,
pytest_collection_modifyitems) only take effect when defined here, in
conftest.py -- defining them inside a regular test file is silently
ignored by pytest, which was verified directly during development (an
earlier version of this suite had them in test_seizure_pipeline.py and
the slow tests ran by default regardless).
"""

import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                      help="run slow, large-scale (Patient-scale) tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselected by default, use --runslow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
