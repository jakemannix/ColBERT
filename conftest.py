import pytest

def pytest_addoption(parser):
    """Add custom command line options to pytest"""
    parser.addoption(
        "--benchmark-data-dir",
        action="store",
        default="",
        help="Path to the benchmark data directory for faceting tests"
    )

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "slow: mark test as slow to run")