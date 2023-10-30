import pytest
from _pytest.tmpdir import _mk_tmp


def pytest_addoption(parser):
    parser.addoption(
        "--fast", action="store_true", default=False, help="run only fast tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--fast"):
        skip_slow = pytest.mark.skip(reason="--fast option was passed")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture(scope="module")
def module_tmp_path(request, tmp_path_factory):
    # Need to implement our own version because the built-in tmp_path only supports
    # function scopes, and can't be used in any module-scoped fixtures.
    # See https://github.com/pytest-dev/pytest/issues/363
    # Taken from https://docs.pytest.org/en/6.2.x/_modules/_pytest/tmpdir.html#tmp_path
    return _mk_tmp(request, tmp_path_factory)
