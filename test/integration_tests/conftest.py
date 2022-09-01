import shutil

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--use-tmp-hub-dir",
        action="store_true",
        help=(
            "When provided, tests will use temporary directory as Torch Hub directory. "
            "Downloaded models will be deleted after each test."
        ),
    )


@pytest.fixture(scope="class")
def temp_hub_dir(tmp_path_factory, pytestconfig):
    if not pytestconfig.getoption("--use-tmp-hub-dir"):
        yield
    else:
        tmp_dir = tmp_path_factory.mktemp("hub", numbered=True).resolve()
        org_dir = torch.hub.get_dir()
        torch.hub.set_dir(tmp_dir)
        yield
        torch.hub.set_dir(org_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
