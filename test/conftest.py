def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="Run slow tests")
