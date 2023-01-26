"""Run smoke tests"""


import torchdata
import torchtext
import torchtext.version  # noqa: F401


def validateTorchdataVersion():
    from datetime import datetime, timedelta

    # validate torchdata version is within 3 days of current day
    torchdata_date_str = torchdata.__version__[-8:]
    torchdata_datetime = datetime(
        year=int(torchdata_date_str[:4]), month=int(torchdata_date_str[4:6]), day=int(torchdata_date_str[6:])
    )

    assert torchdata_datetime >= datetime.now() - timedelta(
        days=3
    ), f"torchdata package version '{torchdata.__version__}' cannot be older than 3 days"


validateTorchdataVersion()
print("torchtext version is ", torchtext.__version__)
