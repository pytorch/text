import re

import requests

# This is to allow monkey-patching in fbcode
from torch.hub import load_state_dict_from_url  # noqa
from torchdata.datapipes.iter import HttpReader, GDriveReader  # noqa F401
from tqdm import tqdm


def _stream_response(r, chunk_size=16 * 1024):
    total_size = int(r.headers.get("Content-length", 0))
    with tqdm(total=total_size, unit="B", unit_scale=1) as t:
        for chunk in r.iter_content(chunk_size):
            if chunk:
                t.update(len(chunk))
                yield chunk


def _get_response_from_google_drive(url):
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v
    if confirm_token is None:
        if "Quota exceeded" in str(response.content):
            raise RuntimeError(
                "Google drive link {} is currently unavailable, because the quota was exceeded.".format(url)
            )
        else:
            raise RuntimeError("Internal error: confirm_token was not found in Google drive link.")

    url = url + "&confirm=" + confirm_token
    response = session.get(url, stream=True)

    if "content-disposition" not in response.headers:
        raise RuntimeError("Internal error: headers don't contain content-disposition.")

    filename = re.findall('filename="(.+)"', response.headers["content-disposition"])
    if filename is None:
        raise RuntimeError("Filename could not be autodetected")
    filename = filename[0]

    return response, filename


class DownloadManager:
    def get_local_path(self, url, destination):
        if "drive.google.com" not in url:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
        else:
            response, filename = _get_response_from_google_drive(url)

        with open(destination, "wb") as f:
            for chunk in _stream_response(response):
                f.write(chunk)


_DATASET_DOWNLOAD_MANAGER = DownloadManager()
_TEST_DOWNLOAD_MANAGER = DownloadManager()
