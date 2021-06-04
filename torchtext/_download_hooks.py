from typing import List, Optional, Union, IO, Dict, Any
import requests
import os
import logging
import uuid
import re
import shutil
from tqdm import tqdm
from iopath.common.file_io import (
    PathHandler,
    PathManager,
    get_cache_dir,
    file_lock,
    HTTPURLHandler,
)


def _stream_response(r, chunk_size=16 * 1024):
    total_size = int(r.headers.get('Content-length', 0))
    with tqdm(total=total_size, unit='B', unit_scale=1) as t:
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
                "Google drive link {} is currently unavailable, because the quota was exceeded.".format(
                    url
                ))
        else:
            raise RuntimeError("Internal error: confirm_token was not found in Google drive link.")

    url = url + "&confirm=" + confirm_token
    response = session.get(url, stream=True)

    if 'content-disposition' not in response.headers:
        raise RuntimeError("Internal error: headers don't contain content-disposition.")

    filename = re.findall("filename=\"(.+)\"", response.headers['content-disposition'])
    if filename is None:
        raise RuntimeError("Filename could not be autodetected")
    filename = filename[0]

    return response, filename


class GoogleDrivePathHandler(PathHandler):
    """
    Download URLs and cache them to disk.
    """

    MAX_FILENAME_LEN = 250

    def __init__(self) -> None:
        self.cache_map: Dict[str, str] = {}

    def _get_supported_prefixes(self) -> List[str]:
        return ["https://drive.google.com"]

    def _get_local_path(
        self,
        path: str,
        force: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        This implementation downloads the remote resource from google drive and caches it locally.
        The resource will only be downloaded if not previously requested.
        """
        self._check_kwargs(kwargs)
        if (
            force
            or path not in self.cache_map
            or not os.path.exists(self.cache_map[path])
        ):
            logger = logging.getLogger(__name__)
            dirname = get_cache_dir(cache_dir)

            response, filename = _get_response_from_google_drive(path)
            if len(filename) > self.MAX_FILENAME_LEN:
                filename = filename[:100] + "_" + uuid.uuid4().hex

            cached = os.path.join(dirname, filename)
            with file_lock(cached):
                if not os.path.isfile(cached):
                    logger.info("Downloading {} ...".format(path))
                    with open(cached, 'wb') as f:
                        for data in _stream_response(response):
                            f.write(data)
            logger.info("URL {} cached in {}".format(path, cached))
            self.cache_map[path] = cached
        return self.cache_map[path]

    def _open(
        self, path: str, mode: str = "r", buffering: int = -1, **kwargs: Any
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a google drive path. The resource is first downloaded and cached
        locally.
        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): Not used for this PathHandler.
        Returns:
            file: a file-like object.
        """
        self._check_kwargs(kwargs)
        assert mode in ("r", "rb"), "{} does not support open with {} mode".format(
            self.__class__.__name__, mode
        )
        assert (
            buffering == -1
        ), f"{self.__class__.__name__} does not support the `buffering` argument"
        local_path = self._get_local_path(path, force=False)
        return open(local_path, mode)


class CombinedInternalPathhandler(PathHandler):
    def __init__(self):
        path_manager = PathManager()
        path_manager.register_handler(HTTPURLHandler())
        path_manager.register_handler(GoogleDrivePathHandler())
        self.path_manager = path_manager

    def _get_supported_prefixes(self) -> List[str]:
        return ["https://", "http://"]

    def _get_local_path(
        self,
        path: str,
        force: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> str:

        destination = kwargs["destination"]

        local_path = self.path_manager.get_local_path(path, force)

        shutil.move(local_path, destination)

        return destination

    def _open(
        self, path: str, mode: str = "r", buffering: int = -1, **kwargs: Any
    ) -> Union[IO[str], IO[bytes]]:
        self._check_kwargs(kwargs)
        assert mode in ("r", "rb"), "{} does not support open with {} mode".format(
            self.__class__.__name__, mode
        )
        assert (
            buffering == -1
        ), f"{self.__class__.__name__} does not support the `buffering` argument"
        local_path = self._get_local_path(path, force=False)
        return open(local_path, mode)


_PATH_MANAGER = PathManager()
_PATH_MANAGER.register_handler(CombinedInternalPathhandler())
