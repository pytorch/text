import hashlib
import logging
import os

import torch
from torchtext import _CACHE_DIR

from ._download_hooks import _DATASET_DOWNLOAD_MANAGER


def reporthook(t):
    """
    https://github.com/tqdm/tqdm.
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def validate_file(file_obj, hash_value, hash_type="sha256"):
    """Validate a given file object with its hash.

    Args:
        file_obj: File object to read from.
        hash_value (str): Hash for url.
        hash_type (str, optional): Hash type, among "sha256" and "md5" (Default: ``"sha256"``).
    Returns:
        bool: return True if its a valid file, else False.

    """

    if hash_type == "sha256":
        hash_func = hashlib.sha256()
    elif hash_type == "md5":
        hash_func = hashlib.md5()
    else:
        raise ValueError

    while True:
        # Read by chunk to avoid filling memory
        chunk = file_obj.read(1024 ** 2)
        if not chunk:
            break
        hash_func.update(chunk)
    return hash_func.hexdigest() == hash_value


def _check_hash(path, hash_value, hash_type):
    logging.info("Validating hash {} matches hash of {}".format(hash_value, path))
    with open(path, "rb") as file_obj:
        if not validate_file(file_obj, hash_value, hash_type):
            raise RuntimeError(
                "The hash of {} does not match. Delete the file manually and retry.".format(os.path.abspath(path))
            )


def download_from_url(url, path=None, root=".data", overwrite=False, hash_value=None, hash_type="sha256"):
    """Download file, with logic (from tensor2tensor) for Google Drive. Returns
    the path to the downloaded file.

    Args:
        url: the url of the file from URL header. (None)
        path: path where file will be saved
        root: download folder used to store the file in (.data)
        overwrite: overwrite existing files (False)
        hash_value (str, optional): hash for url (Default: ``None``).
        hash_type (str, optional): hash type, among "sha256" and "md5" (Default: ``"sha256"``).

    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> torchtext.utils.download_from_url(url)
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> torchtext.utils.download_from_url(url)
        >>> '.data/validation.tar.gz'

    """
    # figure out filename and root
    if path is None:
        _, filename = os.path.split(url)
        root = os.path.abspath(root)
        path = os.path.join(root, filename)
    else:
        path = os.path.abspath(path)
        root, filename = os.path.split(os.path.abspath(path))

    # skip download if path exists and overwrite is not True
    if os.path.exists(path):
        logging.info("File %s already exists." % path)
        if not overwrite:
            if hash_value:
                _check_hash(path, hash_value, hash_type)
            return path

    # make root dir if does not exist
    if not os.path.exists(root):
        try:
            os.makedirs(root)
        except OSError:
            raise OSError("Can't create the download directory {}.".format(root))

    # download data and move to path
    _DATASET_DOWNLOAD_MANAGER.get_local_path(url, destination=path)

    logging.info("File {} downloaded.".format(path))

    # validate
    if hash_value:
        _check_hash(path, hash_value, hash_type)

    # all good
    return path


def _log_class_usage(klass):
    identifier = "torchtext"
    if klass and hasattr(klass, "__name__"):
        identifier += f".{klass.__name__}"
    torch._C._log_api_usage_once(identifier)


def get_asset_local_path(asset_path: str) -> str:
    """Get local path for assets. Download if path does not exost locally

    Args:
        asset_path: Local path to asset or remote URL
    Returns:
        bool: local path of the asset after downloading or reading from cache

    Examples:
        >>> url = 'http://<HOST>/file.txt'
        >>> torchtext.utils.get_asset_local_path(url)
        >>> '.data/file.txt'
        >>> torchtext.utils.get_asset_local_path('/home/user/file.txt')
        >>> '/home/user/file.txt'
    """
    if os.path.exists(asset_path):
        local_path = asset_path
    else:
        local_path = download_from_url(url=asset_path, root=_CACHE_DIR)
    return local_path
