import requests
import csv
import hashlib
from tqdm import tqdm
import os
import tarfile
import logging
import re
import sys
import zipfile
import gzip


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


def download_from_url(url, path=None, root='.data', overwrite=False, hash_value=None,
                      hash_type="sha256"):
    """Download file, with logic (from tensor2tensor) for Google Drive. Returns
    the path to the downloaded file.

    Arguments:
        url: the url of the file from URL header. (None)
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
    def _check_hash(path):
        if hash_value:
            with open(path, "rb") as file_obj:
                if not validate_file(file_obj, hash_value, hash_type):
                    raise RuntimeError("The hash of {} does not match. Delete the file manually and retry.".format(path))

    def _process_response(r, root, filename):
        chunk_size = 16 * 1024
        total_size = int(r.headers.get('Content-length', 0))
        if filename is None:
            d = r.headers['content-disposition']
            filename = re.findall("filename=\"(.+)\"", d)
            if filename is None:
                raise RuntimeError("Filename could not be autodetected")
            filename = filename[0]
        path = os.path.join(root, filename)
        if os.path.exists(path):
            logging.info('File %s already exists.' % path)
            if not overwrite:
                _check_hash(path)
                return path
            logging.info('Overwriting file %s.' % path)
        logging.info('Downloading file {} to {}.'.format(filename, path))
        with open(path, "wb") as file:
            with tqdm(total=total_size, unit='B',
                      unit_scale=1, desc=path.split('/')[-1]) as t:
                for chunk in r.iter_content(chunk_size):
                    if chunk:
                        file.write(chunk)
                        t.update(len(chunk))
        logging.info('File {} downloaded.'.format(path))

        _check_hash(path)
        return path

    if path is None:
        _, filename = os.path.split(url)
    else:
        root, filename = os.path.split(path)

    if not os.path.exists(root):
        try:
            os.makedirs(root)
        except OSError:
            print("Can't create the download directory {}.".format(root))
            raise

    if filename is not None:
        path = os.path.join(root, filename)
    #skip requests.get if path exists and not overwrite.
    if os.path.exists(path):
        logging.info('File %s already exists.' % path)
        if not overwrite:
            _check_hash(path)
            return path

    if 'drive.google.com' not in url:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
        return _process_response(response, root, filename)
    else:
        # google drive links get filename from google drive
        filename = None

    logging.info('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    return _process_response(response, root, filename)


def unicode_csv_reader(unicode_csv_data, **kwargs):
    r"""Since the standard csv library does not handle unicode in Python 2, we need a wrapper.
    Borrowed and slightly modified from the Python docs:
    https://docs.python.org/2/library/csv.html#csv-examples

    Arguments:
        unicode_csv_data: unicode csv data (see example below)

    Examples:
        >>> from torchtext.utils import unicode_csv_reader
        >>> import io
        >>> with io.open(data_path, encoding="utf8") as f:
        >>>     reader = unicode_csv_reader(f)

    """

    # Fix field larger than field limit error
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)
    csv.field_size_limit(maxInt)

    for line in csv.reader(unicode_csv_data, **kwargs):
        yield line


def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')


def extract_archive(from_path, to_path=None, overwrite=False):
    """Extract archive.

    Arguments:
        from_path: the path of the archive.
        to_path: the root path of the extracted files (directory of from_path)
        overwrite: overwrite existing files (False)

    Returns:
        List of paths to extracted files even if not overwritten.

    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> from_path = './validation.tar.gz'
        >>> to_path = './'
        >>> torchtext.utils.download_from_url(url, from_path)
        >>> torchtext.utils.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']
        >>> torchtext.utils.download_from_url(url, from_path)
        >>> torchtext.utils.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']

    """

    if to_path is None:
        to_path = os.path.dirname(from_path)

    if from_path.endswith(('.tar.gz', '.tgz')):
        logging.info('Opening tar file {}.'.format(from_path))
        with tarfile.open(from_path, 'r') as tar:
            files = []
            for file_ in tar:
                file_path = os.path.join(to_path, file_.name)
                if file_.isfile():
                    files.append(file_path)
                    if os.path.exists(file_path):
                        logging.info('{} already extracted.'.format(file_path))
                        if not overwrite:
                            continue
                tar.extract(file_, to_path)
            return files

    elif from_path.endswith('.zip'):
        assert zipfile.is_zipfile(from_path), from_path
        logging.info('Opening zip file {}.'.format(from_path))
        with zipfile.ZipFile(from_path, 'r') as zfile:
            files = []
            for file_ in zfile.namelist():
                file_path = os.path.join(to_path, file_)
                files.append(file_path)
                if os.path.exists(file_path):
                    logging.info('{} already extracted.'.format(file_path))
                    if not overwrite:
                        continue
                zfile.extract(file_, to_path)
        files = [f for f in files if os.path.isfile(f)]
        return files

    elif from_path.endswith('.gz'):
        default_block_size = 65536
        filename = from_path[:-3]
        files = [filename]
        with gzip.open(from_path, 'rb') as gzfile, \
                open(filename, 'wb') as d_file:
            while True:
                block = gzfile.read(default_block_size)
                if not block:
                    break
                else:
                    d_file.write(block)
            d_file.write(block)
        return files

    else:
        raise NotImplementedError(
            "We currently only support tar.gz, .tgz, .gz and zip achives.")


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
