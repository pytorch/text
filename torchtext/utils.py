import six
import requests
import csv
from tqdm import tqdm
import os
import errno
import tarfile


def reporthook(t):
    """https://github.com/tqdm/tqdm"""
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


def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    def process_response(r):
        chunk_size = 16 * 1024
        total_size = int(r.headers.get('Content-length', 0))
        with open(path, "wb") as file:
            with tqdm(total=total_size, unit='B',
                      unit_scale=1, desc=path.split('/')[-1]) as t:
                for chunk in r.iter_content(chunk_size):
                    if chunk:
                        file.write(chunk)
                        t.update(len(chunk))

    if 'drive.google.com' not in url:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
        process_response(response)
        return

    print('downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    process_response(response)


def unicode_csv_reader(unicode_csv_data, **kwargs):
    """Since the standard csv library does not handle unicode in Python 2, we need a wrapper.
    Borrowed and slightly modified from the Python docs:
    https://docs.python.org/2/library/csv.html#csv-examples"""
    if six.PY2:
        # csv.py doesn't do Unicode; encode temporarily as UTF-8:
        csv_reader = csv.reader(utf_8_encoder(unicode_csv_data), **kwargs)
        for row in csv_reader:
            # decode UTF-8 back to Unicode, cell by cell:
            yield [cell.decode('utf-8') for cell in row]
    else:
        for line in csv.reader(unicode_csv_data, **kwargs):
            yield line


def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_and_extract_archive(url, download_root, extract_root=None, filename=None,
                                 md5=None, remove_finished=False):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under.
            If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    try:
        print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(
            url, fpath,
            reporthook=gen_bar_updater()
        )
    except OSError:
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Failed download. Trying https -> http instead.'
                  ' Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_targz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update
