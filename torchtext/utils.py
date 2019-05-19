import six
import requests
import csv
import shutil
import os
from tqdm import tqdm

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


def download_from_url(url, destination):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    def process_response(r, first_byte):
        
        # Since requests doesn't support local file reading
        # we check if the protocol is file://
        if url.startswith('file://'):
            url_no_protocol = url.replace('file://', '', count=1)
            if os.path.exists(url_no_protocol):
                print('File already exists, no need to download')
                return
            else:
                raise Exception('File not found at %s' % url_no_protocol)
        
        # Don't download if the file exists
        if os.path.exists(destination):
            print('File already exists, no need to download')
            return
        
        chunk_size = 1024 ** 2 # 1MB
        file_mode = 'ab' if first_byte else 'wb'
        file_size = len(r.content)
        if file_size < 0:
            raise Exception('Error getting file from server: %s' % url)

        print('Download from ' + url)
        print('Starting download at %.1fMB' % (first_byte / chunk_size))
        print('File size is %.1fMB' % (file_size / chunk_size))

        with tqdm(initial=first_byte, total=file_size, unit='bit', unit_scale=True) as pbar:
            with open(tmp_file_path, file_mode) as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Rename the temp file to the correct name if fully downloaded
        if file_size == os.path.getsize(tmp_file_path):
            shutil.move(tmp_file_path, destination)
        else:
            raise Exception('Error getting file from server: %s' % url)

    tmp_file_path = destination + '.part'
    first_byte = os.path.getsize(tmp_file_path) if os.path.exists(tmp_file_path) else 0
    
    # Set headers: this will tell the server to start download from the specified byte
    headers = {"Range": "bytes=%s-" % first_byte}

    if 'drive.google.com' not in url:
        headers.update({'User-Agent': 'Mozilla/5.0'})
        response = requests.get(url, headers=headers, stream=True)
        process_response(response, first_byte)
        return

    print('downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, headers=headers, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, headers={"Range": "bytes=%s-" % first_byte}, stream=True)

    process_response(response, first_byte)


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
