import os
from torchtext import utils
from .common.torchtext_test_case import TorchtextTestCase


def conditional_remove(f):
    if os.path.isfile(f):
        os.remove(f)


class TestUtils(TorchtextTestCase):

    def test_download_extract_tar(self):
        # create root directory for downloading data
        root = '.data'
        _ = os.makedirs(root, exist_ok=True)

        # ensure archive is not already downloaded, if it is then delete
        url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        target_archive_path = os.path.join(root, 'validation.tar.gz')
        conditional_remove(target_archive_path)

        # download archive and ensure is in correct location
        archive_path = utils.download_from_url(url)
        assert target_archive_path == archive_path

        # extract files and ensure they are correct
        files = utils.extract_archive(archive_path)
        assert files == [os.path.join(root, 'val.de'),
                         os.path.join(root, 'val.en')]

        # remove files and archive
        for f in files:
            conditional_remove(f)
        conditional_remove(archive_path)

    def test_download_extract_zip(self):
        # create root directory for downloading data
        root = '.data'
        _ = os.makedirs(root, exist_ok=True)

        # ensure archive is not already downloaded, if it is then delete
        url = 'https://bitbucket.org/sivareddyg/public/downloads/en-ud-v2.zip'
        target_archive_path = os.path.join(root, 'en-ud-v2.zip')
        conditional_remove(target_archive_path)

        # download archive and ensure is in correct location
        archive_path = utils.download_from_url(url)
        assert target_archive_path == archive_path

        # extract files and ensure they are correct
        files = utils.extract_archive(archive_path)
        assert files == ['en-ud-v2/',
                         'en-ud-v2/en-ud-tag.v2.dev.txt',
                         'en-ud-v2/en-ud-tag.v2.test.txt',
                         'en-ud-v2/en-ud-tag.v2.train.txt',
                         'en-ud-v2/LICENSE.txt',
                         'en-ud-v2/README.txt']

        # remove files and archive
        for f in files:
            conditional_remove(os.path.join(root, f))
        conditional_remove(archive_path)

    def test_download_extract_to_path(self):
        # create root directory for downloading data
        root = '.data'
        _ = os.makedirs(root, exist_ok=True)

        # create directory to extract archive to
        to_path = '.new_data'
        _ = os.makedirs(to_path, exist_ok=True)

        # ensure archive is not already downloaded, if it is then delete
        url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        target_archive_path = os.path.join(root, 'validation.tar.gz')
        conditional_remove(target_archive_path)

        # download archive and ensure is in correct location
        archive_path = utils.download_from_url(url)
        assert target_archive_path == archive_path

        # extract files and ensure they are in the to_path directory
        files = utils.extract_archive(archive_path, to_path)
        assert files == [os.path.join(to_path, 'val.de'),
                         os.path.join(to_path, 'val.en')]

        # remove files and archive
        for f in files:
            conditional_remove(f)
        conditional_remove(archive_path)

    def test_extract_non_tar_zip(self):
        # create root directory for downloading data
        root = '.data'
        _ = os.makedirs(root, exist_ok=True)

        # ensure file is not already downloaded, if it is then delete
        url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec'
        target_archive_path = os.path.join(root, 'wiki.simple.vec')
        conditional_remove(target_archive_path)

        # download file and ensure is in correct location
        archive_path = utils.download_from_url(url)
        assert target_archive_path == archive_path

        # assert that non-valid file (not an archive) raises error
        with self.assertRaises(NotImplementedError):
            utils.extract_archive(archive_path)

        # remove file
        conditional_remove(archive_path)
