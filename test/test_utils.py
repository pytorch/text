#!/usr/bin/env python3
# Note that all the tests in this module require dataset (either network access or cached)
import os
from torchtext import utils
from .common.torchtext_test_case import TorchtextTestCase
from test.common.assets import get_asset_path
import shutil


def conditional_remove(f):
    if os.path.isfile(f):
        os.remove(f)


class TestUtils(TorchtextTestCase):

    def test_download_extract_tar(self):
        # create root directory for downloading data
        root = '.data'
        if not os.path.exists(root):
            os.makedirs(root)

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

        # extract files with overwrite option True
        files = utils.extract_archive(archive_path, overwrite=True)
        assert files == [os.path.join(root, 'val.de'),
                         os.path.join(root, 'val.en')]

        # remove files and archive
        for f in files:
            conditional_remove(f)
        conditional_remove(archive_path)

    def test_download_extract_gz(self):
        # create root directory for downloading data
        root = '.data'
        if not os.path.exists(root):
            os.makedirs(root)

        # ensure archive is not already downloaded, if it is then delete
        url = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.5.en.gz'
        target_archive_path = os.path.join(root, 'val.5.en.gz')
        conditional_remove(target_archive_path)

        # download archive and ensure is in correct location
        archive_path = utils.download_from_url(url)
        assert target_archive_path == archive_path

        # extract files and ensure they are correct
        files = utils.extract_archive(archive_path)
        assert files == [os.path.join(root, 'val.5.en')]

        # extract files with overwrite option True
        files = utils.extract_archive(archive_path, overwrite=True)
        assert files == [os.path.join(root, 'val.5.en')]

        # remove files and archive
        for f in files:
            conditional_remove(f)
        conditional_remove(archive_path)

    def test_download_extract_zip(self):
        # create root directory for downloading data
        root = '.data'
        if not os.path.exists(root):
            os.makedirs(root)

        # ensure archive is not already downloaded, if it is then delete
        url = 'https://bitbucket.org/sivareddyg/public/downloads/en-ud-v2.zip'
        target_archive_path = os.path.join(root, 'en-ud-v2.zip')
        conditional_remove(target_archive_path)

        # download archive and ensure is in correct location
        archive_path = utils.download_from_url(url)
        assert target_archive_path == archive_path

        correct_files = ['en-ud-v2/en-ud-tag.v2.dev.txt',
                         'en-ud-v2/en-ud-tag.v2.test.txt',
                         'en-ud-v2/en-ud-tag.v2.train.txt',
                         'en-ud-v2/LICENSE.txt',
                         'en-ud-v2/README.txt']
        # extract files and ensure they are correct
        files = utils.extract_archive(archive_path)
        assert files == [os.path.join(root, f) for f in correct_files]

        # extract files with overwrite option True
        files = utils.extract_archive(archive_path, overwrite=True)
        assert files == [os.path.join(root, f) for f in correct_files]

        # remove files and archive
        for f in files:
            conditional_remove(f)
        os.rmdir(os.path.join(root, 'en-ud-v2'))
        conditional_remove(archive_path)

    def test_no_download(self):
        asset_name = 'glove.840B.300d.zip'
        asset_path = get_asset_path(asset_name)
        root = '.data'
        if not os.path.exists(root):
            os.makedirs(root)
        data_path = os.path.join('.data', asset_name)
        shutil.copy(asset_path, data_path)
        file_path = utils.download_from_url('fakedownload/glove.840B.300d.zip')
        self.assertEqual(file_path, data_path)
        conditional_remove(data_path)

    def test_download_extract_to_path(self):
        # create root directory for downloading data
        root = '.data'
        if not os.path.exists(root):
            os.makedirs(root)

        # create directory to extract archive to
        to_path = '.new_data'
        if not os.path.exists(root):
            os.makedirs(root)

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

        # extract files with overwrite option True
        files = utils.extract_archive(archive_path, to_path, overwrite=True)
        assert files == [os.path.join(to_path, 'val.de'),
                         os.path.join(to_path, 'val.en')]

        # remove files and archive
        for f in files:
            conditional_remove(f)
        conditional_remove(archive_path)

    def test_extract_non_tar_zip(self):
        # create root directory for downloading data
        root = '.data'
        if not os.path.exists(root):
            os.makedirs(root)

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
