import collections
import hashlib
import os
import struct
import subprocess
import sys
from functools import partial
from typing import Union, Tuple

from torchtext._internal.module_utils import is_module_available

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import (
        FileOpener,
        IterableWrapper,
        OnlineReader,
        FileLister,
        GDriveReader,
    )
    from torchtext._download_hooks import HttpReader


dm_single_close_quote = "\u2019"  # unicode
dm_double_close_quote = "\u201d"
END_TOKENS = [
    ".",
    "!",
    "?",
    "...",
    "'",
    "`",
    '"',
    dm_single_close_quote,
    dm_double_close_quote,
    ")",
]  # acceptable ways to end a sentence
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"
DATASET_NAME = "CNNDM"

URL_LIST = {
    "train": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_train.txt",
    "val": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_val.txt",
    "test": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_test.txt",
}

URL_LIST_MD5 = {
    "train": "c8ca98cfcb6cf3f99a404552568490bc",
    "val": "83a3c483b3ed38b1392285bed668bfee",
    "test": "4f3ac04669934dbc746b7061e68a0258",
}

STORIES_LIST = {
    "cnn": "https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ",
    "dailymail": "https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs",
}

PATH_LIST = {
    "cnn": "cnn_stories.tgz",
    "dailymail": "dailymail_stories.tgz",
}

STORIES_MD5 = {"cnn": "85ac23a1926a831e8f46a6b8eaf57263", "dailymail": "f9c5f565e8abe86c38bfa4ae8f96fd72"}

_EXTRACTED_FOLDERS = {
    "cnn": os.path.join("cnn", "stories"),
    "daily_mail": os.path.join("dailymail", "stories"),
}


def _filepath_fn(root: str, source: str, _=None):
    return os.path.join(root, PATH_LIST[source])


def _extracted_file_path_fn(root: str, source: str, t):
    return os.path.join(root, _EXTRACTED_FOLDERS[source])


def _modify_res(t):
    return t[1]


def _get_url_list(split: str):

    url_dp = IterableWrapper([URL_LIST[split]])
    online_dp = OnlineReader(url_dp)
    return online_dp.readlines().map(_modify_res)


def _get_stories(root: str, source: str):
    story_dp = IterableWrapper([STORIES_LIST[source]])

    cache_compressed_dp = story_dp.on_disk_cache(
        filepath_fn=partial(_filepath_fn, root, source),
        hash_dict={_filepath_fn(root, source): STORIES_MD5[source]},
        hash_type="md5",
    )

    cache_compressed_dp = GDriveReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)
    # cache_decompressed_dp = cache_compressed_dp.on_disk_cache()
    cache_decompressed_dp = FileOpener(cache_compressed_dp, mode="b").load_from_tar()
    # .map(lambda t: (os.path.join(root, source, "stories", os.path.basename(t[0])), t[1]))
    # cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=False)
    stories = cache_decompressed_dp
    stories = stories.map(lambda t: _get_art_abs(t[1]))
    return stories


def _hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()


def _get_url_hashes(url_list):
    return [_hashhex(url) for url in url_list]


def _fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    # print line[-1]
    return line + " ."


def _get_art_abs(story_file):
    lines = story_file.readlines()
    # Lowercase everything
    lines = [line.decode().lower() for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [_fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = " ".join(article_lines)

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    abstract = " ".join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

    return article, abstract


def _get_story_files(url_hash):
    return url_hash + ".story"


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def CNNDM(root: str, split: Union[Tuple[str], str]):
    urls = list(_get_url_list(split))
    
    cnn_stories = _get_stories(root, "cnn")
    # dm_stories = _get_stories(root, 'dailymail')

    # Things to figure out
    # * combine the contents of cnn/dm tars to get all (filepaths, streams)
    # * filter files based on which split we're working with
    # * gow to split up these helper functions
    # * store the .story filenames corresponding to each split on disk so we can pass that into the filepath_fn of the on_disk_cache_dp which caches the files extracted from the tar
    # * how to cache the contents of the extracted tar file

    return cnn_stories

    url_hashes = _get_url_hashes(urls)
    story_fnames = url_hashes.map(_get_story_files)
    # story_fnames = [s+".story" for s in url_hashes]

    # for story in story_fnames:
    def _parse_story_file(story):
        if os.path.join(root, _EXTRACTED_FILES["cnn"], story) in cnn_stories:
            story_file = cnn_stories[os.path.join(root, _EXTRACTED_FILES["cnn"], story)]
        elif os.path.join(root, _EXTRACTED_FILES["dailymail"], story) in cnn_stories:
            story_file = dm_stories[os.path.join(root, _EXTRACTED_FILES["cnn"], story)]
        else:
            print(
                f"Error: Couldn't find story file {story} in either cnn or dailymail directories. Was there an error when loading the files?"
            )

        return _get_art_abs(story_file)

    return story_fnames.map(_parse_story_file)


if __name__ == "__main__":

    print("start")
    out = CNNDM(os.path.expanduser("~/.torchtext/cache"), "train")
    # print(out['/data/home/pabbo/.torchtext/cache/dailymail_stories.tgz/dailymail/stories/70afd5d444a63b17ae663b1ff5b13d4fe1d507f6.story'].read().decode())
    # print(out['/data/home/pabbo/.torchtext/cache/cnn_stories.tgz/cnn/stories/ee8871b15c50d0db17b0179a6d2beab35065f1e9.story'].read().decode())
    # print(out[:10])
