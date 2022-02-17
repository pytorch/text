import codecs
import functools
import inspect
import io
import os

import torch
from torch.utils.data import functional_datapipe, IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper
from torchtext.utils import download_from_url, extract_archive, validate_file

try:
    import defusedxml.ElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from torchtext import _CACHE_DIR

"""
These functions and classes are meant solely for use in torchtext.datasets and not
for public consumption yet.
"""


def _clean_inner_xml_file(outfile, stream):
    """Accepts an output filename and a stream of the byte contents of an XML file
    and writes the cleaned contents to a new file on disk.

    Args:
        outfile: the path to which the modified stream should be written
        stream: the byte datapipe of the contents of the XML file

    Returns: the path to the newly-written file and the new StreamWrapper for appropriate caching
    """
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with codecs.open(outfile, mode="w", encoding="utf-8") as fd_txt:
        root = ET.fromstring(stream.read().decode("utf-8"))[0]
        for doc in root.findall("doc"):
            for e in doc.findall("seg"):
                fd_txt.write(e.text.strip() + "\n")
    return outfile, StreamWrapper(open(outfile, "rb"))


def _clean_inner_tags_file(outfile, stream):
    """Accepts an output filename and a stream of the byte contents of a tags file
    and writes the cleaned contents to a new file on disk.

    Args:
        outfile: the path to which the modified stream should be written
        stream: the byte datapipe of the contents of the tags file

    Returns: the path to the newly-written file and the new StreamWrapper for appropriate caching
    """
    xml_tags = [
        "<url",
        "<keywords",
        "<talkid",
        "<description",
        "<reviewer",
        "<translator",
        "<title",
        "<speaker",
        "<doc",
        "</doc",
    ]
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with codecs.open(outfile, mode="w", encoding="utf-8") as fd_txt:
        for line in stream.readlines():
            if not any(tag in line.decode("utf-8") for tag in xml_tags):
                # TODO: Fix utf-8 next line mark
                #                fd_txt.write(l.strip() + '\n')
                #                fd_txt.write(l.strip() + u"\u0085")
                #                fd_txt.write(l.lstrip())
                fd_txt.write(line.decode("utf-8").strip() + "\n")
    return outfile, StreamWrapper(open(outfile, "rb"))


def _rewrite_text_file(outfile, stream):
    """Accepts an output filename and a stream of the byte contents of a text file
    and writes the cleaned contents to a new file on disk.

    Args:
        outfile: the path to which the modified stream should be written
        stream: the byte datapipe of the contents of the text file

    Returns: the path to the newly-written file and the new StreamWrapper for appropriate caching
    """
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        for line in stream.readlines():
            f.write(line.decode("utf-8") + "\n")
    return outfile, StreamWrapper(open(outfile, "rb"))


def _clean_files(outfile, fname, stream):
    if "xml" in fname:
        return _clean_inner_xml_file(outfile, stream)
    elif "tags" in fname:
        return _clean_inner_tags_file(outfile, stream)
    return _rewrite_text_file(outfile, stream)


def _read_text_iterator(path):
    with io.open(path, encoding="utf8") as f:
        for row in f:
            yield row


def _check_default_set(split, target_select, dataset_name):
    # Check whether given object split is either a tuple of strings or string
    # and represents a valid selection of options given by the tuple of strings
    # target_select.
    if isinstance(split, str):
        split = (split,)
    if isinstance(target_select, str):
        target_select = (target_select,)
    if not isinstance(split, tuple):
        raise ValueError("Internal error: Expected split to be of type tuple.")
    if not set(split).issubset(set(target_select)):
        raise TypeError(
            "Given selection {} of splits is not supported for dataset {}. Please choose from {}.".format(
                split, dataset_name, target_select
            )
        )
    return split


def _wrap_datasets(datasets, split):
    # Wrap return value for _setup_datasets functions to support singular values instead
    # of tuples when split is a string.
    if isinstance(split, str):
        if len(datasets) != 1:
            raise ValueError("Internal error: Expected number of datasets is not 1.")
        return datasets[0]
    return datasets


def _dataset_docstring_header(fn, num_lines=None, num_classes=None):
    """
    Returns docstring for a dataset based on function arguments.

    Assumes function signature of form (root='.data', split=<some tuple of strings>, **kwargs)
    """
    argspec = inspect.getfullargspec(fn)
    if not (argspec.args[0] == "root" and argspec.args[1] == "split"):
        raise ValueError("Internal Error: Given function {} did not adhere to standard signature.".format(fn))
    default_split = argspec.defaults[1]

    if not (isinstance(default_split, tuple) or isinstance(default_split, str)):
        raise ValueError("default_split type expected to be of string or tuple but got {}".format(type(default_split)))

    header_s = fn.__name__ + " dataset\n"

    if isinstance(default_split, tuple):
        header_s += "\nSeparately returns the {} split".format("/".join(default_split))

    if isinstance(default_split, str):
        header_s += "\nOnly returns the {} split".format(default_split)

    if num_lines is not None:
        header_s += "\n\nNumber of lines per split:"
        for k, v in num_lines.items():
            header_s += "\n    {}: {}\n".format(k, v)

    if num_classes is not None:
        header_s += "\n\nNumber of classes"
        header_s += "\n    {}\n".format(num_classes)

    args_s = "\nArgs:"
    args_s += "\n    root: Directory where the datasets are saved."
    args_s += "\n        Default: .data"

    if isinstance(default_split, tuple):
        args_s += "\n    split: split or splits to be returned. Can be a string or tuple of strings."
        args_s += "\n        Default: {}" "".format(str(default_split))

    if isinstance(default_split, str):
        args_s += "\n     split: Only {default_split} is available."
        args_s += "\n         Default: {default_split}.format(default_split=default_split)"

    return "\n".join([header_s, args_s]) + "\n"


def _add_docstring_header(docstring=None, num_lines=None, num_classes=None):
    def docstring_decorator(fn):
        old_doc = fn.__doc__
        fn.__doc__ = _dataset_docstring_header(fn, num_lines, num_classes)
        if docstring is not None:
            fn.__doc__ += docstring
        if old_doc is not None:
            fn.__doc__ += old_doc
        return fn

    return docstring_decorator


def _wrap_split_argument_with_fn(fn, splits):
    """
    Wraps given function of specific signature to extend behavior of split
    to support individual strings. The given function is expected to have a split
    kwarg that accepts tuples of strings, e.g. ('train', 'valid') and the returned
    function will have a split argument that also accepts strings, e.g. 'train', which
    are then turned single entry tuples. Furthermore, the return value of the wrapped
    function is unpacked if split is only a single string to enable behavior such as

    train = AG_NEWS(split='train')
    train, valid = AG_NEWS(split=('train', 'valid'))
    """
    argspec = inspect.getfullargspec(fn)
    if not (
        argspec.args[0] == "root"
        and argspec.args[1] == "split"
        and argspec.varargs is None
        and argspec.varkw is None
        and len(argspec.kwonlyargs) == 0
    ):
        raise ValueError("Internal Error: Given function {} did not adhere to standard signature.".format(fn))

    @functools.wraps(fn)
    def new_fn(root=_CACHE_DIR, split=splits, **kwargs):
        result = []
        for item in _check_default_set(split, splits, fn.__name__):
            result.append(fn(root, item, **kwargs))
        return _wrap_datasets(tuple(result), split)

    new_sig = inspect.signature(new_fn)
    new_sig_params = new_sig.parameters
    new_params = []
    new_params.append(new_sig_params["root"].replace(default=".data"))
    new_params.append(new_sig_params["split"].replace(default=splits))
    new_params += [entry[1] for entry in list(new_sig_params.items())[2:]]
    new_sig = new_sig.replace(parameters=tuple(new_params))
    new_fn.__signature__ = new_sig

    return new_fn


def _wrap_split_argument(splits):
    def new_fn(fn):
        return _wrap_split_argument_with_fn(fn, splits)

    return new_fn


def _create_dataset_directory(dataset_name):
    def decorator(fn):
        argspec = inspect.getfullargspec(fn)
        if not (
            argspec.args[0] == "root"
            and argspec.varargs is None
            and argspec.varkw is None
            and len(argspec.kwonlyargs) == 0
        ):
            raise ValueError("Internal Error: Given function {} did not adhere to standard signature.".format(fn))

        @functools.wraps(fn)
        def wrapper(root=_CACHE_DIR, *args, **kwargs):
            new_root = os.path.join(root, dataset_name)
            if not os.path.exists(new_root):
                os.makedirs(new_root)
            return fn(root=new_root, *args, **kwargs)

        return wrapper

    return decorator


def _download_extract_validate(
    root, url, url_md5, downloaded_file, extracted_file, extracted_file_md5, hash_type="sha256"
):
    root = os.path.abspath(root)
    downloaded_file = os.path.abspath(downloaded_file)
    extracted_file = os.path.abspath(extracted_file)
    if os.path.exists(extracted_file):
        with open(os.path.join(root, extracted_file), "rb") as f:
            if validate_file(f, extracted_file_md5, hash_type):
                return extracted_file

    dataset_tar = download_from_url(
        url, path=os.path.join(root, downloaded_file), hash_value=url_md5, hash_type=hash_type
    )
    extracted_files = extract_archive(dataset_tar)
    assert os.path.exists(extracted_file), "extracted_file [{}] was not found in the archive [{}]".format(
        extracted_file, extracted_files
    )

    return extracted_file


class _RawTextIterableDataset(torch.utils.data.IterableDataset):
    """Defines an abstraction for raw text iterable datasets."""

    def __init__(self, description, full_num_lines, iterator):
        """Initiate the dataset abstraction."""
        super(_RawTextIterableDataset, self).__init__()
        self.description = description
        self.full_num_lines = full_num_lines
        self._iterator = iterator
        self.num_lines = full_num_lines
        self.current_pos = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_pos == self.num_lines - 1:
            raise StopIteration
        item = next(self._iterator)
        if self.current_pos is None:
            self.current_pos = 0
        else:
            self.current_pos += 1
        return item

    def __len__(self):
        return self.num_lines

    def pos(self):
        """
        Returns current position of the iterator. This returns None
        if the iterator hasn't been used yet.
        """
        return self.current_pos

    def __str__(self):
        return self.description


def _generate_iwslt_files_for_lang_and_split(year, src_language, tgt_language, valid_set, test_set):
    train_filenames = (
        "train.{}-{}.{}".format(src_language, tgt_language, src_language),
        "train.{}-{}.{}".format(src_language, tgt_language, tgt_language),
    )
    valid_filenames = (
        "IWSLT{}.TED.{}.{}-{}.{}".format(year, valid_set, src_language, tgt_language, src_language),
        "IWSLT{}.TED.{}.{}-{}.{}".format(year, valid_set, src_language, tgt_language, tgt_language),
    )
    test_filenames = (
        "IWSLT{}.TED.{}.{}-{}.{}".format(year, test_set, src_language, tgt_language, src_language),
        "IWSLT{}.TED.{}.{}-{}.{}".format(year, test_set, src_language, tgt_language, tgt_language),
    )

    src_train, tgt_train = train_filenames
    src_eval, tgt_eval = valid_filenames
    src_test, tgt_test = test_filenames

    uncleaned_train_filenames = (
        "train.tags.{}-{}.{}".format(src_language, tgt_language, src_language),
        "train.tags.{}-{}.{}".format(src_language, tgt_language, tgt_language),
    )
    uncleaned_valid_filenames = (
        "IWSLT{}.TED.{}.{}-{}.{}.xml".format(year, valid_set, src_language, tgt_language, src_language),
        "IWSLT{}.TED.{}.{}-{}.{}.xml".format(year, valid_set, src_language, tgt_language, tgt_language),
    )
    uncleaned_test_filenames = (
        "IWSLT{}.TED.{}.{}-{}.{}.xml".format(year, test_set, src_language, tgt_language, src_language),
        "IWSLT{}.TED.{}.{}-{}.{}.xml".format(year, test_set, src_language, tgt_language, tgt_language),
    )

    uncleaned_src_train, uncleaned_tgt_train = uncleaned_train_filenames
    uncleaned_src_eval, uncleaned_tgt_eval = uncleaned_valid_filenames
    uncleaned_src_test, uncleaned_tgt_test = uncleaned_test_filenames

    file_path_by_lang_and_split = {
        src_language: {
            "train": src_train,
            "valid": src_eval,
            "test": src_test,
        },
        tgt_language: {
            "train": tgt_train,
            "valid": tgt_eval,
            "test": tgt_test,
        },
    }

    uncleaned_filenames_by_lang_and_split = {
        src_language: {
            "train": uncleaned_src_train,
            "valid": uncleaned_src_eval,
            "test": uncleaned_src_test,
        },
        tgt_language: {
            "train": uncleaned_tgt_train,
            "valid": uncleaned_tgt_eval,
            "test": uncleaned_tgt_test,
        },
    }

    return file_path_by_lang_and_split, uncleaned_filenames_by_lang_and_split


@functional_datapipe("read_squad")
class _ParseSQuADQAData(IterDataPipe):
    r"""Iterable DataPipe to parse the contents of a stream of JSON objects
    as provided by SQuAD QA. Used in SQuAD1 and SQuAD2.
    """

    def __init__(self, source_datapipe) -> None:
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for _, stream in self.source_datapipe:
            raw_json_data = stream["data"]
            for layer1 in raw_json_data:
                for layer2 in layer1["paragraphs"]:
                    for layer3 in layer2["qas"]:
                        _context, _question = layer2["context"], layer3["question"]
                        _answers = [item["text"] for item in layer3["answers"]]
                        _answer_start = [item["answer_start"] for item in layer3["answers"]]
                        if len(_answers) == 0:
                            _answers = [""]
                            _answer_start = [-1]
                        yield _context, _question, _answers, _answer_start


@functional_datapipe("read_iob")
class _ParseIOBData(IterDataPipe):
    """A datapipe responsible for reading sep-delimited IOB data from a stream.

    Used for CONLL 2000 and UDPOS."""

    def __init__(self, dp, sep: str = "\t") -> None:
        self.dp = dp
        self.sep = sep

    def __iter__(self):
        columns = []
        for filename, line in self.dp:
            line = line.strip()
            if line == "":
                if columns:
                    yield columns
                columns = []
            else:
                for i, column in enumerate(line.split(self.sep)):
                    if len(columns) < i + 1:
                        columns.append([])
                    columns[i].append(column)
        if len(columns) > 0:
            yield columns
