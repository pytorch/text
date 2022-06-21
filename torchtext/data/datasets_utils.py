import codecs
import functools
import inspect
import os

from torch.utils.data import functional_datapipe, IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper

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
            new_root = os.path.join(root, "datasets", dataset_name)
            if not os.path.exists(new_root):
                os.makedirs(new_root, exist_ok=True)
            return fn(root=new_root, *args, **kwargs)

        return wrapper

    return decorator


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


@functional_datapipe("parse_cnndm_data")
class _ParseCNNDMData(IterDataPipe):
    """Iterable DataPipe to parse the article and abstract from a CNNDM data stream.
    Code is inspired from https://github.com/abisee/cnn-dailymail/blob/master/make_datafiles.py"""

    dm_single_close_quote = "\u2019"  # unicode
    dm_double_close_quote = "\u201d"
    # acceptable ways to end a sentence
    END_TOKENS = [".", "!", "?", "...", "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")", "\n"]

    def __init__(self, source_datapipe) -> None:
        self.source_datapipe = source_datapipe

    def _fix_missing_period(self, line):
        """Adds a period to a line that is missing a period"""
        if "@highlight" in line:
            return line
        if line == "":
            return line
        if line[-1] in self.END_TOKENS:
            return line
        return line + " ."

    def __iter__(self):
        for _, stream in self.source_datapipe:
            lines = stream.readlines()
            lines = [line.decode("utf-8").strip() for line in lines]

            # put periods on the ends of lines that are missing them
            # this is a problem in the dataset because many image captions don't end in periods
            # consequently they end up in the body of the article as run-on sentences
            lines = [self._fix_missing_period(line) for line in lines]

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

            article = " ".join(article_lines)
            abstract = " ".join(highlights)
            yield article, abstract
