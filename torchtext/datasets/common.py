import torch
import inspect
import functools


def check_default_set(split, target_select, dataset_name):
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
        raise TypeError('Given selection {} of splits is not supported for dataset {}. Please choose from {}.'.format(
            split, dataset_name, target_select))
    return split


def wrap_datasets(datasets, split):
    # Wrap return value for _setup_datasets functions to support singular values instead
    # of tuples when split is a string.
    if isinstance(split, str):
        if len(datasets) != 1:
            raise ValueError("Internal error: Expected number of datasets is not 1.")
        return datasets[0]
    return datasets


def find_match(match, lst):
    """
    Searches list of strings and returns first entry that partially or fully
    contains the given string match.
    """
    for element in lst:
        if element.find(match) != -1:
            return element
    return None


def dataset_docstring_header(fn):
    """
    Returns docstring for a dataset based on function arguments.

    Assumes function signature of form (root='.data', split=<some tuple of strings>, offset=0, **kwargs)
    """
    argspec = inspect.getfullargspec(fn)
    if not (argspec.args[0] == "root" and
            argspec.args[1] == "split" and
            argspec.args[2] == "offset"):
        raise ValueError("Internal Error: Given function {} did not adhere to standard signature.".format(fn))
    default_split = argspec.defaults[1]

    if isinstance(default_split, tuple):
        example_subset = default_split[:2]
        if len(default_split) < 3:
            example_subset = (default_split[1],)
        return """{} dataset

        Separately returns the {} split

        Args:
            root: Directory where the datasets are saved.
                Default: ".data"
            split: split or splits to be returned. Can be a string or tuple of strings.
                By default, all three datasets are generated. Users
                could also choose any subset of them, for example {} or just 'train'.
                Default: {}
            offset: the number of the starting line.
                Default: 0
        """.format(fn.__name__, "/".join(default_split), str(example_subset), str(default_split))

    if isinstance(default_split, str):
        return """{} dataset

        Only returns the {default_split} split

        Args:
            root: Directory where the datasets are saved.
                Default: ".data"
            split: Only {default_split} is available.
                Default: {default_split}
            offset: the number of the starting line.
                Default: 0""".format(fn.__name__, default_split=default_split)

    raise ValueError("default_split type expected to be of string or tuple but got {}".format(type(default_split)))


def add_docstring_header(docstring=None):
    def docstring_decorator(fn):
        old_doc = fn.__doc__
        fn.__doc__ = dataset_docstring_header(fn)
        if docstring is not None:
            fn.__doc__ += docstring
        if old_doc is not None:
            fn.__doc__ += old_doc
        return fn
    return docstring_decorator


def wrap_split_argument(fn):
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
    if not (argspec.args[0] == "root" and
            argspec.args[1] == "split" and
            argspec.args[2] == "offset" and
            argspec.defaults[0] == ".data" and
            argspec.defaults[2] == 0 and
            argspec.varargs is None and
            argspec.varkw is None and
            len(argspec.kwonlyargs) == 0 and
            argspec.kwonlydefaults is None and
            len(argspec.annotations) == 0
            ):
        raise ValueError("Internal Error: Given function {} did not adhere to standard signature.".format(fn))

    # functools.wraps only forwards __module__, __name__, etc
    # (see https://docs.python.org/3/library/functools.html#functools.update_wrapper)
    # but not default values of arguments. The wrapped function fn is assumed to have
    # keyword arguments with default values only, so only  a dictionary of default
    # values is needed to support that behavior for new_fn as well.
    fn_kwargs_dict = {}
    for arg, default in zip(argspec.args[3:], argspec.defaults[3:]):
        fn_kwargs_dict[arg] = default

    @functools.wraps(fn)
    def new_fn(root='.data', split=argspec.defaults[1], offset=0, **kwargs):
        for arg in fn_kwargs_dict:
            if arg not in kwargs:
                kwargs[arg] = fn_kwargs_dict[arg]
        kwargs["root"] = root
        kwargs["offset"] = offset
        kwargs["split"] = check_default_set(split, argspec.defaults[1], fn.__name__)
        result = fn(**kwargs)
        return wrap_datasets(tuple(result), split)

    return new_fn


class RawTextIterableDataset(torch.utils.data.IterableDataset):
    """Defines an abstraction for raw text iterable datasets.
    """

    def __init__(self, name, full_num_lines, iterator, offset=0):
        """Initiate text-classification dataset.
        """
        super(RawTextIterableDataset, self).__init__()
        self.name = name
        self.full_num_lines = full_num_lines
        self._iterator = iterator
        self.start = offset
        if offset < 0:
            raise ValueError("Given offset must be non-negative, got {} instead.".format(offset))
        self.num_lines = full_num_lines - offset

    def __iter__(self):
        for i, item in enumerate(self._iterator):
            if i < self.start:
                continue
            if self.num_lines and i >= (self.start + self.num_lines):
                break
            yield item

    def __next__(self):
        item = next(self._iterator)
        return item

    def __len__(self):
        return self.num_lines

    def get_iterator(self):
        return self._iterator
