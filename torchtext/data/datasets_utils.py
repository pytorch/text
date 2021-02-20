import torch
import inspect
import functools

"""
These functions and classes are meant solely for use in torchtext.datasets and not
for public consumption yet.
"""


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
        if match in element:
            return element
    return None


def dataset_docstring_header(fn):
    """
    Returns docstring for a dataset based on function arguments.

    Assumes function signature of form (root='.data', split=<some tuple of strings>, **kwargs)
    """
    argspec = inspect.getfullargspec(fn)
    if not (argspec.args[0] == "root" and
            argspec.args[1] == "split"):
        raise ValueError("Internal Error: Given function {} did not adhere to standard signature.".format(fn))
    default_split = argspec.defaults[1]

    if isinstance(default_split, tuple):
        example_subset = default_split[:2]
        if len(default_split) < 3:
            example_subset = (default_split[0],)
        return """{} dataset

        Separately returns the {} split

        Args:
            root: Directory where the datasets are saved.
                Default: ".data"
            split: split or splits to be returned. Can be a string or tuple of strings.
                By default, all three datasets are generated. Users
                could also choose any subset of them, for example {} or just 'train'.
                Default: {}
        """.format(fn.__name__, "/".join(default_split), str(example_subset), str(default_split))

    if isinstance(default_split, str):
        return """{} dataset

        Only returns the {default_split} split

        Args:
            root: Directory where the datasets are saved.
                Default: ".data"
            split: Only {default_split} is available.
                Default: {default_split}""".format(fn.__name__, default_split=default_split)

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


def _wrap_split_argument(fn, splits):
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
            argspec.varargs is None and
            argspec.varkw is None and
            len(argspec.kwonlyargs) == 0 and
            len(argspec.annotations) == 0
            ):
        raise ValueError("Internal Error: Given function {} did not adhere to standard signature.".format(fn))

    @functools.wraps(fn)
    def new_fn(root='.data', split=splits, **kwargs):
        result = []
        for item in check_default_set(split, splits, fn.__name__):
            result.append(fn(root, item, **kwargs))
        return wrap_datasets(tuple(result), split)

    new_sig = inspect.signature(new_fn)
    new_sig_params = new_sig.parameters
    new_params = []
    new_params.append(new_sig_params['root'].replace(default='.data'))
    new_params.append(new_sig_params['split'].replace(default=splits))
    new_params += [entry[1] for entry in list(new_sig_params.items())[2:]]
    new_sig = new_sig.replace(parameters=tuple(new_params))
    new_fn.__signature__ = new_sig

    return new_fn


def wrap_split_argument(splits):
    def new_fn(fn):
        return _wrap_split_argument(fn, splits)
    return new_fn


class RawTextIterableDataset(torch.utils.data.IterableDataset):
    """Defines an abstraction for raw text iterable datasets.
    """

    def __init__(self, description, full_num_lines, iterator):
        """Initiate text-classification dataset.
        """
        super(RawTextIterableDataset, self).__init__()
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
        Returns current position of the iterator. This is None
        if the iterator hasn't been used yet.
        """
        return self.current_pos

    def __str__(self):
        return self.description
