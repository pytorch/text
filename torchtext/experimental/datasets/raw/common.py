import torch
import inspect
import functools


def check_default_set(split, target_select, dataset_name):
    # Check whether given object split is either a tuple of strings or string
    # and represents a valid selection of options given by the tuple of strings
    # target_select.
    if isinstance(split, str):
        split = (split,)
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


def dataset_docstring_header(fn):
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
        """.format(fn.__name__, "/".join(default_split), str(example_subset), str(default_split)) + fn.__doc__

    if isinstance(default_split, str):
        return """{} dataset

        Only returns the {default_split} split

        Args:
            root: Directory where the datasets are saved.
                Default: ".data"
            split: Only {default_split} is available.
                Default: {default_split}
            offset: the number of the starting line.
                Default: 0
        """.format(fn.__name__, default_split=default_split) + fn.__doc__

    raise ValueError("default_split type expected to be of string or tuple but got {}".format(type(default_split)))


def add_docstring_header(fn):
    fn.__doc__ = dataset_docstring_header(fn)
    return fn


def wrap_split_argument(fn):
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

    fn_kwargs_dict = {}
    for arg, default in zip(argspec.args, argspec.defaults):
        fn_kwargs_dict[arg] = default

    @functools.wraps(fn)
    def new_fn(**kwargs):
        for arg in fn_kwargs_dict:
            if arg not in kwargs:
                kwargs[arg] = fn_kwargs_dict[arg]
        orig_split = kwargs["split"]
        kwargs["split"] = check_default_set(orig_split, argspec.defaults[1], fn.__name__)
        result = fn(**kwargs)
        return wrap_datasets(tuple(result), orig_split)

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
