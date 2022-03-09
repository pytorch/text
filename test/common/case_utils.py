import random
import os.path
import tempfile
import unittest
from itertools import zip_longest

from torchtext._internal.module_utils import is_module_available


class TempDirMixin:
    """Mixin to provide easy access to temp dir"""

    temp_dir_ = None

    @classmethod
    def get_base_temp_dir(cls):
        # If TORCHTEXT_TEST_TEMP_DIR is set, use it instead of temporary directory.
        # this is handy for debugging.
        key = "TORCHTEXT_TEST_TEMP_DIR"
        if key in os.environ:
            return os.environ[key]
        if cls.temp_dir_ is None:
            cls.temp_dir_ = tempfile.TemporaryDirectory()
        return cls.temp_dir_.name

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if cls.temp_dir_ is not None:
            cls.temp_dir_.cleanup()
            cls.temp_dir_ = None

    def get_temp_path(self, *paths):
        temp_dir = os.path.join(self.get_base_temp_dir(), self.id())
        path = os.path.join(temp_dir, *paths)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


def skipIfNoModule(module, display_name=None):
    display_name = display_name or module
    return unittest.skipIf(
        not is_module_available(module), f'"{display_name}" is not available'
    )


def zip_equal(*iterables):
    """With the regular Python `zip` function, if one iterable is longer than the other,
    the remainder portions are ignored.This is resolved in Python 3.10 where we can use
    `strict=True` in the `zip` function
    """
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def get_random_unicode(length):
    # taken from https://stackoverflow.com/a/21666621/2883245

    # Update this to include code point ranges to be sampled
    include_ranges = [
        (0x0021, 0x0021),
        (0x0023, 0x0026),
        (0x0028, 0x007E),
        (0x00A1, 0x00AC),
        (0x00AE, 0x00FF),
        (0x0100, 0x017F),
        (0x0180, 0x024F),
        (0x2C60, 0x2C7F),
        (0x16A0, 0x16F0),
        (0x0370, 0x0377),
        (0x037A, 0x037E),
        (0x0384, 0x038A),
        (0x038C, 0x038C),
    ]

    alphabet = [chr(code_point) for current_range in include_ranges for code_point in range(current_range[0], current_range[1] + 1)]
    return ''.join(random.choice(alphabet) for i in range(length))
