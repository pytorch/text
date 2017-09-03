# -*- coding: utf-8 -*-
from unittest import TestCase
import logging
import os
import shutil
import subprocess
import tempfile

logger = logging.getLogger(__name__)


class TorchtextTestCase(TestCase):
    # Directory where everything temporary and test-related is written
    test_dir = tempfile.mkdtemp()
    sv_dataset_path = os.path.join(test_dir, "sv_dataset")

    def setUp(self):
        logging.basicConfig(format=('%(asctime)s - %(levelname)s - '
                                    '%(name)s - %(message)s'),
                            level=logging.INFO)

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir)
        except:
            subprocess.call(["rm", "-rf", self.test_dir])

    def write_sv_dataset(self, data_format="csv"):
        if data_format == "csv":
            delim = ","
        elif data_format == "tsv":
            delim = "\t"
        else:
            raise ValueError("Invalid format {}".format(data_format))
        with open(self.sv_dataset_path, "w") as sv_dataset_file:
            sv_dataset_file.write("0{}When do you use シ instead of し?{}"
                                  "When do you use \"&\" instead of \"and\"?{}"
                                  "0\n".format(delim, delim, delim))
            sv_dataset_file.write("1{}Where was Lincoln born?{}"
                                  "Which location was Abraham Lincoln born?{}"
                                  "1\n".format(delim, delim, delim))
            sv_dataset_file.write("2{}What is 2+2{}2+2=?{}1".format(delim, delim, delim))
