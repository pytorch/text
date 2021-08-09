
import csv
import io
import os

from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe('parse_csv_files')
class CSVParserIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for file_name, stream in self.source_datapipe:
            lines = [bytes_line.decode(errors="ignore")
                     for bytes_line in stream.readlines()]
            reader = csv.reader(lines)
            for row in reader:
                yield tuple([file_name] + row)






