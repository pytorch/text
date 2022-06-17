import os, sys
from argparse import ArgumentParser
from functools import partial

import torcharrow.pytorch as tap
import torchtext.functional as F
from benchmark.utils import Timer
from torchtext.datasets import DATASETS

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../examples"))
from data_pipeline.roberta_dataframe import RobertaTransformDataFrameNativeOps, RobertaTransformDataFrameUDF
from data_pipeline.roberta_datapipe import RobertaTransformDataPipe


def benchmark_roberta_datapipe(args):
    print("********Running Benchmark using DataPipes**************\n")
    batch_size = args.batch_size
    dataset_name = args.dataset_name
    columns = args.columns

    # Instantiate transform
    with Timer("Initialize Roberta Transform (for datapipe)"):
        transform = RobertaTransformDataPipe()

    with Timer("Initialize Pipeline"):
        # Create SST2 datapipe and apply pre-processing
        train_dp = DATASETS[dataset_name](split="train")
        train_dp = train_dp.batch(batch_size).rows2columnar(columns)

        # Apply text pre-processing
        train_dp = train_dp.map(transform)

        # convert to Tensor
        train_dp = train_dp.map(partial(F.to_tensor, padding_value=1), input_col="tokens")
        train_dp = train_dp.map(F.to_tensor, input_col="label")

    with Timer("Execute Pipeline"):
        list(train_dp)


def benchmark_roberta_dataframe(args, native_ops):
    print("****************Running Benchmark using TorchArrow Dataframes*********************\n")
    batch_size = args.batch_size
    dataset_name = args.dataset_name
    columns = args.columns

    if native_ops:
        append_text = "as native ops"
    else:
        append_text = "as UDF"

    # Instantiate transform
    with Timer("Initialize Roberta Transform (for DataFrame) {}".format(append_text)):
        if native_ops:
            transform = RobertaTransformDataFrameNativeOps()
        else:
            transform = RobertaTransformDataFrameUDF()

    with Timer("Initialize Pipeline"):
        # Create SST2 datapipe and apply pre-processing
        train_dp = DATASETS[dataset_name](split="train")

        # convert to DataFrame of size batches
        # TODO: Figure out how to create DataFrame of larger size and create smaller batches
        train_dp = train_dp.dataframe(columns=columns, dataframe_size=batch_size)

        # Apply transformation on DataFrame
        train_dp = train_dp.map(transform)

        # Remove not required columns
        train_dp = train_dp.map(lambda x: x.drop(["text"]))

        # convert DataFrame to tensor (This will yield named tuple)
        train_dp = train_dp.map(lambda x: x.to_tensor({"tokens": tap.PadSequence(padding_value=1)}))

    with Timer("Execute Pipeline"):
        list(train_dp)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--dataset-name", default="SST2", type=str)
    parser.add_argument("--columns", default=["text", "label"], nargs="+")
    benchmark_roberta_datapipe(parser.parse_args())
    benchmark_roberta_dataframe(parser.parse_args(), native_ops=False)
    benchmark_roberta_dataframe(parser.parse_args(), native_ops=True)
