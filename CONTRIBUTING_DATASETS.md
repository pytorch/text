# Guidelines for adding new dataset

## Background

Torchtext datasets are based on and are built using composition of TorchData’s DataPipes.
[TorchData](https://github.com/pytorch/data) is a library that provides modular/composable primitives, allowing users to
load and transform data in performant data pipelines. With DataPipes, users can easily do data manipulation and
preprocessing using user-defined functions and transformations in a functional style programming. Datasets backed by
DataPipes also enable standard flow-control like batching, collation, shuffling and bucketizing. Collectively, DataPipes
provides a comprehensive experience for data preprocessing and tensorization needs in a Pythonic and flexible way for
model training.

For reference, datasets have been migrated from older-style Iterable Datasets to TorchData’s DataPipes in version 0.12.
You can follow more details in this [github issue](https://github.com/pytorch/text/issues/1494)

## Developers guide

### Before you begin

It’s great that you would like to contribute a new dataset to the repository, we love contributions! But there are few
things to take into account.

- `Dataset Hosting:` Please note that torchtext does not host or provide any hosting services for datasets. It simply
  provides a Dataset API contract to make it easy for end users to consume the dataset from its original source.
- `Dataset Relevance:` Although there are no strict guidelines on what can and cannot be part of the library, we think
  that it is very important to take the dataset relevance into account before adding it to the repository. Some of the
  reference points may include:
  - Whether the dataset provide a good reference benchmarks for any given NLP/Multi-Modal related tasks
  - Number of citations received by the dataset
  - Community needs
- `Licensing concerns:` Last, but not least, make sure there are no licensing concerns over providing access to the
  dataset through torchtext’s datasets API. We have a disclaimer
  [here](https://github.com/pytorch/text#disclaimer-on-datasets) on this too.

If you have any questions or concerns, do not hesitate to open a github issue for seeking feedback from community and
torchtext’s library maintainers.

### Let’s get started!

#### Functional API

TorchText’s datasets API are all functional. To write a new dataset, create the file for the corresponding dataset in
the datasets directory and create the function with `root` as the first argument. The `root` directory is the one used
to cache the dataset. If the dataset consists of splits (`train`, `test`, `val`, `dev` etc), follow `root` by another
keyword argument called `split`. Provide any additional keyword arguments necessary to implement the dataset. Add
following decorators to your function:

- `@_create_dataset_directory:` This decorator will create an appropriate directory in the `root` directory to download
  and cache the dataset.
- `@_wrap_split_argument:` If the dataset consists of split arguments, add this decorator. It will allow the users to
  pass a split argument either as `tuple` or `str`.

Sample code to add function definition:

```python
DATASET_NAME = "MyDataName"

@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", “dev”,"test"))
def MyDataName(root: str, split: Union[Tuple[str], str], …):
    …
```

To make the dataset importable through the torchtext’s datasets package, add it to the datasets `__init__.py` file.

### Dataset Implementation

Building datasets using DataPipes is fun! One just needs to think in terms of various primitive components and
abstractions that are necessary to compose the stack. A typical workflow may look like this:

Data download (+Caching) -> Hash check -> Extraction (+Caching) ->File parsing -> Data organization -> Dataset samples.

We already have a healthy collection of dataset implementations based on DataPipes. It provides a great starter guide to
implement new dataset since they share many of the components. For reference, it is highly recommended to look at some
of these existing datasets implementations. Furthermore, please refer to official torchdata
[documentation](https://pytorch.org/data/beta/index.html) to learn more about available
[Iterable Style DataPipes](https://pytorch.org/data/beta/torchdata.datapipes.iter.html).

Below we provide a bit more details on what each of these components are and how to realize them using DataPipes.

#### Download from source

Typically the first step is to download the dataset from the host to the local machine. The dataset may be present on
different stores like Google Drive, AWS S3 etc and may require different reading mechanisms. TorchData implements a
number of commonly downloading mechanisms like HTTPReader and GDriveReader and can be used for data download. Refer to
the [IO Data Pipes](https://pytorch.org/data/beta/torchdata.datapipes.iter.html#io-datapipes) section for more details.

#### Dataset cache

Downloading data is expensive! Hence it is important to cache the data on disk (unless it is implemented as a streaming
Dataset). TorchData provides
[OnDiskCashHolder](https://pytorch.org/data/beta/generated/torchdata.datapipes.iter.OnDiskCacheHolder.html#torchdata.datapipes.iter.OnDiskCacheHolder)
and
[EndofDiskCashHolder](https://pytorch.org/data/beta/generated/torchdata.datapipes.iter.EndOnDiskCacheHolder.html#torchdata.datapipes.iter.EndOnDiskCacheHolder)
DataPipes to facilitate caching. In short, the datapipe checks whether the file is already available on the local
filesystem and shall trigger download only when it is not present. It is quite important to use caching, otherwise the
data will be downloaded at every epoch. The Datapipe also facilitates data integrity check via hash checking. It is
recommended to do this to ensure that we do not silently ignore changes made in the hosted dataset.

#### Unarchiving and caching compressed files

Typically, the dataset is often stored in archived (zip, tar etc) form. TorchData provides a number of utility datapipe
to uncompress the datasets. Check the available
[archive DataPipes](https://pytorch.org/data/beta/torchdata.datapipes.iter.html#archive-datapipes) to use them in your
dataset implementation. Furthermore, it is also highly recommended to use a caching mechanism (similar to caching
downloaded files) to ensure that data is not decompressed at every epoch. Note that it is not necessary to use hash
check for caching decompressed file(s), since it is already done for compressed file(s).

#### Reading from files

Data is often saved in text files with different structures/formatting like CSV, JSON etc. TorchData provides a number
of standard [text file reading utilities](https://pytorch.org/data/beta/torchdata.datapipes.iter.html#text-datapipes)
that can be conveniently stacked on top of previous IO Stream pipes like
[FileOpener](https://pytorch.org/data/beta/generated/torchdata.datapipes.iter.FileOpener.html#torchdata.datapipes.iter.FileOpener)
that yield data streams.

#### Data organization and returning dataset samples as tuples

In torchtext we follow the convention that samples are returned as tuples. For instance, the samples from classification
datasets are returned as tuples of (int, str) that store labels and text respectively. Similarly for translation dataset
they are tuples of (str, str) that store source and target sentences.

#### Add support for data shuffle and sharding

Finally add support data shuffling and sharding across ranks during distributed training and multi-processing. Follow
this [issue](https://github.com/pytorch/text/issues/1727) for additional details.

### Testing

We use mocking to implement end-2-end testing for the implemented dataset. We avoid testing using a real dataset since
it is expensive to download and/or cache the dataset for testing purposes.

To implement the dataset test, create the corresponding testing file `test_<datasetname>.py` under `tests/datasets`
directory. Do the following:

- Create a function `_get_mock_dataset` that writes a replica of the dataset (albeit with a much smaller number of
  samples, typically 10) in a temporary directory and returns the dataset samples for comparison during testing
- Create the dataset test class `Test<DataSetName>` that tests implementation on following two accounts:
  - Samples returned on iterating over the dataset
  - Dataset returned by passing split argument as `tuple` and as `str`

For detailed examples on how to write the test, please follow the existing test suite under `tests/datasets` directory.

For additional reference, you may also refer to [github issue #1493](https://github.com/pytorch/text/issues/1493) where
we migrated testing of all the datasets from real datasets (that were cached) to mocked one.

### Contribute

Simply create the PR and the torchtext team will help with reviews and get it landed on to the main branch!
