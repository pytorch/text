import logging
import os

import torch
from torch import Tensor
import torch.nn as nn
from tqdm import tqdm

from torchtext.utils import (
    download_from_url,
    extract_archive
)

logger = logging.getLogger(__name__)


def _infer_shape(f):
    num_lines = 0
    for line in f:
        num_lines += 1
    f.seek(0)
    return num_lines


def _load_token_and_vectors_from_file(file_path, delimiter=" "):
    stoi, tokens, vectors, dup_tokens = {}, [], [], []
    dim = None
    with open(file_path, "rb") as f:
        num_lines = _infer_shape(f)
        for line in tqdm(f, unit_scale=0, unit="lines", total=num_lines):
            # token and entries are seperated by delimeter
            token, entries = line.rstrip().split(bytes(delimiter, "utf-8"), 1)
            # we assume entries are always seperated by " "
            entries = entries.split(b" ")

            if dim is None and len(entries) > 1:
                dim = len(entries)
            elif len(entries) == 1:
                logger.warning("Skipping token {} with 1-dimensional "
                               "vector {}; likely a header".format(token, entries))
                continue
            elif dim != len(entries):
                raise RuntimeError(
                    "Vector for token {} has {} dimensions, but previously "
                    "read vectors have {} dimensions. All vectors must have "
                    "the same number of dimensions.".format(token, len(entries),
                                                            dim))

            vector = torch.tensor([float(c) for c in entries], dtype=torch.float)
            try:
                if isinstance(token, bytes):
                    token = token.decode("utf-8")
            except UnicodeDecodeError:
                logger.info("Skipping non-UTF8 token {}".format(repr(token)))
                continue

            if token in stoi:
                dup_tokens.append((token, len(vectors) + 1))
                continue

            stoi[token] = len(vectors)
            tokens.append(token)
            vectors.append(vector)
    return tokens, vectors, dup_tokens


def FastText(language="en", unk_tensor=None, root=".data", validate_file=True):
    r"""Create a FastText Vectors object.

    Args:
        language (str): the language to use for FastText. The list of supported languages options
                        can be found at https://fasttext.cc/docs/en/language-identification.html
        unk_tensor (Tensor): a 1d tensor representing the vector associated with an unknown token
        root (str): folder used to store downloaded files in (.data)
        validate_file (bool): flag to determine whether to validate the downloaded files checksum.
                              Should be `False` when running tests with a local asset.

    Returns:
        Vectors: a Vectors object.

    Raises:
        ValueError: if duplicate tokens are found in FastText file.

    """
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec".format(language)
    file_name = os.path.basename(url)

    cached_vectors_file_path = os.path.join(root, file_name + ".pt")
    if os.path.isfile(cached_vectors_file_path):
        return(torch.load(cached_vectors_file_path))

    checksum = None
    if validate_file:
        checksum = CHECKSUMS_FAST_TEXT.get(url, None)

    downloaded_file_path = download_from_url(url, root=root, hash_value=checksum)
    tokens, vectors, dup_tokens = _load_token_and_vectors_from_file(downloaded_file_path)

    if dup_tokens:
        raise ValueError("Found duplicate tokens in file: {}".format(str(dup_tokens)))

    vectors_obj = Vectors(tokens, vectors, unk_tensor=unk_tensor)
    torch.save(vectors_obj, cached_vectors_file_path)
    return vectors_obj


def GloVe(name="840B", dim=300, unk_tensor=None, root=".data", validate_file=True):
    r"""Create a GloVe Vectors object.

    Args:
        name (str): the name of the GloVe dataset to use. Options are:
            - 42B
            - 840B
            - twitter.27B
            - 6B
        dim (int): the dimension for the GloVe dataset to load. Options are:
            42B:
                - 300
            840B:
                - 300
            twitter.27B:
                - 25
                - 50
                - 100
                - 200
            6B:
                - 50
                - 100
                - 200
                - 300
        unk_tensor (Tensor): a 1d tensor representing the vector associated with an unknown token.
        root (str): folder used to store downloaded files in (.data)
        validate_file (bool): flag to determine whether to validate the downloaded files checksum.
                              Should be `False` when running tests with a local asset.
    Returns:
        Vectors: a Vectors object.

    Raises:
        ValueError: if unexpected duplicate tokens are found in GloVe file.

    """
    dup_token_glove_840b = ("����������������������������������������������������������������������"
                            "����������������������������������������������������������������������"
                            "����������������������������������������������������������������������"
                            "����������������������������������������������������������������������"
                            "������������������������������������������������������", 140649)
    urls = {
        "42B": "https://nlp.stanford.edu/data/glove.42B.300d.zip",
        "840B": "https://nlp.stanford.edu/data/glove.840B.300d.zip",
        "twitter.27B": "https://nlp.stanford.edu/data/glove.twitter.27B.zip",
        "6B": "https://nlp.stanford.edu/data/glove.6B.zip",
    }
    valid_glove_file_names = {
        "glove.42B.300d.txt",
        "glove.840B.300d.txt",
        "glove.twitter.27B.25d.txt",
        "glove.twitter.27B.50d.txt",
        "glove.twitter.27B.100d.txt",
        "glove.twitter.27B.200d.txt",
        "glove.6B.50d.txt",
        "glove.6B.100d.txt",
        "glove.6B.200d.txt",
        "glove.6B.300d.txt"
    }

    file_name = "glove.{}.{}d.txt".format(name, str(dim))
    if file_name not in valid_glove_file_names:
        raise ValueError("Could not find GloVe file with name {}. Please check that `name` and `dim`"
                         "are valid.".format(str(file_name)))

    url = urls[name]
    cached_vectors_file_path = os.path.join(root, file_name + '.pt')
    if os.path.isfile(cached_vectors_file_path):
        return(torch.load(cached_vectors_file_path))

    checksum = None
    if validate_file:
        checksum = CHECKSUMS_GLOVE.get(url, None)

    downloaded_file_path = download_from_url(url, root=root, hash_value=checksum)
    extracted_file_paths = extract_archive(downloaded_file_path)
    # need to get the full path to the correct file in the case when multiple files are extracted with different dims
    extracted_file_path_with_correct_dim = [path for path in extracted_file_paths if file_name in path][0]
    tokens, vectors, dup_tokens = _load_token_and_vectors_from_file(extracted_file_path_with_correct_dim)

    # Ensure there is only 1 expected duplicate token present for 840B dataset
    if dup_tokens:
        if not (len(dup_tokens) == 1 and dup_tokens[0] == dup_token_glove_840b[0] and
           dup_tokens[1] == dup_token_glove_840b[1]):
            raise ValueError("Found duplicate tokens in file: {}".format(str(dup_tokens)))

    vectors_obj = Vectors(tokens, vectors, unk_tensor=unk_tensor)
    torch.save(vectors_obj, cached_vectors_file_path)
    return vectors_obj


def vectors_from_file_object(file_like_object, delimiter=",", unk_tensor=None):
    r"""Create a Vectors object from a csv file like object.

    Note that the tensor corresponding to each vector is of type `torch.float`.

    Format for csv file:
        token1<delimiter>num1 num2 num3
        token2<delimiter>num4 num5 num6
        ...
        token_n<delimiter>num_m num_j num_k

    Args:
        file_like_object (FileObject): a file like object to read data from.
        delimiter (char): a character to delimit between the token and the vector. Default value is ","
        unk_tensor (Tensor): a 1d tensor representing the vector associated with an unknown token.

    Returns:
        Vectors: a Vectors object.

     Raises:
        ValueError: if duplicate tokens are found in FastText file.

    """
    tokens, vectors, dup_tokens = _load_token_and_vectors_from_file(file_like_object.name, delimiter=delimiter)
    if dup_tokens:
        raise ValueError("Found duplicate tokens in file: {}".format(str(dup_tokens)))
    return Vectors(tokens, vectors, unk_tensor=unk_tensor)


class Vectors(nn.Module):
    r"""Creates a vectors object which maps tokens to vectors.

    Arguments:
        tokens (List[str]): a list of tokens.
        vectors (List[torch.Tensor]): a list of 1d tensors representing the vector associated with each token.
        unk_tensor (torch.Tensor): a 1d tensors representing the vector associated with an unknown token.

    Raises:
        ValueError: if `vectors` is empty and a default `unk_tensor` isn't provided.
        RuntimeError: if `tokens` and `vectors` have different sizes or `tokens` has duplicates.
        TypeError: if all tensors within`vectors` are not of data type `torch.float`.
    """

    def __init__(self, tokens, vectors, unk_tensor=None):
        super(Vectors, self).__init__()

        if unk_tensor is None and not vectors:
            raise ValueError("The vectors list is empty and a default unk_tensor wasn't provided.")

        if not all(vector.dtype == torch.float for vector in vectors):
            raise TypeError("All tensors within `vectors` should be of data type `torch.float`.")

        unk_tensor = unk_tensor if unk_tensor is not None else torch.zeros(vectors[0].size(), dtype=torch.float)

        self.vectors = torch.classes.torchtext.Vectors(tokens, vectors, unk_tensor)

    @torch.jit.export
    def __getitem__(self, token: str) -> Tensor:
        r"""
        Args:
            token (str): the token used to lookup the corresponding vector.
        Returns:
            vector (Tensor): a tensor (the vector) corresponding to the associated token.
        """
        return self.vectors.GetItem(token)

    @torch.jit.export
    def __setitem__(self, token: str, vector: Tensor):
        r"""
        Args:
            token (str): the token used to lookup the corresponding vector.
            vector (Tensor): a 1d tensor representing a vector associated with the token.

        Raises:
            TypeError: if `vector` is not of data type `torch.float`.
        """
        if vector.dtype != torch.float:
            raise TypeError("`vector` should be of data type `torch.float` but it's of type " + vector.dtype)

        self.vectors.AddItem(token, vector.float())


CHECKSUMS_GLOVE = {
    "https://nlp.stanford.edu/data/glove.42B.300d.zip":
    "03d5d7fa28e58762ace4b85fb71fe86a345ef0b5ff39f5390c14869da0fc1970",
    "https://nlp.stanford.edu/data/glove.840B.300d.zip":
    "c06db255e65095393609f19a4cfca20bf3a71e20cc53e892aafa490347e3849f",
    "https://nlp.stanford.edu/data/glove.twitter.27B.zip":
    "792af52f795d1a32c9842a3240f5f3fe5e941a8ff6df5eb0f9d668092ebc019c",
    "https://nlp.stanford.edu/data/glove.6B.zip":
    "617afb2fe6cbd085c235baf7a465b96f4112bd7f7ccb2b2cbd649fed9cbcf2fb"
}

CHECKSUMS_FAST_TEXT = {
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.am.vec":
    "b532c57a74628fb110b48b9d8ae2464eb971df2ecc43b89c2eb92803b8ac92bf",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.als.vec":
    "056a359a2651a211817dbb7885ea3e6f69e0d6048d7985eab173858c59ee1adf",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.af.vec":
    "87ecbfea969eb707eab72a7156b4318d341c0652e6e5c15c21bc08f5cf458644",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.an.vec":
    "57db91d8c307c45613092ebfd405061ccfdec5905035d9a8ad364f6b8ce41b29",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ar.vec":
    "5527041ce04fa66e45e27d7bd278f00425d97fde8c67755392d70f112fecc356",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.arz.vec":
    "0b6c261fd179e5d030f2b363f9f7a4db0a52e6241a910b39fb3332d39bcfbec3",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.as.vec":
    "4475daa38bc1e8501e54dfcd79a1a58bb0771b347ad9092ce9e57e9ddfdd3b07",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.av.vec":
    "1292eed7f649687403fac18e0ee97202e163f9ab50f6efa885aa2db9760a967e",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ast.vec":
    "fbba958174ced32fde2593f628c3cf4f00d53cd1d502612a34e180a0d13ce037",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.be.vec":
    "3b36ba86f5b76c40dabe1c7fc3214338d53ce7347c28bb2fba92b6acc098c6ad",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.az.vec":
    "93ebe624677a1bfbb57de001d373e111ef9191cd3186f42cad5d52886b8c6467",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ba.vec":
    "b739fd6f9fe57205314d67a7975a2fc387b55679399a6b2bda0d1835b1fdd5a8",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.azb.vec":
    "05709ce8abc91115777f3cc2574d24d9439d3f6905500163295d695d41260a06",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.bar.vec":
    "3f58304eb0345d96c0abbffb61621c1f6ec2ca39e13272b434cc6cc2bde052a1",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.bcl.vec":
    "309bb74a85647ac3a5be53fd9d3be3196cff385d257561f4183a0d91a67f0c8b",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.bg.vec":
    "16f1a02f3b708f2cbc04971258b0febdfc9ed4e64fcc3818cc6a397e3db5cf81",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.bh.vec":
    "ab0819c155fd1609393f8af74794de8d5b49db0787edf136e938ea2c87993ab5",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.bn.vec":
    "3dd27b9b271c203a452de1c533fdf975ebec121f17f945ef234370358db2bae6",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.bpy.vec":
    "2ba9f046d70bdaae2cbd9d33f9a1d2913637c00126588cc3223ba58ca80d49fe",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.bo.vec":
    "c5ed2a28edf39bc100f4200cdf1c9d3c1448efefcb3d78db8becea613a2fb2eb",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.br.vec":
    "fe858e2be787351cce96c206a9034c361e45f8b9e0a385aacfce3c73f844e923",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.da.vec":
    "397b0c3e18f710fb8aa1caf86441a25af2f247335e8560dbe949feb3613ef5cc",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.bs.vec":
    "ee065fe168c0a4f1a0b9fbd8854be4572c138a414fd7200381d0135ce6c03b49",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.bxr.vec":
    "0bc0e47a669aa0d9ad1c665593f7257c4b27a4e3becce457a7348da716bdabb4",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ca.vec":
    "1600696088c7f2fe555eb6a4548f427f969a450ed0313d68e859d6024242db5f",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.cbk.vec":
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ceb.vec":
    "7fbe4474043e4f656eb2f81ee03d1e863cef8e62ad4e3bd9a3a4143785752568",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ce.vec":
    "2a321e2de98d0abb5a12599d9567dd5ac93f9e2599251237026acff35f23cef8",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.cs.vec":
    "0eba2ac0852b1057909d4e8e5e3fa75470f9cb9408b364433ac4747eb2b568a9",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.cv.vec":
    "67f09d353f2561b16c385187789eb6ff43fa125d3cc81081b2bc7d062c9f0b8a",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.cy.vec":
    "1023affdcb7e84dd59b1b7de892f65888b6403e2ed4fd77cb836face1c70ee68",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.co.vec":
    "7f16f06c19c8528dc48a0997f67bf5f0d79da2d817247776741b54617b6053d9",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ckb.vec":
    "ef3a8472cc2ac86976a1a91cde3edc7fcd1d1affd3c6fb6441451e9fbc6c3ae8",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.de.vec":
    "3020c26e32238ba95a933926763b5c693bf7793bf0c722055cecda1e0283578c",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.diq.vec":
    "6f71204e521e03ae70b4bd8a41c50cc72cd4b8c3e242a4ab5c77670603df1b42",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.dty.vec":
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.dv.vec":
    "2b4f19bfcf0d38e6ab54e53d752847ab60f4880bae955fff2c485135e923501e",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.dsb.vec":
    "ed6699709e0e2f2e3b4a4e32ef3f98f0ccb3f1fed2dad41b7a6deafdc2b32acf",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.el.vec":
    "a397de14c637f0b843fcda8724b406f5a7fe9f3ead7f02cfcaeed43858212da6",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec":
    "ba5420ac217fb34f15f58ded0d911a4370dfb1f3341fa7511a49ae74c87de282",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.eml.vec":
    "a81f0a05c9d3ffd310f6e2d864ee48bff952dbfb2612293b58ab7bc49755cfe6",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.es.vec":
    "cf2e9a1976055a18ad358fb0331bc5f9b2e8541d6d4903b562a63b60f3ae392e",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.et.vec":
    "de15792f8373f27f1053eef28cff4c782c4b440fd57a3472af38e5bf94eafda6",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.eo.vec":
    "a137201c5cf54e218b6bb0bac540beaee2e81e285bf9c59c0d57e0a85e3353c0",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.fi.vec":
    "63017414860020f7409d31c8b65c1f8ed0a64fe11224d4e82e17667ce0fbd0c5",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.fa.vec":
    "da0250d60d159820bf0830499168c2f4f1eaffe74f1508c579ca9b41bae6c53f",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.eu.vec":
    "93f6e44742ea43ff11b5da4c634ebf73f3b1aa3e9485d43eb27bd5ee3979b657",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ilo.vec":
    "e20ac3c7ef6a076315f15d9c326e93b22c2d5eee6bec5caef7bab6faf691b13a",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.frr.vec":
    "a39b393261a8a5c19d97f6a085669daa9e0b9a0cab0b5cf5f7cb23f6084c35e0",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ga.vec":
    "7b33e77e9feb32a6ce2f85ab449516294a616267173c6bbf8f1de5c2b2885699",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.fy.vec":
    "07b695f598e2d51cdd17359814b32f15c56f5beaa7a6b49f69de835e13a212b8",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.fr.vec":
    "bc68b0703375da9e81c3c11d0c28f3f8375dd944c209e697c4075e579455ac2a",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.gd.vec":
    "464e8b97a8b262352a0dc663aa22f98fc0c3f9e7134a749644ad07249dbd42e8",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.gn.vec":
    "5d2ac06649f6199ffad8480efa03f97d2910d1501a4528bfb013524d6f2d6c2b",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.gl.vec":
    "b4b6233b0c650f9d665e5c8aa372f8745d1a40868f93ecf87f026c60b2bb0f9e",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.gu.vec":
    "910296b888e17416e9af43f636f83bbe0b81da68b5e62139ab9c06671dbbacf1",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.gom.vec":
    "20f38a650e90a372a92a1680c6a92fc1d89c21cd41835c8d0e5e42f30d52b7ec",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hi.vec":
    "e5ec503a898207e17a7681d97876607b0481384b6c1cc4c9c6b6aaba7ad293d0",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.gv.vec":
    "b9d6384219d999e43f66ace6decd80eb6359e0956c61cb7049021b194c269ffe",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.he.vec":
    "5d861a705bf541671c0cee731c1b71f6a65d8defd3df2978a7f83e8b0580903b",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hif.vec":
    "445e65668be650f419e0a14791b95c89c3f4142d32371501e53038749eb2c71c",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hsb.vec":
    "27be86ce2435dfeb07d76d406a8ec7b46ebf9b6b8fb5da24208eca1492ffe5bb",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hr.vec":
    "4d42787554747a86253a23e9a830a8571faea0b622e48ed136f8b9817dea9da3",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ht.vec":
    "be5e089f22a43ca00a35467545bc6cca15b5c5951ac34c504a23686ab735e995",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hy.vec":
    "63dc48faeb4f3c5ea2e6f78a0bf4d8bf3d623af52b7f3a9b9e5984dbc79ba66f",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hu.vec":
    "766de324b4783fe2d31df8f78966ea088712a981b6b0b5336bc71938773fe21e",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ia.vec":
    "1ec19501030cafa0fdccf7f4c5794f4cd7e795b015330f6ea6bc9eff97eaeca5",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ie.vec":
    "41c9e34f5445c4aafd7f5d52665b9aa89fb3c76b5262e9401d21b58dd2e53609",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.id.vec":
    "436180ac3d405eefe8c2be20ae3e67cddc866afb94e486afcbaef549c24b7d60",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.io.vec":
    "2bedf13a6d751ad5191474e65b6104fa3175ca4c3f9ade214f25cfeede1c9c8c",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.is.vec":
    "7fe6d8eca113e245ea5467e8f4cab9697dff1d623ac0a8e6fdaca0a93d7fc6f3",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.it.vec":
    "5a9d111edd3f199e7379373ba18f4e5317c6c6c5053a9d6d0a56f32298d3bde4",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ja.vec":
    "b44b2eef8bdcf0739c971c4ff7fcae7a300b5e06cf0e50c5787082957ad9d998",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.jv.vec":
    "4a46ac08781861d6e19fcc70a421340b627889a054279dacee0f32ee12b1f4f7",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.jbo.vec":
    "766c0eb15b1e2cad9a14d0a0937e859e20f6f2ed203ff7ba4f3c70d3b1888d2b",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ka.vec":
    "10b08b9372ef6e44e0e826e6a8d01b3396a319d78ce2db990c51d688c2d0259e",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.kk.vec":
    "2dabc86ed917ba236c96c8c327aa3394f32ea511068a9dce205a46923c5716d1",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.kn.vec":
    "7f9ab4985e0d5f91462fbdcbfbfaeef619d973e638fbc7c928cfcc5bd37d473b",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.km.vec":
    "bf35b294d86fceac916feee3e167fe6aee3fe73380f78e5377c94ff0d023b77c",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ko.vec":
    "44bae904dd7923d1178e83067cc42d9437097f7e86cb83bdd8281febe4b9adaa",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.krc.vec":
    "b0ff031a60938b612f9b0c9612bd206cbb1f5288a6ee3482d116174b81d9269f",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.kv.vec":
    "a6202b11f869683ce75e60bf206c230109f91b651801dc6ea07b3b7f2c5c9b32",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.kw.vec":
    "061e26d970aa7cb3fded9278372a53d0dd8359abc664caa385edaac9aac1359d",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ku.vec":
    "7f117b704d5ac791463796b1ac2d833717c0cfc75dbfb50c2e80aa0c9348c448",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ky.vec":
    "adb5c72c47c514cd5417f46f8a7baba4061063b0e75c2d0b2e42dc08144af6cf",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.lb.vec":
    "566334801777746bc2c1076e1b24a8281e10fe31f0db30a2a4b0b490033e6d04",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.li.vec":
    "50a1054a31a7e11f5bd3fe980e1646205284e540fb1be3ae88f4bf16b0d10301",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.lez.vec":
    "109c3f3fee8970cfab1b7152315816284aa4b5788403d4007866ad417a63b5e6",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.la.vec":
    "ca3b46e03bebf6b937cd7f01c29566b7d48d94d3de033a527ce45744a40ea00a",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.lo.vec":
    "acd1a8cbabbfc50196cb3dfeb9e82c71409c40ae90dc3485044396bbb7350431",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.lmo.vec":
    "26952850a5569e8241d1e6ff2d6877fa51b6715e8fdeec9bf5f9d716e94c958e",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.lt.vec":
    "54bc7d3c1ef600f4710047c4dafd1346e8b53bd29a327bc132f6a9fd0c14b8c7",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.lrc.vec":
    "24d6c530275176cb03e566e9e5737a1680e79854e6c0a2da19a7cb27a029a0ce",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.lv.vec":
    "ed93e318306e19cc18154b095a2494d94ab061009c3a8fa1c3501495f81b7198",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.mg.vec":
    "12ab899108b74bbee8b685ed7f4941719485560c7346875a0be79c7ba6dbec2a",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.mai.vec":
    "387a5d1194e8e441b09c7a215a71cad75e6e1a0777c08f90b2ed5bf4e90423d3",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.mhr.vec":
    "080cb31ff85f0bc21ae75b66311214d594f76a7fdf17699aa5ba8239c6ccd164",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.mk.vec":
    "ea3d8e77ba3cf17c516e7d0c93e45a73a5e54b1b245ddb65351826678fe102d1",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.min.vec":
    "13fac5abbd2053365c5570edea2017e2a6d814e682a8e906d92b3deaa761b741",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ml.vec":
    "69eafbab72db69278acec01ff8883d41d616f8aaa59e473faafc115996db5898",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.mn.vec":
    "a1ec46e780d2f42633ffbe363ce17b1f700fa6744ce40b5a19446a714b9066d8",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.mr.vec":
    "e30ee3d90d6687868cc6dee609e4d487b81362ea231e8456f8265bace55c7ffb",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ms.vec":
    "71ebc8bc0959a592e071db35995119ee33fc17ff61611e6ea09ea6736b317f17",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.mrj.vec":
    "93351fb85f38523fbf3767fac32625f26be37582afbddfef258642f4530f4ab9",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.my.vec":
    "5a8216d0df2d70e5517bcb4cbe523fc03d34f802a83d04a88faadfff7b700b9f",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.mwl.vec":
    "6997a71b0a745c124135d6c52196d14412d4068fca8aa13b2b3b9598b933cf38",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.mt.vec":
    "f07a6071fcb3bcda4c6c5e6a0ebe6f3f5d228e8c1fc7ef5160cc3dd098718e98",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.myv.vec":
    "ffebdb940b95fe76f3885e8853f3d88ebf9a23c24f64ccdf52c9a269a3f4d459",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.nah.vec":
    "2b1fc52e4a4901d824070d1e5fc2196f33c6d787edb8ce3732ace1d05407788e",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.mzn.vec":
    "692f68fa5537a690720f9c2ce0a2c5edaa0d06fe04b2749d169a178aecf751ad",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.nap.vec":
    "954aa926c6d47882c2397997d57a8afde3e0ca851a42b07280d6e465577f6925",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ne.vec":
    "8d4bf875ca4733d022d4f415777dc2f9e33a93ddc67361add30aed298bc41bc6",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.nds.vec":
    "767dcf37a6018cce9f885b31b3c54671199c0f9554ffb09112130b62144556db",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.nl.vec":
    "d0601975d00d672ad03a3b146c13c4b6240111d286834e385853e2a25f4fb66a",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.new.vec":
    "b7328d5408d91bbdc7ee7a9fd6761af322ea8ddb35a405a60826a5b7e327dd29",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.nn.vec":
    "c2d2617c932bb49ba64bea9396435ce882fc4238e3983081967658891d18309e",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.no.vec":
    "762670d35c29910a0daa86444a1b32d4fd9c94deff82c53abe751c5463dcb025",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.or.vec":
    "b1d97ba3d93b37903266b551e164fc9e51c7d5a429e77330cb281fb7de28bd71",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.os.vec":
    "c249450a7cb5750c39a9121658b91055a0f5cccfe67c1879706a8bced390bebd",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.oc.vec":
    "a4bb95b2fc28e82c5c976af32d632e5241daeeaea2bca2cb3300ad036619c0f6",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.pam.vec":
    "b0dd33c3f7e85805b1937d95d73194f3834f40a43a92c12544911ab30818cd20",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.pa.vec":
    "61462550fac53d8156c2e61f738b63ef0639949b87d8abeb566194dc86b1a488",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.pl.vec":
    "9c2674431e796595c8c1d3b5e1a941c7e833d23cad223d6e4d1c36447af5f3cc",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.pfl.vec":
    "889a62dbb945033bfc53516b976042df7791c0aa8290dcb92f12240685d2d2c1",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.pnb.vec":
    "7c26c9297b15a75bb1f2bfeb8f11dd3c55821a06bd64fe7a105699ed4d9d794a",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ps.vec":
    "e718dfda7790cb309e37f0d42400afebf6036aa018dcd7eb330d576bb5c55030",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.qu.vec":
    "b71076861dc0221acf540d4abbf6e760a2871c2dc380556fc7bad402d26ec738",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.pms.vec":
    "33a90387e8b75c09980b4c80171cabaae38e9b87de7bf320ecd93c344afaeb39",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.rm.vec":
    "9a7f0690c8b42c96a1ad50bb8e7da5d69a3a9f7f0676289243319553a10aac41",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ro.vec":
    "e19b3e99a6eae03c15dc5f5d7385beb2540528ee102501499b7ca846c2687d83",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.pt.vec":
    "bffbfcafb9f004f13f1be12fa0201c5011324b14a52c2996ae2b97f268819e0c",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.rue.vec":
    "cb0aa15cb7816337509ed1b95c8844928a38d29e392e4e5295f35593e633b222",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.sa.vec":
    "6a303056d4841496599595be06fdcdf28ab5a2fc611c3428d95a3af9d4df0067",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ru.vec":
    "9567b90e037c459eb4be4c2a47a04fffbfd5b0d01b84baf86b16535f0dc3728e",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.sah.vec":
    "670a7c98a6c4bf6444b1a26213a5c8114d41c68006b4f32f6dee96558494076d",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.sc.vec":
    "52abeb74f579f53b3c8bb55ae5cd8bbf8878c7083e61c693c0f7c8d289e80248",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.scn.vec":
    "ad8e57aba916c6ab571157c02098ad1519c8f6ce1e72f35538efe1cb488a1a25",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.sd.vec":
    "7906a45f27aa65ba3d5cb034f56d2852d54d9ec3301b9df345f1a12a6cef9d7a",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.sco.vec":
    "eafc948e3e9e20aac5e7986979b3b3275c1acb2944e07b9b58d964da61408ff7",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.sh.vec":
    "36dc95a0fc0de137421df0b86eb7c55faff04d30b26969ae1fa331631824276d",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.sk.vec":
    "dd9f51e48a55fe63c5cf901c9ce0bd6baab249ac51135a1b4cdb4e12f164687b",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.sl.vec":
    "2ab76744a9d5321b6709b4ff379fb10495e004f72f6f221965028d6ee1cffd1e",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.so.vec":
    "28025afd6be6c8166898af85eb33536b111753fbf30e363beb7c064674c6d3c4",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.si.vec":
    "112246c583380fcf367932b55e5d42d5ffc12b8c206f981deae24fd4c61b7416",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.sq.vec":
    "4b4850d700aa1674e44bf733d6e2f83763b1ce9f0e7dfd524eb2c1b29c782631",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.sr.vec":
    "713f24f861cf540e3e28882915a89023cde222b6edb28fac7fb45b9bd894042e",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.sv.vec":
    "d21b96312bcf64faf1cd972d6eff44cd4a5afc575ff5c8f9b31d2d8819f56fca",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.su.vec":
    "b763d1c6471320b071ad2009a82dc6fb0ffeaf07319562438f84cfcb2718e2a4",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.sw.vec":
    "b9e17565d44cfba3c120274fd359b371c3b8d969b973e77ada3357defa843c79",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.te.vec":
    "2d684ba8af330a716f732f9581c7faee80322232e02713d441130f304af8a897",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.tg.vec":
    "e2ed18d08da76bff25f2170452365aa341e967114a45271a8ba8d9cefc062aef",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.th.vec":
    "079fadf992d34ae885ce5d7c23baa10aea4ee971147993b007d8bf0557906a18",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ta.vec":
    "a3cedbf2ce4adb5a8b3688539ef37c6c59047d8d20ffd74e2e384ffcac588ac1",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.tk.vec":
    "05f0ccf5f6a2bdf6073e16f11c7a2327ebe4d12610af44051872d4fea32591ec",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.tl.vec":
    "a524621cefca337c5b83e6a2849afd12100fcd59bd7f3b228bddb4fb95cb17ea",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.tr.vec":
    "4cf567dbb73053bb7b08370e89ec6a7c5626e397e71de99637e70c68ba4c71d9",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.tt.vec":
    "6dc86b913c0375b204f1c8d7c8543d80888030693ed4ebef10c75e358c17d0fa",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.tyv.vec":
    "b8a687337b3e7f344b9aecff19306c7a1cb432cdc03b46fd2f2e9e376be3073c",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.uk.vec":
    "186523ce3be943f9ecae127155371c494192564d1dffe743ab5db8ba28e50874",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ug.vec":
    "04184a3a6be245e55f09c04856acc14f687adc4b802aaf875bf5883a1669a856",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ur.vec":
    "df38c3cf123edf09366f47ea694c02dec59929df218ca81d5aa69d77552b6865",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.uz.vec":
    "f6289fa8cf2ff936a1716a5cf8fd07da46907af26b1236403a292273f2d8fb55",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.vec.vec":
    "c6d786f4231f30b4116a8dce181b2513b40b55a654c60793a5c0566152287aeb",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.vls.vec":
    "2f430e1d83f0f00fef517f7d35505bcf1445dc7de0db4f051ae7315f1bb0647b",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.vep.vec":
    "81268d74e29bbae9f166d523151d12c246ff26be9cd680344faece7e1ca97ebe",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.vi.vec":
    "206206d496697de7e96c69500e926014c9f71c7115c3844350766ced21d7003f",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.war.vec":
    "51b58d0ace2779b17b88a5b51847a813042e2b018ada573e0bce5a093da5ff4d",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.wa.vec":
    "37aee21a768a5883f6bee4a486040883224a93619b8b03dcefb1e939b655cd1c",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.vo.vec":
    "4fa6a6ff897a1a49470861d343792feac0ca16e02e9ed1917f1506245ac28b2d",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.wuu.vec":
    "09a619a8ef25392bf8905d741cdb63922a115e05b38f56de27c339985691c5d2",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.xal.vec":
    "bf9ad172c55d8910e0156953158a9cb1f9cbcc6b9b1e78cf09c123d3409af5e3",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.yi.vec":
    "75dc1cad2a4dad5ad7d7723ef0b8e87abe3f4b799e9c38c54f4afe51d916a82b",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.yo.vec":
    "c8aa49859debb8b3d1568bb510e12814d55ce5994f0cc6dc43ca9b2c4f739946",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.xmf.vec":
    "94ffed6fc1123523d72e3b92d0d3cc5513c116b9e9b2bba5d8b47f7b6fce6abd",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.yue.vec":
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.zh.vec":
    "76f72bd13269ae492715415ef62afb109046ce557f5af24e822b71f9b9360bef"
}
