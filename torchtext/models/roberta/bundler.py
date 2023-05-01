import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union
from urllib.parse import urljoin

import torch
from torch.nn import Module
from torchtext._download_hooks import load_state_dict_from_url

logger = logging.getLogger(__name__)

import torchtext.transforms as T
from torchtext import _TEXT_BUCKET

from .model import RobertaEncoderConf, RobertaModel


def _is_head_available_in_checkpoint(checkpoint, head_state_dict):
    # ensure all keys are present
    return all(key in checkpoint.keys() for key in head_state_dict.keys())


@dataclass
class RobertaBundle:
    """RobertaBundle(_params: torchtext.models.RobertaEncoderParams, _path: Optional[str] = None, _head: Optional[torch.nn.Module] = None, transform: Optional[Callable] = None)

    Example - Pretrained base xlmr encoder
        >>> import torch, torchtext
        >>> from torchtext.functional import to_tensor
        >>> xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        >>> model = xlmr_base.get_model()
        >>> transform = xlmr_base.transform()
        >>> input_batch = ["Hello world", "How are you!"]
        >>> model_input = to_tensor(transform(input_batch), padding_value=1)
        >>> output = model(model_input)
        >>> output.shape
        torch.Size([2, 6, 768])

    Example - Pretrained large xlmr encoder attached to un-initialized classification head
        >>> import torch, torchtext
        >>> from torchtext.models import RobertaClassificationHead
        >>> from torchtext.functional import to_tensor
        >>> xlmr_large = torchtext.models.XLMR_LARGE_ENCODER
        >>> classifier_head = torchtext.models.RobertaClassificationHead(num_classes=2, input_dim = 1024)
        >>> model = xlmr_large.get_model(head=classifier_head)
        >>> transform = xlmr_large.transform()
        >>> input_batch = ["Hello world", "How are you!"]
        >>> model_input = to_tensor(transform(input_batch), padding_value=1)
        >>> output = model(model_input)
        >>> output.shape
        torch.Size([1, 2])

    Example - User-specified configuration and checkpoint
        >>> from torchtext.models import RobertaEncoderConf, RobertaBundle, RobertaClassificationHead
        >>> model_weights_path = "https://download.pytorch.org/models/text/xlmr.base.encoder.pt"
        >>> encoder_conf = RobertaEncoderConf(vocab_size=250002)
        >>> classifier_head = RobertaClassificationHead(num_classes=2, input_dim=768)
        >>> model = RobertaBundle.build_model(encoder_conf=encoder_conf, head=classifier_head, checkpoint=model_weights_path)
    """

    _encoder_conf: RobertaEncoderConf
    _path: Optional[str] = None
    _head: Optional[Module] = None
    transform: Optional[Callable] = None

    def get_model(
        self,
        *,
        head: Optional[Module] = None,
        load_weights: bool = True,
        freeze_encoder: bool = False,
        dl_kwargs: Dict[str, Any] = None,
    ) -> RobertaModel:
        r"""get_model(head: Optional[torch.nn.Module] = None, load_weights: bool = True, freeze_encoder: bool = False, *, dl_kwargs=None) -> torchtext.models.RobertaModel

        Args:
            head (nn.Module): A module to be attached to the encoder to perform specific task. If provided, it will replace the default member head (Default: ``None``)
            load_weights (bool): Indicates whether or not to load weights if available. (Default: ``True``)
            freeze_encoder (bool): Indicates whether or not to freeze the encoder weights. (Default: ``False``)
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`. (Default: ``None``)
        """

        if load_weights:
            assert (
                self._path is not None
            ), "load_weights cannot be True. The pre-trained model weights are not available for the current object"

        if freeze_encoder:
            if not load_weights or not self._path:
                logger.warn(
                    "The encoder is not loaded with pre-trained weights. Setting freeze_encoder to True will hinder encoder from learning appropriate weights."
                )

        if head is not None:
            input_head = head
            if self._head is not None:
                logger.log("A custom head module was provided, discarding the default head module.")
        else:
            input_head = self._head

        return RobertaBundle.build_model(
            encoder_conf=self._encoder_conf,
            head=input_head,
            freeze_encoder=freeze_encoder,
            checkpoint=self._path if load_weights else None,
            override_checkpoint_head=True,
            strict=False,
            dl_kwargs=dl_kwargs,
        )

    @classmethod
    def build_model(
        cls,
        encoder_conf: RobertaEncoderConf,
        *,
        head: Optional[Module] = None,
        freeze_encoder: bool = False,
        checkpoint: Optional[Union[str, Dict[str, torch.Tensor]]] = None,
        override_checkpoint_head: bool = False,
        strict=False,
        dl_kwargs: Dict[str, Any] = None,
    ) -> RobertaModel:
        """Class builder method

        Args:
            encoder_conf (RobertaEncoderConf): An instance of class RobertaEncoderConf that defined the encoder configuration
            head (nn.Module): A module to be attached to the encoder to perform specific task. (Default: ``None``)
            freeze_encoder (bool): Indicates whether to freeze the encoder weights. (Default: ``False``)
            checkpoint (str or Dict[str, torch.Tensor]): Path to or actual model state_dict. state_dict can have partial weights i.e only for encoder. (Default: ``None``)
            override_checkpoint_head (bool): Override the checkpoint's head state dict (if present) with provided head state dict. (Default: ``False``)
            strict (bool): Passed to :func: `torch.nn.Module.load_state_dict` method. (Default: ``True``)
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`. (Default: ``None``)
        """
        model = RobertaModel(encoder_conf, head, freeze_encoder)
        if checkpoint is not None:
            if torch.jit.isinstance(checkpoint, Dict[str, torch.Tensor]):
                state_dict = checkpoint
            elif isinstance(checkpoint, str):
                dl_kwargs = {} if dl_kwargs is None else dl_kwargs
                state_dict = load_state_dict_from_url(checkpoint, **dl_kwargs)
            else:
                raise TypeError(
                    "checkpoint must be of type `str` or `Dict[str, torch.Tensor]` but got {}".format(type(checkpoint))
                )

            if head is not None:
                regex = re.compile(r"^head\.")
                head_state_dict = {k: v for k, v in model.state_dict().items() if regex.findall(k)}
                # If checkpoint does not contains head_state_dict, then we augment the checkpoint with user-provided head state_dict
                if not _is_head_available_in_checkpoint(state_dict, head_state_dict) or override_checkpoint_head:
                    state_dict.update(head_state_dict)

            model.load_state_dict(state_dict, strict=strict)

        return model

    @property
    def encoderConf(self) -> RobertaEncoderConf:
        return self._encoder_conf


def xlmr_transform(truncate_length: int) -> Module:
    """Standard transform for XLMR models."""
    return T.Sequential(
        T.SentencePieceTokenizer(urljoin(_TEXT_BUCKET, "xlmr.sentencepiece.bpe.model")),
        T.VocabTransform(load_state_dict_from_url(urljoin(_TEXT_BUCKET, "xlmr.vocab.pt"))),
        T.Truncate(truncate_length),
        T.AddToken(token=0, begin=True),
        T.AddToken(token=2, begin=False),
    )


def roberta_transform(truncate_length: int) -> Module:
    """Standard transform for RoBERTa models."""
    return T.Sequential(
        T.GPT2BPETokenizer(
            encoder_json_path=urljoin(_TEXT_BUCKET, "gpt2_bpe_encoder.json"),
            vocab_bpe_path=urljoin(_TEXT_BUCKET, "gpt2_bpe_vocab.bpe"),
        ),
        T.VocabTransform(load_state_dict_from_url(urljoin(_TEXT_BUCKET, "roberta.vocab.pt"))),
        T.Truncate(truncate_length),
        T.AddToken(token=0, begin=True),
        T.AddToken(token=2, begin=False),
    )


XLMR_BASE_ENCODER = RobertaBundle(
    _path=urljoin(_TEXT_BUCKET, "xlmr.base.encoder.pt"),
    _encoder_conf=RobertaEncoderConf(vocab_size=250002),
    transform=lambda: xlmr_transform(254),
)

XLMR_BASE_ENCODER.__doc__ = """
    XLM-R Encoder with Base configuration

    The XLM-RoBERTa model was proposed in `Unsupervised Cross-lingual Representation Learning
    at Scale <https://arxiv.org/abs/1911.02116>`. It is a large multi-lingual language model,
    trained on 2.5TB of filtered CommonCrawl data and based on the RoBERTa model architecture.

    Originally published by the authors of XLM-RoBERTa under MIT License
    and redistributed with the same license.
    [`License <https://github.com/pytorch/fairseq/blob/main/LICENSE>`__,
    `Source <https://github.com/pytorch/fairseq/tree/main/examples/xlmr#pre-trained-models>`__]

    Please refer to :func:`torchtext.models.RobertaBundle` for the usage.
    """


XLMR_LARGE_ENCODER = RobertaBundle(
    _path=urljoin(_TEXT_BUCKET, "xlmr.large.encoder.pt"),
    _encoder_conf=RobertaEncoderConf(
        vocab_size=250002, embedding_dim=1024, ffn_dimension=4096, num_attention_heads=16, num_encoder_layers=24
    ),
    transform=lambda: xlmr_transform(510),
)

XLMR_LARGE_ENCODER.__doc__ = """
    XLM-R Encoder with Large configuration

    The XLM-RoBERTa model was proposed in `Unsupervised Cross-lingual Representation Learning
    at Scale <https://arxiv.org/abs/1911.02116>`. It is a large multi-lingual language model,
    trained on 2.5TB of filtered CommonCrawl data and based on the RoBERTa model architecture.

    Originally published by the authors of XLM-RoBERTa under MIT License
    and redistributed with the same license.
    [`License <https://github.com/pytorch/fairseq/blob/main/LICENSE>`__,
    `Source <https://github.com/pytorch/fairseq/tree/main/examples/xlmr#pre-trained-models>`__]

    Please refer to :func:`torchtext.models.RobertaBundle` for the usage.
    """


ROBERTA_BASE_ENCODER = RobertaBundle(
    _path=urljoin(_TEXT_BUCKET, "roberta.base.encoder.pt"),
    _encoder_conf=RobertaEncoderConf(vocab_size=50265),
    transform=lambda: roberta_transform(254),
)

ROBERTA_BASE_ENCODER.__doc__ = """
    Roberta Encoder with Base configuration

    RoBERTa iterates on BERT's pretraining procedure, including training the model longer,
    with bigger batches over more data; removing the next sentence prediction objective;
    training on longer sequences; and dynamically changing the masking pattern applied
    to the training data.

    The RoBERTa model was pretrained on the reunion of five datasets: BookCorpus,
    English Wikipedia, CC-News, OpenWebText, and STORIES. Together theses datasets
    contain over a 160GB of text.

    Originally published by the authors of RoBERTa under MIT License
    and redistributed with the same license.
    [`License <https://github.com/pytorch/fairseq/blob/main/LICENSE>`__,
    `Source <https://github.com/pytorch/fairseq/tree/main/examples/roberta#pre-trained-models>`__]

    Please refer to :func:`torchtext.models.RobertaBundle` for the usage.
    """


ROBERTA_LARGE_ENCODER = RobertaBundle(
    _path=urljoin(_TEXT_BUCKET, "roberta.large.encoder.pt"),
    _encoder_conf=RobertaEncoderConf(
        vocab_size=50265,
        embedding_dim=1024,
        ffn_dimension=4096,
        num_attention_heads=16,
        num_encoder_layers=24,
    ),
    transform=lambda: roberta_transform(510),
)

ROBERTA_LARGE_ENCODER.__doc__ = """
    Roberta Encoder with Large configuration

    RoBERTa iterates on BERT's pretraining procedure, including training the model longer,
    with bigger batches over more data; removing the next sentence prediction objective;
    training on longer sequences; and dynamically changing the masking pattern applied
    to the training data.

    The RoBERTa model was pretrained on the reunion of five datasets: BookCorpus,
    English Wikipedia, CC-News, OpenWebText, and STORIES. Together theses datasets
    contain over a 160GB of text.

    Originally published by the authors of RoBERTa under MIT License
    and redistributed with the same license.
    [`License <https://github.com/pytorch/fairseq/blob/main/LICENSE>`__,
    `Source <https://github.com/pytorch/fairseq/tree/main/examples/roberta#pre-trained-models>`__]

    Please refer to :func:`torchtext.models.RobertaBundle` for the usage.
    """


ROBERTA_DISTILLED_ENCODER = RobertaBundle(
    _path=urljoin(_TEXT_BUCKET, "roberta.distilled.encoder.pt"),
    _encoder_conf=RobertaEncoderConf(
        num_encoder_layers=6,
        padding_idx=1,
    ),
    transform=lambda: roberta_transform(510),
)

ROBERTA_DISTILLED_ENCODER.__doc__ = """
    Roberta Encoder with Distilled Weights

    DistilRoBERTa is trained using knowledge distillation, a technique to compress a large
    model called the teacher into a smaller model called the student. By distillating RoBERTa,
    a smaller and faster Transformer model is obtained while maintaining most of the performance.

    DistilRoBERTa was pretrained solely on OpenWebTextCorpus, a reproduction of OpenAI's WebText dataset.
    On average DistilRoBERTa is twice as fast as RoBERTa Base.

    Originally published by Hugging Face under the Apache 2.0 License
    and redistributed with the same license.
    [`License <https://www.apache.org/licenses/LICENSE-2.0>`__,
    `Source <https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation>`__]

    Please refer to :func:`torchtext.models.RobertaBundle` for the usage.
    """
