from dataclasses import dataclass
from functools import partial
from urllib.parse import urljoin

from typing import Optional, Callable, Dict, Union, Any
from torchtext._download_hooks import load_state_dict_from_url
from torch.nn import Module
import torch
import logging
import re
logger = logging.getLogger(__name__)

from .model import (
    RobertaEncoderConf,
    RobertaModel,
    _get_model,
)

from .transforms import get_xlmr_transform

from torchtext import _TEXT_BUCKET


def _is_head_available_in_checkpoint(checkpoint, head_state_dict):
    # ensure all keys are present
    return all(key in checkpoint.keys() for key in head_state_dict.keys())


@dataclass
class RobertaModelBundle:
    """RobertaModelBundle(_params: torchtext.models.RobertaEncoderParams, _path: Optional[str] = None, _head: Optional[torch.nn.Module] = None, transform: Optional[Callable] = None)

    Example - Pretrained encoder
        >>> import torch, torchtext
        >>> xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        >>> model = xlmr_base.get_model()
        >>> transform = xlmr_base.transform()
        >>> model_input = torch.tensor(transform(["Hello World"]))
        >>> output = model(model_input)
        >>> output.shape
        torch.Size([1, 4, 768])
        >>> input_batch = ["Hello world", "How are you!"]
        >>> from torchtext.functional import to_tensor
        >>> model_input = to_tensor(transform(input_batch), padding_value=transform.pad_idx)
        >>> output = model(model_input)
        >>> output.shape
        torch.Size([2, 6, 768])

    Example - Pretrained encoder attached to un-initialized classification head
        >>> import torch, torchtext
        >>> xlmr_large = torchtext.models.XLMR_LARGE_ENCODER
        >>> classifier_head = torchtext.models.RobertaClassificationHead(num_classes=2, input_dim = xlmr_large.encoderConf.embedding_dim)
        >>> classification_model = xlmr_large.get_model(head=classifier_head)
        >>> transform = xlmr_large.transform()
        >>> model_input = torch.tensor(transform(["Hello World"]))
        >>> output = classification_model(model_input)
        >>> output.shape
        torch.Size([1, 2])

    Example - User-specified configuration and checkpoint
        >>> from torchtext.models import RobertaEncoderConf, RobertaModelBundle, RobertaClassificationHead
        >>> model_weights_path = "https://download.pytorch.org/models/text/xlmr.base.encoder.pt"
        >>> roberta_encoder_conf = RobertaEncoderConf(vocab_size=250002)
        >>> roberta_bundle = RobertaModelBundle(_encoder_conf=roberta_encoder_conf, _path=model_weights_path)
        >>> encoder = roberta_bundle.get_model()
        >>> classifier_head = RobertaClassificationHead(num_classes=2, input_dim=768)
        >>> classifier = roberta_bundle.get_model(head=classifier_head)
        >>> # using from_config
        >>> encoder = RobertaModelBundle.from_config(config=roberta_encoder_conf, checkpoint=model_weights_path)
        >>> classifier = RobertaModelBundle.from_config(config=roberta_encoder_conf, head=classifier_head, checkpoint=model_weights_path)
    """
    _encoder_conf: RobertaEncoderConf
    _path: Optional[str] = None
    _head: Optional[Module] = None
    transform: Optional[Callable] = None

    def get_model(self,
                  head: Optional[Module] = None,
                  load_weights: bool = True,
                  freeze_encoder: bool = False,
                  *,
                  dl_kwargs=None) -> RobertaModel:
        r"""get_model(head: Optional[torch.nn.Module] = None, load_weights: bool = True, freeze_encoder: bool = False, *, dl_kwargs=None) -> torctext.models.RobertaModel

        Args:
            head (nn.Module): A module to be attached to the encoder to perform specific task. If provided, it will replace the default member head (Default: ``None``)
            freeze_encoder (bool): Indicates whether to freeze the encoder weights. (Default: ``False``)
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`. (Default: ``None``)
        """

        if load_weights:
            assert self._path is not None, "load_weights cannot be True. The pre-trained model weights are not available for the current object"

        if freeze_encoder:
            if not load_weights or not self._path:
                logger.warn("The encoder is not loaded with pre-trained weights. Setting freeze_encoder to True will hinder encoder from learning appropriate weights.")

        if head is not None:
            input_head = head
            if self._head is not None:
                logger.log("A custom head module was provided, discarding the default head module.")
        else:
            input_head = self._head

        return RobertaModelBundle.from_config(encoder_conf=self._encoder_conf,
                                              head=input_head,
                                              freeze_encoder=freeze_encoder,
                                              checkpoint=self._path,
                                              override_head=True,
                                              dl_kwargs=dl_kwargs)

    @classmethod
    def from_config(
        cls,
        encoder_conf: RobertaEncoderConf,
        head: Optional[Module] = None,
        freeze_encoder: bool = False,
        checkpoint: Optional[Union[str, Dict[str, torch.Tensor]]] = None,
        *,
        override_head: bool = False,
        dl_kwargs: Dict[str, Any] = None,
    ) -> RobertaModel:
        """Class method to create model with user-defined encoder configuration and checkpoint

        Args:
            encoder_conf (RobertaEncoderConf): An instance of class RobertaEncoderConf that defined the encoder configuration
            head (nn.Module): A module to be attached to the encoder to perform specific task. (Default: ``None``)
            freeze_encoder (bool): Indicates whether to freeze the encoder weights. (Default: ``False``)
            checkpoint (str or Dict[str, torch.Tensor]): Path to or actual model state_dict. state_dict can have partial weights i.e only for encoder. (Default: ``None``)
            override_head (bool): Override the checkpoint's head state dict (if present) with provided head state dict. (Default: ``False``)
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`. (Default: ``None``)
        """
        model = _get_model(encoder_conf, head, freeze_encoder)
        if checkpoint is not None:
            if torch.jit.isinstance(checkpoint, Dict[str, torch.Tensor]):
                state_dict = checkpoint
            elif isinstance(checkpoint, str):
                dl_kwargs = {} if dl_kwargs is None else dl_kwargs
                state_dict = load_state_dict_from_url(checkpoint, **dl_kwargs)
            else:
                raise TypeError("checkpoint must be of type `str` or `Dict[str, torch.Tensor]` but got {}".format(type(checkpoint)))

            if head is not None:
                regex = re.compile(r"^head\.")
                head_state_dict = {k: v for k, v in model.state_dict().items() if regex.findall(k)}
                # If checkpoint does not contains head_state_dict, then we augment the checkpoint with user-provided head state_dict
                if not _is_head_available_in_checkpoint(state_dict, head_state_dict) or override_head:
                    state_dict.update(head_state_dict)

            model.load_state_dict(state_dict, strict=True)

        return model

    @property
    def encoderConf(self) -> RobertaEncoderConf:
        return self._encoder_conf


XLMR_BASE_ENCODER = RobertaModelBundle(
    _path=urljoin(_TEXT_BUCKET, "xlmr.base.encoder.pt"),
    _encoder_conf=RobertaEncoderConf(vocab_size=250002),
    transform=partial(get_xlmr_transform,
                      vocab_path=urljoin(_TEXT_BUCKET, "xlmr.vocab.pt"),
                      spm_model_path=urljoin(_TEXT_BUCKET, "xlmr.sentencepiece.bpe.model"),
                      )
)

XLMR_BASE_ENCODER.__doc__ = (
    '''
    XLM-R Encoder with base configuration

    Please refer to :func:`torchtext.models.RobertaModelBundle` for the usage.
    '''
)


XLMR_LARGE_ENCODER = RobertaModelBundle(
    _path=urljoin(_TEXT_BUCKET, "xlmr.large.encoder.pt"),
    _encoder_conf=RobertaEncoderConf(vocab_size=250002, embedding_dim=1024, ffn_dimension=4096, num_attention_heads=16, num_encoder_layers=24),
    transform=partial(get_xlmr_transform,
                      vocab_path=urljoin(_TEXT_BUCKET, "xlmr.vocab.pt"),
                      spm_model_path=urljoin(_TEXT_BUCKET, "xlmr.sentencepiece.bpe.model"),
                      )
)

XLMR_LARGE_ENCODER.__doc__ = (
    '''
    XLM-R Encoder with Large configuration

    Please refer to :func:`torchtext.models.RobertaModelBundle` for the usage.
    '''
)
