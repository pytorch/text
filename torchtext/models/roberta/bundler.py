from dataclasses import dataclass
from functools import partial
from urllib.parse import urljoin

from typing import Optional, Callable, Dict, Union, Any
from torchtext._download_hooks import load_state_dict_from_url
from torch.nn import Module
import torch
import logging

logger = logging.getLogger(__name__)

from .model import (
    RobertaEncoderConf,
    RobertaModel,
    _get_model,
)

from .transforms import get_xlmr_transform

from torchtext import _TEXT_BUCKET


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

        model = _get_model(self._encoder_conf, input_head, freeze_encoder)

        if not load_weights:
            return model

        dl_kwargs = {} if dl_kwargs is None else dl_kwargs
        state_dict = load_state_dict_from_url(self._path, **dl_kwargs)
        if input_head is not None:
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=True)
        return model

    @classmethod
    def from_config(
        self,
        config: RobertaEncoderConf,
        head: Optional[Module] = None,
        freeze_encoder: bool = False,
        checkpoint: Optional[Union[str, Dict[str, torch.Tensor]]] = None,
        *,
        dl_kwargs: Dict[str, Any] = None,
    ) -> RobertaModel:
        """Class method to intantiate model with user-defined encoder configuration and checkpoint

        Args:
            config (RobertaEncoderConf): An instance of class RobertaEncoderConf that defined the encoder configuration
            head (nn.Module, optional): A module to be attached to the encoder to perform specific task
            freeze_encoder (bool): Indicates whether to freeze the encoder weights
            checkpoint (str or Dict[str, torch.Tensor], optional): Path to or actual model state_dict. state_dict can have partial weights i.e only for encoder.
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`.
        """
        model = _get_model(config, head, freeze_encoder)
        if checkpoint is not None:
            if torch.jit.isinstance(checkpoint, Dict[str, torch.Tensor]):
                state_dict = checkpoint
            elif isinstance(checkpoint, str):
                dl_kwargs = {} if dl_kwargs is None else dl_kwargs
                state_dict = load_state_dict_from_url(checkpoint, **dl_kwargs)
            else:
                raise TypeError("checkpoint must be of type `str` or `Dict[str, torch.Tensor]` but got {}".format(type(checkpoint)))
            if head is not None:
                model.load_state_dict(state_dict, strict=False)
            else:
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
