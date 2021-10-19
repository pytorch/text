
import os
from dataclasses import dataclass
from functools import partial

from typing import Optional, Callable
from torch.hub import load_state_dict_from_url
from torch.nn import Module
import logging

logger = logging.getLogger(__name__)

from .model import (
    RobertaEncoderParams,
    RobertaModel,
    _get_model,
)

from .transforms import get_xlmr_transform

from torchtext import _TEXT_BUCKET


@dataclass
class RobertaModelBundle:
    """
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
        >>> classifier_head = torchtext.models.RobertaClassificationHead(num_classes=2, input_dim = xlmr_large.params.embedding_dim)
        >>> classification_model = xlmr_large.get_model(head=classifier_head)
        >>> transform = xlmr_large.transform()
        >>> model_input = torch.tensor(transform(["Hello World"]))
        >>> output = classification_model(model_input)
        >>> output.shape
        torch.Size([1, 2])
    """
    _params: RobertaEncoderParams
    _path: Optional[str] = None
    _head: Optional[Module] = None
    transform: Optional[Callable] = None

    def get_model(self, head: Optional[Module] = None, *, dl_kwargs=None) -> RobertaModel:

        if head is not None:
            input_head = head
            if self._head is not None:
                logger.log("A custom head module was provided, discarding the default head module.")
        else:
            input_head = self._head

        model = _get_model(self._params, input_head)

        dl_kwargs = {} if dl_kwargs is None else dl_kwargs
        state_dict = load_state_dict_from_url(self._path, **dl_kwargs)
        if input_head is not None:
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=True)
        return model

    @property
    def params(self) -> RobertaEncoderParams:
        return self._params


XLMR_BASE_ENCODER = RobertaModelBundle(
    _path=os.path.join(_TEXT_BUCKET, "xlmr.base.encoder.pt"),
    _params=RobertaEncoderParams(vocab_size=250002),
    transform=partial(get_xlmr_transform,
                      vocab_path=os.path.join(_TEXT_BUCKET, "xlmr.vocab.pt"),
                      spm_model_path=os.path.join(_TEXT_BUCKET, "xlmr.sentencepiece.bpe.model"),
                      )
)

XLMR_LARGE_ENCODER = RobertaModelBundle(
    _path=os.path.join(_TEXT_BUCKET, "xlmr.large.encoder.pt"),
    _params=RobertaEncoderParams(vocab_size=250002, embedding_dim=1024, ffn_dimension=4096, num_attention_heads=16, num_encoder_layers=24),
    transform=partial(get_xlmr_transform,
                      vocab_path=os.path.join(_TEXT_BUCKET, "xlmr.vocab.pt"),
                      spm_model_path=os.path.join(_TEXT_BUCKET, "xlmr.sentencepiece.bpe.model"),
                      )
)
