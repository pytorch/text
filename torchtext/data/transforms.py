import torch.nn as nn
import sentencepiece as spm
from .functional import sentencepiece_encode_as_ids


class SentencePieceTransform(nn.Module):
    """A tokenizer transform based on SentencePiece model.

    Arguments:
        filename: the data file for saving SentencePiece model.

    Examples:
        >>> from torchtext.data.transforms import SentencePieceTransform
        >>> sp_transform = SentencePieceTransform('m_user.model')
        >>> sp_transform('SentencePiece is an unsupervised text tokenizer')
    """

    def __init__(self, filename):
        super(SentencePieceTransform, self).__init__()
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(filename)

    def forward(self, txt_str):
        """Transform a sentence text into a list of integer.

        Arguments:
            txt_str: input sentence text.

        Outputs:
            output: a list of integers based on SentencePiece model.
        """

        return sentencepiece_encode_as_ids(self.sp_model, txt_str)
