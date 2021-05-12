from torchtext.vocab import Vocab
from torchtext._torchtext import (
    _build_vocab_from_text_file
)


def build_vocab_from_text_file(file_path, jited_tokenizer, min_freq=1, num_cpus=4):
    r"""Create a `Vocab` object from a raw text file.

    The `file_path` can contain any raw text. This function applies a generic JITed tokenizer in
    parallel to the text.

    Args:
        file_object (FileObject): a file object to read data from.
        jited_tokenizer (ScriptModule): a tokenizer that has been JITed using `torch.jit.script`
        min_freq: The minimum frequency needed to include a token in the vocabulary.
            Values less than 1 will be set to 1. Default: 1.
        num_cpus (int): the number of cpus to use when loading the vectors from file. Default: 4.

    Returns:
        torchtext.vocab.Vocab: a `Vocab` object.

    Examples:
        >>> from torchtext.vocab import build_vocab_from_text_file
        >>> from torchtext.experimental.transforms import basic_english_normalize
        >>> tokenizer = basic_english_normalize()
        >>> tokenizer = basic_english_normalize()
        >>> jit_tokenizer = torch.jit.script(tokenizer)
        >>> v = build_vocab_from_text_file('vocab.txt', jit_tokenizer)
    """
    vocab_obj = _build_vocab_from_text_file(file_path, min_freq, num_cpus, jited_tokenizer)
    return Vocab(vocab_obj)
