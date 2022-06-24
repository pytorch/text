def __getattr__(name):

    moved_apis = [
        "basic_english_normalize",
        "BasicEnglishNormalize",
        "PRETRAINED_SP_MODEL",
        "load_sp_model",
        "sentencepiece_tokenizer",
        "SentencePieceTokenizer",
        "sentencepiece_processor",
        "SentencePieceProcessor",
        "VocabTransform",
        "VectorTransform",
    ]

    if name in moved_apis:
        import warnings

        warnings.warn(
            "experimental package has been moved to prototype. You may change all imports from `torchtext.experimental` to `torchtext.prototype`",
            UserWarning,
        )

        if name == "basic_english_normalize":
            from torchtext.prototype.transforms import basic_english_normalize

            return basic_english_normalize
        elif name == "BasicEnglishNormalize":
            from torchtext.prototype.transforms import BasicEnglishNormalize

            return BasicEnglishNormalize
        elif name == "PRETRAINED_SP_MODEL":
            from torchtext.prototype.transforms import PRETRAINED_SP_MODEL

            return PRETRAINED_SP_MODEL
        elif name == "load_sp_model":
            from torchtext.prototype.transforms import load_sp_model

            return load_sp_model
        elif name == "sentencepiece_tokenizer":
            from torchtext.prototype.transforms import sentencepiece_tokenizer

            return sentencepiece_tokenizer
        elif name == "SentencePieceTokenizer":
            from torchtext.prototype.transforms import SentencePieceTokenizer

            return SentencePieceTokenizer
        elif name == "sentencepiece_processor":
            from torchtext.prototype.transforms import sentencepiece_processor

            return sentencepiece_processor
        elif name == "VocabTransform":
            from torchtext.prototype.transforms import VocabTransform

            return VocabTransform
        else:
            from torchtext.prototype.transforms import VectorTransform

            return VectorTransform

    raise AttributeError(f"module {__name__} has no attribute {name}")
