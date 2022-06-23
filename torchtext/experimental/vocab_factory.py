def __getattr__(name):
    moved_apis = ["build_vocab_from_text_file", "load_vocab_from_file"]
    if name in moved_apis:
        import warnings

        warnings.warn(
            "experimental package has been moved to prototype. You may change all imports from `torchtext.experimental` to `torchtext.prototype`",
            UserWarning,
        )

        if name == "build_vocab_from_text_file":
            from torchtext.prototype.vocab_factory import build_vocab_from_text_file

            return build_vocab_from_text_file
        else:
            from torchtext.prototype.vocab_factory import load_vocab_from_file

            return load_vocab_from_file

    raise AttributeError(f"module {__name__} has no attribute {name}")
