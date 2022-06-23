def __getattr__(name):
    if name == "build_vocab_from_text_file":
        import warnings

        from torchtext.prototype.vocab_factory import build_vocab_from_text_file

        warnings.warn(
            "experimental package has been moved to prototype. You may change all imports from `experimental` to `prototype`",
            UserWarning,
        )

        return build_vocab_from_text_file

    raise AttributeError(f"module {__name__} has no attribute {name}")
