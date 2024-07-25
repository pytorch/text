def __getattr__(name):

    moved_apis = ["FastText", "GloVe", "load_vectors_from_file_path", "build_vectors", "Vectors"]

    if name in moved_apis:
        import warnings

        warnings.warn(
            "experimental package has been moved to prototype. You may change all imports from `torchtext.experimental` to `torchtext.prototype`",
            UserWarning,
        )

        if name == "FastText":
            from torchtext.prototype.vectors import FastText

            return FastText
        elif name == "GloVe":
            from torchtext.prototype.vectors import GloVe

            return GloVe
        elif name == "load_vectors_from_file_path":
            from torchtext.prototype.vectors import load_vectors_from_file_path

            return load_vectors_from_file_path
        elif name == "build_vectors":
            from torchtext.prototype.vectors import build_vectors

            return build_vectors
        else:
            from torchtext.prototype.vectors import Vectors

            return Vectors

    raise AttributeError(f"module {__name__} has no attribute {name}")
