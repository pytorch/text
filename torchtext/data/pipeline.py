class Pipeline(object):
    """Defines a pipeline for transforming sequence data."""

    def __init__(self, convert_token=None):
        if convert_token is None:
            self.convert_token = lambda x: x
        elif callable(convert_token):
            self.convert_token = convert_token
        else:
            raise ValueError("Pipeline input convert_token {} is not None "
                             "or callable".format(convert_token))
        self.pipes = [self]

    def __call__(self, x, *args):
        for pipe in self.pipes:
            x = pipe.call(x)
        return x

    def call(self, x, *args):
        if isinstance(x, list):
            return [self(tok, *args) for tok in x]
        return self.convert_token(x, *args)

    def add_before(self, pipeline):
        """Add `pipeline` before this processing pipeline."""
        if not isinstance(pipeline, Pipeline):
            pipeline = Pipeline(pipeline)
        self.pipes = pipeline.pipes[:] + self.pipes[:]
        return self

    def add_after(self, pipeline):
        """Add `pipeline` after this processing pipeline."""
        if not isinstance(pipeline, Pipeline):
            pipeline = Pipeline(pipeline)
        self.pipes = self.pipes[:] + pipeline.pipes[:]
        return self
