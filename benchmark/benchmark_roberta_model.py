from argparse import ArgumentParser

import torch
import torchtext
from benchmark.utils import Timer
from torchtext.functional import to_tensor


def _train(model):
    model_input = torch.tensor(
        [
            [
                0,
                1,
                2,
                3,
                4,
                5,
            ]
        ]
    )
    logits = model(model_input)
    loss = torch_F.cross_entropy(logits, target)
    loss.backward()


def run(args):
    encoder = args.encoder
    num_passes = args.num_passes

    if encoder == "xlmr_base":
        encoder = torchtext.models.XLMR_BASE_ENCODER
    elif encoder == "xlmr_large":
        encoder = torchtext.models.XLMR_LARGE_ENCODER
    elif encoder == "roberta_base":
        encoder = torchtext.models.ROBERTA_BASE_ENCODER
    elif encoder == "roberta_large":
        encoder = torchtext.models.ROBERTA_LARGE_ENCODER

    model = encoder.get_model()
    transform = encoder.transform()
    input_batch = ["Hello world", "How are you!"]
    model_input = to_tensor(transform(input_batch), padding_value=1)

    model.eval()

    with Timer("Executing model forward"):
        for _ in range(num_passes):
            _ = model(model_input)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--encoder", default="xlmr_base", type=str)
    parser.add_argument("--num-passes", default=10, type=int)
    run(parser.parse_args())
