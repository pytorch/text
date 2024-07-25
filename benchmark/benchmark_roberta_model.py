from argparse import ArgumentParser

import torch
from benchmark.utils import Timer
from torchtext.functional import to_tensor
from torchtext.models import XLMR_BASE_ENCODER, XLMR_LARGE_ENCODER, ROBERTA_BASE_ENCODER, ROBERTA_LARGE_ENCODER

ENCODERS = {
    "xlmr_base": XLMR_BASE_ENCODER,
    "xlmr_large": XLMR_LARGE_ENCODER,
    "roberta_base": ROBERTA_BASE_ENCODER,
    "roberta_large": ROBERTA_LARGE_ENCODER,
}


def basic_model_input(encoder):
    transform = encoder.transform()
    input_batch = ["Hello world", "How are you!"]
    return to_tensor(transform(input_batch), padding_value=1)


def _train(model, model_input):
    model_out = model(model_input)
    model_out.backward(torch.ones_like(model_out))
    model.zero_grad()


def run(args):
    encoder_name = args.encoder
    num_passes = args.num_passes
    warmup_passes = args.num_passes
    model_input = args.model_input

    encoder = ENCODERS.get(encoder_name, None)
    if not encoder:
        raise NotImplementedError("Given encoder [{}] is not available".format(encoder_name))

    model = encoder.get_model()
    if model_input == "basic":
        model_input = basic_model_input(encoder)
    else:
        raise NotImplementedError("Given model input [{}] is not available".format(model_input))

    model.eval()
    for _ in range(warmup_passes):
        model(model_input)

    with Timer("Executing model forward"):
        with torch.no_grad():
            for _ in range(num_passes):
                model(model_input)

    model.train()
    with Timer("Executing model forward/backward"):
        for _ in range(num_passes):
            _train(model, model_input)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--encoder", default="xlmr_base", type=str)
    parser.add_argument("--num-passes", default=50, type=int)
    parser.add_argument("--warmup-passes", default=10, type=int)
    parser.add_argument("--model-input", default="basic", type=str)
    run(parser.parse_args())
