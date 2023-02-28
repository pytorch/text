import time
from functools import partial

from torch.utils.data import DataLoader
from torcheval.metrics.functional import word_error_rate
from torchtext.data.metrics import bleu_score
from torchtext.datasets import CNNDM
from torchtext.datasets import Multi30k
from torchtext.models import T5_BASE_GENERATION
from torchtext.prototype.generate import GenerationUtils

multi_batch_size = 5
language_pair = ("en", "de")
multi_datapipe = Multi30k(split="test", language_pair=language_pair)
task = "translate English to German"


def apply_prefix(task, x):
    return f"{task}: " + x[0], x[1]


multi_datapipe = multi_datapipe.map(partial(apply_prefix, task))
multi_datapipe = multi_datapipe.batch(multi_batch_size)
multi_datapipe = multi_datapipe.rows2columnar(["english", "german"])
multi_dataloader = DataLoader(multi_datapipe, batch_size=None)


def benchmark_beam_search_wer():
    model = T5_BASE_GENERATION.get_model()
    transform = T5_BASE_GENERATION.transform()

    seq_generator = GenerationUtils(model)

    batch = next(iter(multi_dataloader))
    input_text = batch["english"]
    target = batch["german"]
    beam_size = 4

    model_input = transform(input_text)
    model_output = seq_generator.generate(model_input, num_beams=beam_size, vocab_size=model.config.vocab_size)
    output_text = transform.decode(model_output.tolist())

    print(word_error_rate(output_text, target))


if __name__ == "__main__":
    benchmark_beam_search_wer()
