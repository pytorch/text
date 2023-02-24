import tempfile

import pytest  # noqa: F401
import torch
from parameterized import parameterized_class
from torchtext.models import T5Bundle
from torchtext.models import (
    T5_BASE,
    T5_BASE_ENCODER,
    T5_BASE_GENERATION,
    T5_LARGE,
    T5_LARGE_ENCODER,
    T5_LARGE_GENERATION,
    T5_SMALL,
    T5_SMALL_ENCODER,
    T5_SMALL_GENERATION,
    FLAN_T5_BASE_ENCODER,
    FLAN_T5_BASE,
    FLAN_T5_BASE_GENERATION,
)
from torchtext_unittest.common.assets import get_asset_path
from torchtext_unittest.common.parameterized_utils import nested_params
from torchtext_unittest.common.torchtext_test_case import TorchtextTestCase

BUNDLERS = {
    "base_model": T5_BASE,
    "base_encoder": T5_BASE_ENCODER,
    "base_generation": T5_BASE_GENERATION,
    "small_model": T5_SMALL,
    "small_encoder": T5_SMALL_ENCODER,
    "small_generation": T5_SMALL_GENERATION,
    "large_model": T5_LARGE,
    "large_encoder": T5_LARGE_ENCODER,
    "large_generation": T5_LARGE_GENERATION,
    "flan_base_encoder": FLAN_T5_BASE_ENCODER,
    "flan_base_model": FLAN_T5_BASE,
    "flan_base_generation": FLAN_T5_BASE_GENERATION,
}


@parameterized_class(
    ("model_name",),
    [
        ("base_model",),
        ("base_encoder",),
        ("base_generation",),
        ("small_model",),
        ("small_encoder",),
        ("small_generation",),
        ("large_model",),
        ("large_encoder",),
        ("large_generation",),
        ("flan_base_encoder",),
        ("flan_base_model",),
        ("flan_base_generation",),
    ],
)
class TestT5Model(TorchtextTestCase):
    def _t5_model(self, is_jit, t5_model, expected_asset_name, test_text):
        """Verify that pre-trained T5 models in torchtext produce
        the same output as the HuggingFace reference implementation.
        """
        expected_asset_path = get_asset_path(expected_asset_name)
        transform = t5_model.transform()
        model = t5_model.get_model()
        model = model.eval()

        if is_jit:
            transform = torch.jit.script(transform)
            model = torch.jit.script(model)

        model_input = transform(test_text)
        if model.encoder_only:
            actual = model(encoder_tokens=model_input)["encoder_output"]
            if not is_jit:
                self._t5_get_encoder(model, model_input, actual)
        else:
            actual = model(encoder_tokens=model_input)["decoder_output"]

        expected = torch.load(expected_asset_path)
        torch.testing.assert_close(actual, expected, atol=1e-04, rtol=2.5e-06)

    def _t5_get_encoder(self, model, model_input, encoder_output):
        encoder = model.get_encoder()
        # Need to set the key_padding_mask to ensure the same results
        encoder_padding_mask = model_input.eq(model.padding_idx)
        output_from_get_encoder = encoder(model_input, src_key_padding_mask=encoder_padding_mask)["encoder_output"]
        assert torch.all(output_from_get_encoder.eq(encoder_output))

    @nested_params(["not_jit", "jit"])
    def test_t5_model(self, name) -> None:
        names = self.model_name.split("_")

        num_names = len(names)

        if num_names == 3:
            # Handled slightly differently for Flan-T5 model naming
            configuration = names[1]
            type = names[2]
            expected_asset_name = f"t5.flan.{configuration}.{type}.output.pt"
            t5_model = BUNDLERS["flan_" + configuration + "_" + type]
        elif num_names == 2:
            configuration = names[0]
            type = names[1]
            expected_asset_name = f"t5.{configuration}.{type}.output.pt"
            t5_model = BUNDLERS[configuration + "_" + type]
        else:
            raise RuntimeError(f"Unknown model name: {self.model_name}")

        test_text = ["Hello world", "Attention rocks!"]
        is_jit = name == "jit"
        self._t5_model(is_jit=is_jit, t5_model=t5_model, expected_asset_name=expected_asset_name, test_text=test_text)

@parameterized_class(
    ("model",),
    [
        ("hf_t5_small_encoder",),
        ("hf_t5_small",),
        ("hf_t5_small_generation",),
        ("hf_flan_base_encoder",),
        ("hf_flan_base",),
        ("hf_flan_base_generation",),
    ],
)
class TestLoadFromHFCheckpoints(TorchtextTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.encoder_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 0, 0]])
        self.encoder_padding_mask = torch.tensor(
            [[False, False, False, False, False, False], [False, False, False, True, True, True]]
        )
        self.decoder_input_ids = torch.tensor([[7, 8, 9, 0, 0, 0], [10, 11, 12, 0, 0, 0]])
        self.decoder_padding_mask = torch.tensor(
            [[False, False, False, True, True, True], [False, False, False, True, True, True]]
        )

    def test_t5_bundler_load_hf_ckpt_pretrained(self) -> None:
        names = self.model.split("_")
        is_encoder_only = names[-1] == "encoder"

        model_path = get_asset_path(self.model)

        model = T5Bundle.build_model_from_huggingface_ckpt(model_path, encoder_only=is_encoder_only)
        if is_encoder_only:
            model(self.encoder_input_ids, encoder_padding_mask=self.encoder_padding_mask)
        else:
            model(
                self.encoder_input_ids,
                self.decoder_input_ids,
                encoder_padding_mask=self.encoder_padding_mask,
                decoder_padding_mask=self.decoder_padding_mask,
            )
