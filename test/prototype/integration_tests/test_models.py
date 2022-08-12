import torch
from parameterized import parameterized
from test.common.assets import get_asset_path
from test.common.torchtext_test_case import TorchtextTestCase
from torchtext.prototype.models import T5_BASE_ENCODER, T5_BASE, T5_BASE_GENERATION, T5Conf, T5Transform
from torchtext.prototype.models.t5.wrapper import T5Wrapper


class TestT5(TorchtextTestCase):
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
            actual = model(model_input)["encoder_output"]
        else:
            actual = model(model_input)["decoder_output"]

        expected = torch.load(expected_asset_path)
        torch.testing.assert_close(actual, expected, atol=1e-04, rtol=2.5e-06)

    @parameterized.expand([("jit", True), ("not_jit", False)])
    def test_t5_base_encoder_model(self, name, is_jit) -> None:
        expected_asset_name = "t5.base.encoder.output.pt"
        test_text = ["Hello world", "Attention rocks!"]
        self._t5_model(
            is_jit=is_jit, t5_model=T5_BASE_ENCODER, expected_asset_name=expected_asset_name, test_text=test_text
        )

    @parameterized.expand([("jit", True), ("not_jit", False)])
    def test_t5_base_model(self, name, is_jit) -> None:
        expected_asset_name = "t5.base.output.pt"
        test_text = ["Hello world", "Attention rocks!"]
        self._t5_model(is_jit=is_jit, t5_model=T5_BASE, expected_asset_name=expected_asset_name, test_text=test_text)

    @parameterized.expand([("jit", True), ("not_jit", False)])
    def test_t5_base_generation_model(self, name, is_jit) -> None:
        expected_asset_name = "t5.base.generation.output.pt"
        test_text = ["Hello world", "Attention rocks!"]
        self._t5_model(
            is_jit=is_jit, t5_model=T5_BASE_GENERATION, expected_asset_name=expected_asset_name, test_text=test_text
        )

    @parameterized.expand([("jit", True), ("not_jit", False)])
    def test_t5_wrapper(self, name, is_jit) -> None:
        test_text = ["translate English to French: I want to eat pizza for dinner."]
        expected_text = ["Je veux manger de la pizza pour le dîner."]
        beam_size = 3
        max_seq_len = 512
        model = T5Wrapper(configuration="base")
        if is_jit:
            model = torch.jit.script(model)

        output_text = model(test_text, beam_size, max_seq_len)
        self.assertEqual(output_text, expected_text)

    @parameterized.expand([("jit", True), ("not_jit", False)])
    def test_t5_wrapper_checkpoing(self, name, is_jit) -> None:
        test_text = ["translate English to French: I want to eat pizza for dinner."]
        expected_text = ["Je veux manger de la pizza pour le dîner."]
        beam_size = 3
        max_seq_len = 512
        config = T5Conf(encoder_only=False, linear_head=True)
        transform = T5Transform(
            "https://download.pytorch.org/models/text/t5_tokenizer_base.model",
            max_seq_len=512,
            eos_idx=1,
            padding_idx=0,
        )
        model = T5Wrapper(
            checkpoint="https://download.pytorch.org/models/text/t5.base.generation.pt",
            t5_config=config,
            transform=transform,
            freeze_model=True,
            strict=True,
        )
        if is_jit:
            model = torch.jit.script(model)

        output_text = model(test_text, beam_size, max_seq_len)
        self.assertEqual(output_text, expected_text)
