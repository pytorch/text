from unittest.mock import patch

import torch
from torchtext.prototype.generate import GenerationUtil
from torchtext.prototype.models import T5_BASE_GENERATION
from torchtext_unittest.common.torchtext_test_case import TorchtextTestCase


class TestGenerationUtil(TorchtextTestCase):
    def setUp(self) -> None:
        super().setUp()
        t5_base = T5_BASE_GENERATION
        self.transform = t5_base.transform()
        self.model = t5_base.get_model()
        self.model.eval()
        # Examples taken from T5 Paper and Huggingface
        self.inputs = self.transform(
            [
                "summarize: studies have shown that owning a dog is good for you",
                "translate English to German: That is good.",
                "cola sentence: The course is jumping well.",
                "stsb sentence1: The rhino grazed on the grass. sentence2: A rhino is grazing in a field.",
                "summarize: state authorities dispatched emergency crews tuesday to survey the damage after an onslaught of severe weather in mississippi...",
            ]
        )
        torch.manual_seed(0)

    def test_greedy_generate_with_t5(self) -> None:
        generation_model = GenerationUtil(self.model)

        tokens = generation_model.generate(self.inputs, num_beams=1, max_len=30)
        generated_text = self.transform.decode(tokens.tolist())

        expected_generated_text = [
            "a dog is good for you, according to studies . owning a dog is good for you, according to studies .",
            "Das ist gut.",
            "acceptable",
            "4.0",
            "mississippi authorities dispatch emergency crews to survey damage . severe weather in mississippi has caused extensive damage",
        ]

        self.assertEqual(generated_text, expected_generated_text)

    def test_generate_errors_with_incorrect_beams(self) -> None:
        generation_model = GenerationUtil(self.model, is_encoder_decoder=True)

        with self.assertRaises(ValueError):
            generation_model.generate(self.inputs, num_beams=0)

    @patch("logging.Logger.warning")
    def test_warns_when_no_max_len_provided(self, mock) -> None:
        generation_model = GenerationUtil(self.model)
        generation_model.generate(self.inputs)
        mock.assert_called_with("`max_len` was not specified. Defaulting to 100 tokens.")
