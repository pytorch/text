from unittest.mock import patch

import torch
from torchtext.models import T5_BASE_GENERATION
from torchtext.prototype.generate import DEFAULT_MAX_SEQ_LEN, GenerationUtils
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
        generation_model = GenerationUtils(self.model)

        tokens = generation_model.generate(self.inputs, num_beams=1)
        generated_text = self.transform.decode(tokens.tolist())

        expected_generated_text = [
            "a dog is good for you, according to studies . owning a dog is good for you, according to studies .",
            "Das ist gut.",
            "acceptable",
            "4.0",
            "mississippi authorities dispatch emergency crews to survey damage . severe weather in mississippi has caused extensive damage",
        ]

        self.assertEqual(generated_text, expected_generated_text)

    def test_beam_search_generate_t5(self) -> None:
        generation_model = GenerationUtils(self.model)

        tokens = generation_model.generate(
            self.inputs, num_beams=3, vocab_size=self.model.config.vocab_size, max_length=30
        )
        generated_text = self.transform.decode(tokens.tolist())

        expected_generated_text = [
            "kate mccartney: a dog is good for you . she says studies have shown that dog ownership is good for",
            "Das ist gut.",
            "acceptable",
            "4.0",
            "a tornado ripped through a swath of a lake in southeastern michigan . a spokesman",
        ]

        self.assertEqual(generated_text, expected_generated_text)

    def test_beam_search_generate_t5_small_batch_size(self) -> None:
        generation_model = GenerationUtils(self.model)

        tokens = generation_model.generate(
            self.inputs, num_beams=3, vocab_size=self.model.config.vocab_size, max_length=30, max_inference_batch_size=3
        )
        generated_text = self.transform.decode(tokens.tolist())

        expected_generated_text = [
            "kate mccartney: a dog is good for you . she says studies have shown that dog ownership is good for",
            "Das ist gut.",
            "acceptable",
            "4.0",
            "a tornado ripped through a swath of a lake in southeastern michigan . a spokesman",
        ]

        self.assertEqual(generated_text, expected_generated_text)

    def test_beam_search_generate_t5_with_small_beam_threshold(self) -> None:
        generation_model = GenerationUtils(self.model)

        tokens = generation_model.generate(
            self.inputs, num_beams=3, vocab_size=self.model.config.vocab_size, max_length=30, beam_threshold=5
        )
        generated_text = self.transform.decode(tokens.tolist())

        expected_text = [
            "kate mccartney: a dog is good for you . kate mccartney: dogs",
            "Das ist gut.",
            "acceptable",
            "4.0",
            "a tornado ripped through a swath of a lake in southeastern mississippi, causing",
        ]

        self.assertEqual(generated_text, expected_text)

    def test_beam_search_generate_t5_large_num_beams(self) -> None:
        generation_model = GenerationUtils(self.model)

        tokens = generation_model.generate(
            self.inputs, num_beams=25, vocab_size=self.model.config.vocab_size, max_length=30
        )
        generated_text = self.transform.decode(tokens.tolist())

        expected_text = [
            "aaron carroll, aaron jones, aaron jones and aaron jones",
            "Das ist gut.",
            "acceptable",
            "4.0",
            "a blizzard and power outages have prompted a blizzard and power outages, a spokesman says",
        ]

        self.assertEqual(generated_text, expected_text)

    def test_beam_search_generate_t5_large_num_beams_eos_score(self) -> None:
        generation_model = GenerationUtils(self.model)

        tokens = generation_model.generate(
            self.inputs, num_beams=25, vocab_size=self.model.config.vocab_size, max_length=30, eos_score=10.0
        )
        generated_text = self.transform.decode(tokens.tolist())

        expected_text = ["", "Das ist gut.", "acceptable", "4.0", ""]

        self.assertEqual(generated_text, expected_text)

    def test_generate_errors_with_incorrect_beams(self) -> None:
        generation_model = GenerationUtils(self.model, is_encoder_decoder=True)

        with self.assertRaises(ValueError):
            generation_model.generate(self.inputs, num_beams=0)

    @patch("warnings.warn")
    def test_warns_when_no_max_len_provided(self, mock) -> None:
        generation_model = GenerationUtils(self.model)
        generation_model.generate(self.inputs)
        mock.assert_called_with(f"`max_length` was not specified. Defaulting to {DEFAULT_MAX_SEQ_LEN} tokens.")
