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
        self.inputs = [
            "summarize: studies have shown that owning a dog is good for you",
            "translate English to German: That is good.",
            "cola sentence: The course is jumping well.",
            "stsb sentence1: The rhino grazed on the grass. sentence2: A rhino is grazing in a field.",
            "summarize: state authorities dispatched emergency crews tuesday to survey the damage after an onslaught of severe weather in mississippi...",
        ]
        self.transformed_inputs = self.transform(self.inputs)
        torch.manual_seed(0)

    def test_greedy_generate_with_t5(self) -> None:
        generation_model = GenerationUtils(self.model)

        tokens = generation_model.generate(self.transformed_inputs, num_beams=1, max_length=30)
        generated_text = self.transform.decode(tokens.tolist())

        expected_generated_text = [
            "owning a dog is good for you, according to studies . a dog is a good companion for a variety of reasons",
            "Das ist gut so.",
            "acceptable",
            "4.0",
            "mississippi authorities dispatch emergency crews to survey damage . severe weather in mississippi has caused extensive damage",
        ]

        self.assertEqual(generated_text, expected_generated_text)

        inputs = self.transform([self.inputs[0]])

        tokens_for_single_example = generation_model.generate(inputs, num_beams=1, max_length=30)
        generated_text_for_single_example = self.transform.decode(tokens_for_single_example.tolist())

        self.assertEqual(generated_text[0], generated_text_for_single_example[-1])

    def test_generate_errors_with_incorrect_beams(self) -> None:
        generation_model = GenerationUtils(self.model, is_encoder_decoder=True)

        with self.assertRaises(ValueError):
            generation_model.generate(self.transformed_inputs, num_beams=0)

    @patch("logging.Logger.warning")
    def test_warns_when_no_max_len_provided(self, mock) -> None:
        generation_model = GenerationUtils(self.model)
        generation_model.generate(self.transformed_inputs)
        mock.assert_called_with(f"`max_length` was not specified. Defaulting to {DEFAULT_MAX_SEQ_LEN} tokens.")

    def test_get_top_k_restriction(self) -> None:
        generation_model = GenerationUtils(self.model)
        scores = torch.arange(0, 20).reshape(2, -1)

        top_k = 5
        expected = torch.zeros_like(scores)
        expected[0, 5:] = torch.arange(5, 10)
        expected[1, 5:] = torch.arange(15, 20)
        output = generation_model._get_top_k_restriction(scores, top_k)
        self.assertEqual(output, expected)

        top_k = 11
        output = generation_model._get_top_k_restriction(scores, top_k)
        self.assertEqual(output, scores)

        top_k = 0
        with self.assertRaises(ValueError):
            generation_model._get_top_k_restriction(scores, top_k)

    def test_get_top_p_restriction(self) -> None:
        generation_model = GenerationUtils(self.model)
        input_probs = (torch.arange(1, 5) / 10).repeat(2).reshape(2, -1)

        top_p = 0.75
        expected = torch.tensor([0, 0, 0.3, 0.4]).repeat(2).reshape(2, -1)
        output = generation_model._get_top_p_restriction(input_probs, top_p)
        self.assertEqual(output, expected)

        # test min_to_keep parameter
        top_p = 0.2

        # min_tokens_to_keep defaults to 1
        expected_default = torch.tensor([0, 0, 0, 0.4]).repeat(2).reshape(2, -1)
        output_default = generation_model._get_top_p_restriction(input_probs, top_p)
        self.assertEqual(output_default, expected_default)

        output_with_min_2 = generation_model._get_top_p_restriction(input_probs, top_p, min_tokens_to_keep=2)
        self.assertEqual(output_with_min_2, expected)

        top_p = -1
        with self.assertRaises(ValueError):
            generation_model._get_top_p_restriction(input_probs, top_p)

    def test_apply_temperature(self) -> None:
        generation_model = GenerationUtils(self.model)
        input_probs = torch.ones((2, 5))

        # testing valid temperature
        temperature = 2
        valid_output = generation_model._apply_temperature(input_probs, temperature)
        expected_output = torch.ones((2, 5)) / 2

        self.assertEqual(valid_output, expected_output)

        # testing invalid temperature
        temperature = 0
        with self.assertRaises(ValueError):
            generation_model._apply_temperature(input_probs, temperature)
        temperature = -1
        with self.assertRaises(ValueError):
            generation_model._apply_temperature(input_probs, temperature)
