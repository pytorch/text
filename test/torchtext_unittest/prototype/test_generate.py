from unittest.mock import patch
from torchtext.prototype.generate import GenerationUtil
from torchtext.prototype.models import T5_BASE_GENERATION
from torchtext_unittest.common.torchtext_test_case import TorchtextTestCase
import torch


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

    @patch("warnings.warn")
    def test_warns_when_no_max_len_provided(self, mock) -> None:
        generation_model = GenerationUtil(self.model)
        generation_model.generate(self.inputs)
        mock.assert_called_with("`max_len` was not specified. Defaulting to 256 tokens.")

    def test_warns_when_mp_with_greedy(self, mock) -> None:
        pass

    def test_beam_search_with_t5_(self) -> None:
        generation_model = GenerationUtil(self.model)
        tokens = generation_model.generate(
            self.inputs, num_beams=3, max_len=30, beam_size_token=self.model.config.vocab_size
        )
        generated_text = self.transform.decode(tokens.tolist())

        expected_generated_text = [
            "kate mccartney: a dog is good for you . she says studies have shown that dog ownership is good for",
            "Das ist gut.",
            "acceptable",
            "4.0",
            "a tornado ripped through a swath of a lake in st. louis . a s",
        ]

        self.assertEqual(generated_text, expected_generated_text)

    def test_hf_DELETE(self) -> None:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        from torchtext.prototype.generate import GenerationUtil

        t5 = T5ForConditionalGeneration.from_pretrained("t5-base")
        test_sequence = [
            "summarize: studies have shown that owning a dog is good for you"
        ]  # , "Q: what is the capital of Alaska?"]
        generative_hf_t5 = GenerationUtil(t5, is_encoder_decoder=True, is_huggingface_model=True)
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        test_sequence_tk = t5_tokenizer(test_sequence, padding=True, return_tensors="pt").input_ids
        import time

        start = time.time()
        tokens = generative_hf_t5.generate(
            test_sequence_tk,
            max_len=100,
            pad_idx=t5.config.pad_token_id,
            num_beams=7,
            beam_size_token=t5.config.vocab_size,
        )
        end = time.time() - start
        print(t5_tokenizer.batch_decode(tokens, skip_special_tokens=True), end)
        exit()

    def test_jit_generate(self) -> None:
        # jitted_model = torch.jit.script(self.model)
        # encoder = jitted_model.get_encoder()
        
        
        generation_model = GenerationUtil(self.model)
        torch.jit.script(generation_model)

    def test_beam_search_speed(self) -> None:
        pass
