import torch

from ..common.parameterized_utils import nested_params
from ..common.torchtext_test_case import TorchtextTestCase


class TestTransformers(TorchtextTestCase):
    @nested_params(
        [True, False],
        [True, False],
    )
    def test_padded_input_inference(self, with_no_grad, return_all_layers):
        """test transformerencoder inference same with and without padding"""
        from torchtext.models import RobertaEncoderConf, RobertaModel

        def encoder_inference(encoder, input_lst, with_no_grad):
            if with_no_grad:
                with torch.no_grad():
                    res = [encoder(eval_input) for eval_input in input_lst]
            else:
                res = [encoder(eval_input) for eval_input in input_lst]
            return res

        # Roberta config except for less layers (2 instead of 12)
        pad_idx = 1
        encoder_conf = RobertaEncoderConf(
            vocab_size=250002,
            embedding_dim=768,
            ffn_dimension=3072,
            padding_idx=pad_idx,
            max_seq_len=514,
            num_attention_heads=12,
            num_encoder_layers=2,
            dropout=0.1,
            scaling=None,
            normalize_before=False,
        )
        model = RobertaModel(encoder_conf)
        model = model.eval()
        # TODO: make return_all_layers a property of RobertaEncoderConf so it can be passed as arg
        model.encoder.transformer.return_all_layers = return_all_layers

        # result from converting string "some text" to tensor using xlmr_base embeddings
        input_no_pad = torch.Tensor([[0, 3060, 7986, 2]]).to(torch.int)
        data_len = input_no_pad.shape[1]  # sequence length of non-pad data
        # add two padding tokens to input_no_pad
        input_pad = torch.Tensor([[0, 3060, 7986, 2, pad_idx, pad_idx]]).to(torch.int)
        input_lst = [input_no_pad, input_pad]

        output_no_pad, output_pad = encoder_inference(model, input_lst, with_no_grad)
        torch.testing.assert_close(output_no_pad, output_pad[:, :data_len, :])
