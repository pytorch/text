import torchtext
import torch
import urllib
from torchtext import _TEXT_BUCKET
from ..common.torchtext_test_case import TorchtextTestCase
from ..common.assets import get_asset_path


class TestModules(TorchtextTestCase):
    def test_self_attn_mask(self):
        from torchtext.models.roberta.modules import MultiheadSelfAttention
        embed_dim, batch_size, num_heads, source_len = 4, 1, 2, 2
        mha = MultiheadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        query = torch.ones((source_len, batch_size, embed_dim))
        query[0, ...] = 0
        key_padding_mask = torch.zeros((batch_size, source_len))
        attn_mask = torch.zeros((source_len, source_len))
        attn_mask[0][1] = -1e8
        with torch.no_grad():
            mha.input_projection.weight.fill_(1. / embed_dim)
            mha.input_projection.bias.fill_(0.)
            mha.output_projection.weight.fill_(1. / embed_dim)
            mha.output_projection.bias.fill_(0.)

            # with attention mask
            output = mha(query, key_padding_mask, attn_mask)
            actual = output[0].flatten()
            expected = torch.tensor([0., 0., 0., 0])
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


class TestModels(TorchtextTestCase):
    def test_xlmr_base_output(self):
        asset_name = "xlmr.base.output.pt"
        asset_path = get_asset_path(asset_name)
        xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        model = xlmr_base.get_model()
        model = model.eval()
        model_input = torch.tensor([[0, 43523, 52005, 3647, 13293, 113307, 40514, 2]])
        actual = model(model_input)
        expected = torch.load(asset_path)
        torch.testing.assert_close(actual, expected)

    def test_xlmr_base_jit_output(self):
        asset_name = "xlmr.base.output.pt"
        asset_path = get_asset_path(asset_name)
        xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        model = xlmr_base.get_model()
        model = model.eval()
        model_jit = torch.jit.script(model)
        model_input = torch.tensor([[0, 43523, 52005, 3647, 13293, 113307, 40514, 2]])
        actual = model_jit(model_input)
        expected = torch.load(asset_path)
        torch.testing.assert_close(actual, expected)

    def test_xlmr_large_output(self):
        asset_name = "xlmr.large.output.pt"
        asset_path = get_asset_path(asset_name)
        xlmr_base = torchtext.models.XLMR_LARGE_ENCODER
        model = xlmr_base.get_model()
        model = model.eval()
        model_input = torch.tensor([[0, 43523, 52005, 3647, 13293, 113307, 40514, 2]])
        actual = model(model_input)
        expected = torch.load(asset_path)
        torch.testing.assert_close(actual, expected)

    def test_xlmr_large_jit_output(self):
        asset_name = "xlmr.large.output.pt"
        asset_path = get_asset_path(asset_name)
        xlmr_base = torchtext.models.XLMR_LARGE_ENCODER
        model = xlmr_base.get_model()
        model = model.eval()
        model_jit = torch.jit.script(model)
        model_input = torch.tensor([[0, 43523, 52005, 3647, 13293, 113307, 40514, 2]])
        actual = model_jit(model_input)
        expected = torch.load(asset_path)
        torch.testing.assert_close(actual, expected)

    def test_xlmr_transform(self):
        xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        transform = xlmr_base.transform()
        test_text = "XLMR base Model Comparison"
        actual = transform([test_text])
        expected = [[0, 43523, 52005, 3647, 13293, 113307, 40514, 2]]
        torch.testing.assert_close(actual, expected)

    def test_xlmr_transform_jit(self):
        xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        transform = xlmr_base.transform()
        transform_jit = torch.jit.script(transform)
        test_text = "XLMR base Model Comparison"
        actual = transform_jit([test_text])
        expected = [[0, 43523, 52005, 3647, 13293, 113307, 40514, 2]]
        torch.testing.assert_close(actual, expected)

    def test_roberta_bundler_from_config(self):
        from torchtext.models import RobertaEncoderConf
        asset_name = "xlmr.base.output.pt"
        asset_path = get_asset_path(asset_name)
        model_path = urllib.parse.urljoin(_TEXT_BUCKET, "xlmr.base.encoder.pt")
        model = torchtext.models.RobertaModelBundle.from_config(config=RobertaEncoderConf(vocab_size=250002), checkpoint=model_path)
        model = model.eval()
        model_input = torch.tensor([[0, 43523, 52005, 3647, 13293, 113307, 40514, 2]])
        actual = model(model_input)
        expected = torch.load(asset_path)
        torch.testing.assert_close(actual, expected)
