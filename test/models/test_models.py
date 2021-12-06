import torchtext
import torch
from torch.nn import functional as torch_F
import copy
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

    def test_roberta_bundler_build_model(self):
        from torchtext.models import RobertaEncoderConf, RobertaClassificationHead, RobertaModel, RobertaModelBundle
        dummy_encoder_conf = RobertaEncoderConf(vocab_size=10, embedding_dim=16, ffn_dimension=64, num_attention_heads=2, num_encoder_layers=2)

        # case: user provide encoder checkpoint state dict
        dummy_encoder = RobertaModel(dummy_encoder_conf)
        model = RobertaModelBundle.build_model(encoder_conf=dummy_encoder_conf,
                                               checkpoint=dummy_encoder.state_dict())
        self.assertEqual(model.state_dict(), dummy_encoder.state_dict())

        # case: user provide classifier checkpoint state dict when head is given and override_head is False (by default)
        dummy_classifier_head = RobertaClassificationHead(num_classes=2, input_dim=16)
        another_dummy_classifier_head = RobertaClassificationHead(num_classes=2, input_dim=16)
        dummy_classifier = RobertaModel(dummy_encoder_conf, dummy_classifier_head)
        model = RobertaModelBundle.build_model(encoder_conf=dummy_encoder_conf,
                                               head=another_dummy_classifier_head,
                                               checkpoint=dummy_classifier.state_dict())
        self.assertEqual(model.state_dict(), dummy_classifier.state_dict())

        # case: user provide classifier checkpoint state dict when head is given and override_head is set True
        another_dummy_classifier_head = RobertaClassificationHead(num_classes=2, input_dim=16)
        model = RobertaModelBundle.build_model(encoder_conf=dummy_encoder_conf,
                                               head=another_dummy_classifier_head,
                                               checkpoint=dummy_classifier.state_dict(),
                                               override_checkpoint_head=True)
        self.assertEqual(model.head.state_dict(), another_dummy_classifier_head.state_dict())

        # case: user provide only encoder checkpoint state dict when head is given
        dummy_classifier_head = RobertaClassificationHead(num_classes=2, input_dim=16)
        dummy_classifier = RobertaModel(dummy_encoder_conf, dummy_classifier_head)
        encoder_state_dict = {}
        for k, v in dummy_classifier.encoder.state_dict().items():
            encoder_state_dict['encoder.' + k] = v
        model = torchtext.models.RobertaModelBundle.build_model(encoder_conf=dummy_encoder_conf, head=dummy_classifier_head, checkpoint=encoder_state_dict)
        self.assertEqual(model.state_dict(), dummy_classifier.state_dict())

    def test_roberta_bundler_train(self):
        from torchtext.models import RobertaEncoderConf, RobertaClassificationHead, RobertaModel, RobertaModelBundle
        dummy_encoder_conf = RobertaEncoderConf(vocab_size=10, embedding_dim=16, ffn_dimension=64, num_attention_heads=2, num_encoder_layers=2)
        from torch.optim import SGD

        def _train(model):
            optim = SGD(model.parameters(), lr=1)
            model_input = torch.tensor([[0, 1, 2, 3, 4, 5]])
            target = torch.tensor([0])
            logits = model(model_input)
            loss = torch_F.cross_entropy(logits, target)
            loss.backward()
            optim.step()

        # does not freeze encoder
        dummy_classifier_head = RobertaClassificationHead(num_classes=2, input_dim=16)
        dummy_classifier = RobertaModel(dummy_encoder_conf, dummy_classifier_head)
        model = RobertaModelBundle.build_model(encoder_conf=dummy_encoder_conf,
                                               head=dummy_classifier_head,
                                               freeze_encoder=False,
                                               checkpoint=dummy_classifier.state_dict())

        encoder_current_state_dict = copy.deepcopy(model.encoder.state_dict())
        head_current_state_dict = copy.deepcopy(model.head.state_dict())

        _train(model)

        self.assertNotEqual(model.encoder.state_dict(), encoder_current_state_dict)
        self.assertNotEqual(model.head.state_dict(), head_current_state_dict)

        # freeze encoder
        dummy_classifier_head = RobertaClassificationHead(num_classes=2, input_dim=16)
        dummy_classifier = RobertaModel(dummy_encoder_conf, dummy_classifier_head)
        model = RobertaModelBundle.build_model(encoder_conf=dummy_encoder_conf,
                                               head=dummy_classifier_head,
                                               freeze_encoder=True,
                                               checkpoint=dummy_classifier.state_dict())

        encoder_current_state_dict = copy.deepcopy(model.encoder.state_dict())
        head_current_state_dict = copy.deepcopy(model.head.state_dict())

        _train(model)

        self.assertEqual(model.encoder.state_dict(), encoder_current_state_dict)
        self.assertNotEqual(model.head.state_dict(), head_current_state_dict)
