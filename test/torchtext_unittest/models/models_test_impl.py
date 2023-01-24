import copy
from unittest.mock import patch

import torch
from torch.nn import functional as torch_F

from ..common.case_utils import TestBaseMixin


class BaseTestModels(TestBaseMixin):
    def get_model(self, encoder_conf, head=None, freeze_encoder=False, checkpoint=None, override_checkpoint_head=False):
        from torchtext.models import RobertaBundle

        model = RobertaBundle.build_model(
            encoder_conf=encoder_conf,
            head=head,
            freeze_encoder=freeze_encoder,
            checkpoint=checkpoint,
            override_checkpoint_head=override_checkpoint_head,
        )
        model.to(device=self.device, dtype=self.dtype)
        return model

    def test_roberta_bundler_build_model(self) -> None:
        from torchtext.models import RobertaClassificationHead, RobertaEncoderConf, RobertaModel

        dummy_encoder_conf = RobertaEncoderConf(
            vocab_size=10, embedding_dim=16, ffn_dimension=64, num_attention_heads=2, num_encoder_layers=2
        )

        # case: user provide encoder checkpoint state dict
        dummy_encoder = RobertaModel(dummy_encoder_conf)
        model = self.get_model(encoder_conf=dummy_encoder_conf, checkpoint=dummy_encoder.state_dict())
        self.assertEqual(model.state_dict(), dummy_encoder.state_dict())

        # case: user provide classifier checkpoint state dict when head is given and override_head is False (by default)
        dummy_classifier_head = RobertaClassificationHead(num_classes=2, input_dim=16)
        another_dummy_classifier_head = RobertaClassificationHead(num_classes=2, input_dim=16)
        dummy_classifier = RobertaModel(dummy_encoder_conf, dummy_classifier_head)
        model = self.get_model(
            encoder_conf=dummy_encoder_conf,
            head=another_dummy_classifier_head,
            checkpoint=dummy_classifier.state_dict(),
        )
        self.assertEqual(model.state_dict(), dummy_classifier.state_dict())

        # case: user provide classifier checkpoint state dict when head is given and override_head is set True
        another_dummy_classifier_head = RobertaClassificationHead(num_classes=2, input_dim=16)
        model = self.get_model(
            encoder_conf=dummy_encoder_conf,
            head=another_dummy_classifier_head,
            checkpoint=dummy_classifier.state_dict(),
            override_checkpoint_head=True,
        )
        self.assertEqual(model.head.state_dict(), another_dummy_classifier_head.state_dict())

        # case: user provide only encoder checkpoint state dict when head is given
        dummy_classifier_head = RobertaClassificationHead(num_classes=2, input_dim=16)
        dummy_classifier = RobertaModel(dummy_encoder_conf, dummy_classifier_head)
        encoder_state_dict = {}
        for k, v in dummy_classifier.encoder.state_dict().items():
            encoder_state_dict["encoder." + k] = v
        model = self.get_model(
            encoder_conf=dummy_encoder_conf, head=dummy_classifier_head, checkpoint=encoder_state_dict
        )
        self.assertEqual(model.state_dict(), dummy_classifier.state_dict())

    def test_roberta_bundler_train(self) -> None:
        from torchtext.models import RobertaClassificationHead, RobertaEncoderConf, RobertaModel

        dummy_encoder_conf = RobertaEncoderConf(
            vocab_size=10, embedding_dim=16, ffn_dimension=64, num_attention_heads=2, num_encoder_layers=2
        )
        from torch.optim import SGD

        def _train(model):
            optim = SGD(model.parameters(), lr=1)
            model_input = torch.tensor([[0, 1, 2, 3, 4, 5]]).to(device=self.device)
            target = torch.tensor([0]).to(device=self.device)
            logits = model(model_input)
            loss = torch_F.cross_entropy(logits, target)
            loss.backward()
            optim.step()

        # does not freeze encoder
        dummy_classifier_head = RobertaClassificationHead(num_classes=2, input_dim=16)
        dummy_classifier = RobertaModel(dummy_encoder_conf, dummy_classifier_head)
        model = self.get_model(
            encoder_conf=dummy_encoder_conf,
            head=dummy_classifier_head,
            freeze_encoder=False,
            checkpoint=dummy_classifier.state_dict(),
        )

        encoder_current_state_dict = copy.deepcopy(model.encoder.state_dict())
        head_current_state_dict = copy.deepcopy(model.head.state_dict())

        _train(model)

        self.assertNotEqual(model.encoder.state_dict(), encoder_current_state_dict)
        self.assertNotEqual(model.head.state_dict(), head_current_state_dict)

        # freeze encoder
        dummy_classifier_head = RobertaClassificationHead(num_classes=2, input_dim=16)
        dummy_classifier = RobertaModel(dummy_encoder_conf, dummy_classifier_head)
        model = self.get_model(
            encoder_conf=dummy_encoder_conf,
            head=dummy_classifier_head,
            freeze_encoder=True,
            checkpoint=dummy_classifier.state_dict(),
        )

        encoder_current_state_dict = copy.deepcopy(model.encoder.state_dict())
        head_current_state_dict = copy.deepcopy(model.head.state_dict())

        _train(model)

        self.assertEqual(model.encoder.state_dict(), encoder_current_state_dict)
        self.assertNotEqual(model.head.state_dict(), head_current_state_dict)

    @patch("logging.Logger.warning")
    def test_roberta_bundler_get_model(self, mock):
        from torchtext.models import RobertaEncoderConf, RobertaBundle

        dummy_encoder_conf = RobertaEncoderConf(
            vocab_size=10, embedding_dim=16, ffn_dimension=64, num_attention_heads=2, num_encoder_layers=2
        )
        model_bundle = RobertaBundle(dummy_encoder_conf)
        model_bundle.get_model(load_weights=False, freeze_encoder=True)
        mock.assert_called_with(
            "The encoder is not loaded with pre-trained weights. Setting freeze_encoder to True will hinder encoder from learning appropriate weights."
        )

    def test_roberta_bundler_raise_checkpoint(self) -> None:
        from torchtext.models import RobertaClassificationHead, RobertaEncoderConf, RobertaBundle

        with self.assertRaises(TypeError):
            dummy_encoder_conf = RobertaEncoderConf(
                vocab_size=10, embedding_dim=16, ffn_dimension=64, num_attention_heads=2, num_encoder_layers=2
            )
            dummy_classifier_head = RobertaClassificationHead(num_classes=2, input_dim=16)
            RobertaBundle.build_model(
                encoder_conf=dummy_encoder_conf,
                head=dummy_classifier_head,
                freeze_encoder=True,
                checkpoint=1,
            )

    def test_roberta_bundler_encode_conf_property(self) -> None:
        from torchtext.models import RobertaEncoderConf, RobertaBundle

        dummy_encoder_conf = RobertaEncoderConf(
            vocab_size=10, embedding_dim=16, ffn_dimension=64, num_attention_heads=2, num_encoder_layers=2
        )
        model_bundle = RobertaBundle(dummy_encoder_conf)
        self.assertTrue(isinstance(model_bundle.encoderConf, RobertaEncoderConf))
