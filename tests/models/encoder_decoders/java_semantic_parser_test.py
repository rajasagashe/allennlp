# pylint: disable=invalid-name,no-self-use,protected-access

from allennlp.common.testing import ModelTestCase

class JavaSemanticParserTest(ModelTestCase):
    def setUp(self):
        super(JavaSemanticParserTest, self).setUp()
        self.set_up_model("tests/fixtures/encoder_decoder/java_parser/experiment.json",
                          "tests/fixtures/encoder_decoder/java_parser/sample_data_longer.json")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)