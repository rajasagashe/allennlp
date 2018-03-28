# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers import JavaDatasetReader

class TestJavaDatasetReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = JavaDatasetReader(lazy=lazy)
        instances = ensure_list(reader.read('tests/fixtures/data/java.json'))

        assert len(instances) == 3

        # Test variable types lowercased
        assert [t.text for t in instances[0].fields["variable_types"].tokens[:3]] == ['object', 'int', 'defaultmapbag']

        # Test code summary camel case split
        assert [t.text for t in instances[1].fields["code_summary"].tokens[:3]] == ['compareTo', 'compare', 'To']
        assert [t.text for t in instances[2].fields["code_summary"].tokens[9:12]] == ['the', 'NLS', 'element']

        # Test method and variable name camel case split
        text_fields = instances[0].fields["variable_names"].field_list
        assert [token.text for field in text_fields for token in field.tokens][0:2] == ['_current', '_total']
        text_fields = instances[1].fields["method_names"].field_list
        # pylint: disable=line-too-long
        assert [token.text for field in text_fields for token in field.tokens][:3] == ['signum', 'leftLinearCombination', 'left']

    def test_can_build_from_params(self):
        reader = JavaDatasetReader.from_params(Params({}))
        # pylint: disable=protected-access
        assert reader._tokenizer.__class__.__name__ == 'WordTokenizer'
        assert reader._token_indexers["tokens"].__class__.__name__ == 'SingleIdTokenIndexer'
