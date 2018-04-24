# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers import JavaDatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer


class TestJavaDatasetReader:
    def test_read_from_file(self):
        reader = JavaDatasetReader.from_params(Params({
            "utterance_indexers": {"tokens": {"namespace": "utterance"}},
            "type_indexers": {"tokens": {"namespace": "type"}},
            "min_identifier_count": 3,
            "num_dataset_instances": -1,
            "linking_feature_extractors": [
                "exact_token_match",
                "contains_exact_token_match",
                "edit_distance",
                "span_overlap_fraction"
            ]
        }))
        instances = ensure_list(reader.read('tests/fixtures/encoder_decoder/java_parser/sample_data_longer.json'))

        assert len(instances) == 11

        # Test variable types lowercased
        print([t.text for t in instances[0].fields["variable_types"].tokens[:3]])
        assert [t.text for t in instances[0].fields["variable_types"].tokens[:3]] == ['container', 'boolean', 'long']

        # Test method and variable name camel case split
        text_fields = instances[0].fields["variable_names"].field_list
        assert [token.text for field in text_fields for token in field.tokens][0:4] == ['parent', 'isparsed', 'is', 'parsed']

        text_fields = instances[1].fields["method_names"].field_list
        assert [token.text for field in text_fields for token in field.tokens][:6] == ['removewhenresourceremoved', 'remove', 'when', 'resource', 'removed', 'getcompilationunit']

    def test_split_camel_case(self):
        reader = JavaDatasetReader.from_params(Params({
            "utterance_indexers": {"tokens": {"namespace": "utterance"}},
            "type_indexers": {"tokens": {"namespace": "type"}},
            "min_identifier_count": 3,
            "num_dataset_instances": -1,
            "linking_feature_extractors": [
                "exact_token_match",
                "contains_exact_token_match",
                "edit_distance",
                "span_overlap_fraction"
            ]
        }))

        assert reader.split_camel_case('isParsedWell') == ['isparsedwell', 'is', 'parsed', 'well']
        assert reader.split_camel_case('_compute') == ['_compute']
        assert reader.split_camel_case('LOG') == ['log']

    def test_can_build_from_params(self):
        reader = JavaDatasetReader.from_params(Params({
            "utterance_indexers": {"tokens": {"namespace": "utterance"}},
            "type_indexers": {"tokens": {"namespace": "type"}},
            "min_identifier_count": 3,
            "num_dataset_instances": -1,
            "linking_feature_extractors": [
                "exact_token_match",
                "contains_exact_token_match",
                "edit_distance",
                "span_overlap_fraction"
            ]
        }))
        # pylint: disable=protected-access
        assert reader._tokenizer.__class__.__name__ == 'WordTokenizer'
