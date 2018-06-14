# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers import JavaDatasetReader
from allennlp.common.testing import AllenNlpTestCase

class TestJavaDatasetReader(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        self.reader = JavaDatasetReader.from_params(Params({
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

    def test_read_from_file(self):
        instances = ensure_list(self.reader.read('allennlp/tests/fixtures/data/java/sample_data.json'))

        assert len(instances) == 11

        # Test variable types lowercased
        # print([t.text for t in instances[0].fields["variable_types"].tokens[:3]])
        # assert [t.text for t in instances[0].fields["variable_types"].tokens[:3]] == ['container', 'boolean', 'long']

        # Test method and variable name camel case split
        # text_fields = instances[0].fields["variable_names"].field_list
        # assert [token.text for field in text_fields for token in field.tokens][0:4] == ['parent', 'isparsed', 'is', 'parsed']

        # text_fields = instances[1].fields["method_names"].field_list
        # assert [token.text for field in text_fields for token in field.tokens][:6] == ['removewhenresourceremoved', 'remove', 'when', 'resource', 'removed', 'getcompilationunit']

    def test_split_camel_case(self):
        assert self.reader.split_camel_case_add_original('isParsedWell') == ['isParsedWell', 'is', 'Parsed', 'Well']
        assert self.reader.split_camel_case('_compute') == ['_compute']
        assert self.reader.split_camel_case('LOG') == ['LOG']
        #todo(rajas) get this to work
        # assert self.reader.split_camel_case('setDirectionA') == ['set', 'Direction', 'A']

    def test_split_types(self):
        assert self.reader.split_type('FileReader') == ['FileReader']
        assert self.reader.split_type('List<String>') == ['List', 'String']
        assert self.reader.split_type('byte[]') == ['byte']
        assert self.reader.split_type('Deque<Map<String,Object>>') == ['Deque','Map','String','Object']
        assert self.reader.split_type('Object[][]') == ['Object']
        assert self.reader.split_type('ITrace2D') == ['ITrace2D']

    def test_split_identifier_rule_into_multiple(self):
        dataset = [{'rules':[
            "MemberDeclaration-->MethodDeclaration",
            "MethodDeclaration-->TypeTypeOrVoid___IdentifierNT___FormalParameters___MethodBody",
            "TypeTypeOrVoid-->TypeType",
            "TypeType-->Nt_41",
            "Nt_41-->ClassOrInterfaceType",
            "ClassOrInterfaceType-->IdentifierNT",
            "IdentifierNT-->File",
            "IdentifierNT-->function",
            "FormalParameters-->(___)",
            "MethodBody-->Block",
            "Block-->{___Star_26___}",
            "Star_26-->BlockStatement",
            "BlockStatement-->Statement",
            "Statement-->return___Expression___;",
            "Expression-->Primary",
            "Primary-->IdentifierNT",
            "IdentifierNT-->sriniclass_libraryFile"
        ]}]
        split_rules = self.reader._optimize_grammar([dataset[0]['rules']])
        print(split_rules)
        gold_rules = [['MemberDeclaration-->MethodDeclaration',
                            'MethodDeclaration-->TypeTypeOrVoid___function___FormalParameters___{___Star_26___}',
                       'TypeTypeOrVoid-->TypeType',
                       'TypeType-->Nt_41',
                       'Nt_41-->ClassOrInterfaceType',
                       'ClassOrInterfaceType-->IdentifierNTClassOrInterfaceType',
                       'IdentifierNTClassOrInterfaceType-->File',
                       # 'IdentifierNTOther-->function',
                       'FormalParameters-->(___)',
                       # 'MethodBody-->Block',
                       # 'Block-->{___Star_26___}',
                       'Star_26-->BlockStatement',
                       'BlockStatement-->Statement',
                       'Statement-->return___Expression___;',
                       'Expression-->Primary',
                       'Primary-->IdentifierNTPrimary',
                       'IdentifierNTPrimary-->sriniclass_libraryFile']]
        assert split_rules == gold_rules
    def test_can_build_from_params(self):
        # pylint: disable=protected-access
        assert self.reader._tokenizer.__class__.__name__ == 'WordTokenizer'
