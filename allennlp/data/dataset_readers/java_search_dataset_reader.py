import random
from collections import defaultdict, Counter
from typing import Dict, List
import logging
import json
import os

import re
import numpy as np
from overrides import overrides
from nltk.tokenize import RegexpTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ProductionRuleField, MetadataField, KnowledgeGraphField
from allennlp.data.fields import Field, NoCountListField, ArrayField
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

# from java_programmer.fields import JavaProductionRuleField, JavaGlobalProductionRuleField, ProductionRuleField
from allennlp.semparse import KnowledgeGraph
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
IdentifierNT = 'IdentifierNT'
IdentifierOther = IdentifierNT + 'Other'
# LITERALS_TO_TRIM = [IdentifierNT, "Nt_decimal_literal", "Nt_char_literal",   "Nt_float_literal", "Nt_hex_literal", "Nt_oct_literal"]
IDENTIFIER_TYPES = ['Primary', 'ClassOrInterfaceType', 'Nt_33']

# todo(pr): Literals to trim shouoldn't have the modified identifier types or
# at least should have an if else allowing the modified vs normal IdentifierNT to be selected
LITERALS_TO_TRIM = [IdentifierNT, IdentifierNT+IDENTIFIER_TYPES[0], IdentifierNT+IDENTIFIER_TYPES[1], IdentifierNT+IDENTIFIER_TYPES[2], IdentifierOther, "Nt_decimal_literal", "Nt_char_literal",   "Nt_float_literal", "Nt_hex_literal", "Nt_oct_literal"]
UNK = '<UNK>'
DUMMY = '<DUMMY>'
PLAIN_IDENTIFIER_RULE = "<SOME_NAME>-->" + "<SOME_NAME>"
STOPS = set(stopwords.words("english"))
RULES_FILE = os.path.join('debug/', 'grammar_rules.txt')

@DatasetReader.register("java_search")
class JavaSearchDatasetReader(DatasetReader):
    def __init__(self,
                 utterance_indexers: Dict[str, TokenIndexer],
                 code_indexers: Dict[str, TokenIndexer],
                 min_identifier_count: int,
                 num_dataset_instances: int,
                 tokenizer: Tokenizer = None,
                 linking_feature_extractors: List[str] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._utterance_indexers = utterance_indexers
        # self._code_indexers = code_indexers
        self._code_indexers = utterance_indexers

        self._environment_token_indexers = {"tokens": SingleIdTokenIndexer()}

        self._environment_token_indexers = {"tokens": SingleIdTokenIndexer()}

        self._min_identifier_count = min_identifier_count
        self._num_dataset_instances = num_dataset_instances
        self._tokenizer = tokenizer or WordTokenizer()
        self._linking_feature_extractors = linking_feature_extractors

        # We first delete the file since it could've been created by another job.
        # read_rules_from_file
        self.read_rules_from_file = True
        if not self.read_rules_from_file and os.path.exists(RULES_FILE):
            os.remove(RULES_FILE)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)

        # dataset = dataset[:2000]
        if self._num_dataset_instances != -1:
            dataset = dataset[:self._num_dataset_instances]
        dataset = [self.return_stripped_rec(r) for r in dataset]

        contexts = []
        for rec in dataset:
            contexts.append(self.get_context_field(rec))
        all_context_field = NoCountListField(contexts)

        is_validation = False
        if os.path.basename(file_path).startswith('train'):
            self.train_dataset = dataset
            # todo: Remove this
            # dataset = dataset[:5000]
        else:
            # The Validation datasets could be very large!
            dataset = dataset[:5000]
            is_validation = True

            codes = []
            for rec in self.train_dataset:
                preprocessed_code = self.preprocess_code(rec['code'])
                codes.append(TextField([Token(c) for c in preprocessed_code],
                                     self._code_indexers))
            all_code_field = NoCountListField(codes)

        all_code_str_field = MetadataField(self.train_dataset)



        # todo(pr):

        logger.info("Reading the dataset")
        for i, record in enumerate(dataset):
            # instance = self.text_to_instance(record['nl'],
            #                                  record['varNames'],
            #                                  record['varTypes'],
            #                                  record['methodNames'],
            #                                  record['methodReturns'],
            #                                  methodName=record['methodName'],
            #                                  className=record['className'],
            #                                  code=record['code'],
            #                                  path="")

            # random_num = i
            # while random_num == i:
            #     random_num = random.randint(0, len(dataset)-1)
            # random_record = dataset[random_num]

            # random_record = random.sample(dataset, 1)[0]
            if is_validation:
                instance = self.text_to_instance(record=record,
                                                 # random_record=random_record,
                                                 index=i,
                                                 all_code_field=all_code_field,
                                                 all_code_str_field=all_code_str_field,
                                                 all_context_field=all_context_field)
            else:
                instance = self.text_to_instance(record=record,
                                                 # random_record=random_record,
                                                 index=i,
                                                 all_code_str_field=all_code_str_field,
                                                 all_context_field=all_context_field)
            yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         record,
                         # random_record,
                         index,
                         all_code_field=None,
                         all_code_str_field=None,
                         all_context_field=None
                         ) -> Instance:

        correct_context_field = self.get_context_field(record)
        # random_context_field = self.get_context_field(random_record)

        preprocessed_code = self.preprocess_code(record['code'])
        code_field = TextField([Token(c) for c in preprocessed_code], self._code_indexers)

        # print('Target field', record)
        # print('Random field', random_record)

        fields = {"context": correct_context_field,
                  # "random_context": random_context_field,
                  "code": code_field,
                  "record_indices": MetadataField(index)
                 }

        if all_code_field is not None:
            fields.update({'all_code': all_code_field,
                           'all_train_records': all_code_str_field,
                           'current_record': MetadataField(record)})
        if all_code_str_field is not None:
            fields.update({'all_train_records': all_code_str_field})

        if all_context_field is not None:
            fields.update({'all_context': all_context_field})

        return Instance(fields)

    def preprocess_code(self, code):
        return [c for c in code if len(c) > 1][:75]


    def return_stripped_rec(self, rec):
        # Saves memory by not keeping other keys
        new_rec = {'nl': rec['nl'],
                'code': rec['code'],
                'method_name': rec['methodName'],
                'class_name': rec['className'],
                'path': rec['path'],
                }
        if 'orig_code' in rec:
            tokenized = RegexpTokenizer(r'\w+').tokenize(rec['orig_code'])
            new_rec.update({'orig_code': rec['orig_code'],
                            'orig_code_split': tokenized})
        return new_rec

    def get_context_field(self, record):
        utterance = record['nl']
        # record['varNames'],
        # record['varTypes'],
        # record['methodNames'],
        # record['methodReturns'],
        method_name=record['method_name']
        class_name=record['class_name']

        utterance_tokens = [t for t in utterance if t not in STOPS]
        utterance_tokens = utterance_tokens[:20]
        utterance_tokens = [t.lower() for t in utterance_tokens]
        if len(utterance_tokens) < 1:
            utterance_tokens = ['a']

        context_lst = ['|MethodName|']
        context_lst.extend(self.split_camel_case_add_original(method_name))

        # context_lst.append('|ClassName|')
        # context_lst.extend(self.split_camel_case_add_original(class_name))

        # context_lst = ['|VariableName|']
        # for name in variable_names[:5]:
        #     context_lst.extend(self.split_camel_case_add_original(name))
        # context_lst.append('|VariableType|')
        # for name in variable_types[:5]:
        #     context_lst.extend(self.split_camel_case_add_original(name))
        # context_lst.append('|MethodName|')
        # for name in method_names[:5]:
        #     context_lst.extend(self.split_camel_case_add_original(name))
        # context_lst.append('|MethodType|')
        # for name in method_types[:5]:
        #     context_lst.extend(self.split_camel_case_add_original(name))

        context_lst.append('|Utterance|')
        context_lst.extend(utterance_tokens)

        context_field = TextField([Token(t) for t in context_lst], self._utterance_indexers)
        return context_field

    @staticmethod
    def split_rule(rule):
        return rule.split('-->')
    @staticmethod
    def create_unk_rule(nt):
        return nt + '-->' + UNK

    # def get_field_from_method_variable_names(self, words: List[str]) -> ListField:
    #     # For each variable or method name, this method splits it
    #     # on camel case then generates a TextField for each one.
    #     fields: List[Field] = []
    #     for word in words:
    #         tokens = [Token(text=w.lower()) for w in self.split_camel_case_add_original(word)]
    #         fields.append(TextField(tokens, self._utterance_indexers))
    #     return fields
    #     # return ListField(fields)

    def split_camel_case_add_original(self, name: str) -> List[str]:
        # Returns the string and its camel case split version.
        tokens = self.split_camel_case(name)
        lower_tokens = [t.lower() for t in tokens]
        if len(lower_tokens) > 1:
            lower_tokens = [name] + lower_tokens
        return lower_tokens

    @staticmethod
    def split_camel_case(name: str) -> List[str]:
        # Returns the string and its camel case split version.
        tokens = re.sub('(?!^)([A-Z][a-z]+)', r' \1', name).split()
        return tokens

    @classmethod
    def from_params(cls, params: Params) -> 'JavaSearchDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        # todo(rajas): utterance indexer should be renamed to identifier indexer
        utterance_indexers = TokenIndexer.dict_from_params(params.pop('utterance_indexers'))
        min_identifier_count = params.pop_int('min_identifier_count')
        num_dataset_instances = params.pop_int('num_dataset_instances', -1)
        linking_feature_extracters = params.pop('linking_feature_extractors', None)
        # identifier_indexers = TokenIndexer.dict_from_params(params.pop('identifier_indexers'))
        code_indexers = TokenIndexer.dict_from_params(params.pop('code_indexers'))
        params.assert_empty(cls.__name__)
        return cls(utterance_indexers=utterance_indexers,
                   min_identifier_count=min_identifier_count,
                   num_dataset_instances=num_dataset_instances,
                   linking_feature_extractors=linking_feature_extracters,
                   # identifier_indexers=identifier_indexers,
                   code_indexers=code_indexers,
                   tokenizer=tokenizer,
                   lazy=lazy)