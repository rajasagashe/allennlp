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
from allennlp.data.fields import Field, NoCountListField, ArrayField, NoCountListFieldBatch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

# from java_programmer.fields import JavaProductionRuleField, JavaGlobalProductionRuleField, ProductionRuleField
from allennlp.semparse import KnowledgeGraph
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

NUM_PROTOTYPES = 3
@DatasetReader.register("java_search")
class JavaSearchDatasetReader(DatasetReader):
    def __init__(self,
                 utterance_indexers: Dict[str, TokenIndexer],
                 code_indexers: Dict[str, TokenIndexer],
                 num_dataset_instances: int,
                 tokenizer: Tokenizer = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._utterance_indexers = utterance_indexers
        # self._code_indexers = code_indexers
        self._code_indexers = utterance_indexers
        self._num_dataset_instances = num_dataset_instances
        self._tokenizer = tokenizer or WordTokenizer()

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

        # Remove records which don't have prototype keys. This can happen since
        # LSH wasn't able to find any neighbors
        dataset = [r for r in dataset if self.has_prototype(r)]


        if os.path.basename(file_path).startswith('train'):
            self.train_dataset = dataset


        # We are evaluating, so load it in
        if not hasattr(self, 'train_dataset'):
            with open('/home/rajas/final/data/official/train-3proto-oraclebleu-all.dataset') as dataset_file:
                self.train_dataset = json.load(dataset_file)
            self.train_dataset = [r for r in self.train_dataset if self.has_prototype(r)]


        contexts = []
        # Very important that train_dataset is used here since during validation the
        # best context from train is used not validation!
        for rec in self.train_dataset:
            contexts.append(self.get_context_field(rec['nl']))
        all_context_field = NoCountListField(contexts)

        all_train_records = MetadataField(self.train_dataset)

        logger.info("Reading the dataset")
        for i, record in enumerate(dataset):
            instance = self.text_to_instance(record=record,
                                             index=i,
                                             all_train_records=all_train_records,
                                             all_context_field=all_context_field)
            yield instance

    def has_prototype(self, rec):
        return 'prototype_nl0' in rec

    @overrides
    def text_to_instance(self,  # type: ignore
                         record,
                         # random_record,
                         index,
                         all_context_field,
                         # all_code_field=None,
                         all_train_records=None,
                         ) -> Instance:

        curr_context_field = self.get_context_field(record['nl'])
        prototype_context_field, prototype_bleu_field = self.get_prototype_context_fields(record)
        random_context_field, random_bleu_field = self.get_random_context_fields(record['nl'])


        fields = {'current_context': curr_context_field,
                  "prototype_context": prototype_context_field,
                  # "random_context": random_context_field,
                  'random_bleu': random_bleu_field,
                  'prototype_bleu': prototype_bleu_field,
                  "record_indices": MetadataField(index),
                  'all_context': all_context_field,
                  'all_train_records': all_train_records,
                  'current_record': MetadataField(record)
                 }

        return Instance(fields)

    def get_random_context_fields(self, curr_nl):
        fields = []
        bleus = []

        for i in range(NUM_PROTOTYPES):
            rec = random.choice(self.train_dataset)
            fields.append(self.get_context_field(rec['nl']))
            bleus.append(self._get_bleu(curr_nl, rec['nl']))

        # print('Random bleus', bleus)
        return NoCountListFieldBatch(fields), ArrayField(np.array(bleus))

    def get_prototype_context_fields(self, rec):
        fields = []
        bleus = []

        for i in range(NUM_PROTOTYPES):
            protokey = 'prototype_nl' + str(i)
            fields.append(self.get_context_field(rec[protokey]))
            # if self._get_bleu(rec['nl'], rec[protokey]) == 1.0:
                # print(rec[protokey])
                # print(rec['code'])
                # print(rec['prototype_code0'])
            bleus.append(self._get_bleu(rec['nl'], rec[protokey]))

        # print('Prototype bleus', bleus)
        return NoCountListFieldBatch(fields), ArrayField(np.array(bleus))

    def get_context_field(self, utterance):
        # record['varNames'],
        # record['varTypes'],
        # record['methodNames'],
        # record['methodReturns'],

        # method_name=record['method_name']
        # class_name=record['class_name']

        # utterance_tokens = [t for t in utterance if t not in STOPS]
        utterance_tokens = utterance[:25]
        utterance_tokens = [t.lower() for t in utterance_tokens]
        if len(utterance_tokens) < 1:
            utterance_tokens = ['a']

        # context_lst = ['|MethodName|']
        # context_lst.extend(self.split_camel_case_add_original(method_name))

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

        context_lst = utterance_tokens
        # context_lst.extend(utterance_tokens)

        context_field = TextField([Token(t) for t in context_lst], self._utterance_indexers)
        return context_field


    @staticmethod
    def _get_bleu(gold_seq, pred_seq):
        # This is how Ling et al. compute bleu score.
        sm = SmoothingFunction()
        ngram_weights = [0.25] * min(4, len(gold_seq))
        return sentence_bleu([gold_seq], pred_seq, weights=ngram_weights, smoothing_function=sm.method3)

    # def preprocess_code(self, code):
    #     return [c for c in code if len(c) > 1][:75]
    # def return_stripped_rec(self, rec):
    #     # Saves memory by not keeping other keys
    #     new_rec = {'nl': rec['nl'],
    #             'code': rec['code'],
    #             'method_name': rec['methodName'],
    #             'class_name': rec['className'],
    #             'path': rec['path'],
    #             }
    #     if 'orig_code' in rec:
    #         tokenized = RegexpTokenizer(r'\w+').tokenize(rec['orig_code'])
    #         new_rec.update({'orig_code': rec['orig_code'],
    #                         'orig_code_split': tokenized})
    #     return new_rec

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
        num_dataset_instances = params.pop_int('num_dataset_instances', -1)
        code_indexers = TokenIndexer.dict_from_params(params.pop('code_indexers'))

        params.assert_empty(cls.__name__)
        return cls(utterance_indexers=utterance_indexers,
                   num_dataset_instances=num_dataset_instances,
                   # identifier_indexers=identifier_indexers,
                   code_indexers=code_indexers,
                   tokenizer=tokenizer,
                   lazy=lazy)