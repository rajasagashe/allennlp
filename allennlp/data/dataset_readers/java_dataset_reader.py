from collections import defaultdict, Counter
from typing import Dict, List
import logging
import json

import re
import numpy as np
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ProductionRuleField
from allennlp.data.fields import Field, ListField, ArrayField
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

# from java_programmer.fields import JavaProductionRuleField, JavaGlobalProductionRuleField, ProductionRuleField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
LITERALS_TO_TRIM = ["IdentifierNT", "Nt_decimal_literal", "Nt_char_literal",   "Nt_float_literal", "Nt_hex_literal", "Nt_oct_literal"]

@DatasetReader.register("java")
class JavaDatasetReader(DatasetReader):
    def __init__(self,
                 utterance_indexers: Dict[str, TokenIndexer],
                 min_identifier_count: int,
                 num_dataset_instances: int,
                 tokenizer: Tokenizer = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._utterance_indexers = utterance_indexers
        self._min_identifier_count = min_identifier_count
        self._num_dataset_instances = num_dataset_instances
        self._tokenizer = tokenizer or WordTokenizer()

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)

        if self._num_dataset_instances != -1:
            dataset = dataset[0:self._num_dataset_instances]

        # Prepare global production rules to be used in JavaGrammarState.
        lhs2unique_rhs = self.trim_grammar_identifiers_literals(dataset)

        global_production_rule_field, rule2index = self.get_grammar_fields(lhs2unique_rhs)


        logger.info("Reading the dataset")
        for record in dataset:
            instance = self.text_to_instance(record['nl'],
                                             record['varNames'],
                                             record['varTypes'],
                                             record['methodNames'],
                                             record['methodReturns'],
                                             record['code'],
                                             record['rules'],
                                             global_production_rule_field,
                                             rule2index)
            yield instance

    def get_grammar_fields(self, lhs2all_rhs):
        # Converts all the rules into ProductionRuleFields and returns a dictionary
        # from the rule to its index in the ListField of ProductionRuleFields.
        production_rule_fields = []

        rule2index = {}
        index = 0
        for lhs, all_rhs in lhs2all_rhs.items():
            for rhs in all_rhs:
                rule = lhs+'-->'+'___'.join(rhs)
                field = ProductionRuleField(rule, is_global_rule=True)
                production_rule_fields.append(field)

                rule2index[rule] = index
                index += 1
        return ListField(production_rule_fields), rule2index

    def trim_grammar_identifiers_literals(self, dataset):
        # This is used remove identifier and literal rules, if the occur below
        # a self._min_identifier_count.

        # First get all unique rhsides and their counts.
        lhs2unique_rhs = defaultdict(set)
        rhs_token2count = Counter()
        for record in dataset:
            rules = record['rules']
            for rule in rules:
                lhs, rhs = rule.split('-->')
                rhs_tokens = rhs.split('___')
                lhs2unique_rhs[lhs].add(tuple(rhs_tokens))
                rhs_token2count.update(rhs_tokens)
        for lhs in lhs2unique_rhs:
            # We only trim from a predefined set of identifiers and literals. This means that
            # some literals aren't trimmed, but that's because the expand to a small set of
            # terminals so it's not necessary.
            if lhs in LITERALS_TO_TRIM:
                unique_rhs = lhs2unique_rhs[lhs]
                trimmed_rhs = []
                for rhs_tup in unique_rhs:
                    # We assume that the identifier and literal right hand sides
                    # will only have one value in the tuple.
                    if rhs_token2count[rhs_tup[0]] >= self._min_identifier_count:
                        trimmed_rhs.append(rhs_tup)
                lhs2unique_rhs[lhs] = trimmed_rhs

        # Now add the unk rules for each literal
        for trimmed_nonterminal in LITERALS_TO_TRIM:
            if trimmed_nonterminal in lhs2unique_rhs:
                lhs2unique_rhs[trimmed_nonterminal].append(tuple(['<UNK>']))
        return lhs2unique_rhs

    @overrides
    def text_to_instance(self,  # type: ignore
                         utterance: str,
                         variable_names: List[str],
                         variable_types: List[str],
                         method_names: List[str],
                         method_return_types: List[str],
                         code: str,
                         rules: List[str],
                         global_rule_field: ListField,
                         rule2index: Dict[str, int]) -> Instance:

        variable_name_fields = self.add_camel_case_split_tokens(variable_names)
        method_name_fields = self.add_camel_case_split_tokens(method_names)

        # todo(rajas) change the indexers below to type indexers
        variable_types_field = TextField([Token(t.lower()) for t in variable_types], self._utterance_indexers)
        method_return_types_field = TextField([Token(t.lower()) for t in method_return_types], self._utterance_indexers)

        utterance = [t for word in utterance for t in self.split_camel_case(word)]
        utterance_field = TextField([Token(t) for t in utterance], self._utterance_indexers)
        # code_field = TextField([Token(t) for t in code], self._utterance_indexers)

        # todo(rajas) remove unks when the environment copying added
        rule_indexes = []
        for rule in rules:
            if rule in rule2index:

                rule_indexes.append(rule2index[rule])
            else:
                lhs, rhs = rule.split('-->')
                rule_indexes.append(rule2index[lhs+'--><UNK>'])

        # todo(rajas) convert to an index field
        rule_field = ArrayField(np.array(rule_indexes), padding_value=-1)

        fields = {"utterance": utterance_field,
                  "variable_names": variable_name_fields,
                  "variable_types": variable_types_field,
                  "method_names": method_name_fields,
                  "method_return_types": method_return_types_field,
                  # "code": code_field,
                  "rules": rule_field,
                  "actions": global_rule_field}

        return Instance(fields)

    def add_camel_case_split_tokens(self, words: List[str]) -> ListField:
        fields: List[Field] = []
        for word in words:
            tokens = [Token(text=w) for w in self.split_camel_case(word)]
            fields.append(TextField(tokens, self._utterance_indexers)) # todo change this
        return ListField(fields)

    @staticmethod
    def split_camel_case(name: str) -> List[str]:
        """
        """
        # TODO Lowercase them all
        tokens = re.sub('(?!^)([A-Z][a-z]+)', r' \1', name).split()
        if len(tokens) > 1:
            return [name] + tokens
        return tokens


    @classmethod
    def from_params(cls, params: Params) -> 'JavaDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        utterance_indexers = TokenIndexer.dict_from_params(params.pop('utterance_indexers'))
        min_identifier_count = params.pop_int('min_identifier_count')
        num_dataset_instances = params.pop_int('num_dataset_instances', -1)
        # identifier_indexers = TokenIndexer.dict_from_params(params.pop('identifier_indexers'))
        # type_indexers = TokenIndexer.dict_from_params(params.pop('type_indexers'))
        params.assert_empty(cls.__name__)
        return cls(utterance_indexers=utterance_indexers,
                   min_identifier_count=min_identifier_count,
                   num_dataset_instances=num_dataset_instances,
                   # identifier_indexers=identifier_indexers,
                   # type_indexers=type_indexers,
                   tokenizer=tokenizer,
                   lazy=lazy)