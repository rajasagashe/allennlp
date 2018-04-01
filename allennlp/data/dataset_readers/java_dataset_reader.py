"""
Reader for Java Dataset
"""

from typing import Dict, List, Union
import gzip
import logging
import os
import json

import re
import numpy as np
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common import Params
from allennlp.common.util import JsonDict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, IndexField, KnowledgeGraphField, ListField, ArrayField
from allennlp.data.fields import MetadataField, ProductionRuleField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.semparse.knowledge_graphs import TableQuestionKnowledgeGraph
from allennlp.semparse.type_declarations import wikitables_type_declaration as wt_types
from allennlp.semparse.worlds import WikiTablesWorld
from allennlp.semparse.worlds.world import ParsingError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("java")
class JavaDatasetReader(DatasetReader):
    """


    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._nonterminal_indexers = {"tokens": SingleIdTokenIndexer("rule_labels")}
        self._terminal_indexers = {"token_characters": TokenCharactersIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        logger.info("Reading the dataset")
        for record in dataset:
            instance = self.text_to_instance(record['nl'],
                                             record['varNames'],
                                             record['varTypes'],
                                             record['methodNames'],
                                             record['methodReturns'],
                                             record['code'],
                                             record['rules'])
            yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         code_summary: str,
                         variable_names: List[str],
                         variable_types: List[str],
                         method_names: List[str],
                         method_return_types: List[str],
                         code: str,
                         rules: List[str]) -> Instance:

        variable_name_fields = self.add_camel_case_split_tokens(variable_names)
        method_name_fields = self.add_camel_case_split_tokens(method_names)

        variable_types_field = TextField([Token(t.lower()) for t in variable_types], self._token_indexers)
        method_return_types_field = TextField([Token(t.lower()) for t in method_return_types], self._token_indexers)

        code_summary = [t for word in code_summary for t in self.split_camel_case(word)]
        code_summary_field = TextField([Token(t) for t in code_summary], self._token_indexers)
        code_field = TextField([Token(t) for t in code], self._token_indexers)

        pre_rules = [r.replace('-->', ' -> ') for r in rules]
        production_rule_fields: List[Field] = []
        for production_rule in pre_rules:
            field = ProductionRuleField(production_rule,
                                        terminal_indexers=self._terminal_indexers,
                                        nonterminal_indexers=self._nonterminal_indexers,
                                        # todo refine this
                                        is_nonterminal=lambda x: (x[0].isupper() or x.startswith("return")))

            production_rule_fields.append(field)

        # Compute parent rules
        nt2rule_index = {}
        for i, rule in enumerate(rules):
            nt2rule_index[rule.split('-->')[0]] = i
        rule_parent_index = np.zeros(len(rules)) - 1
        # print(rules)
        # print("len rules", len(rules))
        # print("len nt2rule", len(nt2rule_index))
        self.compute_rule_parent(rules, rule_parent_index, 0, -1, nt2rule_index)
        # Pad with -1 since 0 is valid index to rule 0.
        rule_parent_index_field = ArrayField(rule_parent_index, padding_value=-1)

        fields = {"utterance": code_summary_field,
                  "variable_names": variable_name_fields,
                  "variable_types": variable_types_field,
                  "method_names": method_name_fields,
                  "method_return_types": method_return_types_field,
                  "code": code_field,
                  "rules": ListField(production_rule_fields),
                  "rule_parent_index": rule_parent_index_field}

        return Instance(fields)

    def compute_rule_parent(self, rules: List[str], rule_parent_index, rule_index, parent_index, nt2rule_index):
        # print(rule_index)
        # print(rules[rule_index])
        rule_parent_index[rule_index] = parent_index
        lhs, rhs = rules[rule_index].split('-->')
        for child in rhs.split('___'):
            if lhs != "IdentifierNT" and child[0].isupper():
                # Verify that child is a non terminal.
                child_index = nt2rule_index[child]
                self.compute_rule_parent(rules, rule_parent_index, child_index, rule_index, nt2rule_index)


    def add_camel_case_split_tokens(self, words: List[str]) -> ListField:
        fields: List[Field] = []
        for word in words:
            tokens = [Token(text=w) for w in self.split_camel_case(word)]
            fields.append(TextField(tokens, self._token_indexers))
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
    def from_params(cls, params: Params) -> 'WikiTablesDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return cls(tokenizer=tokenizer, token_indexers=token_indexers, lazy=lazy)