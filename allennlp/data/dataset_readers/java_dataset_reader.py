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
from allennlp.data.fields import ProductionRuleField, MetadataField, KnowledgeGraphField
from allennlp.data.fields import Field, ListField, ArrayField
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

# from java_programmer.fields import JavaProductionRuleField, JavaGlobalProductionRuleField, ProductionRuleField
from allennlp.semparse import KnowledgeGraph

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
LITERALS_TO_TRIM = ["IdentifierNT", "Nt_decimal_literal", "Nt_char_literal",   "Nt_float_literal", "Nt_hex_literal", "Nt_oct_literal"]

@DatasetReader.register("java")
class JavaDatasetReader(DatasetReader):
    def __init__(self,
                 utterance_indexers: Dict[str, TokenIndexer],
                 type_indexers: Dict[str, TokenIndexer],
                 min_identifier_count: int,
                 num_dataset_instances: int,
                 tokenizer: Tokenizer = None,
                 linking_feature_extractors: List[str] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._utterance_indexers = utterance_indexers
        self._type_indexers = type_indexers

        self._environment_token_indexers = {"tokens": SingleIdTokenIndexer()}

        self._min_identifier_count = min_identifier_count
        self._num_dataset_instances = num_dataset_instances
        self._tokenizer = tokenizer or WordTokenizer()
        self._linking_feature_extractors = linking_feature_extractors

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)

        if self._num_dataset_instances != -1:
            dataset = dataset[0:self._num_dataset_instances]

        global_production_rule_fields, global_rule2index = self.get_global_rule_fields(dataset)


        logger.info("Reading the dataset")
        for record in dataset:
            instance = self.text_to_instance(record['nl'],
                                             record['varNames'],
                                             record['varTypes'],
                                             record['methodNames'],
                                             record['methodReturns'],
                                             global_production_rule_fields,
                                             global_rule2index,
                                             record['code'],
                                             record['rules']
                                             )
            yield instance

    def get_global_rule_fields(self, dataset):
        # Prepare global production rules to be used in JavaGrammarState.
        lhs2all_rhs = self.trim_grammar_identifiers_literals(dataset)

        # Converts all the rules into ProductionRuleFields and returns a dictionary
        # from the rule to its index in the ListField of ProductionRuleFields.
        production_rule_fields = []

        rule2index = {}
        index = 0
        for lhs, all_rhs in lhs2all_rhs.items():
            for rhs in all_rhs:
                rule = lhs + '-->' + '___'.join(rhs)
                if 'sriniclass' in rule or 'srinifunc' in rule:
                    # Names from environment shouldn't be added to global rules.
                    continue
                field = ProductionRuleField(rule, is_global_rule=True)
                production_rule_fields.append(field)

                rule2index[rule] = index
                index += 1
        return production_rule_fields, rule2index

    @overrides
    def text_to_instance(self,  # type: ignore
                         utterance: List[str],
                         variable_names: List[str],
                         variable_types: List[str],
                         method_names: List[str],
                         method_types: List[str],
                         global_rule_fields: List[ProductionRuleField],
                         globalrule2index: Dict[str, int],
                         code: str = None,
                         rules: List[str] = None,
                         ) -> Instance:
        fields = {}

        variable_name_fields = self.get_field_from_method_variable_names(variable_names)
        method_name_fields = self.get_field_from_method_variable_names(method_names)

        variable_types_field = TextField([Token(t.lower()) for t in variable_types], self._type_indexers)
        method_return_types_field = TextField([Token(t.lower()) for t in method_types], self._type_indexers)


        # todo(rajas) add camel casing for utterance tokens as well
        # utterance = [t for word in utterance for t in self.split_camel_case(word)]
        utterance_tokens = [Token(t.lower()) for t in utterance]
        utterance_field = TextField(utterance_tokens, self._utterance_indexers)

        metadata_dict = {'utterance':  utterance,
                         'variableNames':  variable_names,
                         'variableTypes':  variable_types,
                         'methodNames':  method_names,
                         'methodTypes':  method_types}
        if code is not None:
            metadata_dict['code'] = code
        fields['metadata'] = MetadataField(metadata_dict)

        knowledge_graph = self.get_java_class_knowledge_graph(method_names=method_names, variable_names=variable_names)

        # We need to iterate over knowledge_graph's entities since these are sorted and each entity in
        # entity_tokens needs to correspond to knowledge_graph's entities.
        entity_tokens = []
        for entity in knowledge_graph.entities:
            entity_text = knowledge_graph.entity_text[entity]
            # Entity tokens should contain the camel case split of entity
            # e.g. isParsed -> is, Parsed and isParsed
            # So if the utterance has word parsed, it will appear in the exact text set
            # of knowledge graph field.
            entity_camel_split = self.split_camel_case(entity_text)
            entity_tokens.append([Token(e) for e in entity_camel_split])

        # todo(rajas): add a feature that filters out trivial links between utterance
        # and variables when the utterance has word 'is' and variable is 'isParsed'.
        # however if utterance has 'parsed' then it should be linked with 'isParsed'.
        java_class_field = KnowledgeGraphField(knowledge_graph=knowledge_graph,
                                               utterance_tokens=utterance_tokens,
                                               token_indexers=self._environment_token_indexers,
                                               entity_tokens=entity_tokens,
                                               feature_extractors=self._linking_feature_extractors)

        environment_rules, environmentrule2index = self.get_java_class_specific_rules(knowledge_graph)

        if rules is not None:
            target_rule_field = self.get_target_rule_field(rules, globalrule2index, environmentrule2index)
            fields['rules'] = target_rule_field


        entity_field = MetadataField(knowledge_graph.entities)

        fields.update({"utterance": utterance_field,
                       "variable_names": variable_name_fields,
                       "variable_types": variable_types_field,
                       "method_names": method_name_fields,
                       "method_return_types": method_return_types_field,
                       "actions": ListField(global_rule_fields + environment_rules),
                       "java_class": java_class_field,
                       "entities": entity_field })

        return Instance(fields)



    def trim_grammar_identifiers_literals(self, dataset):
        # This is used remove identifier and literal rules, if they occur below
        # a self._min_identifier_count. This is needed since the action_indexer
        # indexes the entire rule, so instead of having it map the entire rule
        # to an unk we want the rule to become Identifier-->UNK

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


    def get_java_class_knowledge_graph(self, method_names, variable_names):
        # Note that sriniclass_ prefix means that the name is copied from the
        # java class.
        # todo(rajas): add a class field for sriniclass and srinifunc
        entities = []
        entity_text = {}
        for name in (variable_names):
            entity = 'sriniclass_' + name
            entities.append(entity)
            entity_text[entity] = name
        for name in (method_names):
            entity = 'srinifunc_' + name
            entities.append(entity)
            entity_text[entity] = name

        neighbors = {entity: [] for entity in entity_text}
        knowledge_graph = KnowledgeGraph(entities=entities, entity_text=entity_text, neighbors=neighbors)
        return knowledge_graph

    def get_java_class_specific_rules(self, knowledge_graph: KnowledgeGraph):
        # Returns the rules unique per training instance, which copy variable
        # and method names from the java class.
        java_class_rules = []
        javaclassrule2index = {}
        for i, entity in enumerate(knowledge_graph.entities):
            rule = 'IdentifierNT-->' + entity
            field = ProductionRuleField(rule, is_global_rule=False)
            java_class_rules.append(field)
            javaclassrule2index[rule] = i
        return java_class_rules, javaclassrule2index

    def get_target_rule_field(self, rules, globalrule2index, environmentrule2index):
        rule_indexes = []
        for rule in rules:
            if rule in environmentrule2index:
                # Environment rule indexes come after last global index since the
                # environment rules are appended to the global rules list.
                rule_indexes.append(len(globalrule2index) + environmentrule2index[rule])
            elif rule in globalrule2index:
                rule_indexes.append(globalrule2index[rule])
            else:
                lhs, _ = rule.split('-->')
                rule_indexes.append(globalrule2index[lhs + '--><UNK>'])

        # todo(rajas) convert to an index field
        rule_field = ArrayField(np.array(rule_indexes), padding_value=-1)
        return rule_field

    def get_field_from_method_variable_names(self, words: List[str]) -> ListField:
        # For each variable or method name, this method splits it
        # on camel case then generates a TextField for each one.
        fields: List[Field] = []
        for word in words:
            tokens = [Token(text=w) for w in self.split_camel_case(word)]
            fields.append(TextField(tokens, self._utterance_indexers))
        return ListField(fields)

    @staticmethod
    def split_camel_case(name: str) -> List[str]:
        # Returns the string and its camel case split version all lowercased.
        tokens = re.sub('(?!^)([A-Z][a-z]+)', r' \1', name).split()
        if len(tokens) > 1:
            tokens = [name] + tokens
        tokens = [t.lower() for t in tokens]
        return tokens

    @classmethod
    def from_params(cls, params: Params) -> 'JavaDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        # todo(rajas): utterance indexer should be renamed to identifier indexer
        utterance_indexers = TokenIndexer.dict_from_params(params.pop('utterance_indexers'))
        min_identifier_count = params.pop_int('min_identifier_count')
        num_dataset_instances = params.pop_int('num_dataset_instances', -1)
        linking_feature_extracters = params.pop('linking_feature_extractors', None)
        # identifier_indexers = TokenIndexer.dict_from_params(params.pop('identifier_indexers'))
        type_indexers = TokenIndexer.dict_from_params(params.pop('type_indexers'))
        params.assert_empty(cls.__name__)
        return cls(utterance_indexers=utterance_indexers,
                   min_identifier_count=min_identifier_count,
                   num_dataset_instances=num_dataset_instances,
                   linking_feature_extractors=linking_feature_extracters,
                   # identifier_indexers=identifier_indexers,
                   type_indexers=type_indexers,
                   tokenizer=tokenizer,
                   lazy=lazy)