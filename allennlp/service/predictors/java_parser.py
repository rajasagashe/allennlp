import json
from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor

PLAIN_IDENTIFIER_RULE = "<SOME_NAME>-->" + "<SOME_NAME>"
@Predictor.register('java-parser')
class JavaParserPredictor(Predictor):
    """
    Wrapper for the
    :class:`~allennlp.models.encoder_decoders.wikitables_semantic_parser.WikiTablesSemanticParser`
    model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

        file_path = "/home/rajas/semparse/java-programmer/data/added-path-methodname/train-with-edits.json"
        extra_file_path = "/home/rajas/semparse/java-programmer/data/added-path-methodname/extra-fromtrain-with-edits.json"


        # with open(file_path) as dataset_file:
        #     p2methods = json.load(dataset_file)
        #
        # dataset = []
        # for _, methods in p2methods.items():
        #     dataset += methods
        #     if self._num_dataset_instances != -1:
        #         if len(dataset) > self._num_dataset_instances:
        #             break
        # print('num_dataset_instances', dataset_reader._num_dataset_instances)
        # # if dataset_reader._num_dataset_instances != -1:
        # #     dataset = dataset[0:dataset_reader._num_dataset_instances]
        #
        # modified_rules = dataset_reader.split_identifier_rule_into_multiple([d['rules'] for d in dataset])
        # self.global_production_rule_fields, self.global_rule2index = dataset_reader.get_global_rule_fields(modified_rules)

        with open(file_path) as dataset_file:
            p2methods = json.load(dataset_file)
        train_dataset = []
        for _, methods in p2methods.items():
            train_dataset += methods
            if self._dataset_reader._num_dataset_instances != -1:
                if len(train_dataset) > self._dataset_reader._num_dataset_instances:
                    break
        with open(extra_file_path) as dataset_file:
            p2methods = json.load(dataset_file)
        self.extra_dataset = []
        for _, methods in p2methods.items():
            self.extra_dataset += methods

        modified_rules = self._dataset_reader.split_identifier_rule_into_multiple(
            [d['rules'] for d in train_dataset]
        )

        self.global_production_rule_fields, self.global_rule2index = self._dataset_reader.get_global_rule_fields(modified_rules)



    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"question": "...", "table": "..."}``.
        The format of the table string is:
            methodNames parseDetails submit ...
            variableNames isReady counter ...
        """
        table_text = json_dict["table"]

        # utterance_text = json_dict["question"]
        # java_class = {'variable_names': [],
        #               'variable_types': [],
        #               'method_names': [],
        #               'method_types': []}
        #
        # class_category = ''

        lst = table_text.split('\n')
        methodName = lst[0]
        path = lst[1]

        for record in self.extra_dataset:
            if (record['methodName'] == methodName and
                record['path'] == path):
                break

        # now fetch record from validation dataset
        # nl = table_text[0]

        # for row_index, line in enumerate(table_text.split('\n')):
        #     line = line.rstrip('\n')
            # if ':' in line:
            #     class_category = line
            # else:
            #     if class_category != ''
            #     name, type = line.split('(')
            #     type = type.split(')')[0]
            #     if class_category == 'Variables:':
            #         java_class['variable_names'].append(name)
            #         java_class['variable_types'].append(type)
            #     else:
            #         java_class['method_names'].append(name)
            #         java_class['method_types'].append(type)
        modified_targ_rules = self._dataset_reader.split_identifier_rule_into_multiple([record['rules']])
        modified_proto_rules = self._dataset_reader.split_identifier_rule_into_multiple([record['prototype_rules']])

        for rules in modified_proto_rules:
            for i,rule in enumerate(rules):
                # lhs, rhs = rule.split('-->')
                # No class field/func identifier rules since these cannot be embedded
                if rule not in self.global_rule2index:
                    rules[i] = PLAIN_IDENTIFIER_RULE
        instance = self._dataset_reader.text_to_instance(record['nl'],
                              record['varNames'],
                              record['varTypes'],
                              record['methodNames'],
                              record['methodReturns'],
                              self.global_production_rule_fields,
                              self.global_rule2index,

                              # Pass the modified identifier rules.
                              rules=modified_targ_rules[0],
                              proto_rules=modified_proto_rules[0],
                              proto_tokens=record['prototype_nl'],
                              proto_code=record['prototype_code'],
                              proto_og_rules=record['prototype_rules'],
                              protoMethodName=record['prototype_methodName'],
                              methodName=record['methodName'],
                              code=record['code'],
                              path=record['path'])
        print("Processed tokens", instance.fields['metadata'].metadata['utterance'])
        extra_info = {'question_tokens': instance.fields['metadata'].metadata['utterance'],
                      'prototype_rules': modified_proto_rules[0]}  #record['nl']}
        return instance, extra_info

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        instance, return_dict = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance, cuda_device)
        return_dict.update(outputs)
        return sanitize(return_dict)