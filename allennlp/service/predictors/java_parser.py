import json
from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('java-parser')
class JavaParserPredictor(Predictor):
    """
    Wrapper for the
    :class:`~allennlp.models.encoder_decoders.wikitables_semantic_parser.WikiTablesSemanticParser`
    model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

        file_path = "/home/rajas/semparse/java-programmer/data/train.dataset"
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        self.global_production_rule_fields, self.global_rule2index = dataset_reader.get_global_rule_fields(dataset)


    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"question": "...", "table": "..."}``.
        The format of the table string is:
            methodNames parseDetails submit ...
            variableNames isReady counter ...
        """
        utterance_text = json_dict["question"]
        table_text = json_dict["table"]

        java_class = {'methodNames': [], 'variableNames': []}

        for row_index, line in enumerate(table_text.split('\n')):
            line = line.rstrip('\n')
            tokens = line.split()
            java_class[tokens[0]].extend(tokens[1:])

        tokenized_utterance = utterance_text.split()


        instance = self._dataset_reader.text_to_instance(tokenized_utterance,
                                                         java_class['variableNames'],
                                                         java_class['methodNames'],
                                                         self.global_production_rule_fields,
                                                         self.global_rule2index)

        extra_info = {'question_tokens': tokenized_utterance}
        return instance, extra_info

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        instance, return_dict = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance, cuda_device)
        return_dict.update(outputs)
        return sanitize(return_dict)