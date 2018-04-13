from collections import defaultdict
from typing import Dict, Tuple, List, Set, Any

from allennlp.training.metrics import Average
from overrides import overrides

import torch
from torch.autograd import Variable
from torch.nn.modules.rnn import LSTMCell

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.nn.util import get_text_field_mask
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.nn.decoding import BeamSearch, JavaGrammarState, MaximumLikelihood, RnnState
from allennlp.models.encoder_decoders.java.java_decoder_state import JavaDecoderState
from allennlp.models.encoder_decoders.java.java_decoder_step import JavaDecoderStep

# from java_programmer.allennlp_in_progress.data.vocabulary import Vocabulary
# from java_programmer.allennlp_in_progress.nn import util
# from java_programmer.allennlp_in_progress.beam_search import BeamSearch
# from java_programmer.allennlp_in_progress.rnn_state import RnnState
# from java_programmer.fields.java_production_rule_field import ProductionRuleArray
# from java_programmer.models.java_decoder_step import JavaDecoderStep
# from java_programmer.models.java_decoder_state import JavaDecoderState
# from java_programmer.grammar.java_grammar_state import JavaGrammarState
# from java_programmer.models.maximum_likelihood import MaximumLikelihood

START_SYMBOL = "MemberDeclaration"

@Model.register("java_parser")
class JavaSemanticParser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 utterance_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 # nonterminal_embedder: TextFieldEmbedder,
                 max_decoding_steps: int,
                 decoder_beam_search: BeamSearch,
                 action_embedding_dim: int,
                 dropout: float = 0.0,
                 num_linking_features: int = 8,
                 rule_namespace: str = 'rule_labels',
                 attention_function: SimilarityFunction = None) -> None:
        super(JavaSemanticParser, self).__init__(vocab)
        self._utterance_embedder = utterance_embedder
        # self._embed_terminals = True
        # self._nonterminal_embedder = nonterminal_embedder
        # self._terminal_embedder = nonterminal_embedder

        self._max_decoding_steps = max_decoding_steps
        self._decoder_beam_search = decoder_beam_search

        self._encoder = encoder

        # self._embedding_dim = self._nonterminal_embedder.get_output_dim()
        # self._decoder_output_dim = self._encoder.get_output_dim()

        self._input_attention = Attention(attention_function)

        # self._decoder_step = JavaDecoderStep(encoder_output_dim=self._encoder.get_output_dim(),
        #                                      attention_function=attention_function,
        #                                      nonterminal_embedder=nonterminal_embedder)
        self._decoder_trainer = MaximumLikelihood()
        self._action_sequence_accuracy = Average()

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._rule_namespace = rule_namespace

        self._action_padding_index = -1  # the padding value used by IndexField
        self._action_embedder = Embedding(num_embeddings=vocab.get_vocab_size(self._rule_namespace),
                                          embedding_dim=action_embedding_dim)

        # This is what we pass as input in the first step of decoding, when we don't have a
        # previous action, or a previous question attention.
        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        self._first_attended_question = torch.nn.Parameter(torch.FloatTensor(encoder.get_output_dim()))
        torch.nn.init.normal(self._first_action_embedding)
        torch.nn.init.normal(self._first_attended_question)

        self._decoder_step = JavaDecoderStep(encoder_output_dim=self._encoder.get_output_dim(),
                                                   action_embedding_dim=action_embedding_dim,
                                                   attention_function=attention_function,
                                                   dropout=dropout)

    @overrides
    def forward(self,  # type: ignore
                utterance: Dict[str, torch.LongTensor],
                variable_names: Dict[str, torch.LongTensor],
                variable_types: Dict[str, torch.LongTensor],
                method_names: Dict[str, torch.LongTensor],
                method_return_types: Dict[str, torch.LongTensor],
                actions: List[List[ProductionRuleArray]],
                rules: List[torch.Tensor]) -> Dict[str, torch.Tensor]:

        # Encode summary, variables, methods with bi lstm.
        ##################################################

        # (batch_size, input_sequence_length, encoder_output_dim)
        embedded_utterance = self._utterance_embedder(utterance)
        batch_size, _, _ = embedded_utterance.size()
        utterance_mask = get_text_field_mask(utterance)

        # (batch_size, question_length, encoder_output_dim)
        # todo(rajas): add the dropout back in
        # encoder_outputs = self._dropout(self._encoder(embedded_utterance, utterance_mask))
        encoder_outputs = self._encoder(embedded_utterance, utterance_mask)

        # This will be our initial hidden state and memory cell for the decoder LSTM.
        # todo(rajas) change back to encoder is bidirectional from True
        final_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                             utterance_mask,
                                                             True)
        memory_cell = Variable(encoder_outputs.data.new(batch_size, self._encoder.get_output_dim()).fill_(0))

        initial_score = Variable(embedded_utterance.data.new(batch_size).fill_(0))

        action_embeddings, action_indices = self._embed_actions(actions)

        # _, num_entities, num_question_tokens = linking_scores.size()
        # flattened_linking_scores, actions_to_entities = self._map_entity_productions(linking_scores,
        #                                                                              world,
        #                                                                              actions)

        # if target_action_sequences is not None:
        #     # Remove the trailing dimension (from ListField[ListField[IndexField]]).
        #     target_action_sequences = target_action_sequences.squeeze(-1)
        #     target_mask = target_action_sequences != self._action_padding_index
        # else:
        #     target_mask = None

        # To make grouping states together in the decoder easier, we convert the batch dimension in
        # all of our tensors into an outer list.  For instance, the encoder outputs have shape
        # `(batch_size, question_length, encoder_output_dim)`.  We need to convert this into a list
        # of `batch_size` tensors, each of shape `(question_length, encoder_output_dim)`.  Then we
        # won't have to do any index selects, or anything, we'll just do some `torch.cat()`s.
        initial_score_list = [initial_score[i] for i in range(batch_size)]
        encoder_output_list = [encoder_outputs[i] for i in range(batch_size)]
        question_mask_list = [utterance_mask[i] for i in range(batch_size)]
        initial_rnn_state = []
        for i in range(batch_size):
            initial_rnn_state.append(RnnState(final_encoder_output[i],
                                              memory_cell[i],
                                              self._first_action_embedding,
                                              self._first_attended_question,
                                              encoder_output_list,
                                              question_mask_list))
        initial_grammar_state = [self._create_grammar_state(actions[i])
                                 for i in range(batch_size)]
        initial_state = JavaDecoderState(batch_indices=list(range(batch_size)),
                                               action_history=[[] for _ in range(batch_size)],
                                               score=initial_score_list,
                                               rnn_state=initial_rnn_state,
                                               grammar_state=initial_grammar_state,
                                               action_embeddings=action_embeddings,
                                               action_indices=action_indices,
                                               possible_actions=actions,
                                               flattened_linking_scores=flattened_linking_scores,
                                               actions_to_entities=actions_to_entities,
                                               entity_types=entity_type_dict,
                                               debug_info=None)


        if self.training:
            return self._decoder_trainer.decode(initial_state,
                                                self._decoder_step,
                                                rules,
                                                action_map)
        else:
            outputs = {}
            if rules is not None:
                outputs['loss'] = self._decoder_trainer.decode(initial_state,
                                             self._decoder_step,
                                             rules)['loss']
            # print("Decoding------------")
            # print('nonterminal action indices', nonterminals_action_indices)
            num_steps = self._max_decoding_steps
            best_final_states = self._decoder_beam_search.search(num_steps,
                                                                 initial_state,
                                                                 self._decoder_step,
                                                                 keep_final_unfinished_states=False)
            # print("BEST FINAL STATES", len(best_final_states))
            outputs['rules'] = []
            for i in range(batch_size):
                if i in best_final_states:
                    credit = 0
                    if rules is not None:
                        credit = self._action_history_match(best_final_states[i][0].action_history[0],
                                                            rules[i])
                    self._action_sequence_accuracy(credit)
                    outputs['rules'].append(best_final_states[i][0].grammar_state[0]._action_history[0])
                # todo rajas remove else
                # else:
                #     credit = 0
                #     if rules is not None:
                #         credit = self._action_history_match(best_final_states[i].action_history[0],
                #                                             rules[i])
                #     self._action_sequence_accuracy(credit)
                #     outputs['rules'].append(best_final_states[i][0].grammar_state[0]._action_history[0])
            return outputs

    @staticmethod
    def _action_history_match(predicted: List[int], targets: torch.LongTensor) -> int:
        # Check if target is big enough to cover prediction (including start/end symbols)

        # print("Predicted", predicted)
        # print("Target", targets)
        # if len(predicted) > targets.size(0):
        #     return 0
        min_length = min(len(predicted), targets.size(0))

        # todo(rajas) target is padded with 0's so you need to remove those firstly:
        # todo change the padding to -1 as it's more clear

        # predicted_tensor = targets.data.new(predicted)
        predicted_tensor = Variable(targets.data.new(predicted))
        predicted_tensor = predicted_tensor[:min_length].long()
        targets_trimmed = targets[:min_length].long()
        # Return 1 if the predicted sequence is anywhere in the list of targets.
        x = targets_trimmed.eq(predicted_tensor)
        num_correct = torch.sum(x)
        num_correct = num_correct.data.cpu().numpy().tolist()[0]
        # print('numcorrect', num_correct)
        return num_correct / min_length


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'parse_acc': self._action_sequence_accuracy.get_metric(reset)
        }
        # return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    @staticmethod
    def _create_grammar_state(possible_actions: List[ProductionRuleArray]) -> JavaGrammarState:
        # valid_actions = world.get_valid_actions()
        # action_mapping = {}
        # for i, action in enumerate(possible_actions):
        #     action_string = action[0]
        #     action_mapping[action_string] = i
        # translated_valid_actions = {}
        # for key, action_strings in valid_actions.items():
        #     translated_valid_actions[key] = [action_mapping[action_string]
        #                                      for action_string in action_strings]
        # return JavaGrammarState([START_SYMBOL], {}, translated_valid_actions, action_mapping)
        nonterminal2action_index = defaultdict(list)
        for i, action in enumerate(possible_actions):
            lhs = action[0].split(' -> ')[0]
            nonterminal2action_index[lhs].append(i)
        return JavaGrammarState([START_SYMBOL], nonterminal2action_index)

    def _embed_actions(self, actions: List[List[ProductionRuleArray]]) -> Tuple[torch.Tensor,
                                                                                Dict[Tuple[int, int], int]]:
        """
        Given all of the possible actions for all batch instances, produce an embedding for them.
        There will be significant overlap in this list, as the production rules from the grammar
        are shared across all batch instances.  Our returned tensor has an embedding for each
        `unique` action, so we also need to return a mapping from the original ``(batch_index,
        action_index)`` to our new ``global_action_index``, so that we can get the right action
        embedding during decoding.

        Returns
        -------
        action_embeddings : ``torch.Tensor``
            Has shape ``(num_unique_actions, action_embedding_dim)``.
        action_map : ``Dict[Tuple[int, int], int]``
            Maps ``(batch_index, action_index)`` in the input action list to ``action_index`` in
            the ``action_embeddings`` tensor.  All non-embeddable actions get mapped to `-1` here.
        """
        embedded_actions = self._action_embedder.weight

        # Now we just need to make a map from `(batch_index, action_index)` to
        # `global_action_index`.  global_action_ids has the list of all unique actions; here we're
        # going over all of the actions for each batch instance so we can map them to the global
        # action ids.
        action_vocab = self.vocab.get_token_to_index_vocabulary(self._rule_namespace)
        action_map: Dict[Tuple[int, int], int] = {}
        for batch_index, instance_actions in enumerate(actions):
            for action_index, action in enumerate(instance_actions):
                if not action[0]:
                    # This rule is padding.
                    continue
                global_action_id = action_vocab.get(action[0], -1)
                action_map[(batch_index, action_index)] = global_action_id
        return embedded_actions, action_map


    @classmethod
    def from_params(cls, vocab, params: Params) -> 'SimpleSeq2Seq':
        utterance_embedder_params = params.pop("utterance_embedder")
        utterance_embedder = TextFieldEmbedder.from_params(vocab, utterance_embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        max_decoding_steps = params.pop("max_decoding_steps")
        action_embedding_dim = params.pop_int("action_embedding_dim")
        decoder_beam_search = BeamSearch.from_params(params.pop("decoder_beam_search"))

        # nonterminal_embedder = TextFieldEmbedder.from_params(vocab, params.pop("nonterminal_embedder"))
        # terminal_embedder_params = params.pop('terminal_embedder', None)
        # if terminal_embedder_params:
        #     terminal_embedder = TextFieldEmbedder.from_params(vocab, terminal_embedder_params)
        # else:
        #     terminal_embedder = None
        # If no attention function is specified, we should not use attention, not attention with
        # default similarity function.
        attention_function_type = params.pop("attention_function", None)
        if attention_function_type is not None:
            attention_function = SimilarityFunction.from_params(attention_function_type)
        else:
            attention_function = None
        scheduled_sampling_ratio = params.pop_float("scheduled_sampling_ratio", 0.0)
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   utterance_embedder=utterance_embedder,
                   encoder=encoder,
                   attention_function=attention_function,
                   # nonterminal_embedder=nonterminal_embedder,
                   # terminal_embedder=terminal_embedder,
                   action_embedding_dim=action_embedding_dim,
                   max_decoding_steps=max_decoding_steps,
                   decoder_beam_search=decoder_beam_search)
