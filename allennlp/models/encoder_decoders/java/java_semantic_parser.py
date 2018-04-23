import json
from collections import defaultdict, Counter
from typing import Dict, Tuple, List, Set, Any
import gc
import re

from allennlp.training.metrics import Average
from overrides import overrides

import torch
from torch.autograd import Variable
from torch.nn.modules.rnn import LSTMCell
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from allennlp.common.util import timeit, debug_print
from allennlp.common import Params
from allennlp.common.util import timeit
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder, FeedForward
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
# START_SYMBOL = "MethodDeclaration"

@Model.register("java_parser")
class JavaSemanticParser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 utterance_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 mixture_feedforward: FeedForward,
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
        self._partial_production_rule_accuracy = Average()
        self._code_bleu = Average()
        self._exact_match_accuracy = Average()

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
        # torch.nn.init.normal(self._first_action_embedding)
        # torch.nn.init.normal(self._first_attended_question)
        torch.nn.init.constant(self._first_action_embedding, 0.0)
        torch.nn.init.constant(self._first_attended_question, 0.0)

        self._linking_params = torch.nn.Linear(num_linking_features, 1)

        self._decoder_step = JavaDecoderStep(encoder_output_dim=self._encoder.get_output_dim(),
                                             mixture_feedforward=mixture_feedforward,
                                             action_embedder=self._action_embedder,
                                             action_embedding_dim=action_embedding_dim,
                                             attention_function=attention_function,
                                             dropout=dropout)

        f = open('debug/pred_target_strings.csv', 'w+')
        f.write('Target, Predicted\n')
        f.close()

        codef = open('debug/pred_target_code.txt', 'w+')
        codef.write('Targ and Predicted Code\n')
        codef.close()

    @overrides
    # @timeit
    def forward(self,  # type: ignore
                utterance: Dict[str, torch.LongTensor],
                # variable_names: Dict[str, torch.LongTensor],
                # variable_types: Dict[str, torch.LongTensor],
                # method_names: Dict[str, torch.LongTensor],
                # method_return_types: Dict[str, torch.LongTensor],
                actions: List[List[ProductionRuleArray]],
                java_class: Dict[str, torch.LongTensor],
                entities: List[Set[str]],
                rules: List[torch.Tensor] = None,
                code: List[Dict[str, str]] = None
                ) -> Dict[str, torch.Tensor]:

        # Encode summary, variables, methods with bi lstm.
        ##################################################

        # (batch_size, input_sequence_length, encoder_output_dim)
        embedded_utterance = self._utterance_embedder(utterance)
        batch_size, _, _ = embedded_utterance.size()
        utterance_mask = get_text_field_mask(utterance)

        # (batch_size, question_length, encoder_output_dim)
        encoder_outputs = self._dropout(self._encoder(embedded_utterance, utterance_mask))

        # This will be our initial hidden state and memory cell for the decoder LSTM.
        # todo(rajas) change back to encoder is bidirectional from True
        final_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                             utterance_mask,
                                                             True)
        memory_cell = Variable(encoder_outputs.data.new(batch_size, self._encoder.get_output_dim()).fill_(0))

        # print("utterance weight", self._utterance_embedder._token_embedders)
        debug_print("utterance weight", self._utterance_embedder._token_embedders['tokens'].weight.size())
        # exit()
        initial_score = Variable(embedded_utterance.data.new(batch_size).fill_(0))
        _, actionidx2vocabidx = self._embed_actions(actions)

        linking_features = java_class['linking']
        linking_scores = self._linking_params(linking_features).squeeze(3)

        flattened_linking_scores, actions_to_entities = self._map_entity_productions(linking_scores,
                                                                                     entities,
                                                                                     actions)

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
                                              question_mask_list,
                                              parent_states=[self._first_action_embedding]))
        initial_grammar_state = [self._create_grammar_state(actions[i])
                                 for i in range(batch_size)]
        initial_state = JavaDecoderState(batch_indices=list(range(batch_size)),
                                         action_history=[[] for _ in range(batch_size)],
                                         score=initial_score_list,
                                         rnn_state=initial_rnn_state,
                                         grammar_state=initial_grammar_state,
                                         # action_embeddings=action_embeddings,
                                         action_indices=actionidx2vocabidx,
                                         possible_actions=actions,
                                         flattened_linking_scores=flattened_linking_scores,
                                         actions_to_entities=actions_to_entities,
                                         # entity_types=entity_type_dict,
                                         debug_info=None)


        if self.training:
            return self._decoder_trainer.decode(initial_state,
                                                self._decoder_step,
                                                rules)
        else:
            action_mapping = {}
            for batch_index, batch_actions in enumerate(actions):
                for action_index, action in enumerate(batch_actions):
                    action_mapping[(batch_index, action_index)] = action[0]
            outputs: Dict[str, Any] = {'action_mapping': action_mapping}
            if rules is not None:
                outputs['loss'] = self._decoder_trainer.decode(initial_state,
                                             self._decoder_step,
                                             rules)['loss']
            num_steps = self._max_decoding_steps
            initial_state.debug_info = [[] for _ in range(batch_size)]
            best_final_states = self._decoder_beam_search.search(num_steps,
                                                                 initial_state,
                                                                 self._decoder_step,
                                                                 keep_final_unfinished_states=False)

            outputs['best_action_sequence'] = []
            outputs['debug_info'] = []
            outputs['entities'] = []
            outputs['linking_scores'] = linking_scores
            # if self._linking_params is not None:
            #     outputs['feature_scores'] = feature_scores
            # todo(rajas) remove similarity scores
            outputs['similarity_scores'] = linking_scores
            outputs['logical_form'] = []

            outputs['rules'] = []

            for i in range(batch_size):
                if i in best_final_states:
                    em = 0
                    partial_parse_acc = 0
                    bleu = 0
                    pred_rules = self._get_rules_from_action_history(best_final_states[i][0].action_history[0], actions[i])
                    if rules is not None:
                        # todo(rajas): average action history function could
                        # become a one liner with pred_rules and targ_rules
                        partial_parse_acc = self._average_action_history_match(best_final_states[i][0].action_history[0], rules[i])
                        targ_action_history = rules[i].long().data.cpu().numpy().tolist()
                        targ_rules = self._get_rules_from_action_history(targ_action_history, actions[i])

                        pred_code = self._gen_code_from_rules(pred_rules)
                        targ_code = code[i]['code']

                        bleu = self._get_bleu(targ_code, pred_code)
                        self._log_code(targ_code, pred_rules, i)
                        if pred_code == targ_code:
                            em = 1


                    self._partial_production_rule_accuracy(partial_parse_acc)
                    self._code_bleu(bleu)
                    self._exact_match_accuracy(em)
                    outputs['rules'].append(best_final_states[i][0].grammar_state[0]._action_history[0])

                    outputs['best_action_sequence'].append(pred_rules)
                    outputs['logical_form'].append(self._gen_code_from_rules(pred_rules))
                    # print('best final states', best_final_states[i][0])
                    # print('best final states', best_final_states[i][0].debug_info)

                    outputs['debug_info'].append(best_final_states[i][0].debug_info[0])  # type: ignore
                    outputs['entities'].append(entities[i])

            return outputs


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions.  This is (confusingly) a separate notion from the "decoder"
        in "encoder/decoder", where that decoder logic lives in ``WikiTablesDecoderStep``.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        action_mapping = output_dict['action_mapping']
        best_actions = output_dict["best_action_sequence"]
        debug_infos = output_dict['debug_info']
        batch_action_info = []
        for batch_index, (predicted_actions, debug_info) in enumerate(zip(best_actions, debug_infos)):
            instance_action_info = []
            for predicted_action, action_debug_info in zip(predicted_actions, debug_info):
                action_info = {}
                action_info['predicted_action'] = predicted_action
                considered_actions = action_debug_info['considered_actions']
                probabilities = action_debug_info['probabilities']
                actions = []
                for action, probability in zip(considered_actions, probabilities):
                    if action != -1:
                        actions.append((action_mapping[(batch_index, action)], probability))
                actions.sort()
                considered_actions, probabilities = zip(*actions)
                action_info['considered_actions'] = considered_actions
                action_info['action_probabilities'] = probabilities
                action_info['question_attention'] = action_debug_info['question_attention']
                instance_action_info.append(action_info)
            batch_action_info.append(instance_action_info)
        output_dict["predicted_actions"] = batch_action_info
        return output_dict


    @staticmethod
    def _get_bleu(gold_seq, pred_seq):
        # This is how Ling et al. compute bleu score.
        sm = SmoothingFunction()
        ngram_weights = [0.25] * min(4, len(gold_seq))
        return sentence_bleu([gold_seq], pred_seq, weights=ngram_weights, smoothing_function=sm.method3)

    @staticmethod
    def _average_action_history_match(predicted: List[int], targets: torch.LongTensor) -> int:
        # Since targets is padded with -1's we find the target's length by the first index
        # where the -1's start to occur.

        # todo(Rajas) add full sequence, exact match
        min_val, unpadded_targ_length = torch.min(targets, 0)
        if min_val.data.cpu().numpy().tolist()[0] == -1:
            unpadded_targ_length = unpadded_targ_length.data.cpu().numpy().tolist()[0]
        else:
            unpadded_targ_length = targets.size(0)
        min_length = min(len(predicted), unpadded_targ_length)
        max_length = max(len(predicted), unpadded_targ_length)

        predicted_tensor = Variable(targets.data.new(predicted))
        predicted_tensor = predicted_tensor[:min_length].long()
        targets_trimmed = targets[:min_length].long()

        correct_tensor = targets_trimmed.eq(predicted_tensor)
        del predicted_tensor
        num_correct = torch.sum(correct_tensor)
        num_correct = num_correct.data.cpu().numpy().tolist()[0]
        return num_correct / max_length


    def _is_terminal_rule(self, rule):
        return (
               "IdentifierNT-->" in rule and rule != 'IdentifierNT-->VarCopy' and rule != 'IdentifierNT-->MethodCopy') \
               or re.match(r"^Nt_.*_literal-->.*", rule) \
               or rule == "<unk>"

    def _gen_code_from_rules(self, rules):
        stack = []
        code = []
        for i in range(0, len(rules)):
            _, rhs = rules[i].split('-->')
            if not self._is_terminal_rule(rules[i]):
                stack.extend(rhs.split('___')[::-1])
            else:
                code.append(rhs)
            try:
                top = stack.pop()
                while not top[0].isupper():
                    code.append(top)
                    if len(stack) == 0:
                        break
                    top = stack.pop()
            except:
                pass
        return code

    def _get_rules_from_action_history(self, action_history: List[int], actions: List[ProductionRuleArray]):
        rules = []
        for a in action_history:
            rules.append(actions[a][0])
        return rules

    # def _get_rules_from_action_history(self, target_actions :List[int], predicted_actions: List[int], actions: List[ProductionRuleArray]):
    #     predicted_rules = []
    #     target_rules = []
    #     for a in target_actions:
    #         target_rules.append(actions[a][0])
    #     for a in predicted_actions:
    #         predicted_rules.append(actions[a][0])
    #     return target_rules, predicted_rules

    def _log_rules(self, target_rules, predicted_rules, batch):
        f = open('debug/pred_target_strings.csv', 'a')
        f.write("batch, " + str(batch) + '\n')
        for i in range(min(len(target_rules), len(predicted_rules))):  #todo(rajas) handle bounds
            f.write(target_rules[i] + ", " + predicted_rules[i] + '\n')
        f.close()

    def _log_code(self, real_target_code, predicted_rules, batch):
        codef = open('debug/pred_target_code.txt', 'a')
        codef.write("batch, " + str(batch) + '\n')

        predicted_code = self._gen_code_from_rules(predicted_rules)
        # codef.write('Target\n')
        # codef.write(' '.join(real_target_code['code']) + '\n')
        # codef.write('Prediction\n')
        # codef.write(' '.join(predicted_code) + '\n\n')
        # codef.close()
        print('==============' * 4)
        print('Prediction====')
        print(' '.join(predicted_code) + '')
        print('Target========')
        print(' '.join(real_target_code) + '')

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'bleu': self._code_bleu.get_metric(reset),
            'em': self._exact_match_accuracy.get_metric(reset),
            'partial_acc': self._partial_production_rule_accuracy.get_metric(reset)

        }

    def print_gbc(self):
        print('Printing gbc')
        num_tensors = 0
        sum_sizes = 0
        tensor_size_counter = Counter()
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    num_tensors += 1
                    size = 1
                    for dim in list(obj.size()):
                        size *= dim
                    sum_sizes += size
                    obj_key = str(type(obj)) + str(list(obj.size()))
                    tensor_size_counter[obj_key] = size
                    # print(type(obj), obj.size())
            except:
                y = 3
        print('Num tensors', num_tensors)
        print('Total size', sum_sizes)
        print('Total size mb', (sum_sizes*4)/(1024*1024))
        print('Largest tensors', tensor_size_counter.most_common(30))


    @staticmethod
    @timeit
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
            lhs = action[0].split('-->')[0]
            nonterminal2action_index[lhs].append(i)
        return JavaGrammarState([START_SYMBOL], nonterminal2action_index)

    @timeit
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
        # todo(rajas): optimize the code to stop at global actions and repeat for each batch
        action_map: Dict[Tuple[int, int], int] = {}
        for batch_index, instance_actions in enumerate(actions):
            for action_index, action in enumerate(instance_actions):
                if not action[0]:
                    # This rule is padding.
                    continue
                global_action_id = action_vocab.get(action[0], -1)
                action_map[(batch_index, action_index)] = global_action_id
        return embedded_actions, action_map

    @staticmethod
    @timeit
    def _map_entity_productions(linking_scores: torch.FloatTensor,
                                batch_entities: List[Set[str]],
                                actions: List[List[ProductionRuleArray]]) -> Tuple[torch.Tensor,
                                                                                   Dict[Tuple[int, int], int]]:
        """
        Constructs a map from ``(batch_index, action_index)`` to ``(batch_index * entity_index)``.
        That is, some actions correspond to terminal productions of entities from our table.  We
        need to find those actions and map them to their corresponding entity indices, where the
        entity index is its position in the list of entities returned by the ``world``.  This list
        is what defines the second dimension of the ``linking_scores`` tensor, so we can use this
        index to look up linking scores for each action in that tensor.

        For easier processing later, the mapping that we return is `flattened` - we really want to
        map ``(batch_index, action_index)`` to ``(batch_index, entity_index)``, but we are going to
        have to use the result of this mapping to do ``index_selects`` on the ``linking_scores``
        tensor.  You can't do ``index_select`` with tuples, so we flatten ``linking_scores`` to
        have shape ``(batch_size * num_entities, num_question_tokens)``, and return shifted indices
        into this flattened tensor.

        Parameters
        ----------
        linking_scores : ``torch.Tensor``
            A tensor representing linking scores between each table entity and each question token.
            Has shape ``(batch_size, num_entities, num_question_tokens)``.
        worlds : ``List[WikiTablesWorld]``
            The ``World`` for each batch instance.  The ``World`` contains a reference to the
            ``TableKnowledgeGraph`` that defines the set of entities in the linking.
        actions : ``List[List[ProductionRuleArray]]``
            The list of possible actions for each batch instance.  Our action indices are defined
            in terms of this list, so we'll find entity productions in this list and map them to
            entity indices from the entity list we get from the ``World``.

        Returns
        -------
        flattened_linking_scores : ``torch.Tensor``
            A flattened version of ``linking_scores``, with shape ``(batch_size * num_entities,
            num_question_tokens)``.
        actions_to_entities : ``Dict[Tuple[int, int], int]``
            A mapping from ``(batch_index, action_index)`` to ``(batch_size * num_entities)``,
            representing which action indices correspond to which entity indices in the returned
            ``flattened_linking_scores`` tensor.
        """
        batch_size, num_entities, num_question_tokens = linking_scores.size()
        entity_map: Dict[Tuple[int, str], int] = {}
        for batch_index, entities in enumerate(batch_entities):
            for entity_index, entity in enumerate(entities):
                entity_map[(batch_index, entity)] = batch_index * num_entities + entity_index
        actions_to_entities: Dict[Tuple[int, int], int] = {}
        for batch_index, action_list in enumerate(actions):
            for action_index, action in enumerate(action_list):
                if not action[0]:
                    # This action is padding.
                    continue
                _, production = action[0].split('-->')
                entity_index = entity_map.get((batch_index, production), None)
                if entity_index is not None:
                    actions_to_entities[(batch_index, action_index)] = entity_index
        flattened_linking_scores = linking_scores.view(batch_size * num_entities, num_question_tokens)
        return flattened_linking_scores, actions_to_entities


    @classmethod
    def from_params(cls, vocab, params: Params) -> 'SimpleSeq2Seq':
        utterance_embedder_params = params.pop("utterance_embedder")
        utterance_embedder = TextFieldEmbedder.from_params(vocab, utterance_embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        max_decoding_steps = params.pop("max_decoding_steps")
        action_embedding_dim = params.pop_int("action_embedding_dim")
        dropout = params.pop_float('dropout', 0.0)
        num_linking_features = params.pop_int('num_linking_features', 8)
        decoder_beam_search = BeamSearch.from_params(params.pop("decoder_beam_search"))
        mixture_feedforward_type = params.pop('mixture_feedforward', None)
        if mixture_feedforward_type is not None:
            mixture_feedforward = FeedForward.from_params(mixture_feedforward_type)
        else:
            mixture_feedforward = None
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
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   utterance_embedder=utterance_embedder,
                   encoder=encoder,
                   attention_function=attention_function,
                   mixture_feedforward=mixture_feedforward,
                   # nonterminal_embedder=nonterminal_embedder,
                   # terminal_embedder=terminal_embedder,
                   action_embedding_dim=action_embedding_dim,
                   max_decoding_steps=max_decoding_steps,
                   decoder_beam_search=decoder_beam_search,
                   num_linking_features=num_linking_features,
                   dropout=dropout)