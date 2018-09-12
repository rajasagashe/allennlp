from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
from copy import deepcopy

import time
from overrides import overrides

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear

from allennlp.common import util as common_util
from allennlp.common.util import timeit, debug_print
from allennlp.nn import util
from allennlp.modules import Attention, FeedForward, Embedding
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.similarity_functions import CosineSimilarity
from allennlp.common.checks import check_dimensions_match

from allennlp.nn.decoding import RnnState, DecoderStep
from allennlp.models.encoder_decoders.java.java_decoder_state import JavaDecoderState

# from allennlp.models.encoder_decoders.wikitables.wikitables_decoder_state import JavaDecoderState
# from java_programmer.models.java_decoder_state import JavaDecoderState
# from java_programmer.allennlp_in_progress.decoder_state import DecoderState
# from java_programmer.allennlp_in_progress.decoder_step import DecoderStep
# from java_programmer.allennlp_in_progress.rnn_state import RnnState
# from allennlp.nn.decoding import DecoderStep, RnnState
from allennlp.nn.util import batched_index_select

class JavaDecoderStep(DecoderStep[JavaDecoderState]):
    def __init__(self,
                 encoder_output_dim: int,
                 action_embedding_dim: int,
                 attention_function: SimilarityFunction,
                 mixture_feedforward: FeedForward = None,
                 prototype_feedforward: FeedForward = None,
                 dropout: float = 0.0,
                 should_copy_proto_actions: bool = True,
                 seq2seq_baseline: bool = False) -> None:
        super(JavaDecoderStep, self).__init__()
        self._mixture_feedforward = mixture_feedforward
        self._prototype_feedforward = prototype_feedforward
        self._should_copy_proto_actions = should_copy_proto_actions
        self._seq2seq_baseline = seq2seq_baseline

        self._input_attention = Attention(attention_function)

        self._identifier_literal_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        torch.nn.init.normal(self._identifier_literal_action_embedding)
        # self._num_start_types = num_start_types
        # self._start_type_predictor = Linear(encoder_output_dim, num_start_types)

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        output_dim = encoder_output_dim
        input_dim = output_dim
        # Our decoder input will be the concatenation of the decoder hidden state and the previous
        # action embedding, and we'll project that down to the decoder's `input_dim`, which we
        # arbitrarily set to be the same as `output_dim`.
        self._input_projection_layer = Linear(output_dim + 2 * action_embedding_dim, input_dim)
        # Before making a prediction, we'll compute an attention over the input given our updated
        # hidden state.  Then we concatenate that with the decoder state and project to
        # `action_embedding_dim` to make a prediction.
        self._output_projection_layer = Linear(output_dim + encoder_output_dim, action_embedding_dim)

        # self.utt_sim_linear = Linear(1, 5)
        # self.hidden_copy_proto_rule_mlp = Linear(encoder_output_dim, 1)

        # TODO(pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(input_dim, output_dim)

        if mixture_feedforward is not None:
            # check_dimensions_match(output_dim, mixture_feedforward.get_input_dim(),
            #                        "hidden state embedding dim", "mixture feedforward input dim")
            check_dimensions_match(mixture_feedforward.get_output_dim(), 1,
                                   "mixture feedforward output dim", "dimension for scalar value")

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

    @overrides
    def take_step(self,
                  state: JavaDecoderState,
                  max_actions: int = None,
                  allowed_actions: List[str] = None) -> List[JavaDecoderState]:
        # Taking a step in the decoder consists of three main parts.  First, we'll construct the
        # input to the decoder and update the decoder's hidden state.  Second, we'll use this new
        # hidden state (and maybe other information) to predict an action.  Finally, we will
        # construct new states for the next step.  Each new state corresponds to one valid action
        # that can be taken from the current state, and they are ordered by their probability of
        # being selected.
        updated_state = self._update_decoder_state(state)
        batch_results = self._compute_action_probabilities(state,
                                                           updated_state['predicted_action_embeddings'])
        new_states = self._construct_next_states(state,
                                                 updated_state,
                                                 batch_results,
                                                 max_actions,
                                                 allowed_actions)

        return new_states

        # actions_to_link = False
        # if actions_to_link:
        #     # entity_action_logits: (group_size, num_entity_actions)
        #     # entity_action_mask: (group_size, num_entity_actions)
        #     entity_action_logits, entity_action_mask = \
        #         self._get_entity_action_logits(state, actions_to_link, attention_weights)
        #
        #     # The `action_embeddings` tensor gets used later as the input to the next decoder step.
        #     # For linked actions, we don't have any action embedding, so we use the entity type
        #     # instead.
        #     # action_embeddings = torch.cat([action_embeddings, entity_type_embeddings], dim=1)
        #     if self._mixture_feedforward is not None:
        #         # The entity and action logits are combined with a mixture weight to prevent the
        #         # entity_action_logits from dominating the embedded_action_logits if a softmax
        #         # was applied on both together.
        #         mixture_weight = self._mixture_feedforward(hidden_state)
        #         mix1 = torch.log(mixture_weight)
        #         mix2 = torch.log(1 - mixture_weight)
        #
        #         entity_action_probs = util.masked_log_softmax(entity_action_logits,
        #                                                       entity_action_mask.float()) + mix1
        #         embedded_action_probs = util.masked_log_softmax(embedded_action_logits,
        #                                                         embedded_action_mask.float()) + mix2
        #
        #         if self._should_copy_proto_actions:
        #             entity_action_probs = entity_action_probs + log_proto_mix1
        #             embedded_action_probs = embedded_action_probs + log_proto_mix1
        #
        #             embedded_action_probs, proto_action_probs = self.combine_embedded_proto_action_probs(
        #                 embedded_action_probs, proto_action_probs, proto_action_probs_mask.float(), action_indices,
        #                 proto_mix2)
        #
        #         current_log_probs = torch.cat([embedded_action_probs, entity_action_probs], dim=1)
        #     else:
        #         action_logits = torch.cat([embedded_action_logits, entity_action_logits], dim=1)
        #         action_mask = torch.cat([embedded_action_mask, entity_action_mask], dim=1).float()
        #         current_log_probs = util.masked_log_softmax(action_logits, action_mask)
        #
        #
        # proto_attention_weights = None
        # proto_action_probs = None
        # proto_action_probs_mask = None
        #
        #
        #
        # if allowed_actions is not None:
        #     return self._construct_next_states(updated_rnn_state=None,
        #                                        batch_action_probs=batch_results,
        #                                        max_actions=None,
        #                                        allowed_actions=allowed_actions)


            # This method is slow but integrates well with beam search, so use it
            # for inference.
            # return self._compute_new_states_optimized(state,
            #                                           log_probs,
            #                                           hidden_state,
            #                                           memory_cell,
            #                                           action_embeddings,
            #                                           attended_question,
            #                                           allowed_actions,
            #                                           self._identifier_literal_action_embedding)

        # else:
        #     return self._compute_new_states(state,
        #                                     log_probs,
        #                                     current_log_probs,
        #                                     hidden_state,
        #                                     memory_cell,
        #                                     action_embeddings,
        #                                     attended_question,
        #                                     attention_weights,
        #                                     allowed_actions,
        #                                     self._identifier_literal_action_embedding,
        #                                     prototype_attention_weights=proto_attention_weights,
        #                                     max_actions=max_actions,
        #                                     proto_action_probs=proto_action_probs,
        #                                     proto_action_probs_mask=proto_action_probs_mask,
        #                                     action_indices=action_indices,
        #                                     should_copy_proto_actions=self._should_copy_proto_actions)


    @timeit
    def _update_decoder_state(self, state: JavaDecoderState) -> Dict[str, torch.Tensor]:
        # For updating the decoder, we're doing a bunch of tensor operations that can be batched
        # without much difficulty.  So, we take all group elements and batch their tensors together
        # before doing these decoder operations.

        group_size = len(state.batch_indices)
        attended_question = torch.stack([rnn_state.attended_input for rnn_state in state.rnn_state])
        hidden_state = torch.stack([rnn_state.hidden_state for rnn_state in state.rnn_state])
        memory_cell = torch.stack([rnn_state.memory_cell for rnn_state in state.rnn_state])
        previous_action_embedding = torch.stack([rnn_state.previous_action_embedding
                                                 for rnn_state in state.rnn_state])
        # The parent is the top most element on the parent_states stack.
        parent_action_embedding = torch.stack([rnn_state.parent_states[-1]
                                               for rnn_state in state.rnn_state])

        # (group_size, decoder_input_dim)
        projected_input = self._input_projection_layer(torch.cat([attended_question,
                                                                  previous_action_embedding, parent_action_embedding], -1))
        # decoder_input = self._activation(projected_input)
        # todo(pr): look into the benefits of this activation

        decoder_input = projected_input
        hidden_state, memory_cell = self._decoder_cell(decoder_input, (hidden_state, memory_cell))
        hidden_state = self._dropout(hidden_state)

        # (group_size, encoder_output_dim)
        encoder_outputs = torch.stack([state.rnn_state[0].encoder_outputs[i] for i in state.batch_indices])
        encoder_output_mask = torch.stack([state.rnn_state[0].encoder_output_mask[i] for i in state.batch_indices])
        attended_question, attention_weights = self.attend_on_question(hidden_state,
                                                                       encoder_outputs,
                                                                       encoder_output_mask.float())
        action_query = torch.cat([hidden_state, attended_question], dim=-1)

        # (group_size, action_embedding_dim)
        # todo(pr): look into activation
        # projected_query = self._activation(self._output_projection_layer(action_query))
        projected_query = (self._output_projection_layer(action_query))
        predicted_action_embeddings = self._dropout(projected_query)

        return {
                'hidden_state': hidden_state,
                'memory_cell': memory_cell,
                'attended_question': attended_question,
                'attention_weights': attention_weights,
                'predicted_action_embeddings': predicted_action_embeddings,
                }



    def attend_on_question(self,
                           query: torch.Tensor,
                           encoder_outputs: torch.Tensor,
                           encoder_output_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a query (which is typically the decoder hidden state), compute an attention over the
        output of the question encoder, and return a weighted sum of the question representations
        given this attention.  We also return the attention weights themselves.

        This is a simple computation, but we have it as a separate method so that the ``forward``
        method on the main parser module can call it on the initial hidden state, to simplify the
        logic in ``take_step``.
        """
        # (group_size, question_length)
        question_attention_weights = self._input_attention(query,
                                                           encoder_outputs,
                                                           encoder_output_mask)
        # (group_size, encoder_output_dim)
        attended_question = util.weighted_sum(encoder_outputs, question_attention_weights)
        return attended_question, question_attention_weights

    # def _get_entity_action_logits(self,
    #                               state: JavaDecoderState,
    #                               actions_to_link: List[List[int]],
    #                               attention_weights: torch.Tensor) -> Tuple[torch.FloatTensor,
    #                                                                         torch.LongTensor,
    #                                                                         torch.FloatTensor]:
    #     """
    #     Returns scores for each action in ``actions_to_link`` that are derived from the linking
    #     scores between the question and the table entities, and the current attention on the
    #     question.  The intuition is that if we're paying attention to a particular word in the
    #     question, we should tend to select entity productions that we think that word refers to.
    #     We additionally return a mask representing which elements in the returned ``action_logits``
    #     tensor are just padding, and an embedded representation of each action that can be used as
    #     input to the next step of the encoder.  That embedded representation is derived from the
    #     type of the entity produced by the action.
    #
    #     The ``actions_to_link`` are in terms of the `batch` action list passed to
    #     ``model.forward()``.  We need to convert these integers into indices into the linking score
    #     tensor, which has shape (batch_size, num_entities, num_question_tokens), look up the
    #     linking score for each entity, then aggregate the scores using the current question
    #     attention.
    #
    #     Parameters
    #     ----------
    #     state : ``JavaDecoderState``
    #         The current state.  We'll use this to get the linking scores.
    #     actions_to_link : ``List[List[int]]``
    #         A list of _batch_ action indices for each group element.  Should have shape
    #         (group_size, num_actions), unpadded.  This is expected to be output from
    #         :func:`_get_actions_to_consider`.
    #     attention_weights : ``torch.Tensor``
    #         The current attention weights over the question tokens.  Should have shape
    #         ``(group_size, num_question_tokens)``.
    #
    #     Returns
    #     -------
    #     action_logits : ``torch.FloatTensor``
    #         A score for each of the given actions.  Shape is ``(group_size, num_actions)``, where
    #         ``num_actions`` is the maximum number of considered actions for any group element.
    #     action_mask : ``torch.LongTensor``
    #         A mask of shape ``(group_size, num_actions)`` indicating which ``(group_index,
    #         action_index)`` pairs were merely added as padding.
    #     type_embeddings : ``torch.LongTensor``
    #         A tensor of shape ``(group_size, num_actions, action_embedding_dim)``, with an embedded
    #         representation of the `type` of the entity corresponding to each action.
    #     """
    #     # First we map the actions to entity indices, using state.actions_to_entities, and find the
    #     # type of each entity using state.entity_types.
    #     action_entities: List[List[int]] = []
    #     entity_types: List[List[int]] = []
    #     for batch_index, action_list in zip(state.batch_indices, actions_to_link):
    #         action_entities.append([])
    #         entity_types.append([])
    #         for action_index in action_list:
    #             entity_index = state.actions_to_entities[(batch_index, action_index)]
    #             action_entities[-1].append(entity_index)
    #             # All entities will have the same type here. In the java paper all the previous
    #             # action embeddings for rules identifier/literal are denoted by one special rule
    #             # "IdentifierOrLiteral" to collapse the large number of rules into 1.
    #             entity_types[-1].append(0)  # state.entity_types[entity_index])
    #
    #     # Then we create a padded tensor suitable for use with
    #     # `state.flattened_linking_scores.index_select()`.
    #     num_actions = [len(action_list) for action_list in action_entities]
    #     max_num_actions = max(num_actions)
    #     padded_actions = [common_util.pad_sequence_to_length(action_list, max_num_actions)
    #                       for action_list in action_entities]
    #     # padded_types = [common_util.pad_sequence_to_length(type_list, max_num_actions)
    #     #                 for type_list in entity_types]
    #     # Shape: (group_size, num_actions)
    #     action_tensor = Variable(state.score[0].data.new(padded_actions).long())
    #     # type_tensor = Variable(state.score[0].data.new(padded_types).long())
    #
    #     # To get the type embedding tensor, we just use an embedding matrix on the list of entity
    #     # types.
    #     # type_embeddings = self._entity_type_embedding(type_tensor)
    #
    #     # `state.flattened_linking_scores` is shape (batch_size * num_entities, num_question_tokens).
    #     # We want to select from this using `action_tensor` to get a tensor of shape (group_size,
    #     # num_actions, num_question_tokens).  Unfortunately, the index_select functions in nn.util
    #     # don't do this operation.  So we'll do some reshapes and do the index_select ourselves.
    #     group_size = len(state.batch_indices)
    #     num_question_tokens = state.flattened_linking_scores.size(-1)
    #     flattened_actions = action_tensor.view(-1)
    #     # (group_size * num_actions, num_question_tokens)
    #     flattened_action_linking = state.flattened_linking_scores.index_select(0, flattened_actions)
    #     # (group_size, num_actions, num_question_tokens)
    #     action_linking = flattened_action_linking.view(group_size, max_num_actions, num_question_tokens)
    #
    #     # Now we get action logits by weighting these entity x token scores by the attention over
    #     # the question tokens.  We can do this efficiently with torch.bmm.
    #     action_logits = action_linking.bmm(attention_weights.unsqueeze(-1)).squeeze(-1)
    #
    #     # Finally, we make a mask for our action logit tensor.
    #     sequence_lengths = Variable(action_linking.data.new(num_actions))
    #     action_mask = util.get_mask_from_sequence_lengths(sequence_lengths, max_num_actions)
    #     return action_logits, action_mask  # , type_embeddings

    @timeit
    def _compute_action_probabilities(self,
                                      state: JavaDecoderState,
                                      predicted_action_embeddings: torch.Tensor
                                     ) -> Dict[int, Tuple[int, Any, Any, List[str]]]:
        # We take a couple of extra arguments here because subclasses might use them.
        # pylint: disable=unused-argument,no-self-use

        # In this section we take our predicted action embedding and compare it to the available
        # actions in our current state (which might be different for each group element).  For
        # computing action scores, we'll forget about doing batched / grouped computation, as it
        # adds too much complexity and doesn't speed things up, anyway, with the operations we're
        # doing here.  This means we don't need any action masks, as we'll only get the right
        # lengths for what we're computing.

        group_size = len(state.batch_indices)
        actions, group_action_embeddings = state.get_valid_actions_embeddings()

        batch_results: Dict[int, List[Tuple[int, torch.Tensor, torch.Tensor, List[int]]]] = defaultdict(list)
        for group_index in range(group_size):
            predicted_action_embedding = predicted_action_embeddings[group_index]
            action_embeddings = group_action_embeddings[group_index]

            # This is just a matrix product between a (num_actions, embedding_dim) matrix and an
            # (embedding_dim, 1) matrix.
            # print('action', action_embeddings)
            # print('pred', predicted_action_embeddings)
            action_logits = action_embeddings.mm(predicted_action_embedding.unsqueeze(-1)).squeeze(-1)
            current_log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)

            # This is now the total score for each state after taking each action.  We're going to
            # sort by this later, so it's important that this is the total score, not just the
            # score for the current action.
            log_probs = state.score[group_index] + current_log_probs
            batch_results[state.batch_indices[group_index]] = (group_index,
                                                                    log_probs,
                                                                    action_embeddings,
                                                                    actions[group_index])
        return batch_results

    @timeit
    def _construct_next_states(self,
                               state: JavaDecoderState,
                               updated_rnn_state: Dict[str, torch.Tensor],
                               batch_action_probs: Dict[int, List[Tuple[int, Any, Any, List[int]]]],
                               max_actions: int,
                               allowed_actions: List[str]):
        # pylint: disable=no-self-use

        # We'll yield a bunch of states here that all have a `group_size` of 1, so that the
        # learning algorithm can decide how many of these it wants to keep, and it can just regroup
        # them later, as that's a really easy operation.
        #
        # We first define a `make_state` method, as in the logic that follows we want to create
        # states in a couple of different branches, and we don't want to duplicate the
        # state-creation logic.  This method creates a closure using variables from the method, so
        # it doesn't make sense to pull it out of here.

        # Each group index here might get accessed multiple times, and doing the slicing operation
        # each time is more expensive than doing it once upfront.  These three lines give about a
        # 10% speedup in training time.

        start = time.time()
        group_size = len(state.batch_indices)
        hidden_state = [x.squeeze(0) for x in updated_rnn_state['hidden_state'].chunk(group_size, 0)]
        memory_cell = [x.squeeze(0) for x in updated_rnn_state['memory_cell'].chunk(group_size, 0)]
        attended_question = [x.squeeze(0) for x in updated_rnn_state['attended_question'].chunk(group_size, 0)]
        end = time.time()
        debug_print('Time to squeeze and chunk', (end-start)*1000)
        # @timeit
        def make_state(group_index: int,
                       action: str,
                       new_score: torch.Tensor,
                       action_embedding: torch.Tensor) -> JavaDecoderState:



            lhs, _ = action.split('-->')
            if 'IdentifierNT' in lhs or '_literal' in lhs:
                action_embedding = self._identifier_literal_action_embedding

            # Pop the old parent states and push the new ones one num nonterminals times since
            # each of the nonterminals will use it.
            new_parent_states = [p for p in state.rnn_state[group_index].parent_states]
            new_parent_states.pop()
            num_nonterminals_in_rhs = state.grammar_states[0].number_nonterminals_in_rhs(action)
            new_parent_states = new_parent_states + ([action_embedding] * num_nonterminals_in_rhs)

            new_rnn_state = RnnState(hidden_state[group_index],
                                     memory_cell[group_index],
                                     action_embedding,
                                     attended_question[group_index],
                                     state.rnn_state[group_index].encoder_outputs,
                                     state.rnn_state[group_index].encoder_output_mask,
                                     new_parent_states,
                                     state.rnn_state[group_index].proto_rules_encoder_outputs,
                                     state.rnn_state[group_index].proto_rules_encoder_output_mask,
                                     utt_final_encoder_outputs=state.rnn_state[group_index].utt_final_encoder_outputs)


            considered_actions = None
            probabilities = None
            if state.debug_info is not None:
                batch_index = state.batch_indices[group_index]
                # These indices are the locations in the tuple as created in
                # _compute_action_probabilities
                considered_actions = batch_action_probs[batch_index][3]
                probabilities = batch_action_probs[batch_index][1].exp().cpu()
            return state.new_state_from_group_index(group_index,
                                                    action,
                                                    new_score,
                                                    new_rnn_state,
                                                    considered_actions,
                                                    probabilities,
                                                    updated_rnn_state['attention_weights'])

        new_states = []

        for bindex, results in batch_action_probs.items():
            if allowed_actions: # todo(pr); is this ok to comment? and not max_actions:
                # action_taken = False

                # If we're given a set of allowed actions, and we're not just keeping the top k of
                # them, we don't need to do any sorting, so we can speed things up quite a bit.
                group_index, log_probs, action_embeddings, actions = results
                # print('in allowed actions', actions)
                # print('action is', allowed_actions[group_index])
                start3 = time.time()


                # for log_prob, action_embedding, action in zip(log_probs, action_embeddings, actions):
                #     # print(action)
                #     if action == allowed_actions[group_index]:
                #         # print('calling mke state', allowed_actions[group_index])
                #         action_taken = True
                #         new_states.append(make_state(group_index, action, log_prob, action_embedding))

                lhs, _ = allowed_actions[group_index].split('-->')
                aindex = state.nonterminal2action2index[lhs][allowed_actions[group_index]]
                new_states.append(make_state(group_index, actions[aindex], log_probs[aindex], action_embeddings[aindex]))

                # end3 = time.time()
                # debug_print('Inner for loop', (end3-start3)*1000)
                # if False:
                #     print("Something went wrong====================================")
                #     print('Group', group_index, "bindex", bindex)
                #     print(state.grammar_states[group_index]._nonterminal_stack)
                #     print('avail actions', actions)
                #     print('allowed action', allowed_actions[group_index])
                #
                #     # print(state.action_history[group_index])
                #     # for rule in state.action_history[group_index]:
                #     #     print(rule)
                #     exit()
            else:
                group_index, log_probs, action_embeddings, actions = results
                log_probs_cpu = log_probs.data.cpu().numpy().tolist()
                batch_states = []
                for log_prob, action_embedding, action in zip(log_probs, action_embeddings, actions):
                    batch_states.append((log_probs_cpu, group_index, log_prob, action_embedding, action))


                # We use a key here to make sure we're not trying to compare anything on the GPU.
                batch_states.sort(key=lambda x: x[0], reverse=True)
                if max_actions:
                    batch_states = batch_states[:max_actions]
                for _, group_index, log_prob, action_embedding, action in batch_states:
                    new_states.append(make_state(group_index, action, log_prob, action_embedding))

        return new_states


    # @staticmethod
    # # @timeit
    # def _compute_new_states(state: JavaDecoderState,
    #                         log_probs: torch.Tensor,
    #                         current_log_probs: torch.Tensor,
    #                         hidden_state: torch.Tensor,
    #                         memory_cell: torch.Tensor,
    #                         action_embeddings: torch.Tensor,
    #                         attended_question: torch.Tensor,
    #                         attention_weights: torch.Tensor,
    #                         allowed_actions: List[Set[int]],
    #                         identifier_literal_action_embedding: torch.Tensor,
    #                         prototype_attention_weights: torch.Tensor,
    #                         max_actions: int = None,
    #                         proto_action_probs=None,
    #                         proto_action_probs_mask=None,
    #                         action_indices=None,
    #                         should_copy_proto_actions=True
    #                         ) -> List[JavaDecoderState]:
    #     # Each group index here might get accessed multiple times, and doing the slicing operation
    #     # each time is more expensive than doing it once upfront.  These three lines give about a
    #     # 10% speedup in training time.  I also tried this with sorted_log_probs and
    #     # action_embeddings, but those get accessed for _each action_, so doing the splits there
    #     # didn't help.
    #     start1 = time.time()
    #     hidden_state = [x.squeeze(0) for x in hidden_state.split(1, 0)]
    #     memory_cell = [x.squeeze(0) for x in memory_cell.split(1, 0)]
    #     attended_question = [x.squeeze(0) for x in attended_question.split(1, 0)]
    #     end1 = time.time()
    #
    #     start_sort_iter = time.time()
    #     sorted_log_probs, sorted_actions = log_probs.sort(dim=-1, descending=True)
    #     if max_actions is not None:
    #         # We might need a version of `sorted_log_probs` on the CPU later, but only if we need
    #         # to truncate the best states to `max_actions`.
    #         sorted_log_probs_cpu = sorted_log_probs.data.cpu().numpy()
    #     if state.debug_info is not None:
    #         probs_cpu = log_probs.exp().data.cpu().numpy().tolist()
    #         current_probs_cpu = current_log_probs.exp().data.cpu().numpy().tolist()
    #         # prototype_action_probs = [0] * log_probs.size(0)
    #         # prototype_attention_weights
    #         # new_actions = [0] * log_probs.size(0)
    #         considered_proto_actions = None
    #         new_actions = None
    #         if should_copy_proto_actions:
    #             prototype_action_probs = proto_action_probs.data.cpu().numpy().tolist()
    #             prototype_action_probs_mask = proto_action_probs_mask.data.cpu().numpy().tolist()
    #             considered_proto_actions = action_indices.data.cpu().numpy().tolist()
    #
    #             new_actions = []
    #             for group_index, proto_action_index in enumerate(considered_proto_actions):
    #                 new_actions.append([])
    #                 for i in range(len(proto_action_index)):
    #                     # action_id = actions_to_embed[group_index][proto_action_index[i]]
    #                     action_id = considered_actions[group_index][proto_action_index[i]]
    #
    #                     # if action_id not in new_actions[-1]:
    #                     # astr = state.action_mapping[(state.batch_indices[group_index], action_id)]
    #                     # if 'IdentifierNTPrimary-->loc0' in astr:
    #                     #     print(astr)
    #                     # print('yo')
    #
    #                     if prototype_action_probs_mask[group_index][i] != 0:
    #                         new_actions[-1].append(action_id)
    #                     else:
    #                         break
    #                 prototype_action_probs[group_index] = prototype_action_probs[group_index][:len(new_actions[-1])]
    #                 # if len(new_actions[-1]) == 0:
    #                 #     print('yo')
    #         considered_proto_actions = new_actions
    #
    #     sorted_actions = sorted_actions.data.cpu().numpy().tolist()
    #     best_next_states: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    #     for group_index, (batch_index, group_actions) in enumerate(zip(state.batch_indices, sorted_actions)):
    #         for action_index, action in enumerate(group_actions):
    #             # `action` is currently the index in `log_probs`, not the actual action ID.  To get
    #             # the action ID, we need to go through `considered_actions`.
    #             action = considered_actions[group_index][action]
    #             if action == -1:
    #                 # This was padding.
    #                 continue
    #             if allowed_actions is not None and action != allowed_actions[group_index]:
    #                 # This happens when our _decoder trainer_ wants us to only evaluate certain
    #                 # actions, likely because they are the gold actions in this state.  We just skip
    #                 # emitting any state that isn't allowed by the trainer, because constructing the
    #                 # new state can be expensive.
    #                 continue
    #             best_next_states[batch_index].append((group_index, action_index, action))
    #     end_sort_iter = time.time()
    #     debug_print('Time to sort compute new ', (end_sort_iter - start_sort_iter) * 1000)
    #
    #     new_states = []
    #     for batch_index, best_states in sorted(best_next_states.items()):
    #         if max_actions is not None:
    #             # We sorted previously by _group_index_, but we then combined by _batch_index_.  We
    #             # need to get the top next states for each _batch_ instance, so we sort all of the
    #             # instance's states again (across group index) by score.  We don't need to do this
    #             # if `max_actions` is None, because we'll be keeping all of the next states,
    #             # anyway.
    #             best_states.sort(key=lambda x: sorted_log_probs_cpu[x[:2]], reverse=True)
    #             best_states = best_states[:max_actions]
    #         for group_index, action_index, action in best_states:
    #             # We'll yield a bunch of states here that all have a `group_size` of 1, so that the
    #             # learning algorithm can decide how many of these it wants to keep, and it can just
    #             # regroup them later, as that's a really easy operation.
    #             batch_index = state.batch_indices[group_index]
    #             new_action_history = state.action_history[group_index] + [action]
    #             # new_score = state.score[group_index] + sorted_log_probs[group_index, action_index]
    #             new_score = sorted_log_probs[group_index, action_index]
    #
    #             production_rule = state.possible_actions[batch_index][action][0]
    #             new_grammar_state = state.grammar_state[group_index].take_action(production_rule)
    #
    #             lhs, _ = production_rule.split('-->')
    #             if 'IdentifierNT' in lhs or '_literal' in lhs:
    #                 action_embedding = identifier_literal_action_embedding
    #             else:
    #                 # `action_index` is the index in the _sorted_ tensors, but the action embedding
    #                 # matrix is _not_ sorted, so we need to get back the original, non-sorted action
    #                 # index before we get the action embedding.
    #                 action_embedding_index = sorted_actions[group_index][action_index]
    #                 action_embedding = action_embeddings[group_index, action_embedding_index, :]
    #
    #             # Pop the old parent states and push the new ones one num nonterminals times since
    #             # each of the nonterminals will use it.
    #             new_parent_states = [p for p in state.rnn_state[group_index].parent_states]
    #             new_parent_states.pop()
    #             num_nonterminals_in_rhs = new_grammar_state.number_nonterminals_in_rhs(production_rule)
    #             new_parent_states = new_parent_states + ([action_embedding] * num_nonterminals_in_rhs)
    #
    #             if state.debug_info is not None:
    #                 if prototype_attention_weights is None:
    #                     debug_info = {
    #                         'considered_actions': considered_actions[group_index],
    #                         'question_attention': attention_weights[group_index],
    #                         # 'probabilities': probs_cpu[group_index],
    #                         'probabilities': current_probs_cpu[group_index],
    #                     }
    #                 else:
    #                     debug_info = {
    #                         'considered_actions': considered_actions[group_index],
    #                         'considered_prototype_actions': considered_proto_actions[group_index],
    #                         'question_attention': attention_weights[group_index],
    #                         'prototype_attention': prototype_attention_weights[group_index],
    #                         # 'probabilities': probs_cpu[group_index],
    #                         'probabilities': current_probs_cpu[group_index],
    #                         'prototype_action_probs': prototype_action_probs[group_index],
    #                     }
    #                 new_debug_info = [state.debug_info[group_index] + [debug_info]]
    #             else:
    #                 new_debug_info = None
    #
    #             new_rnn_state = RnnState(hidden_state[group_index],
    #                                      memory_cell[group_index],
    #                                      action_embedding,
    #                                      attended_question[group_index],
    #                                      state.rnn_state[group_index].encoder_outputs,
    #                                      state.rnn_state[group_index].encoder_output_mask,
    #                                      new_parent_states,
    #                                      state.rnn_state[group_index].proto_rules_encoder_outputs,
    #                                      state.rnn_state[group_index].proto_rules_encoder_output_mask,
    #                                      utt_final_encoder_outputs=state.rnn_state[
    #                                          group_index].utt_final_encoder_outputs,
    #                                      proto_utt_final_encoder_outputs=state.rnn_state[
    #                                          group_index].proto_utt_final_encoder_outputs,
    #                                      proto_utt_encoder_outputs=state.rnn_state[
    #                                          group_index].proto_utt_encoder_outputs,
    #                                      proto_utt_encoder_output_mask=state.rnn_state[
    #                                          group_index].proto_utt_encoder_output_mask
    #                                      )
    #
    #             new_state = JavaDecoderState(batch_indices=[batch_index],
    #                                          action_history=[new_action_history],
    #                                          score=[new_score],
    #                                          rnn_state=[new_rnn_state],
    #                                          grammar_state=[new_grammar_state],
    #                                          # action_embeddings=state.action_embeddings,
    #                                          action_indices=state.action_indices,
    #                                          possible_actions=state.possible_actions,
    #                                          flattened_linking_scores=state.flattened_linking_scores,
    #                                          actions_to_entities=state.actions_to_entities,
    #                                          proto_actions=[state.proto_actions[group_index]],
    #                                          proto_mask=[state.proto_mask[group_index]],
    #                                          action_mapping=state.action_mapping,
    #                                          # entity_types=state.entity_types,
    #                                          debug_info=new_debug_info)
    #             new_states.append(new_state)
    #     end2 = time.time()
    #     # print('compute new initialize new state time', (end2-time2)*1000)
    #     return new_states
    #
    # @staticmethod
    # # @timeit
    # def _compute_new_states_optimized(state: JavaDecoderState,
    #                                   log_probs: torch.Tensor,
    #                                   hidden_state: torch.Tensor,
    #                                   memory_cell: torch.Tensor,
    #                                   action_embeddings: torch.Tensor,
    #                                   attended_question: torch.Tensor,
    #                                   allowed_actions: List[str],
    #                                   identifier_literal_action_embedding: torch.Tensor) -> List[JavaDecoderState]:
    #     hidden_state = [x.squeeze(0) for x in hidden_state.split(1, 0)]
    #     memory_cell = [x.squeeze(0) for x in memory_cell.split(1, 0)]
    #     attended_question = [x.squeeze(0) for x in attended_question.split(1, 0)]
    #
    #     new_states = []
    #
    #     # todo(pr) new
    #     # for group_index, action in allowed_actions:
    #
    #
    #     # todo(pr)Old
    #
    #     for group_index, allowed_action_index in enumerate(allowed_action_indices):
    #         # index into scores
    #         log_probs_index = allowed_logit_indices[group_index]
    #
    #         # todo(rajas): come clean this up
    #         # update the scores
    #         batch_index = state.batch_indices[group_index]
    #         new_action_history = state.action_history[group_index] + [allowed_action_index]
    #
    #         # new_score = state.score[group_index] + log_probs[group_index, log_probs_index]
    #         new_score = log_probs[group_index, log_probs_index]
    #
    #         production_rule = state.possible_actions[batch_index][allowed_action_index][0]
    #         new_grammar_state = state.grammar_state[group_index].take_action(production_rule)
    #
    #         lhs, _ = production_rule.split('-->')
    #         if 'IdentifierNT' in lhs or '_literal' in lhs:
    #             action_embedding = identifier_literal_action_embedding
    #         else:
    #             # todo(rajas) confirm the log_probs index is correct here
    #             action_embedding = action_embeddings[group_index, log_probs_index, :]
    #
    #         # Pop the old parent states and push the new ones one num nonterminals times since
    #         # each of the nonterminals will use it.
    #         new_parent_states = [p for p in state.rnn_state[group_index].parent_states]
    #         new_parent_states.pop()
    #         num_nonterminals_in_rhs = new_grammar_state.number_nonterminals_in_rhs(production_rule)
    #         new_parent_states = new_parent_states + ([action_embedding] * num_nonterminals_in_rhs)
    #
    #         new_rnn_state = RnnState(hidden_state[group_index],
    #                                  memory_cell[group_index],
    #                                  action_embedding,
    #                                  attended_question[group_index],
    #                                  state.rnn_state[group_index].encoder_outputs,
    #                                  state.rnn_state[group_index].encoder_output_mask,
    #                                  new_parent_states,
    #                                  state.rnn_state[group_index].proto_rules_encoder_outputs,
    #                                  state.rnn_state[group_index].proto_rules_encoder_output_mask,
    #                                  utt_final_encoder_outputs=state.rnn_state[group_index].utt_final_encoder_outputs,
    #                                  proto_utt_final_encoder_outputs=state.rnn_state[
    #                                      group_index].proto_utt_final_encoder_outputs,
    #                                  proto_utt_encoder_outputs=state.rnn_state[group_index].proto_utt_encoder_outputs,
    #                                  proto_utt_encoder_output_mask=state.rnn_state[
    #                                      group_index].proto_utt_encoder_output_mask
    #                                  )
    #
    #         new_state = JavaDecoderState(batch_indices=[batch_index],
    #                                      action_history=[new_action_history],
    #                                      score=[new_score],
    #                                      rnn_state=[new_rnn_state],
    #                                      grammar_state=[new_grammar_state],
    #                                      # action_embeddings=state.action_embeddings,
    #                                      action_indices=state.action_indices,
    #                                      possible_actions=state.possible_actions,
    #                                      flattened_linking_scores=state.flattened_linking_scores,
    #                                      actions_to_entities=state.actions_to_entities,
    #                                      proto_actions=[state.proto_actions[group_index]],
    #                                      proto_mask=[state.proto_mask[group_index]],
    #                                      action_mapping=state.action_mapping,
    #                                      # entity_types=state.entity_types,
    #                                      debug_info=None)
    #         new_states.append(new_state)
    #
    #     return new_states
