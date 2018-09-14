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


    #@timeit
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

    #@timeit
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
            action_logits = action_embeddings.mm(predicted_action_embedding.unsqueeze(-1))#.squeeze(-1)
            # Shape: (num_actions, 1)
            current_log_probs = torch.nn.functional.log_softmax(action_logits, dim=0)

            # This is now the total score for each state after taking each action.  We're going to
            # sort by this later, so it's important that this is the total score, not just the
            # score for the current action.
            log_probs = state.score[group_index] + current_log_probs
            batch_results[group_index] = (log_probs,
                                          action_embeddings,
                                          actions[group_index])
        return batch_results

    #@timeit
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

        group_size = len(state.batch_indices)
        hidden_state = [x.squeeze(0) for x in updated_rnn_state['hidden_state'].chunk(group_size, 0)]
        memory_cell = [x.squeeze(0) for x in updated_rnn_state['memory_cell'].chunk(group_size, 0)]
        attended_question = [x.squeeze(0) for x in updated_rnn_state['attended_question'].chunk(group_size, 0)]
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
                # These indices are the locations in the tuple as created in
                # _compute_action_probabilities
                considered_actions = batch_action_probs[group_index][2]

                probabilities = batch_action_probs[group_index][0].squeeze(-1).data.exp().cpu().numpy().tolist()
            return state.new_state_from_group_index(group_index,
                                                    action,
                                                    new_score,
                                                    new_rnn_state,
                                                    considered_actions,
                                                    probabilities,
                                                    updated_rnn_state['attention_weights'])

        new_states = []

        for group_index, results in batch_action_probs.items():
            log_probs, action_embeddings, actions = results
            if allowed_actions: # todo(pr); is this ok to comment? and not max_actions:
                lhs, _ = allowed_actions[group_index].split('-->')
                aindex = state.nonterminal2action2index[lhs][allowed_actions[group_index]]
                new_states.append(make_state(group_index, actions[aindex], log_probs[aindex], action_embeddings[aindex]))
            else:
                batch_states = []
                # Shape (num_actions)
                log_probs = log_probs.squeeze(-1)
                if len(actions) > max_actions:
                    # efficient code for identifiers
                    log_probs_cpu = log_probs.data.cpu().numpy()
                    top_indices = np.argpartition(log_probs_cpu, -max_actions)[-max_actions:]

                    # actually sort the indices
                    sorted_top_indices = top_indices[np.argsort(log_probs_cpu[top_indices])]
                    # Need descending order.
                    sorted_top_indices = sorted_top_indices[::-1]

                    for top_index in sorted_top_indices.tolist():
                        batch_states.append((log_probs_cpu[top_index], group_index, log_probs[top_index],
                                            action_embeddings[top_index], actions[top_index]))
                else:
                    log_probs_cpu = log_probs.data.cpu().numpy().tolist()
                    batch_states = []

                    for i in range(len(actions)):
                        batch_states.append((log_probs_cpu[i], group_index, log_probs[i], action_embeddings[i], actions[i]))

                    # We use a key here to make sure we're not trying to compare anything on the GPU.
                    batch_states.sort(key=lambda x: x[0], reverse=True)

                if max_actions:
                    batch_states = batch_states[:max_actions]
                for _, group_index, log_prob, action_embedding, action in batch_states:
                    new_states.append(make_state(group_index, action, log_prob, action_embedding))

        return new_states