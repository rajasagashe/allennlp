from collections import defaultdict
import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import time
import torch
from torch.autograd import Variable

from allennlp.nn.decoding.decoder_step import DecoderStep
from allennlp.nn.decoding.decoder_state import DecoderState
from allennlp.nn.decoding.decoder_trainer import DecoderTrainer

# from java_programmer.allennlp_in_progress.decoder_step import DecoderStep
# from java_programmer.allennlp_in_progress.decoder_state import DecoderState
# from java_programmer.allennlp_in_progress.decoder_trainer import DecoderTrainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class MaximumLikelihood(DecoderTrainer[Tuple[torch.Tensor, torch.Tensor]]):
    def decode(self,
               initial_state: DecoderState,
               decode_step: DecoderStep,
               batch_rules: List[Tuple[str, Tuple[str, ...]]]) -> Dict[str, torch.Tensor]:
        # targets, target_mask = supervision
        # allowed_transitions = self._create_allowed_transitions(targets, target_mask)
        finished_states = []
        states = [initial_state]
        step_num = 0
        while states:
            # allowed_actions = [rules[step_num].data.cpu().numpy().tolist()[0] for rules in batch_rules]

            # print("Step num", step_num, "-------------")
            # print(batch_rules)
            next_states = []
            # We group together all current states to get more efficient (batched) computation.

            # start = time.time()
            grouped_state = states[0].combine_states(states)
            # end = time.time() - start
            # print("Time to combine states", end)

            allowed_actions = self._get_allowed_actions(grouped_state, step_num, batch_rules)
            step_num += 1


            # This will store a set of (batch_index, action_history) tuples, and we'll check it
            # against the allowed actions to make sure we're actually scoring all of the actions we
            # are supposed to.
            actions_taken: Set[Tuple[int, Tuple[int, ...]]] = set()
            for next_state in decode_step.take_step(grouped_state, allowed_actions=allowed_actions):
                actions_taken.add((next_state.batch_indices[0], tuple(next_state.action_history[0])))
                if next_state.is_finished():
                    finished_states.append(next_state)
                else:
                    next_states.append(next_state)
            # print(states[0].grammar_state[0].)
            states = next_states

        # This is a dictionary of lists - for each batch instance, we want the score of all
        # finished states.  So this has shape (batch_size, num_target_action_sequences), though
        # it's not actually a tensor, because different batch instance might have different numbers
        # of finished states.

        # loss = 0
        # for state in finished_states:
        #     if len(state.score) > 1:
        #         exit()
        #     loss += -state.score[0] / len(state.action_history)

        batch_scores = self._group_scores_by_batch(finished_states)
        loss = 0
        for score in batch_scores.values():  # we don't care about the batch index, just the scores
            loss += -score
        return {'loss': loss / len(finished_states)}

    @staticmethod
    def _get_allowed_actions(state: DecoderState, step_num: int, batch_rules):
        # This method just returns index of local actions, since considered uses local indices.
        allowed_actions = []
        for batch_index in state.batch_indices:
            # [rules[step_num].data.cpu().numpy().tolist()[0] for rules in batch_rules]
            action = batch_rules[batch_index][step_num].data.cpu().numpy().tolist()[0]
            # global_action = action_map[(batch_index, action)]
            allowed_actions.append(action)
            # allowed_actions.append(allowed_transitions[batch_index][tuple(action_history)])
        return allowed_actions

    @staticmethod
    def _group_scores_by_batch(finished_states: List[DecoderState]) -> Dict[int, List[Variable]]:
        """
        Takes a list of finished states and groups all final scores for each batch element into a
        list.  This is not trivial because the instances in the batch all might "finish" at
        different times, so we re-batch them during the training process.  We need to recover the
        original batch grouping so we can compute the loss correctly.
        """
        batch_scores: Dict[int, List[Variable]] = defaultdict(list)
        for state in finished_states:
            for score, batch_index in zip(state.score, state.batch_indices):
                batch_scores[batch_index].append(score)
        return batch_scores
