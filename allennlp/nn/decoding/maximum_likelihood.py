from collections import defaultdict
import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import time
import torch
from torch.autograd import Variable

from allennlp.nn.decoding.decoder_step import DecoderStep
from allennlp.nn.decoding.decoder_state import DecoderState
from allennlp.nn.decoding.decoder_trainer import DecoderTrainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class MaximumLikelihood(DecoderTrainer[Tuple[torch.Tensor, torch.Tensor]]):
    def decode(self,
               initial_state: DecoderState,
               decode_step: DecoderStep,
               batch_actions: List[List[str]]) -> Dict[str, torch.Tensor]:

        finished_states = []
        states = [initial_state]
        step_num = 0
        while states:
            # print('step num', step_num)
            next_states = []
            # We group together all current states to get more efficient (batched) computation.
            grouped_state = states[0].combine_states(states)

            allowed_actions = self._get_allowed_actions(grouped_state, step_num, batch_actions)
            step_num += 1

            # print('stepnum', step_num, grouped_state.batch_indices, grouped_state.grammar_states[0]._nonterminal_stack)

            # todo(pr) assert that next states and finished sum up to batch size
            for next_state in decode_step.take_step(grouped_state, allowed_actions=allowed_actions):
                if next_state.is_finished():
                    finished_states.append(next_state)
                else:
                    next_states.append(next_state)
            states = next_states

        batch_scores = self._group_scores_by_batch(finished_states)
        loss = 0
        for score in batch_scores.values():  # we don't care about the batch index, just the scores
        # for state in finished_states:
        #     loss += -state.score[0]
            loss += -score[0]

        # print('unprocessd', batch_scores)
        for b in batch_scores.keys():
            # print(b)
            # x = batch_scores[b][0].data.cpu().numpy().tolist()[0]
            # print('batch', x)
            # if type(x) == list:
            #     exit()
            batch_scores[b] = batch_scores[b][0].data.cpu().numpy().tolist()[0]

        if len(batch_scores) != len(batch_actions):
            print('lens')
            print(len(batch_scores), len(batch_actions))

            for state in finished_states:
                print('finished', state.batch_indices, state.grammar_states[0]._nonterminal_stack)
            for i in range(len(batch_actions)):
                if i not in batch_scores:
                    print('Batch index not complete', i)

            exit()
        # print('batch scores', batch_scores)
        return {'loss': loss / len(finished_states), 'batch_scores': batch_scores}

    @staticmethod
    def _get_allowed_actions(state: DecoderState, step_num: int, batch_actions: List[List[str]]):
        # This method just returns index of local actions, since considered uses local indices.
        allowed_actions = []
        # print('==================================')
        for group, batch_index in enumerate(state.batch_indices):
            # print('group', group, batch_index, '-----------')
            # print(len(batch_actions[group]), step_num)
            # print(state.grammar_states[group]._nonterminal_stack)


            action = batch_actions[batch_index][step_num]
            # print('action', action)
            # print('rules for batch', batch_index)
            # for rule in batch_actions[batch_index]:
            #     print(rule)

            allowed_actions.append(action)
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
