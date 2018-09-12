from typing import Dict, List, Tuple

import torch

from allennlp.nn.decoding import JavaGrammarState, RnnState, DecoderState
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.common.util import timeit, debug_print

class JavaDecoderState(DecoderState['JavaDecoderState']):
    def __init__(self,
                 batch_indices: List[int],
                 action_history: List[List[str]],
                 score: List[torch.Tensor],
                 rnn_state: List[RnnState],
                 grammar_states: List[JavaGrammarState],
                 nonterminal2actions: Dict[str, List[str]],
                 nonterminal2action_embeddings: Dict[str, torch.Tensor],
                 nonterminal2action2index: Dict[str, torch.Tensor],
                 # flattened_linking_scores: torch.FloatTensor,
                 # actions_to_entities: Dict[Tuple[int, int], int],

                 # proto_actions:List[List[int]],
                 # proto_mask:List[List[int]],
                 debug_info: List = None
                 ) -> None:
        super(JavaDecoderState, self).__init__(batch_indices, action_history, score)
        self.rnn_state = rnn_state
        self.grammar_states = grammar_states
        self.nonterminal2actions = nonterminal2actions
        self.nonterminal2action_embeddings = nonterminal2action_embeddings
        self.nonterminal2action2index = nonterminal2action2index

        # self.flattened_linking_scores = flattened_linking_scores
        # self.actions_to_entities = actions_to_entities
        self.debug_info = debug_info
        # self.proto_actions = proto_actions
        # self.proto_mask = proto_mask

    def get_valid_actions_embeddings(self) -> Tuple[list, list]:
        """
        Returns a list of valid actions for each element of the group
                a list of embeddings for those valid actions
        """
        nonterminals = [state.get_current_nonterminal() for state in self.grammar_states]

        actions = [self.nonterminal2actions[nt] for nt in nonterminals]
        embeddings = [self.nonterminal2action_embeddings[nt] for nt in nonterminals]
        return actions, embeddings

    # @timeit
    def new_state_from_group_index(self,
                                   group_index: int,
                                   action: str,
                                   new_score: torch.Tensor,
                                   new_rnn_state: RnnState,
                                   considered_actions: List[int] = None,
                                   action_probabilities: List[float] = None,
                                   attention_weights: torch.Tensor = None) -> 'GrammarBasedDecoderState':


        batch_index = self.batch_indices[group_index]
        new_action_history = self.action_history[group_index] + [action]
        new_grammar_state = self.grammar_states[group_index].take_action(action)

        if self.debug_info is not None:
            attention = attention_weights[group_index] if attention_weights is not None else None
            debug_info = {
                    'considered_actions': considered_actions,
                    'question_attention': attention,
                    'probabilities': action_probabilities,
                    }
            new_debug_info = [self.debug_info[group_index] + [debug_info]]
        else:
            new_debug_info = None


        return JavaDecoderState(batch_indices=[batch_index],
                                 action_history=[new_action_history],
                                 score=[new_score],
                                 rnn_state=[new_rnn_state],
                                 grammar_states=[new_grammar_state],
                                nonterminal2actions=self.nonterminal2actions,
                                nonterminal2action_embeddings=self.nonterminal2action_embeddings,
                                nonterminal2action2index=self.nonterminal2action2index,
                                 debug_info=new_debug_info)

        # return GrammarBasedDecoderState(batch_indices=[batch_index],
        #                                 action_history=[new_action_history],
        #                                 score=[new_score],
        #                                 rnn_state=[new_rnn_state],
        #                                 grammar_state=[new_grammar_state],
        #                                 possible_actions=self.possible_actions,
        #                                 extras=self.extras,
        #                                 debug_info=new_debug_info)

    def is_finished(self) -> bool:
        if len(self.batch_indices) != 1:
            raise RuntimeError("is_finished() is only defined with a group_size of 1")
        return self.grammar_states[0].is_finished()

    @classmethod
    def combine_states(cls, states: List['JavaDecoderState']) -> 'JavaDecoderState':
        batch_indices = [batch_index for state in states for batch_index in state.batch_indices]
        action_histories = [action_history for state in states for action_history in state.action_history]
        scores = [score for state in states for score in state.score]
        rnn_state = [rnn_state for state in states for rnn_state in state.rnn_state]
        grammar_states = [grammar_state for state in states for grammar_state in state.grammar_states]

        # proto_actions = [proto_actions for state in states for proto_actions in state.proto_actions]
        # proto_actions_mask = [proto_mask for state in states for proto_mask in state.proto_mask]

        if states[0].debug_info is not None:
            debug_info = [debug_info for state in states for debug_info in state.debug_info]
        else:
            debug_info = None



        return JavaDecoderState(batch_indices=batch_indices,
                                action_history=action_histories,
                                score=scores,
                                rnn_state=rnn_state,
                                nonterminal2actions=states[0].nonterminal2actions,
                                nonterminal2action_embeddings=states[0].nonterminal2action_embeddings,
                                nonterminal2action2index=states[0].nonterminal2action2index,
                                grammar_states=grammar_states,
                                # flattened_linking_scores=states[0].flattened_linking_scores,
                                # actions_to_entities=states[0].actions_to_entities,
                                # proto_actions=proto_actions,
                                # proto_mask=proto_actions_mask,
                                debug_info=debug_info)
