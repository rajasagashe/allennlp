from typing import Dict, List, Tuple

import torch

from allennlp.nn.decoding import JavaGrammarState, RnnState, DecoderState
from allennlp.data.fields.production_rule_field import ProductionRuleArray

# from java_programmer.allennlp_in_progress.decoder_state import DecoderState
# from java_programmer.allennlp_in_progress.rnn_state import RnnState
# from java_programmer.fields.java_production_rule_field import ProductionRuleArray
# from java_programmer.grammar.java_grammar_state import JavaGrammarState


class JavaDecoderState(DecoderState['JavaDecoderState']):
    def __init__(self,
                 batch_indices: List[int],
                 action_history: List[List[str]],
                 score: List[torch.Tensor],
                 rnn_state: List[RnnState],
                 grammar_state: List[JavaGrammarState],
                 # nonterminal_action_indices: List[int],
                 # action_embeddings: torch.Tensor,
                 action_indices: Dict[Tuple[int, int], int],
                 possible_actions: List[List[ProductionRuleArray]],
                 flattened_linking_scores: torch.FloatTensor,
                 actions_to_entities: Dict[Tuple[int, int], int],
                 debug_info: List = None
                 # prev_rules: List[torch.Tensor],
                 # nonterminal2parent_rules: List[Dict[str, torch.LongTensor]],
                 # nonterminal2parent_states: List[Dict[str, torch.LongTensor]]
                 ) -> None:
        super(JavaDecoderState, self).__init__(batch_indices, action_history, score)
        self.rnn_state = rnn_state
        self.grammar_state = grammar_state
        # self.action_embeddings = action_embeddings
        self.action_indices = action_indices
        self.possible_actions = possible_actions
        self.flattened_linking_scores = flattened_linking_scores
        self.actions_to_entities = actions_to_entities
        self.debug_info = debug_info
        # self.nonterminal_action_indices = nonterminal_action_indices
        # self.prev_rules = prev_rules
        # self.nonterminal2parent_rules = nonterminal2parent_rules
        # self.nonterminal2parent_states = nonterminal2parent_states

    def get_valid_actions(self) -> List[torch.Tensor]:
        """
        Returns a list of valid actions for each element of the group.
        """
        return [state.get_valid_actions() for state in self.grammar_state]

    def is_finished(self) -> bool:
        if len(self.batch_indices) != 1:
            raise RuntimeError("is_finished() is only defined with a group_size of 1")
        return self.grammar_state[0].is_finished()

    @classmethod
    def combine_states(cls, states: List['JavaDecoderState']) -> 'JavaDecoderState':
        batch_indices = [batch_index for state in states for batch_index in state.batch_indices]
        action_histories = [action_history for state in states for action_history in state.action_history]
        scores = [score for state in states for score in state.score]
        rnn_state = [rnn_state for state in states for rnn_state in state.rnn_state]
        grammar_states = [grammar_state for state in states for grammar_state in state.grammar_state]
        if states[0].debug_info is not None:
            debug_info = [debug_info for state in states for debug_info in state.debug_info]
        else:
            debug_info = None
        # nonterminals = [nonterminal for state in states for nonterminal in state.nonterminals]
        # prev_rules = [prev_rule for state in states for prev_rule in state.prev_rules]
        # nonterminal_action_indices = [nonterminal_action_index for state in states for nonterminal_action_index in state.nonterminal_action_indices]
        # print("Combine len prev rules", len(prev_rules))
        # print("Combine len states", len(states))
        # nonterminal2parent_rules = [nonterminal2parent_rule for state in states for nonterminal2parent_rule in state.nonterminal2parent_rules]
        # nonterminal2parent_states = [nonterminal2parent_state for state in states for nonterminal2parent_state in state.nonterminal2parent_states]

        return JavaDecoderState(batch_indices=batch_indices,
                                action_history=action_histories,
                                score=scores,
                                rnn_state=rnn_state,
                                # action_embeddings=states[0].action_embeddings,
                                action_indices=states[0].action_indices,
                                grammar_state=grammar_states,
                                possible_actions=states[0].possible_actions,
                                flattened_linking_scores=states[0].flattened_linking_scores,
                                actions_to_entities=states[0].actions_to_entities,
                                debug_info=debug_info
                                # nonterminals=nonterminals,
                                # nonterminal_action_indices=nonterminal_action_indices,
                                # prev_rules=prev_rules,
                                # nonterminal2parent_rules=nonterminal2parent_rules,
                                # nonterminal2parent_states=nonterminal2parent_states,
                                # lhs2indexed=states[0].lhs2indexed,
                                # rhs2indexed=states[0].rhs2indexed
                                )
