from typing import Dict, List, Tuple

import torch

from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.modules import TextFieldEmbedder
from allennlp.nn.decoding import DecoderState, RnnState
from allennlp.semparse.type_declarations import JavaGrammarState


class JavaDecoderState(DecoderState['JavaDecoderState']):
    def __init__(self,
                 action_history: List[int],
                 score: torch.Tensor,
                 rnn_state: RnnState,
                 grammar_state: JavaGrammarState,
                 nonterminal: List[Tuple[str, bool, Dict[str, torch.Tensor]]],
                 # todo change the types on the following 3 parameters
                 prev_rules: torch.Tensor,
                 nonterminal2parent_rule: Dict[torch.LongTensor, torch.LongTensor],
                 nonterminal2parent_state: Dict[torch.LongTensor, torch.LongTensor]) -> None:
        super(JavaDecoderState, self).__init__(None, action_history, score)
        self._grammar_state = grammar_state
        self._rnn_state = rnn_state
        self._batch_size = rnn_state.hidden_state.size(0)
        if len(action_history) == 0:
            # todo insert the method decl starting action in action history
            print('todo')

        self._nonterminal = nonterminal
        self._prev_rule = prev_rules
        self._nonterminal2parent_rule = nonterminal2parent_rule
        self._nonterminal2parent_state = nonterminal2parent_state


    def get_valid_actions(self) -> List[List[int]]:
        """
        Returns a list of valid actions for each element of the group.
        """
        return self._grammar_state.get_valid_actions()
        # return [state.get_valid_actions() for state in self.grammar_state]