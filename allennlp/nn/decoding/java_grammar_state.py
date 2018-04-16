from copy import deepcopy
from typing import List, Dict, Tuple

from allennlp.data.fields.production_rule_field import ProductionRuleArray
# from java_programmer.fields.java_production_rule_field import ProductionRuleArray

class JavaGrammarState:
    def __init__(self,
                 nonterminal_stack: List[str],
                 valid_actions: Dict[str, List[int]],
                 action_history: List[str] = None) -> None:
        self._nonterminal_stack = nonterminal_stack
        self._valid_actions = valid_actions
        self._action_history = action_history or []
        # self._action_indices = action_indices


    def is_finished(self) -> bool:
        """
        Have we finished producing our logical form?  We have finished producing the logical form
        if and only if there are no more non-terminals on the stack.
        """
        return len(self._nonterminal_stack) == 0
        # return not self._nonterminal_stack

    def number_nonterminals_in_rhs(self, production_rule: str):
        _, rhs = production_rule.split('-->')
        rhs_elements = rhs.split('___')
        return sum((element in self._valid_actions.keys()) for element in rhs_elements)

    def get_valid_actions(self) -> List[int]:
        """
        Returns a list of valid actions as integers.
        """
        return self._valid_actions[self._nonterminal_stack[-1]]

    def take_action(self, rule: str) -> 'GrammarState':
        left_side, right_side = rule.split('-->')

        assert self._nonterminal_stack[-1] == left_side, (f"Tried to expand {self._nonterminal_stack[-1]}"
                                                          "but got rule f{left_side}->f{right_side}")

        new_stack = deepcopy(self._nonterminal_stack)
        new_stack.pop()

        new_action_history = deepcopy(self._action_history)
        new_action_history.append(rule)

        right_side_list = right_side.split('___')
        right_side_list.reverse()
        new_stack = new_stack + right_side_list

        # now consume the terminals added to the stack
        nonterminals = self._valid_actions.keys()
        for i in range(len(new_stack)):
            if new_stack[-1] not in nonterminals:
                # Its a terminal
                new_stack.pop()
            else:
                break

        return JavaGrammarState(nonterminal_stack=new_stack,
                                valid_actions=self._valid_actions,
                                action_history=new_action_history)

