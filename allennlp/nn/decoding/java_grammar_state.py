from copy import deepcopy
from typing import List, Dict, Tuple

from allennlp.data.fields.production_rule_field import ProductionRuleArray
# from java_programmer.fields.java_production_rule_field import ProductionRuleArray
START_SYMBOL = "MemberDeclaration"
IdentifierNT = 'IdentifierNT'
class JavaGrammarState:
    def __init__(self,
                 nonterminals: Dict[str, str],
                 nonterminal_stack: List[str] = None,
                 # action_history: List[str] = None
                 ) -> None:
        self._nonterminal_stack = [START_SYMBOL] if nonterminal_stack is None else nonterminal_stack
        self._nonterminals = nonterminals
        # self._action_history = action_history or []
        # self._action_indices = action_indices


    def is_finished(self) -> bool:
        """
        Have we finished producing our logical form?  We have finished producing the logical form
        if and only if there are no more non-terminals on the stack.
        """
        return len(self._nonterminal_stack) == 0
        # return not self._nonterminal_stack

    def number_nonterminals_in_rhs(self, production_rule: str):
        # todo(rajas): add better checks to guarantee that 0 returned
        # for a rule where rhs are all terminals
        lhs, rhs = production_rule.split('-->')
        if IdentifierNT not in lhs:
            rhs_elements = rhs.split('___')
            return sum((element in self._nonterminals) for element in rhs_elements)
        return 0

    def get_current_nonterminal(self) -> List[int]:
        """
        Returns a list of valid actions as integers.
        """
        # return self._nonterminals[self._nonterminal_stack[-1]]
        return self._nonterminal_stack[-1]

    def take_action(self, rule: str) -> 'GrammarState':
        left_side, right_side = rule.split('-->')

        assert self._nonterminal_stack[-1] == left_side, (f"Tried to expand {self._nonterminal_stack[-1]}"
                                                          "but got rule f{left_side}->f{right_side}")

        new_stack = deepcopy(self._nonterminal_stack)
        new_stack.pop()

        # new_action_history = deepcopy(self._action_history)
        # new_action_history.append(rule)

        right_side_list = right_side.split('___')
        right_side_list.reverse()
        if IdentifierNT not in left_side:
            # IdentifierNT can expand to names which look like nonterminals.
            # For example the production rule IdentifierNT-->Expression is in
            # the dataset. Thus, when the lhs is IdentifierNT, we don't push
            # the rhs on, and this is ok since the rhs is always a terminal.

            # Add only the nonterminals to the stack.
            new_stack += [s for s in right_side_list if s in self._nonterminals]

        return JavaGrammarState(nonterminal_stack=new_stack,
                                nonterminals=self._nonterminals,
                                # action_history=new_action_history
                                )

