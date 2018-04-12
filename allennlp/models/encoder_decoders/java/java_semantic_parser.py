from typing import Dict, Tuple, List, Set

import numpy
from overrides import overrides

import torch
from torch.autograd import Variable
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.seq2seq import START_SYMBOL, END_SYMBOL
from allennlp.models.encoder_decoders.java.java_decoder_step import JavaDecoderStep
from allennlp.models.encoder_decoders.java.java_decoder_state import JavaDecoderState
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder, TimeDistributed
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.nn.decoding import RnnState
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, weighted_sum
from allennlp.semparse.type_declarations import JavaGrammarState


@Model.register("java_parser")
class JavaSemanticParser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 nonterminal_embedder: TextFieldEmbedder,
                 terminal_embedder: TextFieldEmbedder,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 attention_function: SimilarityFunction = None,
                 scheduled_sampling_ratio: float = 0.0) -> None:
        super(JavaSemanticParser, self).__init__(vocab)
        self._source_embedder = source_embedder
        self._nonterminal_embedder = nonterminal_embedder
        self._terminal_embedder = terminal_embedder

        self._encoder = encoder

        self._embedding_dim = self._nonterminal_embedder.get_output_dim()
        self._decoder_output_dim = self._encoder.get_output_dim()

        self._input_attention = Attention(attention_function)
        # TODO (pradeep): Do not hardcode decoder cell type.
        # todo don't hardcode the dim
        self._decoder_cell = LSTMCell(120, self._decoder_output_dim)
        # self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)
        self._decoder_step = JavaDecoderStep(encoder_output_dim=self._encoder.get_output_dim(),
                                                   # action_embedding_dim=action_embedding_dim,
                                                   attention_function=attention_function,
                                                   nonterminal_embedder=nonterminal_embedder)

    @overrides
    def forward(self,  # type: ignore
                utterance: Dict[str, torch.LongTensor],
                variable_names: Dict[str, torch.LongTensor],
                variable_types: Dict[str, torch.LongTensor],
                method_names: Dict[str, torch.LongTensor],
                method_return_types: Dict[str, torch.LongTensor],
                code: Dict[str, torch.LongTensor],
                # rules: Dict[str, torch.LongTensor],
                rules: List[List[ProductionRuleArray]],
                nonterminals: Dict[str, torch.LongTensor],
                prev_rules: Dict[str, torch.LongTensor],
                rule_parent_index: torch.LongTensor) -> Dict[str, torch.Tensor]:

        # (batch_size, input_sequence_length, encoder_output_dim)

        # Encode summary, variables, methods with bi lstm.
        ##################################################
        """
        embedded_code_summary = self._source_embedder(code_summary)
        code_summary_mask = get_text_field_mask(code_summary)

        embedded_variable_names = self._source_embedder(variable_names)
        embedded_variable_types = self._source_embedder(variable_types)

        # embedded_method_names = self._source_embedder(method_names)
        # embedded_method_return_types = self._source_embedder(method_return_types)

        encoder = TimeDistributed(self._encoder)
        variable_name_mask = get_text_field_mask(variable_names, num_wrapping_dims=1)
        encoded_variable_names = encoder(embedded_variable_names, variable_name_mask)

        # method_name_mask = get_text_field_mask(method_names, num_wrapping_dims=1)
        # encoded_method_names = encoder(embedded_method_names, method_name_mask)

        variable_name_type_input = torch.cat((encoded_variable_names, embedded_variable_types.unsqueeze(2)), dim=2)
        variable_type_mask = get_text_field_mask(variable_types).unsqueeze(-1)
        variable_name_type_mask = torch.cat((variable_name_mask, variable_type_mask), dim=2)
        encoded_variable_names_types = encoder(variable_name_type_input, variable_name_type_mask)

        encoded_code_summary = self._encoder(embedded_code_summary, code_summary_mask)


        embedded_input = self._source_embedder(source_tokens)
        batch_size, _, _ = embedded_input.size()
        source_mask = get_text_field_mask(source_tokens)
        encoder_outputs = self._encoder(embedded_input, source_mask)


        encoder_outputs = torch.cat(encoded_code_summary, encoded_variable_names_types, dim=1)

        final_encoder_output = encoded_code_summary[:, -1]  # (batch_size, encoder_output_dim)
        """
        # (batch_size, input_sequence_length, encoder_output_dim)
        embedded_input = self._source_embedder(utterance)
        batch_size, _, _ = embedded_input.size()
        utterance_mask = get_text_field_mask(utterance)
        encoder_outputs = self._encoder(embedded_input, utterance_mask)
        final_encoder_output = encoder_outputs[:, -1]  # (batch_size, encoder_output_dim)
        # Decoder
        # embedded_nonterminals = self._nonterminal_embedder(nonterminals)
        # embedded_rules = self._nonterminal_embedder(rules)
        # batch_size, num_rules, num_rule_tokens, _ = embedded_rules.size()
        #
        # boe = BagOfEmbeddingsEncoder(self._embedding_dim, averaged=True)
        # embedded_nonterminals = embedded_nonterminals.squeeze(2)
        # embedded_rules = boe(embedded_rules.view(-1, num_rule_tokens, self._embedding_dim))
        # embedded_rules = embedded_rules.view(batch_size, num_rules, self._embedding_dim)

        hidden_state = encoder_outputs[:, -1]  # (batch_size, encoder_output_dim)
        memory_cell = Variable(encoder_outputs.data.new(batch_size, self._encoder.get_output_dim()).fill_(0))
        attended_utterance, _ = self._decoder_step.attend_on_utterance(hidden_state,
                                                                       encoder_outputs,
                                                                       utterance_mask)

        # create rnn state, decoder state, decoder step, fire off trainer
        # set initial state
        initial_rnn_state = RnnState(hidden_state,
                                     memory_cell,
                                     None, # TODO do we need previous action embedding?
                                     attended_utterance,
                                     encoder_outputs,
                                     utterance_mask)
        initial_grammar_state = JavaGrammarState([], None) # insert first method decl nonterminal
        initial_score = Variable(encoder_outputs.data.new(batch_size).fill_(0))

        nonterminals = [rules[batch_index][0]['left'] for batch_index in range(batch_size)]
        initial_state = JavaDecoderState([],
                                         initial_score,
                                         initial_rnn_state,
                                         initial_grammar_state,
                                         nonterminals,
                                         prev_rules=None,
                                         nonterminal2parent_rule=None,
                                         nonterminal2parent_state=None) # todo more args

        # nonterminal_embedder: TextFieldEmbedder,
        # nonterminal: torch.Tensor,
        # prev_rule: torch.Tensor,
        # nonterminal2parent_rule: Dict[torch.LongTensor, torch.LongTensor],
        # nonterminal2parent_state: Dict[torch.LongTensor, torch.LongTensor]

        if self.training:
            # embedding for the rhs of gold sequences
            rule_index = 0
            allowed_actions = [rules[batch_index][rule_index]['right'] for batch_index in range(batch_size)]

            state = initial_state
            next_states = []
            for next_state in self._decoder_step.take_step(state, allowed_actions=allowed_actions):
                next_states.append(next_state)
                state = next_state
                rule_index += 1
                allowed_actions = [rules[batch_index][rule_index]['right'] for batch_index in range(batch_size)]

            loss = 0
            scores = [s.score for s in next_states]
            for scores in scores.values():  # we don't care about the batch index, just the scores
                loss += -util.logsumexp(torch.cat(scores))
            return {'loss': loss / len(scores)}

        else:
            print('inference yo')
            # fire beam search
            num_steps = self._max_decoding_steps
            # This tells the state to start keeping track of debug info, which we'll pass along in
            # our output dictionary.

            best_final_states = self._beam_search.search(num_steps,
                                                         initial_state,
                                                         self._decoder_step,
                                                         keep_final_unfinished_states=False)



        # Loss
        ############################################################################
        # todo find loss from the context list
        return {"loss": 0.0}

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.LongTensor:
        """
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        relevant_targets = targets[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        loss = sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)
        return loss

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.data.cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace="target_tokens")
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'SimpleSeq2Seq':
        source_embedder_params = params.pop("source_embedder")
        source_embedder = TextFieldEmbedder.from_params(vocab, source_embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        max_decoding_steps = params.pop("max_decoding_steps")
        target_namespace = params.pop("target_namespace", "tokens")

        nonterminal_embedder = TextFieldEmbedder.from_params(vocab, params.pop("nonterminal_embedder"))
        terminal_embedder_params = params.pop('terminal_embedder', None)
        if terminal_embedder_params:
            terminal_embedder = TextFieldEmbedder.from_params(vocab, terminal_embedder_params)
        else:
            terminal_embedder = None
        # If no attention function is specified, we should not use attention, not attention with
        # default similarity function.
        attention_function_type = params.pop("attention_function", None)
        if attention_function_type is not None:
            attention_function = SimilarityFunction.from_params(attention_function_type)
        else:
            attention_function = None
        scheduled_sampling_ratio = params.pop_float("scheduled_sampling_ratio", 0.0)
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   source_embedder=source_embedder,
                   encoder=encoder,
                   max_decoding_steps=max_decoding_steps,
                   target_namespace=target_namespace,
                   attention_function=attention_function,
                   scheduled_sampling_ratio=scheduled_sampling_ratio,
                   nonterminal_embedder=nonterminal_embedder,
                   terminal_embedder=terminal_embedder)
