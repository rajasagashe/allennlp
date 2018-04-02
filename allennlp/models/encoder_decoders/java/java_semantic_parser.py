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
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder, TimeDistributed
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, weighted_sum


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

    @overrides
    def forward(self,  # type: ignore
                utterance: Dict[str, torch.LongTensor],
                variable_names: Dict[str, torch.LongTensor],
                variable_types: Dict[str, torch.LongTensor],
                method_names: Dict[str, torch.LongTensor],
                method_return_types: Dict[str, torch.LongTensor],
                code: Dict[str, torch.LongTensor],
                rules: Dict[str, torch.LongTensor],
                nonterminals: Dict[str, torch.LongTensor],
                prev_rules: Dict[str, torch.LongTensor],
                rule_parent_index: torch.LongTensor) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : Dict[str, torch.LongTensor]
           The output of ``TextField.as_array()`` applied on the source ``TextField``. This will be
           passed through a ``TextFieldEmbedder`` and then through an encoder.
        target_tokens : Dict[str, torch.LongTensor], optional (default = None)
           Output of ``Textfield.as_array()`` applied on target ``TextField``. We assume that the
           target tokens are also represented as a ``TextField``.
        """
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
        ##################################################
        embedded_nonterminals = self._nonterminal_embedder(nonterminals)
        embedded_prev_rules = self._nonterminal_embedder(prev_rules)
        embedded_rules = self._nonterminal_embedder(rules)
        batch_size, num_rules, num_rule_tokens, _ = embedded_rules.size()

        boe = BagOfEmbeddingsEncoder(self._embedding_dim, averaged=True)
        embedded_nonterminals = embedded_nonterminals.squeeze(2)
        embedded_rules = boe(embedded_rules.view(-1, num_rule_tokens, self._embedding_dim))
        embedded_rules = embedded_rules.view(batch_size, num_rules, self._embedding_dim)
        embedded_prev_rules = boe(embedded_prev_rules.view(-1, num_rule_tokens, self._embedding_dim))
        embedded_prev_rules = embedded_prev_rules.view(batch_size, num_rules, self._embedding_dim)

        rule_parent_index = rule_parent_index.long()
        embedded_parent_rules = util.batched_index_select(embedded_rules, rule_parent_index)



        hidden_state = encoder_outputs[:, -1]  # (batch_size, encoder_output_dim)
        memory_cell = Variable(encoder_outputs.data.new(batch_size, self._encoder.get_output_dim()).fill_(0))

        rule_parent_state = [[0] * num_rules] * batch_size
        context_list = []
        for i, (nt, prev_rule, parent_rule) in enumerate(zip(embedded_nonterminals.split(1, 1), embedded_prev_rules.split(1, 1), embedded_parent_rules.split(1, 1))):

            if i == 0:
                parent_states = Variable(hidden_state.data.new(batch_size, self._encoder.get_output_dim()).fill_(0))
            else:
                parent_states = []
                for batch_index in range(batch_size):
                    parent_index = rule_parent_index[batch_index][i]
                    parent_state = rule_parent_state[batch_index][parent_index.data[0]]
                    if type(parent_state) == type(3):
                        parent_state = Variable(encoder_outputs.data.new(1, self._embedding_dim).fill_(0))
                    parent_states.append(parent_state)
                parent_states = torch.cat(parent_states, 0)
            # print("Sizes")
            # print(nt.size())
            # print(prev_rule.size())
            # print(parent_states.size())
            decoder_input = torch.cat((nt.squeeze(1), prev_rule.squeeze(1), parent_rule.squeeze(1), parent_states), 1)
            hidden_state, memory_cell = self._decoder_cell(decoder_input, (hidden_state, memory_cell))

            attended_utterance, attention_weights = self.attend_on_utterance(hidden_state,
                                                                             encoder_outputs,
                                                                             utterance_mask)
            # todo add the attention over the variables and methods
            # compute context as concatenation of hidden state and attended encoder outputs
            context = torch.cat((hidden_state, attended_utterance), 1)

            # update rule_parent_state with the current state
            for batch_index in range(batch_size):
                rule_parent_state[batch_index][i] = hidden_state[batch_index].unsqueeze(0)

            # add to list for prob
            context_list.append(context)



        # Loss
        ############################################################################
        # todo find loss from the context list
        return {"loss": 0.0}


    def attend_on_utterance(self,
                           query: torch.Tensor,
                           encoder_outputs: torch.Tensor,
                           encoder_output_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a query (which is typically the decoder hidden state), compute an attention over the
        output of the utterance encoder, and return a weighted sum of the utterance representations
        given this attention.  We also return the attention weights themselves.

        This is a simple computation, but we have it as a separate method so that the ``forward``
        method on the main parser module can call it on the initial hidden state, to simplify the
        logic in ``take_step``.
        """
        # (group_size, utterance_length)
        utterance_attention_weights = self._input_attention(query,
                                                            encoder_outputs,
                                                            encoder_output_mask.float())
        # (group_size, encoder_output_dim)
        attended_utterance = util.weighted_sum(encoder_outputs, utterance_attention_weights)
        return attended_utterance, utterance_attention_weights




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
