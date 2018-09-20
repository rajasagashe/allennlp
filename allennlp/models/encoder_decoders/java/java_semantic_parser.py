import json
from collections import defaultdict, Counter
from typing import Dict, Tuple, List, Set, Any
import gc
import re

import os

import time

import numpy as np

from allennlp.modules import TimeDistributed
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.training.metrics import Average
from overrides import overrides

import torch
from torch.autograd import Variable
from torch.nn.modules.rnn import LSTMCell
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from allennlp.common.util import timeit, debug_print
from allennlp.common import Params
from allennlp.common.util import timeit
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder, FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.nn.util import get_text_field_mask
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.nn.decoding import BeamSearch, JavaGrammarState, MaximumLikelihood, RnnState
from allennlp.models.encoder_decoders.java.java_decoder_state import JavaDecoderState
from allennlp.models.encoder_decoders.java.better_step import JavaDecoderStep


START_SYMBOL = "MemberDeclaration"

@Model.register("java_parser")
class JavaSemanticParser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 utterance_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 proto_encoder: Seq2SeqEncoder,
                 mixture_feedforward: FeedForward,
                 prototype_feedforward: FeedForward,
                 max_decoding_steps: int,
                 decoder_beam_search: BeamSearch,
                 action_embedding_dim: int,
                 should_copy_identifiers: bool,
                 use_utterance_embedder_identifiers: bool,
                 dropout: float = 0.0,
                 num_linking_features: int = 8,
                 rule_namespace: str = 'rule_labels',
                 attention_function: SimilarityFunction = None,
                 should_copy_proto_actions: bool = True,
                 seq2seq_baseline: bool = False,
                 serialization_dir: str = None) -> None:
        super(JavaSemanticParser, self).__init__(vocab)
        self._utterance_embedder = utterance_embedder
        self._should_copy_proto_actions = should_copy_proto_actions
        self._should_copy_identifiers = should_copy_identifiers
        self._use_utterance_embedder_identifiers = use_utterance_embedder_identifiers

        self._seq2seq_baseline = seq2seq_baseline
        self._max_decoding_steps = max_decoding_steps
        self._decoder_beam_search = decoder_beam_search

        self._encoder = encoder
        # todo(pr): later perhaps add a LSTM
        embedding_dim = utterance_embedder.get_output_dim()
        self._identifier_encoder = TimeDistributed(BagOfEmbeddingsEncoder(embedding_dim,
                                                                          averaged=True))

        self._boe = BagOfEmbeddingsEncoder(embedding_dim, averaged=True)

        # self._proto_encoder = proto_encoder

        self._input_attention = Attention(attention_function)

        self._decoder_trainer = MaximumLikelihood()
        self._code_bleu = Average()
        self._exact_match_accuracy = Average()
        self._avg_targ_log_probs = Average()
        self._avg_pred_log_probs = Average()

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._rule_namespace = rule_namespace


        self._action_embedding_dim = action_embedding_dim
        # self._action_embedder = Embedding(num_embeddings=vocab.get_vocab_size(self._rule_namespace),
        #                                   embedding_dim=action_embedding_dim)


        # This is what we pass as input in the first step of decoding, when we don't have a
        # previous action, or a previous question attention.
        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        self._first_attended_question = torch.nn.Parameter(torch.FloatTensor(encoder.get_output_dim()))
        # torch.nn.init.normal(self._first_action_embedding)
        # torch.nn.init.normal(self._first_attended_question)
        torch.nn.init.constant(self._first_action_embedding, 0.0)
        torch.nn.init.constant(self._first_attended_question, 0.0)

        # self._linking_params = torch.nn.Linear(num_linking_features, 1)
        # self._proto_params = torch.nn.Linear(2 * self._encoder.get_output_dim(), self._encoder.get_output_dim())

        self._decoder_step = JavaDecoderStep(encoder_output_dim=self._encoder.get_output_dim(),
                                             mixture_feedforward=mixture_feedforward,
                                             prototype_feedforward=prototype_feedforward,
                                             action_embedding_dim=action_embedding_dim,
                                             attention_function=attention_function,
                                             dropout=dropout,
                                             should_copy_identifiers=should_copy_identifiers,
                                             should_copy_proto_actions=should_copy_proto_actions,
                                             seq2seq_baseline=seq2seq_baseline)

        self._serialization_dir = serialization_dir
        self.load_actions(vocab)

    def load_actions(self, vocab):

        # with open(os.path.join(self._serialization_dir, 'grammar_rules.txt'), 'r') as f:
        with open(os.path.join('debug/', 'grammar_rules.txt'), 'r') as f:
            nonterminal2actions = json.load(f)
        self._nonterminal2actions = nonterminal2actions

        self._nonterminal2action2index = defaultdict(dict)
        for nt, actions in nonterminal2actions.items():
            for i, action in enumerate(actions):
                self._nonterminal2action2index[nt][action] = i

        # self._use_utterance_embedder_identifiers = False
        self._nonterminal2action_embeddings = self.get_action_embeddings()

        # if self._use_utterance_embedder_identifiers:
        #     self._identifier_embeddings = self.get_identifier_embeddings(vocab)

    def get_action_embeddings(self):
        # First: Count number of actions and map each nonterminal's actions
        # to an index range.
        num_actions = 0
        nonterminal2range = {}
        for nt, rules in self._nonterminal2actions.items():
            if self._use_utterance_embedder_identifiers and nt == 'IdentifierNT':
                print('continuing', nt)
                exit()
                continue
            nonterminal2range[nt] = (num_actions, num_actions + len(rules))
            num_actions += len(rules)


        # Second: Embed each of the actions based on the range
        self._action_embedder = Embedding(num_embeddings=num_actions,
                                          embedding_dim=self._action_embedding_dim)
        if torch.cuda.is_available():
            self._action_embedder.cuda()

        nonterminal2action_embeddings = {}
        for nt, (start, end) in nonterminal2range.items():
            nonterminal2action_embeddings[nt] = self._action_embedder.weight[start:end, :]
            # print('Device', self._action_embedder.weight[start:end, :].get_device())
        return nonterminal2action_embeddings

    def get_identifier_embeddings(self, vocab):
        # First: split up identifiers on camel case and get their vocabulary
        # indices.
        all_indices = []
        for rule in self._nonterminal2actions['IdentifierNT']:
            _, identifier_name = rule.split('-->')
            tokens = self.split_camel_case_add_original(identifier_name)
            # action_vocab = self.vocab.get_token_to_index_vocabulary(self._rule_namespace)
            # global_action_id = action_vocab.get(action[0])  # , -1)
            all_indices.append([vocab.get_token_index(t, 'utterance') for t in tokens])

        # Embed the tokens of the identifier, based on the indices and then average
        # the embeddings of the tokens for each identifier.
        all_embeddings = []
        for indices in all_indices:
            embeddings = []
            for index in indices:
                embeddings.append(self._utterance_embedder._token_embedders['tokens'].weight[index])
            # embedded = torch.stack(embeddings).squeeze(1)
            # averaged = self._boe(embedded.unsqueeze(0))
            all_embeddings.append(embeddings)
        return all_embeddings



    # def get_identifier_embeddings(self, vocab):
    #     print('get_identifier_embeddings')
    #     # First: split up identifiers on camel case and get their vocabulary
    #     # indices.
    #     all_indices = []
    #     for rule in self._nonterminal2actions['IdentifierNT']:
    #         _, identifier_name = rule.split('-->')
    #         tokens = self.split_camel_case_add_original(identifier_name)
    #         # action_vocab = self.vocab.get_token_to_index_vocabulary(self._rule_namespace)
    #         # global_action_id = action_vocab.get(action[0])  # , -1)
    #         all_indices.append([vocab.get_token_index(t, 'utterance') for t in tokens])
    #
    #     # Embed the tokens of the identifier, based on the indices and then average
    #     # the embeddings of the tokens for each identifier.
    #
    #
    #     self._temp_embedder = Embedding(num_embeddings=4000,
    #                                       embedding_dim=self._action_embedding_dim)
    #     if torch.cuda.is_available():
    #         self._temp_embedder.cuda()
    #
    #     embeddings = []
    #     for indices in all_indices:
    #         # tensor = Variable(self._action_embedder.weight.data.new(len(indices))).long()
    #         # tensor = self._action_embedder.weight.data.new(len(indices)).long()
    #         # for i, index in enumerate(indices):
    #         #     tensor[i] = index
    #         #
    #         # # print(tensor)
    #         # # Shape: (num_tokens, embedding_dim)
    #         # embedded = self._utterance_embedder({'tokens': tensor})
    #         # averaged = self._boe(embedded.unsqueeze(0))
    #         # embeddings.append(averaged)
    #
    #         # alternative to avoid creating a variable
    #         embed = []
    #         for index in indices:
    #             # embed.append(self._utterance_embedder._token_embedders['tokens'].weight[index])
    #             embed.append(self._temp_embedder.weight[index])
    #             # embed.append(self._action_embedder.weight[index])
    #         embedded = torch.stack(embed).squeeze(1)
    #         averaged = self._boe(embedded.unsqueeze(0))
    #         embeddings.append(averaged)
    #     return torch.stack(embeddings).squeeze(1)

    # todo(pr): these are copied from dataset reader, deduplicate these!
    def split_camel_case_add_original(self, name: str) -> List[str]:
        # Returns the string and its camel case split version.
        tokens = self.split_camel_case(name)
        if len(tokens) > 1:
            tokens = [name] + tokens
        return tokens
    @staticmethod
    def split_camel_case(name: str) -> List[str]:
        # Returns the string and its camel case split version.
        tokens = re.sub('(?!^)([A-Z][a-z]+)', r' \1', name).split()
        return tokens


    @overrides
    # @timeit
    def forward(self,  # type: ignore
                utterance: Dict[str, torch.LongTensor],
                # prototype_utterance: Dict[str, torch.LongTensor],
                # variable_names: Dict[str, torch.LongTensor],
                # variable_types: Dict[str, torch.LongTensor],
                # method_names: Dict[str, torch.LongTensor],
                # method_return_types: Dict[str, torch.LongTensor],
                # actions: List[List[ProductionRuleArray]],
                # java_class: Dict[str, torch.LongTensor],
                # entities: List[Set[str]],
                # identifiers: Dict[str, torch.LongTensor],

                rules: List[List[str]] = None,
                copy_identifiers = None,
                copy_identifiers_actions = None,
                prototype_rules: List[torch.Tensor] = None,
                metadata: List[Dict[str, str]] = None
                ) -> Dict[str, torch.Tensor]:

        # (batch_size, input_sequence_length, encoder_output_dim)
        embedded_utterance = self._utterance_embedder(utterance)
        batch_size, _, embedding_dim = embedded_utterance.size()
        utterance_mask = get_text_field_mask(utterance)

        encoder_outputs = self._dropout(self._encoder(embedded_utterance, utterance_mask))
        # proto_utt_encoder_outputs = self._dropout(self._encoder(proto_embedded_utterance, proto_utterance_mask))
        # final_proto_utt_encoder_output = util.get_final_encoder_states(proto_utt_encoder_outputs,
        #                                                      proto_utterance_mask,
        #                                                      True)

        # This will be our initial hidden state and memory cell for the decoder LSTM.

        # print("Sizes")
        # print(embedded_utterance.size())
        # print(utterance_mask.size())
        # print(encoder_outputs.size())
        # print('utterance')
        # print(utterance)
        final_utt_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                                 utterance_mask,
                                                                 True)


        if self._use_utterance_embedder_identifiers:
            exit()
            identifier_embeddings = self.get_identifier_embeddings(self.vocab)
            embeddings = []
            for i in range(len(identifier_embeddings)):
                embedded = torch.stack(identifier_embeddings[i]).unsqueeze(0)
                # print('embed',embedded.size())
                averaged = self._boe(embedded)
                # print('average', averaged.size())
                embeddings.append(averaged)

            all = torch.stack(embeddings).squeeze(1)
            if torch.cuda.is_available():
                all.cuda()
            # print('all', all.size())
            # print('all vals', all[0][:10])
            # print(self._utterance_embedder._token_embedders['tokens'].weight.get_device())
            # print('all vals', all.get_device())
            self._nonterminal2action_embeddings['IdentifierNT'] = all

        # proto_embedded_utterance = self._utterance_embedder(prototype_utterance)
        # proto_utterance_mask = get_text_field_mask(prototype_utterance)
        #
        # _, actionidx2vocabidx = self._embed_actions(actions)
        # actionidx2vocabidx[(0, -1)] = 0
        #
        # start1 = time.time()
        # proto_actions = []
        # proto_mask_lst = []
        # for r in prototype_rules:
        #     lst = r.long().data.cpu().numpy().tolist()
        #     # the -1 padding values need to be 0
        #     proto_mask_lst.append([x+1 for x in lst])
        #
        #     # lst = [v if v!=-1 else 0 for v in lst]
        #     indexes = [actionidx2vocabidx[(0, v)] if (0, v) in actionidx2vocabidx else 0 for v in lst]
        #     proto_actions.append(indexes)
        # proto_tensor = Variable(embedded_utterance.data.new(proto_actions).long())
        # proto_mask_tens = Variable(embedded_utterance.data.new(proto_mask_lst).long())
        # proto_embeddings = self._action_embedder(proto_tensor)
        # proto_mask = get_text_field_mask({'text': proto_mask_tens})
        # end1 = time.time()
        # # print("Time for proto rule embeddings", (end1 - start1) * 1000)
        # encoder_proto_out = self._dropout(self._proto_encoder(proto_embeddings, proto_mask))





        # (batch_size, question_length, encoder_output_dim)


        # todo(pr): for copying
        # textfield of environment names
        # Shape: (batch_size, num_actions, num_tokens, embedding dim)


        identifiers_embedding_list = None

        if self._should_copy_identifiers:
            embedded_identifiers = self._utterance_embedder(copy_identifiers, num_wrapping_dims=1)
            identifiers_mask = get_text_field_mask(copy_identifiers, num_wrapping_dims=1)

            # todo(pr): try an LSTM encoder here

            # Shape: (batch_size, num_actions, embedding dim)
            identifiers_encoded = self._identifier_encoder(embedded_identifiers,
                                                           identifiers_mask)
            # Since there will be many padding actions we trim to the number of actions per batch.
            identifiers_embedding_list = [identifiers_encoded[i][:len(actions)] for i, actions in enumerate(copy_identifiers_actions)]


        memory_cell = Variable(encoder_outputs.data.new(batch_size, self._encoder.get_output_dim()).fill_(0))
        initial_score = Variable(embedded_utterance.data.new(batch_size,1).fill_(0))

        # To make grouping states together in the decoder easier, we convert the batch dimension in
        # all of our tensors into an outer list.  For instance, the encoder outputs have shape
        # `(batch_size, question_length, encoder_output_dim)`.  We need to convert this into a list
        # of `batch_size` tensors, each of shape `(question_length, encoder_output_dim)`.  Then we
        # won't have to do any index selects, or anything, we'll just do some `torch.cat()`s.
        initial_score_list = [initial_score[i] for i in range(batch_size)]
        encoder_output_list = [encoder_outputs[i] for i in range(batch_size)]
        question_mask_list = [utterance_mask[i] for i in range(batch_size)]

        # proto_rules_encoder_output_list = [encoder_proto_out[i] for i in range(batch_size)]
        # proto_rules_encoder_mask_list = [proto_mask[i] for i in range(batch_size)]
        #
        # proto_utt_encoder_output_list = [proto_utt_encoder_outputs[i] for i in range(batch_size)]
        # proto_utt_encoder_mask_list = [proto_utterance_mask[i] for i in range(batch_size)]

        utt_final_encoder_output_list = [final_utt_encoder_output[i] for i in range(batch_size)]
        # proto_utt_final_encoder_output_list = [final_proto_utt_encoder_output[i] for i in range(batch_size)]
        # proto_utt_encoder_mask_list = [proto_mask[i] for i in range(batch_size)]

        initial_rnn_state = []
        for i in range(batch_size):
            initial_rnn_state.append(RnnState(final_utt_encoder_output[i],
                                              memory_cell[i],
                                              self._first_action_embedding,
                                              self._first_attended_question,
                                              encoder_output_list,
                                              question_mask_list,
                                              parent_states=[self._first_action_embedding],
                                            #   proto_rules_encoder_outputs=proto_rules_encoder_output_list,
                                            #   proto_rules_encoder_output_mask=proto_rules_encoder_mask_list,
                                            # proto_utt_encoder_outputs=proto_utt_encoder_output_list,
                                            #   proto_utt_encoder_output_mask=proto_utt_encoder_mask_list,
                                              utt_final_encoder_outputs=utt_final_encoder_output_list,
                                              # proto_utt_final_encoder_outputs=proto_utt_final_encoder_output_list
                                              ))

        nonterminals = self._nonterminal2actions.keys()
        initial_grammar_state = [JavaGrammarState(nonterminals=nonterminals)
                                 for i in range(batch_size)]

        initial_state = JavaDecoderState(batch_indices=list(range(batch_size)),
                                         action_history=[[] for _ in range(batch_size)],
                                         score=initial_score_list,
                                         rnn_state=initial_rnn_state,
                                         grammar_states=initial_grammar_state,
                                         nonterminal2actions=self._nonterminal2actions,
                                         nonterminal2action_embeddings=self._nonterminal2action_embeddings,
                                         nonterminal2action2index=self._nonterminal2action2index,
                                         should_copy_identifiers=self._should_copy_identifiers,
                                         copy_identifier_actions=copy_identifiers_actions,
                                         copy_identifier_embeddings=identifiers_embedding_list,
                                         # flattened_linking_scores=flattened_linking_scores,
                                         # actions_to_entities=actions_to_entities,
                                         # proto_actions=proto_actions,
                                         # proto_mask=proto_mask_lst,
                                         debug_info=None)


        if self.training:
            x = self._decoder_trainer.decode(initial_state,
                                                self._decoder_step,
                                                rules)
            # self._nonterminal2action_embeddings['IdentifierNT'] = None
            return x
        else:
            outputs: Dict[str, Any] = {}

            if rules is not None:
                mle_outputs = self._decoder_trainer.decode(initial_state,
                                             self._decoder_step,
                                             rules)


                outputs['loss'] = mle_outputs['loss']
                outputs['batch_scores'] = mle_outputs['batch_scores']

            # Now remove unk rules from the grammar state for better code gen.
            # initial_state.grammar_state = [self._create_grammar_state_no_unks(a) for a in actions]

            initial_state.grammar_states = [JavaGrammarState(nonterminals=nonterminals)
                                            for _ in range(batch_size)]
            num_steps = self._max_decoding_steps
            initial_state.debug_info = [[] for _ in range(batch_size)]
            best_final_states = self._decoder_beam_search.search(num_steps,
                                                                 initial_state,
                                                                 self._decoder_step,
                                                                 keep_final_unfinished_states=True)

            outputs['best_action_sequence'] = []
            outputs['debug_info'] = []
            outputs['entities'] = []
            outputs['logical_form'] = []
            outputs['rules'] = []

            for i in range(batch_size):
                if i in best_final_states:

                    em = 0
                    bleu = 0
                    log_prob_pred = 0
                    log_prob_target = 0

                    # print('pred rules', pred_rules)
                    pred_rules = best_final_states[i][0].action_history[0]
                    if rules is not None:
                        pred_code = self._gen_code_from_rules(pred_rules)
                        targ_code = metadata[i]['code']

                        bleu = self._get_bleu(targ_code, pred_code)
                        log_prob_target = outputs['batch_scores'][i]
                        log_prob_pred = best_final_states[i][0].score[0].data.cpu().numpy().tolist()[0]
                        self._log_predictions(metadata[i], pred_code, batch=i,
                                              bleu=bleu,
                                              log_prob_target=log_prob_target,
                                              log_prob_pred=log_prob_pred
                                              )
                        if pred_code == targ_code:
                            em = 1


                    # self._partial_production_rule_accuracy(partial_parse_acc)
                    self._code_bleu(bleu)
                    self._exact_match_accuracy(em)
                    # print('***************************************')
                    # print('i', i)
                    # print('best final states', best_final_states.keys())
                    # print('targ log probs', log_prob_target)
                    # print('batch scores', outputs['batch_scores'])
                    # print(best_final_states[i][0].grammar_states)
                    self._avg_targ_log_probs(log_prob_target)
                    self._avg_pred_log_probs(log_prob_pred)
                    outputs['rules'].append(best_final_states[i][0].action_history)

                    outputs['best_action_sequence'].append(pred_rules)
                    outputs['logical_form'].append(self.indent(self._gen_code_from_rules(pred_rules)))
                    # print('best final states', best_final_states[i][0])
                    # print('best final states', best_final_states[i][0].debug_info)

                    outputs['debug_info'].append(best_final_states[i][0].debug_info[0])  # type: ignore
                        # outputs['beam_loss'].append(best_final_states[i][0].score)  # type: ignore
                        # outputs['entities'].append(entities[i])

            # self._nonterminal2action_embeddings['IdentifierNT'] = None
            return outputs


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions.  This is (confusingly) a separate notion from the "decoder"
        in "encoder/decoder", where that decoder logic lives in ``WikiTablesDecoderStep``.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """

        best_actions = output_dict["best_action_sequence"]
        debug_infos = output_dict['debug_info']
        batch_action_info = []
        for batch_index, (predicted_actions, debug_info) in enumerate(zip(best_actions, debug_infos)):
            instance_action_info = []
            for predicted_action, action_debug_info in zip(predicted_actions, debug_info):
                action_info = {}
                action_info['predicted_action'] = predicted_action
                considered_actions = action_debug_info['considered_actions']
                probabilities = action_debug_info['probabilities']
                actions = []


                max_num_actions = len(considered_actions) if len(considered_actions) < 30 else 30
                probs_np = np.array(probabilities)
                top_indices = np.argpartition(probs_np, -max_num_actions)[-max_num_actions:]
                sorted_top_indices = top_indices[np.argsort(probs_np[top_indices])]
                for index in sorted_top_indices[::-1]:
                    actions.append((considered_actions[index], probabilities[index]))

                # for action, probability in zip(considered_actions, probabilities):
                #     print("Action and prob", probability, action)
                #     actions.append((action, probability))
                # actions.sort()

                considered_actions, probabilities = zip(*actions)

                # print('considered actions', considered_actions)

                action_info['considered_actions'] = considered_actions
                action_info['action_probabilities'] = probabilities

                # todo(pr): temporarily commented out prototype stuff
                # considered_prototype_actions = action_debug_info['considered_prototype_actions']
                # prototype_action_probs = action_debug_info['prototype_action_probs']
                # # print('sem parser decode')
                # # print(considered_prototype_actions)
                # # print(prototype_action_probs)
                # actions = []
                # for action, probability in zip(considered_prototype_actions, prototype_action_probs):
                #         actions.append((action, probability))
                # actions.sort()
                # if len(actions) != 0:
                #     considered_prototype_actions, prototype_action_probs = zip(*actions)
                # else:
                #     considered_prototype_actions, prototype_action_probs = [], []
                # action_info['considered_prototype_actions'] = considered_prototype_actions
                # action_info['prototype_action_probs'] = prototype_action_probs
                # action_info['prototype_attention'] = action_debug_info['prototype_attention']

                # todo(pr): hack to get demo to work. remove this
                action_info['considered_prototype_actions'] = considered_actions
                action_info['prototype_action_probs'] = probabilities
                action_info['prototype_attention'] = action_debug_info['question_attention']



                action_info['question_attention'] = action_debug_info['question_attention']
                instance_action_info.append(action_info)
            batch_action_info.append(instance_action_info)
        output_dict["predicted_actions"] = batch_action_info
        return output_dict


    @staticmethod
    def _get_bleu(gold_seq, pred_seq):
        # This is how Ling et al. compute bleu score.
        sm = SmoothingFunction()
        ngram_weights = [0.25] * min(4, len(gold_seq))
        return sentence_bleu([gold_seq], pred_seq, weights=ngram_weights, smoothing_function=sm.method3)

    @staticmethod
    def _average_action_history_match(predicted: List[int], targets: torch.LongTensor) -> int:
        # Since targets is padded with -1's we find the target's length by the first index
        # where the -1's start to occur.

        # todo(Rajas) add full sequence, exact match
        min_val, unpadded_targ_length = torch.min(targets, 0)
        if min_val.data.cpu().numpy().tolist()[0] == -1:
            unpadded_targ_length = unpadded_targ_length.data.cpu().numpy().tolist()[0]
        else:
            unpadded_targ_length = targets.size(0)
        min_length = min(len(predicted), unpadded_targ_length)
        max_length = max(len(predicted), unpadded_targ_length)

        predicted_tensor = Variable(targets.data.new(predicted))
        predicted_tensor = predicted_tensor[:min_length].long()
        targets_trimmed = targets[:min_length].long()

        correct_tensor = targets_trimmed.eq(predicted_tensor)
        del predicted_tensor
        num_correct = torch.sum(correct_tensor)
        num_correct = num_correct.data.cpu().numpy().tolist()[0]
        return num_correct / max_length


    def _is_terminal_rule(self, rule):
        lhs, rhs = rule.split('-->')
        return (
               "IdentifierNT" in lhs) \
               or re.match(r"^Nt_.*_literal-->.*", rule) \
               or rule == "<unk>"

    def _gen_code_from_rules(self, rules):
        stack = []
        code = []
        for i in range(0, len(rules)):
            _, rhs = rules[i].split('-->')
            if not self._is_terminal_rule(rules[i]):
                stack.extend(rhs.split('___')[::-1])
            else:
                code.append(rhs)
            try:
                top = stack.pop()
                while not top[0].isupper():
                    code.append(top)
                    if len(stack) == 0:
                        break
                    top = stack.pop()
            except:
                pass
        return code

    def _get_rules_from_action_history(self, action_history: List[int], actions: List[ProductionRuleArray]):
        rules = []
        for a in action_history:
            rules.append(actions[a][0])
        return rules

    def indent(self, code):
        newlists = [[]]
        for tok in code:
            newlists[-1].append(tok)
            if tok == '{' or tok == ';' or tok == '}':
                newlists.append([])
        indent = 0
        pretty = ""
        for x in newlists:
            if '}' in x:
                indent -= 1
            pretty += ('\t' * indent) + ' '.join(x) + "\n"
            if '{' in x:
                indent += 1
        return pretty.strip('\n')

    def _log_predictions(self, metadata, pred_code, batch, bleu, log_prob_target, log_prob_pred):
        # codef.write("batch, " + str(batch) + '\n')

        if getattr(self, '_on_extra', None) is not None:
            datasetname = 'valid' if not self._on_extra else 'extra'
            em_correct = open(os.path.join(self._serialization_dir, 'epoch%d-%s-em_correct.txt' %(self._epoch_num, datasetname)), 'a')
            em_incorrect = open(os.path.join(self._serialization_dir, 'epoch%d-%s-em_incorrect.txt' %(self._epoch_num, datasetname)), 'a')
            # codef = open(os.path.join(self._serialization_dir, 'epoch%d-%s-all-preds.txt' %(self._epoch_num, datasetname)), 'a')
            high_bleu = open(os.path.join(self._serialization_dir, 'epoch%d-%s-bleu-high.txt' %(self._epoch_num, datasetname)), 'a')
            all_json = open(os.path.join(self._serialization_dir, 'epoch%d-%s-all.json' %(self._epoch_num, datasetname)), 'a')
        else:
            em_correct = open('debug/em_correct.txt', 'a')
            em_incorrect = open('debug/em_incorrect.txt', 'a')
            all_json = open('debug/all.json', 'a')
            high_bleu = open('debug/bleu-high.txt', 'a')
            codef = open('debug/pred_target_code.txt', 'a')

        log = '==============' * 4 + '\n'
        log += metadata['path'] + '\n'
        # log += 'Variables:\n'
        # log += self.combine_name_types(metadata['variableNames'], metadata['variableTypes'])
        # log += 'Methods:\n'
        # log += self.combine_name_types(metadata['methodNames'], metadata['methodTypes'])
        log += 'Prototype========\n'
        log += metadata['prototype_path'] + '\n'
        log += 'NL:' + ' '.join(metadata['prototype_utterance']) + '\n'
        log += 'methodName:' + (metadata['prototype_methodName']) + '\n'
        log += self.indent(metadata['prototype_code']) + '\n'
        log += 'Target========\n'
        log += metadata['path'] + '\n'
        log += 'NL:' + ' '.join(metadata['utterance']) + '\n'
        log += 'methodName:' + (metadata['methodName']) + '\n'
        log += self.indent(metadata['code']) + '\n'
        log += 'Prediction====\n'
        log += self.indent(pred_code) + '\n'
        print(log)


        pred_dict = {}
        pred_dict['pred_code'] = pred_code
        pred_dict['bleu'] = bleu
        pred_dict['pred_prob'] = log_prob_pred
        pred_dict['targ_prob'] = log_prob_target
        all_json.write(json.dumps({**metadata, **pred_dict}, indent=4)+'\n')

        # codef.write(log)
        if metadata['code'] == pred_code:
            em_correct.write(log)
        else:
            em_incorrect.write(log)
        if bleu > .3:
            high_bleu.write(log)

        em_correct.close()
        em_incorrect.close()
        high_bleu.close()
        all_json.close()
        # codef.close()

    def combine_name_types(self, names, types):
        combine_str = ""
        for n, t in zip(names, types):
            combine_str += n + ' (' + t + ')\n'
        return combine_str

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'bleu': self._code_bleu.get_metric(reset),
            'em': self._exact_match_accuracy.get_metric(reset),
            'targ_probs': self._avg_targ_log_probs.get_metric(reset),
            'pred_probs': self._avg_pred_log_probs.get_metric(reset),
            # 'partial_acc': self._partial_production_rule_accuracy.get_metric(reset)

        }

    @staticmethod
    # @timeit
    def _create_grammar_state_no_unks(possible_actions: List[ProductionRuleArray]) -> JavaGrammarState:
        nonterminal2action_index = defaultdict(list)
        for i, action in enumerate(possible_actions):
            if not action[0]: # padding
                continue
            lhs, rhs = action[0].split('-->')
            if 'UNK' in rhs:
                continue
            nonterminal2action_index[lhs].append(i)
        return JavaGrammarState([START_SYMBOL], nonterminal2action_index)

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'SimpleSeq2Seq':
        utterance_embedder_params = params.pop("utterance_embedder")
        utterance_embedder = TextFieldEmbedder.from_params(vocab, utterance_embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        proto_encoder = Seq2SeqEncoder.from_params(params.pop("proto_encoder"))
        max_decoding_steps = params.pop("max_decoding_steps")
        action_embedding_dim = params.pop_int("action_embedding_dim")
        dropout = params.pop_float('dropout', 0.0)
        num_linking_features = params.pop_int('num_linking_features', 8)
        should_copy_proto_actions = params.pop_bool('should_copy_proto_actions', True)
        should_copy_identifiers = params.pop_bool('should_copy_identifiers')
        use_utterance_embedder_identifiers = params.pop_bool('use_utterance_embedder_identifiers')

        is_seq2seq_baseline = params.pop_bool('is_seq2seq_baseline', True)
        decoder_beam_search = BeamSearch.from_params(params.pop("decoder_beam_search"))
        mixture_feedforward_type = params.pop('mixture_feedforward', None)
        if mixture_feedforward_type is not None:
            mixture_feedforward = FeedForward.from_params(mixture_feedforward_type)
        else:
            mixture_feedforward = None

        prototype_feedforward_type = params.pop('prototype_feedforward', None)
        if prototype_feedforward_type is not None:
            prototype_feedforward = FeedForward.from_params(prototype_feedforward_type)
        else:
            prototype_feedforward = None

        serialization_dir = params.pop('serialization_dir', None)
        # nonterminal_embedder = TextFieldEmbedder.from_params(vocab, params.pop("nonterminal_embedder"))
        # terminal_embedder_params = params.pop('terminal_embedder', None)
        # if terminal_embedder_params:
        #     terminal_embedder = TextFieldEmbedder.from_params(vocab, terminal_embedder_params)
        # else:
        #     terminal_embedder = None
        # If no attention function is specified, we should not use attention, not attention with
        # default similarity function.
        attention_function_type = params.pop("attention_function", None)
        if attention_function_type is not None:
            attention_function = SimilarityFunction.from_params(attention_function_type)
        else:
            attention_function = None
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   utterance_embedder=utterance_embedder,
                   encoder=encoder,
                   proto_encoder=proto_encoder,
                   attention_function=attention_function,
                   mixture_feedforward=mixture_feedforward,
                   prototype_feedforward=prototype_feedforward,
                   # nonterminal_embedder=nonterminal_embedder,
                   # terminal_embedder=terminal_embedder,
                   action_embedding_dim=action_embedding_dim,
                   max_decoding_steps=max_decoding_steps,
                   decoder_beam_search=decoder_beam_search,
                   num_linking_features=num_linking_features,
                   should_copy_identifiers=should_copy_identifiers,
                   use_utterance_embedder_identifiers=use_utterance_embedder_identifiers,
                   dropout=dropout,
                   should_copy_proto_actions=should_copy_proto_actions,
                   seq2seq_baseline=is_seq2seq_baseline,
                   serialization_dir=serialization_dir)