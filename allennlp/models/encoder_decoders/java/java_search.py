import json
import threading
from collections import defaultdict, Counter
from typing import Dict, Tuple, List, Set, Any
import gc
import re
import random
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
import torch.nn.functional as F
import nmslib

@Model.register("java_search_model")
class JavaSearchModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 utterance_embedder: TextFieldEmbedder,
                 code_embedder: TextFieldEmbedder,
                 context_encoder: Seq2SeqEncoder,
                 code_encoder: Seq2SeqEncoder,
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
        super(JavaSearchModel, self).__init__(vocab)
        self._utterance_embedder = utterance_embedder
        # self._code_embedder = code_embedder
        self._code_embedder = utterance_embedder

        self._context_encoder = context_encoder
        self._code_encoder = code_encoder
        # todo(pr): later perhaps add a LSTM
        # self._identifier_encoder = TimeDistributed(BagOfEmbeddingsEncoder(embedding_dim,
        #                                                                   averaged=True))

        # self._boe = BagOfEmbeddingsEncoder(embedding_dim, averaged=True)
        self._code_bleu = Average()
        self._encoded_all_code = None
        self._search_index = None
        self._serialization_dir = serialization_dir

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
                context: Dict[str, torch.LongTensor],
                # random_context: Dict[str, torch.LongTensor],
                code: Dict[str, torch.LongTensor] = None,
                current_record: List[Dict[str, str]] = None,
                all_code: Dict[str, torch.LongTensor] = None,
                all_context: Dict[str, torch.LongTensor] = None,
                all_train_records: List[Dict[str, str]] = None,
                record_indices: List[int] = None,
                ) -> Dict[str, torch.Tensor]:

        # Embed the code and utterance
        embedded_context = self._utterance_embedder(context)
        context_mask = get_text_field_mask(context)

        batch_size = embedded_context.size(0)

        contexts = []
        for batch_index in range(batch_size):
            random_num = random.randint(0, all_context['tokens'].size(0)-1)
            while random_num == record_indices[batch_index]:
                random_num = random.randint(0, all_context['tokens'].size(0)-1)

            # print(batch_index)
            # print(record_indices)
            # print(random_num, len(all_train_records))
            # print("Record", all_train_records[0][record_indices[batch_index]])

            # print("Paried with", all_train_records[0][random_num])
            contexts.append(all_context['tokens'][random_num])
        random_context = {'tokens': torch.stack(contexts)}



        embedded_random_context = self._utterance_embedder(random_context)
        random_context_mask = get_text_field_mask(random_context)

        # Feed through RNN
        # Shape: (batch_size, num_tokens, embedding_dim)
        encoded_context = self._context_encoder(embedded_context, context_mask)

        encoded_random_context = self._context_encoder(embedded_random_context, random_context_mask)

        # Shape: (batch_size, embedding_dim)
        final_encoded_context = util.get_final_encoder_states(encoded_context,
                                                              context_mask,
                                                              bidirectional=True)
        final_encoded_random_context = util.get_final_encoder_states(encoded_random_context,
                                                                     random_context_mask,
                                                                     bidirectional=True)

        # Shape: (batch_size, num_tokens(75), embedding_dim)
        embedded_code = self._code_embedder(code)
        code_mask = get_text_field_mask(code)

        encoded_code = self._code_encoder(embedded_code, code_mask)
        final_encoded_code = util.get_final_encoder_states(encoded_code, code_mask, bidirectional=True)

        good_sim = F.cosine_similarity(final_encoded_code, final_encoded_context)
        bad_sim = F.cosine_similarity(final_encoded_code, final_encoded_random_context)

        outputs = {}
        # margin = 0.05
        # loss = (margin - good_sim + bad_sim).clamp(min=1e-6).mean()

        margin = 2.0
        loss = (margin - good_sim + bad_sim).mean()
        #

        # loss = (1.0 - good_sim).mean()
        outputs['loss'] = loss
        # print('Loss', loss)

        if not self.training:
            # print("All code size", all_code['tokens'].size())
            encoded_all_code = self.embed_encode_all_code(all_code)
            # print("encoded code size", encoded_all_code[0].size())
            # print("encoded code size", encoded_all_code[0].shape)

            normalized_context = final_encoded_context / final_encoded_context.norm(dim=-1, keepdim=True)
            normalized_context = normalized_context.data.cpu().numpy()

            use_index = True
            if use_index:
                # print('stacked code size', np.concatenate(encoded_all_code).shape)
                search_index = self._create_index(np.concatenate(encoded_all_code))

                for batch_index in range(batch_size):
                    ids, distances = search_index.knnQuery(normalized_context[batch_index],
                                                           k=8)

                    bleus = []
                    preds = []
                    target_rec = current_record[batch_index]
                    for id in ids:
                        pred = all_train_records[0][id]
                        preds.append(pred)
                        # if 'orig_code_split' in target_rec:
                        #     # For python
                        #     bleus.append(self._get_bleu(target_rec['orig_code_split'],
                        #                                 pred['orig_code_split']))
                        # else:
                        bleus.append(self._get_bleu(target_rec['code'], pred['code']))

                    # Add the best bleu to metrics
                    self._code_bleu(bleus[0])
                    self._log_predictions(rec_target=target_rec,
                                          preds=preds,
                                          bleus=bleus,
                                          distances=distances)

            else:
                batch_code_indices, batch_max_sims_per_chunk = self.search(encoded_all_code,
                                                    normalized_context)

                # print("shape", batch_max_sims_per_chunk.shape)
                # Shape: (batch_size)
                highest_sim_chunk_index = np.argmax(batch_max_sims_per_chunk, axis=1)
                # print('highest_sim_chunk_index', highest_sim_chunk_index.shape)

                # Shape: (batch_size, 1)
                batch_best_code_indices = np.take(batch_code_indices, highest_sim_chunk_index)
                print('best', batch_best_code_indices)


                for batch_index, best_index in enumerate(batch_best_code_indices.tolist()):
                    # compute bleu
                    pred_rec = all_train_records[0][best_index]

                    # the second index of 0 is since validation batch size will be 1
                    target_rec = current_record[batch_index]
                    bleu = self._get_bleu(target_rec['code'], pred_rec['code'])
                    # print('Bleu', bleu, current_record, prototype)
                    self._code_bleu(bleu)
                    self._log_predictions(rec_target=target_rec, bleu=bleu, rec_pred = pred_rec)
        else:
            # Need to clear out the cached encoded all code, since embeddings
            # have been updated!!
            # x = 4
            self._encoded_all_code = None
            self._search_index = None

        return outputs

    def _create_index(self, data):
        if self._search_index is None:
            # print("NOne index!!!")
            index = nmslib.init(method='hnsw', space='cosinesimil')
            index.addDataPointBatch(data)
            index.createIndex({'post': 2}, print_progress=True)
            self._search_index = index
        return self._search_index

    def embed_encode_all_code(self, all_code):
        if self._encoded_all_code is None:
            # all_code['tokens'].squeeze_(0)
            tokens = all_code['tokens']

            print(tokens.size())
            index_chunks = [t for t in torch.chunk(tokens, 75, dim=0)]
            # mask_chunks = [t for t in torch.chunk(self._all_mask, 100, dim=0)]

            encoded_lst = []
            print('============')
            for index_chunk in index_chunks:
                # print('index chunk', )
                dict = {'tokens': index_chunk}
                # print(index_chunk.size())
                # Shape(num_methods, num_tokens, embedding_dim)
                embedded_code = self._code_embedder(dict)  # , num_wrapping_dims=1)
                # Shape(num_methods, num_tokens)
                mask = get_text_field_mask(dict)  # , num_wrapping_dims=1)

                # for code, mask in zip(code_chunks, mask_chunks):
                encoded = self._code_encoder(embedded_code, mask)
                final_encoded = util.get_final_encoder_states(encoded,
                                                          mask,
                                                          bidirectional=True)

                # normalize
                final_encoded = final_encoded / final_encoded.norm(dim=-1, keepdim=True)
                encoded_lst.append(final_encoded.data.cpu().numpy())

            # todo(search): numpy and normalize final encoded, no need to stack??
            # self._final_encoded_lst = torch.stack(encoded_lst)
            # self._final_encoded_lst = torch.stack(encoded_lst)
            self._encoded_all_code = encoded_lst

        return self._encoded_all_code

    def get_best_sim_index_slow(self, ):
        # Deprecated:
        # Write this all to a file or store in memory
        best_sim = -1
        best_index = -1
        for i in range(encoded_all_code.size(0)):
            sim = F.cosine_similarity(final_encoded_context, encoded_all_code[i].unsqueeze(0))
            sim_cpu = sim.data.cpu().numpy().tolist()[0]
            if sim_cpu > best_sim:
                best_index = i
                best_sim = sim_cpu
        return best_index

    def search(self, all_code_vec_chunks, context_vec, n_results=2):
        max_indices = []
        sims = []
        threads = []
        outputs = []
        for i, codevecs_chunk in enumerate(all_code_vec_chunks):
            t = threading.Thread(target=self.search_thread,
                                 args=(outputs, context_vec, codevecs_chunk, n_results))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:  # wait until all sub-threads finish
            t.join()



        for d in outputs:
            max_indices.append(d['code_indices'])
            sims.append(d['max_sims'])

        # (batch_size, chunk)
        max_indices_array = np.array(max_indices).transpose()
        sims_array = np.array(sims).transpose()

        return max_indices, sims_array

    def search_thread(self, outputs, context_vec, code_vecs, n_results):
        # 1. compute code similarities
        # Shape (batch_size, num_codes_in_chunk)
        chunk_sims = self.dot_np(context_vec, code_vecs)
        # print("Search Thread Called")
        # print('context', context_vec.shape)
        # print('code', code_vecs.shape)
        # print('sim', chunk_sims.shape)

        # # 2. choose the top K results
        # negsims = np.negative(chunk_sims[0])
        # maxinds = np.argpartition(negsims, kth=n_results - 1)
        # maxinds = maxinds[:n_results]
        #
        # print('max inds', maxinds)
        # chunk_sims = chunk_sims[0][maxinds]
        # max_indices.extend(maxinds)
        # sims.extend(chunk_sims)

        # Shape (batch_size)
        top_indices = np.argmax(chunk_sims, axis=1)
        # Shape (batch_size)
        max_sims = np.take(chunk_sims, top_indices)

        # print('top indices', top_indices.shape)

        outputs.append({'code_indices': top_indices,
                        'max_sims': max_sims})

        # max_indices.extend(top_indices)
        # sims.extend(max_sims)

    # def cos_np(data1, data2):
    #     """numpy implementation of cosine similarity for matrix"""
    #     dotted = np.dot(data1, np.transpose(data2))
    #     norm1 = np.linalg.norm(data1, axis=1)
    #     norm2 = np.linalg.norm(data2, axis=1)
    #     matrix_vector_norms = np.multiply(norm1, norm2)
    #     neighbors = np.divide(dotted, matrix_vector_norms)
    #     return neighbors

    @staticmethod
    def normalize(data):
        """normalize datarix by rows"""
        normalized_data = data / np.linalg.norm(data, axis=1).reshape((data.shape[0], 1))
        return normalized_data

    @staticmethod
    def dot_np(data1, data2):
        """cosine similarity for normalized vectors"""
        return np.dot(data1, np.transpose(data2))

    def _log_predictions(self, rec_target, preds, bleus, distances=None):
        log = '==============' * 4 + '\n'
        log += 'Target========\n'
        log += rec_target['path'] + '\n'
        log += 'NL:' + ' '.join(rec_target['nl']) + '\n'
        log += 'methodName:' + (rec_target['method_name']) + '\n'
        if 'orig_code' in rec_target:
            # For the python dataset, code is indented already
            log += rec_target['orig_code'] + '\n'
        else:
            log += self.indent(rec_target['code']) + '\n'

        for i, pred in enumerate(preds):
            log += 'Pred' + str(i) + '========\n'
            log += 'Sim ' + str(distances[i]) + '\n'
            # log += pred['path'] + '\n'
            log += 'NL:' + ' '.join(pred['nl']) + '\n'
            log += 'methodName:' + (pred['method_name']) + '\n'
            if 'orig_code' in pred:
                # For the python dataset, code is indented already
                log += pred['orig_code'] + '\n'
            else:
                log += self.indent(pred['code']) + '\n'
            log += "Bleu: " + str(bleus[i]) + '\n'
            log += '\n'

        print(log)

        all_preds = open(
            os.path.join(self._serialization_dir, 'epoch%d-%s-all_preds.txt' % (self._epoch_num, "valid")), 'a')
        high_bleu = open(
            os.path.join(self._serialization_dir, 'epoch%d-%s-bleu-high.txt' % (self._epoch_num, "valid")), 'a')

        all_preds.write(log)
        if bleus[0] > .2:
            high_bleu.write(log)

        high_bleu.close()
        all_preds.close()
        print(log)



    @staticmethod
    def _get_bleu(gold_seq, pred_seq):
        # This is how Ling et al. compute bleu score.
        sm = SmoothingFunction()
        ngram_weights = [0.25] * min(4, len(gold_seq))
        return sentence_bleu([gold_seq], pred_seq, weights=ngram_weights, smoothing_function=sm.method3)

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

    def combine_name_types(self, names, types):
        combine_str = ""
        for n, t in zip(names, types):
            combine_str += n + ' (' + t + ')\n'
        return combine_str

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'bleu': self._code_bleu.get_metric(reset),
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
        code_embedder_params = params.pop("code_embedder")
        code_embedder = TextFieldEmbedder.from_params(vocab, code_embedder_params)

        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        code_encoder = Seq2SeqEncoder.from_params(params.pop("code_encoder"))
        # proto_encoder = Seq2SeqEncoder.from_params(params.pop("proto_encoder"))
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
                   code_embedder=code_embedder,
                   context_encoder=encoder,
                   code_encoder=code_encoder,
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