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
import scipy
import sklearn

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
                 # code_embedder: TextFieldEmbedder,
                 context_encoder: Seq2SeqEncoder,
                 # code_encoder: Seq2SeqEncoder,
                 serialization_dir: str = None) -> None:
        super(JavaSearchModel, self).__init__(vocab)
        self._utterance_embedder = utterance_embedder
        # self._code_embedder = code_embedder
        # self._code_embedder = utterance_embedder

        self._context_encoder = context_encoder
        # self._code_encoder = code_encoder
        # todo(pr): later perhaps add a LSTM
        # self._identifier_encoder = TimeDistributed(BagOfEmbeddingsEncoder(embedding_dim,
        #                                                                   averaged=True))

        # self._boe = BagOfEmbeddingsEncoder(embedding_dim, averaged=True)
        self._code_bleu = Average()
        self._encoded_all_context = None
        self._search_index = None
        self._serialization_dir = serialization_dir


    @overrides
    # @timeit
    def forward(self,  # type: ignore
                current_context: Dict[str, torch.LongTensor],
                prototype_context: Dict[str, torch.LongTensor],
                record_indices: List[int],
                current_record: List[Dict[str, str]],
                all_context: Dict[str, torch.LongTensor],
                all_train_records: List[Dict[str, str]],

                random_context: Dict[str, torch.LongTensor] = None,
                random_bleu:torch.LongTensor = None,
                prototype_bleu:torch.LongTensor = None,

                ) -> Dict[str, torch.Tensor]:

        embedded_context = self._utterance_embedder(current_context)
        context_mask = get_text_field_mask(current_context)
        batch_size = embedded_context.size(0)
        num_prototypes = 3

        # Shape: (batch_size, num_prototypes=3, num_tokens, embedding_dim)
        embedded_prototype_context = self._utterance_embedder(prototype_context, num_wrapping_dims=1)
        prototype_context_mask = get_text_field_mask(prototype_context, num_wrapping_dims=1)

        if random_context is None:
            contexts = []
            for batch_index in range(batch_size):
                temp_list = []
                for i in range(num_prototypes):
                    random_num = random.randint(0, all_context['tokens'].size(0)-1)
                    while random_num == record_indices[batch_index]:
                        random_num = random.randint(0, all_context['tokens'].size(0)-1)
                    temp_list.append(all_context['tokens'][random_num])
                contexts.append(torch.stack(temp_list))

            random_context = {'tokens': torch.stack(contexts)}

        embedded_random_context = self._utterance_embedder(random_context, num_wrapping_dims=1)
        random_context_mask = get_text_field_mask(random_context, num_wrapping_dims=1)


        # Shape: (batch_size, num_tokens, embedding_dim)
        encoded_context = self._context_encoder(embedded_context, context_mask)

        proto_encoder = TimeDistributed(self._context_encoder)
        # Shape: (batch_size, num_prototypes=3, num_tokens, embedding_dim)
        encoded_prototype_context = proto_encoder(embedded_prototype_context, prototype_context_mask)
        encoded_random_context = proto_encoder(embedded_random_context, random_context_mask)

        # Shape: (batch_size, 1, embedding_dim)
        final_encoded_context = util.get_final_encoder_states(encoded_context,
                                                              context_mask,
                                                              bidirectional=True)
        final_encoded_context.unsqueeze_(1)

        proto_seq_len = encoded_prototype_context.size(2)
        rand_seq_len = encoded_random_context.size(2)
        final_dim = encoded_prototype_context.size(-1)
        # Shape: (batch_size * num_prototypes=3, embedding_dim)
        final_encoded_prototype_context = util.get_final_encoder_states(
            encoded_prototype_context.view(batch_size*num_prototypes, proto_seq_len, final_dim),
            prototype_context_mask.view(batch_size*num_prototypes, proto_seq_len),
            True)
        final_encoded_random_context = util.get_final_encoder_states(
            encoded_random_context.view(batch_size*num_prototypes, rand_seq_len, final_dim),
            random_context_mask.view(batch_size*num_prototypes, rand_seq_len),
            True)

        # Shape: (batch_size, num_prototypes=3, embedding_dim)
        final_encoded_prototype_context = final_encoded_prototype_context.view(batch_size, num_prototypes, final_dim)
        final_encoded_random_context = final_encoded_random_context.view(batch_size, num_prototypes, final_dim)

        # # Hyperbolic tangent
        # final_encoded_context = F.tanh(final_encoded_context)
        # final_encoded_prototype_context = F.tanh(final_encoded_prototype_context)
        # final_encoded_random_context = F.tanh(final_encoded_random_context)

        # Shape: (batch_size, num_prototypes=3)
        proto_sim = F.cosine_similarity(final_encoded_context, final_encoded_prototype_context, dim=2)
        random_sim = F.cosine_similarity(final_encoded_context, final_encoded_random_context, dim=2)

        outputs = {}
        if random_bleu is None:
            good_dist = 1-proto_sim
            bad_dist = 1-random_sim
            # margin = 0.05
            # loss = (margin - proto_sim + random_sim).clamp(min=1e-6).mean()
            # margin = 2.0
            # loss = (margin - proto_sim + random_sim).mean()
            loss = (good_dist + (2-bad_dist)).clamp(min=1e-6).mean()
        else:
            # Map the bleus and similarities into the [0,2] range.
            proto_sim_shifted = proto_sim + 1
            random_sim_shifted = random_sim + 1
            # print('=================')
            # print(prototype_bleu)
            # print(random_bleu)
            prototype_bleu = prototype_bleu * 2
            random_bleu = random_bleu * 2

            # print(proto_sim_shifted)
            good_diff = (proto_sim_shifted - prototype_bleu) ** 2
            bad_diff = (random_sim_shifted - random_bleu) ** 2
            # print(good_diff)
            loss = (good_diff + bad_diff).clamp(min=1e-6).mean()

        outputs['loss'] = loss

        if not self.training:
            encoded_all_context_cpu = self.embed_encode_all_code(all_context)

            # normalized_context = final_encoded_context / final_encoded_context.norm(dim=-1, keepdim=True)
            final_encoded_context = final_encoded_context.squeeze(1)
            final_encoded_context_cpu = final_encoded_context.data.cpu().numpy()

            use_index = True
            if use_index:
                print('Encoded All Context', encoded_all_context_cpu.shape)
                search_index = self._create_index(encoded_all_context_cpu)

                for batch_index in range(batch_size):
                    ids, distances = search_index.knnQuery(final_encoded_context_cpu[batch_index],
                                                           k=3)

                    bleus = []
                    preds = []
                    target_rec = current_record[batch_index]
                    for id in ids:
                        # # To verify the cosine similarities
                        # cos = scipy.spatial.distance.cosine(final_encoded_context_cpu[batch_index],
                        #                                     encoded_all_context_cpu[id])
                        #
                        # print('Cosine dist verified per pred', cos)
                        # cossim = sklearn.metrics.pairwise.cosine_similarity(                                                       np.expand_dims(final_encoded_context_cpu[batch_index], axis=0),
                        #         np.expand_dims(encoded_all_context_cpu[id], axis=0)
                        # )
                        #
                        # print('Cosine sim verified per pred', cossim)
                        # normed1 = encoded_all_context_cpu[id] /np.linalg.norm(encoded_all_context_cpu[id])
                        # normed2 = final_encoded_context_cpu[batch_index] /np.linalg.norm(final_encoded_context_cpu[batch_index])
                        # print(np.linalg.norm(encoded_all_context_cpu[id]))
                        # print(np.linalg.norm(final_encoded_context_cpu[batch_index]))
                        # # print('normed1', normed1)
                        # # print('normed2', normed2)
                        # print('doted', np.dot(normed1, normed2))

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
                    # Cosine distance to best prototype
                    model_dist = 1-proto_sim[batch_index].data.cpu().numpy().tolist()[0]
                    self._log_predictions(rec_target=target_rec,
                                          preds=preds,
                                          bleus=bleus,
                                          model_dist=model_dist,
                                          distances=distances)


            # else:
            #     batch_code_indices, batch_max_sims_per_chunk = self.search(encoded_all_code,
            #                                         normalized_context)
            #
            #     # print("shape", batch_max_sims_per_chunk.shape)
            #     # Shape: (batch_size)
            #     highest_sim_chunk_index = np.argmax(batch_max_sims_per_chunk, axis=1)
            #     # print('highest_sim_chunk_index', highest_sim_chunk_index.shape)
            #
            #     # Shape: (batch_size, 1)
            #     batch_best_code_indices = np.take(batch_code_indices, highest_sim_chunk_index)
            #     print('best', batch_best_code_indices)
            #
            #
            #     for batch_index, best_index in enumerate(batch_best_code_indices.tolist()):
            #         # compute bleu
            #         pred_rec = all_train_records[0][best_index]
            #
            #         # the second index of 0 is since validation batch size will be 1
            #         target_rec = current_record[batch_index]
            #         bleu = self._get_bleu(target_rec['code'], pred_rec['code'])
            #         # print('Bleu', bleu, current_record, prototype)
            #         self._code_bleu(bleu)
            #         self._log_predictions(rec_target=target_rec, bleu=bleu, rec_pred = pred_rec)
        else:
            # Need to clear out the cached encoded all code, since embeddings
            # have been updated!!
            # x = 4
            self._encoded_all_context = None
            self._search_index = None

        return outputs

    def _create_index(self, data):
        if self._search_index is None:
            # print("NOne index!!!")
            index = nmslib.init(method='hnsw', space='cosinesimil')
            index.addDataPointBatch(data)
            index.createIndex({'post': 2})#, print_progress=True)
            self._search_index = index
        return self._search_index

    def embed_encode_all_code(self, all_context):
        if self._encoded_all_context is None:
            # all_context['tokens'].squeeze_(0)
            tokens = all_context['tokens']

            index_chunks = [t for t in torch.chunk(tokens, 25, dim=0)]

            encoded_lst = []
            for index_chunk in index_chunks:
                dict = {'tokens': index_chunk}
                # Shape(num_contexts_chunk, num_tokens, embedding_dim)
                embedded_context = self._utterance_embedder(dict)
                # Shape(num_contexts_chunk, num_tokens)
                mask = get_text_field_mask(dict)

                encoded = self._context_encoder(embedded_context, mask)
                final_encoded = util.get_final_encoder_states(encoded,
                                                          mask,
                                                          bidirectional=True)

                # Don't normalize since using nms Lib!!! Do this for parallelization code
                # normalize in advance to prepare for cosine similarity
                # final_encoded = final_encoded / final_encoded.norm(dim=-1, keepdim=True)
                encoded_lst.append(final_encoded.data.cpu().numpy())

            self._encoded_all_context = np.concatenate(encoded_lst)

        return self._encoded_all_context

    # def get_best_sim_index_slow(self, ):
    #     # Deprecated:
    #     # Write this all to a file or store in memory
    #     best_sim = -1
    #     best_index = -1
    #     for i in range(encoded_all_code.size(0)):
    #         sim = F.cosine_similarity(final_encoded_context, encoded_all_code[i].unsqueeze(0))
    #         sim_cpu = sim.data.cpu().numpy().tolist()[0]
    #         if sim_cpu > best_sim:
    #             best_index = i
    #             best_sim = sim_cpu
    #     return best_index
    #
    # def search(self, all_code_vec_chunks, context_vec, n_results=2):
    #     max_indices = []
    #     sims = []
    #     threads = []
    #     outputs = []
    #     for i, codevecs_chunk in enumerate(all_code_vec_chunks):
    #         t = threading.Thread(target=self.search_thread,
    #                              args=(outputs, context_vec, codevecs_chunk, n_results))
    #         threads.append(t)
    #     for t in threads:
    #         t.start()
    #     for t in threads:  # wait until all sub-threads finish
    #         t.join()
    #
    #
    #
    #     for d in outputs:
    #         max_indices.append(d['code_indices'])
    #         sims.append(d['max_sims'])
    #
    #     # (batch_size, chunk)
    #     max_indices_array = np.array(max_indices).transpose()
    #     sims_array = np.array(sims).transpose()
    #
    #     return max_indices, sims_array
    #
    # def search_thread(self, outputs, context_vec, code_vecs, n_results):
    #     # 1. compute code similarities
    #     # Shape (batch_size, num_codes_in_chunk)
    #     chunk_sims = self.dot_np(context_vec, code_vecs)
    #     # print("Search Thread Called")
    #     # print('context', context_vec.shape)
    #     # print('code', code_vecs.shape)
    #     # print('sim', chunk_sims.shape)
    #
    #     # # 2. choose the top K results
    #     # negsims = np.negative(chunk_sims[0])
    #     # maxinds = np.argpartition(negsims, kth=n_results - 1)
    #     # maxinds = maxinds[:n_results]
    #     #
    #     # print('max inds', maxinds)
    #     # chunk_sims = chunk_sims[0][maxinds]
    #     # max_indices.extend(maxinds)
    #     # sims.extend(chunk_sims)
    #
    #     # Shape (batch_size)
    #     top_indices = np.argmax(chunk_sims, axis=1)
    #     # Shape (batch_size)
    #     max_sims = np.take(chunk_sims, top_indices)
    #
    #     # print('top indices', top_indices.shape)
    #
    #     outputs.append({'code_indices': top_indices,
    #                     'max_sims': max_sims})
    #
    #     # max_indices.extend(top_indices)
    #     # sims.extend(max_sims)
    #
    # # def cos_np(data1, data2):
    # #     """numpy implementation of cosine similarity for matrix"""
    # #     dotted = np.dot(data1, np.transpose(data2))
    # #     norm1 = np.linalg.norm(data1, axis=1)
    # #     norm2 = np.linalg.norm(data2, axis=1)
    # #     matrix_vector_norms = np.multiply(norm1, norm2)
    # #     neighbors = np.divide(dotted, matrix_vector_norms)
    # #     return neighbors
    #
    # @staticmethod
    # def normalize(data):
    #     """normalize datarix by rows"""
    #     normalized_data = data / np.linalg.norm(data, axis=1).reshape((data.shape[0], 1))
    #     return normalized_data
    #
    # @staticmethod
    # def dot_np(data1, data2):
    #     """cosine similarity for normalized vectors"""
    #     return np.dot(data1, np.transpose(data2))

    def _log_predictions(self, rec_target, preds, bleus, model_dist, distances):
        log = '==============' * 4 + '\n'
        log += 'Target Code========\n'
        # log += rec_target['path'] + '\n'
        log += 'NL:' + ' '.join(rec_target['nl']) + '\n'
        # log += 'methodName:' + (rec_target['method_name']) + '\n'
        if 'orig_code' in rec_target:
            # For the python dataset, code is indented already
            log += rec_target['orig_code'] + '\n'
        else:
            log += self.indent(rec_target['code']) + '\n'

        log += 'Best Prototype Target========\n'
        # log += rec_target['path'] + '\n'
        log += 'Sim ' + str(model_dist) + '\n'
        log += 'NL:' + ' '.join(rec_target['prototype_nl0']) + '\n'
        log += self.indent(rec_target['prototype_code0']) + '\n'

        for i, pred in enumerate(preds):
            log += 'Pred' + str(i) + '========\n'
            log += 'Sim ' + str(distances[i]) + '\n'
            # log += pred['path'] + '\n'
            log += 'NL:' + ' '.join(pred['nl']) + '\n'
            # log += 'methodName:' + (pred['method_name']) + '\n'
            if 'orig_code' in pred:
                # For the python dataset, code is indented already
                log += pred['orig_code'] + '\n'
            else:
                log += self.indent(pred['code']) + '\n'
            log += "Bleu: " + str(bleus[i]) + '\n'
            log += '\n'

        print(log)

        if hasattr(self, '_epoch_num'):
            # If you run allennlp in evaluate mode it won't have this.
            all_preds = open(
                os.path.join(self._serialization_dir, 'epoch%d-%s-all_preds.txt' % (self._epoch_num, "valid")), 'a')
            high_bleu = open(
                os.path.join(self._serialization_dir, 'epoch%d-%s-bleu-high.txt' % (self._epoch_num, "valid")), 'a')

            all_preds.write(log)
            if bleus[0] > .2:
                high_bleu.write(log)

            high_bleu.close()
            all_preds.close()


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

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'SimpleSeq2Seq':
        utterance_embedder_params = params.pop("utterance_embedder")
        utterance_embedder = TextFieldEmbedder.from_params(vocab, utterance_embedder_params)
        # code_embedder_params = params.pop("code_embedder")
        # code_embedder = TextFieldEmbedder.from_params(vocab, code_embedder_params)

        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        # code_encoder = Seq2SeqEncoder.from_params(params.pop("code_encoder"))


        serialization_dir = params.pop('serialization_dir', None)
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   utterance_embedder=utterance_embedder,
                   # code_embedder=code_embedder,
                   context_encoder=encoder,
                   # code_encoder=code_encoder,
                   serialization_dir=serialization_dir)