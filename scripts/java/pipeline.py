""" This performs the filters from filter-by-num-javadocs.py and pairs with prototypes from pair_with_clossest_comments"""


import json

from collections import defaultdict, Counter

import time

import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.metrics import jaccard_similarity_score
from nltk.corpus import stopwords
from datasketch import MinHashLSHForest, MinHash
import random
STOPS = set(stopwords.words("english"))

def combine_name_types(names, types):
    combine_str = ""
    for n, t in zip(names, types):
        combine_str += n + ' (' + t + ')\n'
    return combine_str
def indent(code):
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
    return pretty
def class_string(methods):
    rec = methods[0]
    log = '-' * 100 + '\n'
    log += "ClassName " + rec['className']
    log += 'Variables:\n'
    log += combine_name_types(rec['varNames'], rec['varTypes'])
    log += 'Methods:\n'
    log += combine_name_types(rec['methodNames'], rec['methodReturns'])
    for m in methods:
        log += 'NL:' + ' '.join(m['nl']) + '\n'
        if 'methodName' in m:
            log += "Method: " + m['methodName'] + '\n'
        log += indent(m['code'])
    return log

def get_bleu(gold_seq, pred_seq):
    # This is how Ling et al. compute bleu score.
    sm = SmoothingFunction()
    ngram_weights = [0.25] * min(4, len(gold_seq))
    return sentence_bleu([gold_seq], pred_seq, weights=ngram_weights, smoothing_function=sm.method3)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print(
                'method time %r  %2.2f ms' % \
                (method.__name__, (te - ts) * 1000))
        return result

    return timed
def split_camel_case(name: str):
    # Returns the string and its camel case split version all lowercased.
    tokens = re.sub('(?!^)([A-Z][a-z]*)', r' \1', name).split()
    # if len(tokens) > 1:
    #     tokens = [name] + tokens
    tokens = [t.lower() for t in tokens]
    return tokens
def preprocess(tok):
    tok = tok.lower()
    return tok
def trunc(seq, length):
    if len(seq) >= length:
        return seq[:length]
    else:
        return seq + [""] * (length-len(seq))
def get_nl_tokens(rec, nl_len):
    """
    Returns the camel case split method name concatenated with the
    truncated nl all lowercased. this is input to jaccard function.
    method names also used since just comments not good enough:

    e.g.
    tgt:returns directionA
    proto1: returns directionB
    proto2: returns date
    Both proto1 and proto2 have same jaccard distance, this tie is broken
    towards the correct proto1 by using method names as well.
    """
    nl_split = [t2 for t in rec['nl'] for t2 in split_camel_case(t)]
    tokens = split_camel_case(rec['methodName']) + nl_split #rec['nl']

    tokens = [t for t in tokens if t not in STOPS]

    temp = [preprocess(tok) for tok in trunc(tokens, nl_len)]

    return temp
    # return trunc(tokens)

def print_len(dataset):
    lens =[len(methods) for _, methods in dataset.items()]
    print(sum(lens))
def flatten(path2methods):
    return [m for _, methods in path2methods.items() for m in methods]
def limit(path2methods, num_recs):
    newpath2methods = {}
    count = 0
    for k, methods in path2methods.items():
        newpath2methods[k] = methods
        count += len(methods)
        if count > num_recs:
            break
    return newpath2methods


@timeit
def get_dataset_class2methods(filename):
    with open(filename) as file:
        dataset = json.load(file)
        print(filename)
        print('num recs', len(dataset))
        class2methods = defaultdict(list)
        for rec in dataset:
            class2methods[rec['path']].append(rec)
        return class2methods

@timeit
def filter_by_javadoc_count(class2methods, count=2):
    return {k: methods for k, methods in class2methods.items() if len(methods) > count}

@timeit
def filter_duplicates(class2methods):
    """ Filters out a method if it occurs in training."""
    all_methods = set()

    num_duplicates = 0
    for key, methods in class2methods.items():
        new_methods = []
        for m in methods:
            m_str = ' '.join([m['methodName']] + m['code'])
            if m_str not in all_methods:
                all_methods.add(m_str)
                new_methods.append(m)
            else:
                num_duplicates += 1
        class2methods[key] = new_methods
    print("Number duplicates", num_duplicates)
    return class2methods

@timeit
def filter_code_length(class2methods):
    """ Min len 11 filters getters, 14 does setters. The 100 is arbitrary."""
    for key in class2methods.keys():
        new_methods = []
        for m in class2methods[key]:
            if 14 < len(m['code']) and len(m['code']) < 120:
                new_methods.append(m)
        class2methods[key] = new_methods
    return class2methods

@timeit
def filter_out_tests(class2methods):
    return {k: methods for k, methods in class2methods.items() if 'test' not in methods[0]['className'].lower()}

def get_filtered_dataset(dataset_filename):

    dataset = get_dataset_class2methods(dataset_filename)
    print_len(dataset)

    dataset = filter_out_tests(dataset)
    print_len(dataset)

    dataset = filter_code_length(dataset)
    print_len(dataset)

    dataset = filter_duplicates(dataset)
    print_len(dataset)

    # # This has to come last, to ensure at least 2
    # # methods present for prototypes within class
    dataset = filter_by_javadoc_count(dataset, 1)
    print_len(dataset)


    # with open(filtered_filename, 'w') as file:
    #     json.dump(dataset, file)
    print(len(flatten(dataset)), "Num records after filtering")

    return dataset


def add_prototype_in_rec(rec, proto_rec):
    rec['prototype_nl'] = proto_rec['nl']
    rec['prototype_code'] = proto_rec['code']
    rec['prototype_methodName'] = proto_rec['methodName']
    rec['prototype_rules'] = proto_rec['rules']
    rec['prototype_path'] = proto_rec['path']

def pair_oracle_best_train_bleu_lsh(dataset,
                                    proto_cand_dataset=None,
                                    num_candidates=200):
    """Prototype is the one maximizes bleu. All records
    from the dataset are valid candidates if
    within_class= False. If it's true then candidates
    are from the same class."""
    if proto_cand_dataset == None:
        proto_cand_dataset = dataset

    forest = MinHashLSHForest(num_perm=256)

    for i, method in enumerate(tqdm(proto_cand_dataset)):
        m1 = MinHash(num_perm=256)
        for d in method['code']:
            m1.update(d.encode('utf8'))
        forest.add(str(i), m1)
    forest.index()

    total_bleu = 0.0
    for i, method in enumerate(tqdm(dataset)):
        m1 = MinHash(num_perm=256)
        for d in method['code']:
            m1.update(d.encode('utf8'))
        result = forest.query(m1, num_candidates)

        best_index = i
        best_bleu = 0.0
        for index_str in result:
            index = int(index_str)
            if index != i:
                bleu = get_bleu(gold_seq=method['code'],
                               pred_seq=proto_cand_dataset[index]['code'])
                if bleu > best_bleu:
                    best_bleu = bleu
                    best_index = index


        if best_index == i:
            # print(i, result)
            best_index = random.randint(0, len(dataset))
            # print(method['nl'])
            # print(method['code'])

        method2 = proto_cand_dataset[best_index]
        total_bleu += get_bleu(gold_seq=method['code'],
                               pred_seq=method2['code'])
        add_prototype_in_rec(rec=method, proto_rec=method2)
    print(total_bleu / len(dataset), "bleu Oracle, from train")

def pair_oracle_best_bleu_in_class(dataset):
    """Prototype is the one maximizes bleu. All records
    from the dataset are valid candidates if
    within_class= False. If it's true then candidates
    are from the same class."""
    total_bleu = 0.0

    for _, methods in tqdm(dataset.items()):
        for i, m in enumerate(methods):
            max_bleu = 0.0
            max_rec = {}
            for j, m2 in enumerate(methods):
                if i == j:
                    continue

                bleu = get_bleu(gold_seq=m['code'],
                                pred_seq=m2['code'])
                if bleu > max_bleu:
                    max_bleu = bleu
                    max_rec = m2

            total_bleu += max_bleu
            add_prototype_in_rec(rec=m, proto_rec=max_rec)
    print(total_bleu / len(flatten(dataset)), "bleu Oracle, within class")

def pair_jaccard_nl_from_train_lsh(dataset, proto_cand_dataset=None, nl_len=25):
    """ proto_cand_dataset is used to pair valid/test
    with prototypes from train."""
    if proto_cand_dataset == None:
        proto_cand_dataset = dataset

    forest = MinHashLSHForest(num_perm=256)

    nl2index = defaultdict(list)
    for i, method in enumerate(tqdm(proto_cand_dataset)):
        m1 = MinHash(num_perm=256)
        nl = get_nl_tokens(method, nl_len)
        nl2index[''.join(nl)].append(i)
        for d in nl:
            m1.update(d.encode('utf8'))
        # forest.add((method['methodName'] +
        #             ' '.join(method['code'])), m1)
        forest.add(str(i), m1)
    forest.index()

    total_bleu = 0.0
    for i, method in enumerate(tqdm(dataset)):
        m1 = MinHash(num_perm=256)
        nl = get_nl_tokens(method, nl_len)
        for d in nl:
            m1.update(d.encode('utf8'))
        result = forest.query(m1, 50)

        best_index = i
        best_jaccard = 0.0
        for index_str in result:
            index = int(index_str)
            if index != i:
                tgt = get_nl_tokens(proto_cand_dataset[index], nl_len)
                jaccard = jaccard_similarity_score(nl, tgt)
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_index = index

        if best_index == i:
            # print(i, result)
            best_index = random.randint(0, len(dataset))
            # print(method['nl'])
            # print(method['code'])

        method2 = proto_cand_dataset[best_index]
        total_bleu += get_bleu(gold_seq=method['code'],
                               pred_seq=method2['code'])
        add_prototype_in_rec(rec=method, proto_rec=method2)
    print(total_bleu / len(dataset), "bleu NL jaccard, from train")


def pair_jaccard_nl_within_class(path2methods, nl_len=25):
    total_bleu = 0.0

    for _, methods in tqdm(path2methods.items()):
        for i, m in enumerate(methods):
            max_jaccard = -1
            max_rec = {}
            src = get_nl_tokens(m, nl_len)
            for j, m2 in enumerate(methods):
                if i == j:
                    continue
                tgt = get_nl_tokens(m2, nl_len)
                score = jaccard_similarity_score(src, tgt)

                if score > max_jaccard:
                    max_jaccard = score
                    max_rec = m2


            total_bleu += get_bleu(gold_seq=m['code'],
                                   pred_seq=max_rec['code'])
            add_prototype_in_rec(rec=m, proto_rec=max_rec)

    print(total_bleu / len(flatten(path2methods)), "bleu NL jaccard, within class")


def dump(dataset, outdir, filename, num):
    outfile = os.path.join(outdir, filename)
    with open(outfile, 'w') as file:
        json.dump(dataset[:num], file)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--indir", dest="indir")
    parser.add_argument("--outdir", dest="outdir")
    args = parser.parse_args()

    dataset_lens = {'train': 100000,
                     'valid': 2000,
                     # "test": 2000
                    }
    datasets = {}
    for dt, num_recs in dataset_lens.items():
        print(dt, '='*30)
        dataset_filename = os.path.join(args.indir,"%s.dataset" % (dt))

        dataset = get_filtered_dataset(dataset_filename)
        datasets[dt] = limit(dataset, num_recs)

    # dump(flatten(datasets['train']), args.outdir, 'train-small.dataset', 8000)

    for dt in dataset_lens.keys():
        print(dt, '=' * 30)
        num_recs = dataset_lens[dt]
        path2methods = datasets[dt]
        dataset_lst = flatten(path2methods)


        pair_jaccard_nl_from_train_lsh(dataset_lst,
                         proto_cand_dataset=flatten(datasets['train']),
                          nl_len=25)
        dump(dataset_lst, args.outdir,
                 "%s-filtered-proto-nljaccard-all.dataset" % (dt),
                 num_recs)

        pair_jaccard_nl_within_class(path2methods, nl_len=25)
        dump(dataset_lst, args.outdir,
             "%s-filtered-proto-nljaccard-class.dataset" % (dt),
             num_recs)

        if dt == 'valid':
            num_candidates = 40000
        elif dt == 'train':
            num_candidates = 2000
        pair_oracle_best_train_bleu_lsh(dataset_lst,
                                    proto_cand_dataset=flatten(datasets['train']),
                                        num_candidates=num_candidates)
        dump(dataset_lst, args.outdir,
             "%s-filtered-proto-oraclebleu-all.dataset" % (dt),
             num_recs)

        pair_oracle_best_bleu_in_class(path2methods)
        dump(dataset_lst, args.outdir,
             "%s-filtered-proto-oraclebleu-class.dataset" % (dt),
             num_recs)





# TODO(VERIFY) that no similarity between train and valid