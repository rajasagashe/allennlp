"""
Takes the same example from two different datasets
and prints the different prototypes from both datasets.
The hope is to discover why the bleu is so much higher for
the oracle datasets.
"""

from argparse import ArgumentParser
import json
import random
from collections import defaultdict

import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def get_dataset_class2methods(filename):
    with open(filename) as file:
        dataset = json.load(file)
        print(filename)
        # print('num recs', len(dataset))
        class2methods = defaultdict(list)
        for rec in dataset:
            class2methods[rec['path']].append(rec)
        return class2methods

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

def class_string(rec):
    log = '-' * 20 + '\n'
    log += "ClassName " + rec['className'] + '\n'
    log += rec['path'] + '\n'
    return log

def prototype_str(rec):
    log = ""
    log += 'ProPath:' + rec['prototype_path'] + '\n'
    log += 'ProMethod:' + rec['prototype_methodName'] + '\n'
    log += 'ProNL:' + ' '.join(rec['prototype_nl']) + '\n'
    log += indent(rec['prototype_code'])
    return log

def target_str(rec):
    log = ""
    log += 'TgtMethod:' + rec['methodName'] + '\n'
    log += 'TgtNL:' + ' '.join(rec['nl']) + '\n'
    log += indent(rec['code']) + '\n'
    return log
def get_bleu(gold_seq, pred_seq):
    # This is how Ling et al. compute bleu score.
    sm = SmoothingFunction()
    ngram_weights = [0.25] * min(4, len(gold_seq))
    return sentence_bleu([gold_seq], pred_seq, weights=ngram_weights, smoothing_function=sm.method3)
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset1", dest="dataset1")
    parser.add_argument("--dataset2", dest="dataset2")
    args = parser.parse_args()


    dataset1 = get_dataset_class2methods(args.dataset1)
    dataset2 = get_dataset_class2methods(args.dataset2)

    classes = random.sample(list(dataset1), 100)
    for c in classes:
        rec1 = dataset1[c][0]
        # get the same record from other dataset
        rec2 = None
        for r in dataset2[c]:
            if (r['methodName'] == rec1['methodName'] and
                r['code'] == rec1['code']):
                rec2 = r
                break
        if rec1['prototype_code'] == rec2['prototype_code']:
            continue

        bleu1 = get_bleu(rec1['code'], rec1['prototype_code'])
        bleu2 = get_bleu(rec2['code'], rec2['prototype_code'])

        if abs(bleu2-bleu1) < .05:
            continue

        print('===========================' * 2)
        print(bleu1, bleu2)
        # pretty print both prototypes and code and NL
        print(class_string(rec1))
        print(args.dataset1)
        print(prototype_str(rec1))
        print(args.dataset2)
        print(prototype_str(rec2))
        print(target_str(rec1))