#!/usr/bin/env python3


def bleu(gold, pred):
    """-> float in [0, 1]; gold, pred : seq (sent : seq (word : str))"""
    from nltk.translate.bleu_score import corpus_bleu
    return corpus_bleu([[g] for g in gold], pred)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description= "corpus-level BLEU score.")
    parser.add_argument('gold', help= "file for the gold-standard sentences")
    parser.add_argument('pred', help= "files for the predicted sentences", nargs= '+')
    parser.add_argument('--ignore-case', action= 'store_true', help= "case insensitive")
    return parser.parse_args()


if '__main__' == __name__:
    args = parse_args()
    from util import comp, partial
    from util_io import load
    proc = str.split
    if args.ignore_case: proc = comp(proc, str.lower)
    load_corpus = comp(list, partial(map, proc), load)
    gold = load_corpus(args.gold)
    for pred in args.pred:
        print(pred, "{:.4f}".format(bleu(gold, load_corpus(pred))))
