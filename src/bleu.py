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
    from utils import load
    proc = (lambda s: s.lower().split()) if args.ignore_case else str.split
    gold = load(args.gold, proc= proc)
    for pred in args.pred:
        print(pred, "{:.4f}".format(bleu(gold, load(pred, proc= proc))))
