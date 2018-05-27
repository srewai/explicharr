#!/usr/bin/env python3


def load(filename, tokenize= str.split):
    with open(filename) as file:
        return list(map(tokenize, file))


def bleu(gold, pred):
    from nltk.translate.bleu_score import corpus_bleu
    return corpus_bleu([[g] for g in gold], pred)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description= "corpus-level BLEU score.")
    parser.add_argument('gold', help= "file for the gold-standard sentences")
    parser.add_argument('pred', help= "file for the predicted sentences")
    return parser.parse_args()


if '__main__' == __name__:
    args = parse_args()
    print(bleu(load(args.gold), load(args.pred)))
