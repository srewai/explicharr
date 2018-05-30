from collections import Counter
import numpy as np


stat = lambda x: dict(min= x.min(), max= x.max(), mean= x.mean())
amap = lambda f, a: np.fromiter(map(f, a), dtype= np.float)

char_count = lambda sent: len(sent) + sum(map(len, sent))
word_count = len


vocab = lambda sents: Counter((word for sent in sents for word in sent))


def alphabet(vocab):
    char2freq = Counter()
    for word, freq in vocab.items():
        char2freq[" "] += freq
        for char in word:
            char2freq[char] += freq
    return char2freq


if '__main__' == __name__:
    from utils import load

    path = "../data/"
    normal = load(path + "train.nen")
    simple = load(path + "train.sen")

    print("normal word", stat(amap(word_count, normal)))
    print("normal char", stat(amap(char_count, normal)))
    print("simple word", stat(amap(word_count, simple)))
    print("simple char", stat(amap(char_count, simple)))

    vocab_normal = vocab(normal)
    vocab_simple = vocab(simple)

    alphabet_normal = alphabet(vocab_normal)
    alphabet_simple = alphabet(vocab_simple)

    char2freq = alphabet_normal.copy()
    char2freq.update(alphabet_simple)
    print("char\tnormal\tsimple")
    for char, _ in char2freq.most_common():
        print("{}\t{}\t{}".format(char, alphabet_normal[char], alphabet_simple[char]))
