# todo fixme due to api changes, this code is not working currently
# we may not need this anymore anyways


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
    from os.path import join
    from util_io import load

    path = "../data"
    src = list(map(str.split, load(join(path, "train.nen"))))
    tgt = list(map(str.split, load(join(path, "train.sen"))))

    print("src word", stat(amap(word_count, src)))
    print("src char", stat(amap(char_count, src)))
    print("tgt word", stat(amap(word_count, tgt)))
    print("tgt char", stat(amap(char_count, tgt)))

    vocab_src = vocab(src)
    vocab_tgt = vocab(tgt)

    alphabet_src = alphabet(vocab_src)
    alphabet_tgt = alphabet(vocab_tgt)

    char2freq = alphabet_src.copy()
    char2freq.update(alphabet_tgt)
    print("char\tsrc\ttgt")
    for char, _ in char2freq.most_common():
        print("{}\t{}\t{}".format(char, alphabet_src[char], alphabet_tgt[char]))
