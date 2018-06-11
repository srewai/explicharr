from collections import defaultdict, Counter
from itertools import chain, repeat, islice
import numpy as np


def batches(size, batch_size):
    i = 0
    while i < size:
        yield i, i + batch_size
        i += batch_size


def batch(pair_list):
    source, target = zip(*pair_list)
    return np.stack(source), np.stack(target)


def build_vocab(sentences, top= 256, special= ('eol', 'padding', 'init')):
    char2freq = Counter(char for sent in sentences for char in sent)
    chars = [k for k, _ in sorted(
        char2freq.items()
        , key= lambda kv: (-kv[1], kv[0])
    )[:top-len(special)]]
    chars.extend(special)
    return {char: idx for idx, char in enumerate(chars)}


def string_to_indices(sentence, vocab):
    unknown = vocab[' ']
    return [vocab.get(s, unknown) for s in sentence]


def indices_to_string(sentence, vocab):
    idx2char = {idx : char for char, idx in vocab.items()}
    sent = []
    for idx in sentence:
        char = idx2char[idx]
        if 'eol' == char: break
        sent.append(char)
    return "".join(sent)


class DataLoader:
    def __init__(self, source_file, target_file, bucket_quant):
        max_len = 297 # 297+3=300 aka 6 buckets of 50
        self.source_lines = []
        self.target_lines = []
        with open(source_file) as fs, open(target_file) as ft:
            for s, t in zip(fs, ft):
                s = s.strip()
                t = t.strip()
                if max(len(s), len(t)) <= max_len:
                    self.source_lines.append(s)
                    self.target_lines.append(t)
        print("Source Sentences", len(self.source_lines))
        print("Target Sentences", len(self.target_lines))
        self.bucket_quant = bucket_quant
        self.source_vocab = build_vocab(self.source_lines)
        self.target_vocab = build_vocab(self.target_lines)
        print("SOURCE VOCAB SIZE", len(self.source_vocab))
        print("TARGET VOCAB SIZE", len(self.target_vocab))

    def load_translation_data(self):
        bucket_quant = self.bucket_quant
        source_vocab = self.source_vocab
        target_vocab = self.target_vocab
        buckets, sents = defaultdict(list), defaultdict(list)
        for src, tgt in zip(self.source_lines, self.target_lines):
            s = string_to_indices(src, source_vocab)
            s.append(source_vocab['eol'])
            t = [target_vocab['init']]
            t.extend(string_to_indices(tgt, target_vocab))
            t.append(target_vocab['eol'])
            new_length = int(bucket_quant * np.ceil(max(len(s), len(t)) / bucket_quant))
            s = np.fromiter(islice(chain(s, repeat(source_vocab['padding'])), new_length), dtype= np.uint8)
            t = np.fromiter(islice(chain(t, repeat(target_vocab['padding'])), new_length + 1), dtype= np.uint8)
            buckets[new_length].append((s, t))
            sents[new_length].append((src, tgt))
        return dict(buckets), dict(sents)

    def decode(self, sent):
        return indices_to_string(sent, self.target_vocab)
