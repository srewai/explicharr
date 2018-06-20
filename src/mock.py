#!/usr/bin/env python3


len_cap = 2**8
path  = "../data"
path2 = "trial/data"
split = 0.01


from os.path import join
from util_io import clean, load

src, tgt = [], []

for line in load(join(path, "aligned-good-0.67.txt")):
    s, t, _ = line.split("\t")
    src.append(clean(s))
    tgt.append(clean(t))

for line in load(join(path, "aligned-good_partial-0.53.txt")):
    s, t, _ = line.split("\t")
    src.append(clean(s))
    tgt.append(clean(t))

len_cap -= 2 # start and end paddings
src_tgt = [(s, t) for s, t in zip(src, tgt) if s != t and max(len(s), len(t)) <= len_cap]

import random
random.seed(0)
random.shuffle(src_tgt)

i = int(len(src_tgt) * split)
valid_src, valid_tgt = zip(*src_tgt[:i])
train_src, train_tgt = zip(*src_tgt[i:])

from util_io import save
save(join(path2, "valid_src"), valid_src)
save(join(path2, "valid_tgt"), valid_tgt)
save(join(path2, "train_src"), train_src)
save(join(path2, "train_tgt"), train_tgt)
