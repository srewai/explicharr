#!/usr/bin/env python3

from os.path import join
from utils import load, chartab, PointedIndex, encode, jagged_array
import numpy as np


path = "trial/data"


src = load(join(path, "train_src"))
tgt = load(join(path, "train_tgt"))

idx2src = PointedIndex(chartab(src))
idx2tgt = PointedIndex(chartab(tgt))
np.save("trial/data/index", dict(idx2src= idx2src.vec, idx2tgt= idx2tgt.vec))

src = [encode(idx2src, sent) for sent in src]
tgt = [encode(idx2tgt, sent) for sent in tgt]
src = jagged_array(src, fill= idx2src("\n"), shape= (len(src), max(map(len, src))), dtype= np.uint8)
tgt = jagged_array(tgt, fill= idx2tgt("\n"), shape= (len(tgt), max(map(len, tgt))), dtype= np.uint8)
np.save("trial/data/train_src", src)
np.save("trial/data/train_tgt", tgt)


src = load(join(path, "valid_src"))
tgt = load(join(path, "valid_tgt"))
src = [encode(idx2src, sent) for sent in src]
tgt = [encode(idx2tgt, sent) for sent in tgt]
src = jagged_array(src, fill= idx2src("\n"), shape= (len(src), max(map(len, src))), dtype= np.uint8)
tgt = jagged_array(tgt, fill= idx2tgt("\n"), shape= (len(tgt), max(map(len, tgt))), dtype= np.uint8)
np.save("trial/data/valid_src", src)
np.save("trial/data/valid_tgt", tgt)
