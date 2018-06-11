#!/usr/bin/env python3

from utils import load, chartab, PointedIndex, encode, jagged_array
import numpy as np


path = "../mock/"


src = load(path + "train.nen")
tgt = load(path + "train.sen")

idx2src = PointedIndex(chartab(src))
idx2tgt = PointedIndex(chartab(tgt))
np.save("trial/data/idx", dict(idx2src= idx2src.vec, idx2tgt= idx2tgt.vec))

src = [encode(idx2src, sent) for sent in src]
tgt = [encode(idx2tgt, sent) for sent in tgt]
src = jagged_array(src, fill= idx2src("\n"), shape= (len(src), max(map(len, src))), dtype= np.uint8)
tgt = jagged_array(tgt, fill= idx2tgt("\n"), shape= (len(tgt), max(map(len, tgt))), dtype= np.uint8)
np.save("trial/data/src_train", src)
np.save("trial/data/tgt_train", tgt)


src = load(path + "test.nen")
tgt = load(path + "test.sen")

src = [encode(idx2src, sent) for sent in src]
tgt = [encode(idx2tgt, sent) for sent in tgt]
src = jagged_array(src, fill= idx2src("\n"), shape= (len(src), max(map(len, src))), dtype= np.uint8)
tgt = jagged_array(tgt, fill= idx2tgt("\n"), shape= (len(tgt), max(map(len, tgt))), dtype= np.uint8)
np.save("trial/data/src_valid", src)
np.save("trial/data/tgt_valid", tgt)
