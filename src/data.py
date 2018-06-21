#!/usr/bin/env python3


path = "trial/data"


from os.path import join
from util import partial, PointedIndex
from util_io import load, chartab, encode
from util_np import np, jagged_array

src = list(load(join(path, "train_src")))
tgt = list(load(join(path, "train_tgt")))

idx_src = PointedIndex(chartab(src))
idx_tgt = PointedIndex(chartab(tgt))
enc_src = partial(encode, idx_src)
enc_tgt = partial(encode, idx_tgt)

assert 1 == idx_src("\n") == idx_tgt("\n")
pack = lambda txt: jagged_array(txt, fill= 1, shape= (len(txt), max(map(len, txt))), dtype= np.uint8)

np.save(join(path, "index_src"), idx_src.vec)
np.save(join(path, "index_tgt"), idx_tgt.vec)
np.save(join(path, "train_src"), pack(list(map(enc_src, src))))
np.save(join(path, "train_tgt"), pack(list(map(enc_tgt, tgt))))
np.save(join(path, "valid_src"), pack(list(map(enc_src, load(join(path, "valid_src"))))))
np.save(join(path, "valid_tgt"), pack(list(map(enc_tgt, load(join(path, "valid_tgt"))))))
