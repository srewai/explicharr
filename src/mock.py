#!/usr/bin/env python3

from os.path import join


def mock(src, tgt, src2, tgt2, len_cap= 512):
    keep = []
    with open(src) as fs, open(tgt) as ft:
        for st in zip(fs, ft):
            ls, lt = map(len, st)
            if max(ls, lt) < len_cap:
                keep.append((ls + lt, ls, lt, st))
    keep.sort()
    with open(src2, 'w') as fs, open(tgt2, 'w') as ft:
        for _, _, _, (s, t) in keep:
            print(s, end= "", file= fs)
            print(t, end= "", file= ft)


len_cap = 2**8


path, path2 = "../data", "trial/data"
mock(len_cap= len_cap
     , src=  join(path,  "train.nen")
     , tgt=  join(path,  "train.sen")
     , src2= join(path2, "train_src")
     , tgt2= join(path2, "train_tgt"))
mock(len_cap= len_cap
     , src=  join(path,  "test.nen")
     , tgt=  join(path,  "test.sen")
     , src2= join(path2, "valid_src")
     , tgt2= join(path2, "valid_tgt"))
