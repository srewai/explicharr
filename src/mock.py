#!/usr/bin/env python3

import os


def mock(pathi, patho, src, tgt, rst= "rest", max_len= 512):
    keep, rest = [], []
    with open(os.path.join(pathi, src)) as fs, \
         open(os.path.join(pathi, tgt)) as ft:
        for st in zip(fs, ft):
            ls, lt = map(len, st)
            (keep if max(ls, lt) <= max_len else rest).append(
                (ls + lt, ls, lt, st))
    keep.sort()
    rest.sort()
    with open(os.path.join(patho, src), 'w') as fs, \
         open(os.path.join(patho, tgt), 'w') as ft:
        for _, _, _, (s, t) in keep:
            print(s, end= "", file= fs)
            print(t, end= "", file= ft)
    with open(os.path.join(patho, rst), 'w') as fr:
        for _, _, _, (s, t) in rest:
            print(s, end= "", file= fr)
            print(t, end= "", file= fr)


if '__main__' == __name__:
    mock(pathi= "../data", patho= "../mock", src= "train.nen", tgt= "train.sen", rst= "train", max_len= 64)
    mock(pathi= "../data", patho= "../mock", src= "test.nen",  tgt= "test.sen",  rst= "test",  max_len= 64)
