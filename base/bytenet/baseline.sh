#!/usr/bin/env bash

# mkdir baseline

# python3 train_translator.py ../../data/train.nen ../../data/train.sen

python3 translate.py --model-path baseline/model/e20_b300.ckpt ../../data/test.nen ../../data/test.sen
