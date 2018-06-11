#!/usr/bin/env bash


# mkdir baseline


# python3 preprocess.py \
    #         -train_src ../../data/train.nen \
    #         -train_tgt ../../data/train.sen \
    #         -valid_src ../../data/test.nen \
    #         -valid_tgt ../../data/test.sen \
    #         -save_data baseline/data.pt


# python3 train.py -data baseline/data.pt \
    #         -save_model baseline/model \
    #         -save_mode best \
    #         -proj_share_weight


# python3 translate.py \
    #         -model baseline/model.chkpt \
    #         -vocab baseline/data.pt \
    #         -src ../../data/test.nen \
    #         -output baseline/pred


python3 ../bleu.py \
        --ignore-case \
        ../../data/test.sen \
        baseline/pred.sen
