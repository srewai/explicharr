#!/usr/bin/env bash


# python3 preprocess.py \
    #         -train_src ../../data/train.nen \
    #         -train_tgt ../../data/train.sen \
    #         -valid_src ../../data/test.nen \
    #         -valid_tgt ../../data/test.sen \
    #         -save_data baseline/data.pt


# python3 train.py -data baseline/mock.pt \
    #         -epoch 3 \
    #         -batch_size 6 \
    #         -save_model baseline/model \
    #         -save_mode best \
    #         -proj_share_weight


python3 train.py -data baseline/data.pt \
        -save_model baseline/model \
        -save_mode best \
        -proj_share_weight


# python translate.py \
    #        -model baseline/model.chkpt \
    #        -vocab baseline/data.pt \
    #        -src ../../data/test.nen \
    #        > baseline/pred
