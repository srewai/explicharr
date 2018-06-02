#!/usr/bin/env bash


# mkdir Data
# mkdir Data/Models
# mkdir Data/Models/translation_model
# mkdir Data/MachineTranslation

# # mkdir Data/tb_summaries
# # mkdir Data/tb_summaries/translator_model


# python3 train_translator.py \
    #         --source_file ../../data/train.nen \
    #         --target_file ../../data/train.sen


python3 translate.py \
        --model_path Data/Models/translation_model/model_epoch_1_213.ckpt \
        --source_file ../../data/test.nen \
        --target_file ../../data/test.sen
