#!/usr/bin/env bash


# mkdir Data
# mkdir Data/MachineTranslation
# mkdir Data/Models
# mkdir Data/Models/translation_model
# mkdir Data/tb_summaries
# mkdir Data/tb_summaries/translator_model


python3 train_translator.py \
        --source_file ../../data/train.nen \
        --target_file ../../data/train.sen
