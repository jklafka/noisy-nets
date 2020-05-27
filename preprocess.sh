#!/bin/bash

python3 Noise/clean_script.py $1
python3 Noise/noise.py clean_$1
python3 Noise/train_test_vocab.py noisy-clean_$1 ${1}_train ${1}_test ${1}_vocab
rm Stimuli/clean_eng-eng.txt Stimuli/eng-eng_vocab.txt Stimuli/noisy-clean_eng-eng.csv
