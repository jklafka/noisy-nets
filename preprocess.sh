#!/bin/bash

python3 clean_script.py $1
python3 Noise/noise.py clean_$1
python3 Noise/bert_preprocess.py noisy-clean_$1 ${1}_train ${1}_test ${1}_vocab
