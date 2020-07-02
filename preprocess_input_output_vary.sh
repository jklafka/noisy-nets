#!/bin/bash
#SBATCH --job-name=noisy-nets-p
#SBATCH --error=noisy-nets-p.err
#SBATCH --output=noisy-nets-p.out
#SBATCH --mail-user=qinxuanwu@uchicago.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=speech-cpu
#SBATCH --nodes=1 --constraint=avx

python3 Noise/clean_script.py $1
python3 Noise/input_output_vary/train_test_vocab.py clean_$1 clean_${1}_train clean_${1}_test ${1}_vocab
python3 Noise/input_output_vary/noise_introduce.py $1
rm Stimuli/clean_$1.txt Stimuli/clean_${1}_train.txt Stimuli/clean_${1}_test.txt
