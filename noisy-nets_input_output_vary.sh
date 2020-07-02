#!/bin/bash
#SBATCH --job-name=noisy-nets-p
#SBATCH --error=noisy-nets-p.err
#SBATCH --output=noisy-nets-p.out
#SBATCH --mail-user=qinxuanwu@uchicago.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=speech-gpu
#SBATCH --nodes=1 --constraint=12g


python3 Models/lm_autoencoder_updated.py ${1}_train_$2 ${1}_test ${1}_vocab noisy-test_$2
