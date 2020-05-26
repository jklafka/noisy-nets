#!/bin/bash
#SBATCH --job-name=lm_autoencoder
#SBATCH --error=noisy-nets.err
#SBATCH --output=noisy-nets.out
#SBATCH --mail-user=jklafka@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

python3 lm_autoencoder.py noisy-train noisy-test vocab
