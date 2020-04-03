#!/bin/bash
#SBATCH --job-name=noisy-nets
#SBATCH --error=noisy-nets.err
#SBATCH --output=noisy-nets.out
#SBATCH --mail-user=jklafka@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

python3 bert_sdae.py noisy-train noisy-test vocab
