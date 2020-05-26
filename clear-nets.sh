#!/bin/bash
#SBATCH --job-name=clear-nets
#SBATCH --error=clear-nets.err
#SBATCH --output=clear-nets.out
#SBATCH --mail-user=jklafka@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

python3 bert_preprocess.py $1 train test vocab
python3 bert_sdae.py train test vocab
