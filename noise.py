import random

NOISE_RATIO = 0.5

# read in stimuli
source_pairs = open("Stimuli/eng-eng.txt", 'r').readlines()
# get vocabulary


noisy_pairs = []
vocabulary = set()

for pair in source_pairs:
    seq1, seq2 = pair.split('\t')
    seq1s = seq1.split()
    if len(seq1s) >= 5:
        vocabulary = vocabulary | set(seq1s)
        if random.random() < NOISE_RATIO: # make edit with prob < NOISE_RATIO
            j = random.randint(0, len(seq1s) - 1)
            if random.random() < .5: # 50% chance insert, 50% chance removes
                # remove word at index j
                seq1s.pop(j)
            else:
                # insert random word at index j
                random_word = random.choice(tuple(vocabulary))
                seq1s.insert(j, random_word)
            seq1 = ' '.join(seq1s)
        noisy_pairs.append(seq1 + '\t' + seq2)

# write to output
with open("Stimuli/noisy-eng-eng.txt", 'w') as f:
    for pair in noisy_pairs:
        f.write("%s\n" % pair)
