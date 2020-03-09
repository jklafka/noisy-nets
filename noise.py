import random, argparse

NOISE_RATIO = 0.5

parser = argparse.ArgumentParser()
parser.add_argument("language_file", help="Name of the (noised) .tsv to use")
args = parser.parse_args()

# read in stimuli
source_pairs = open("Stimuli/" + args.language_file + ".txt", 'r').readlines()
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
            p = random.random()
            if p < .33: # 50% chance insert, 50% chance removes
                # remove word at index j
                seq1s.pop(j)
            elif p >= .33 and p < .66:
                # insert random word at index j
                random_word = random.choice(tuple(vocabulary))
                seq1s.insert(j, random_word)
            else:
                # swap words
                # choose a random word from seq1 that isn't at position j
                indices = list(range(len(seq1s)))
                indices.pop(j)
                i = random.choice(indices)
                seq1s[i], seq1s[j] = seq1s[j], seq1s[i]
            seq1 = ' '.join(seq1s)
        noisy_pairs.append(seq1 + '\t' + seq2)

# write to output
with open("Stimuli/noisy-" + args.language_file + ".txt", 'w') as f:
    for pair in noisy_pairs:
        f.write("%s" % pair)
