import random, argparse, csv

NOISE_RATIO = 0.5

parser = argparse.ArgumentParser()
parser.add_argument("language_file", help="Name of the (un-noised) .txt to use")
args = parser.parse_args()

# read in stimuli
source_pairs = open("Stimuli/" + args.language_file + ".txt", 'r').readlines()
# get vocabulary


noisy_data = []
vocabulary = set()

for pair in source_pairs:
    seq1, seq2 = pair.split('\t')
    seq1s = seq1.split()
    if len(seq1s) >= 5:
        vocabulary = vocabulary | set(seq1s)
        if random.random() < NOISE_RATIO: # make edit with prob < NOISE_RATIO
            j = random.randint(0, len(seq1s) - 1)
            p = random.random()
            if p < .33: # 1/3 insert, 1/3 deletion, 1/3 swap
                # remove word at index j
                seq1s.pop(j)
                edit = "deletion"
            elif p >= .33 and p < .66:
                # insert random word at index j
                random_word = random.choice(tuple(vocabulary))
                seq1s.insert(j, random_word)
                edit = "insertion"
            else:
                # swap words
                # choose a random word from seq1 that isn't at position j
                indices = list(range(len(seq1s)))
                indices.pop(j)
                i = random.choice(indices)
                seq1s[i], seq1s[j] = seq1s[j], seq1s[i]
                edit = "swap"
            seq1 = ' '.join(seq1s)
        else:
            j = -1
            edit = "none"
        # noisy_data.append(seq1 + ',' + seq2 + ',' + edit + ',' + str(j))
        noisy_data.append([seq1, seq2.strip('\n'), edit, str(j)])

# write to output
with open("Stimuli/noisy-" + args.language_file + ".csv", 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerows(noisy_data)
