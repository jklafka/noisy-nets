import random, argparse, csv
from itertools import chain

NOISE_RATIO = 0.5

parser = argparse.ArgumentParser()
parser.add_argument("language_file", help="Name of the (un-noised) .txt to use")
args = parser.parse_args()

# read in stimuli
# source_pairs = open("Stimuli/" + args.language_file + ".txt", 'r').readlines()
# get vocabulary

random.seed(2020)



def noice_introduce(seqs, INS_RATIO = .33, DE_RATIO = .33):
    vocabulary = set()
    j = random.randint(0, len(seqs) - 1) #position
    p = random.random()
    if p < DE_RATIO: # 1/3 insert, 1/3 deletion, 1/3 swap
        # remove word at index j
        seqs.pop(j)
        edit = "deletion"
    elif p >= DE_RATIO and p < (DE_RATIO + INS_RATIO):
        # insert random word at index j
        vocabulary = vocabulary | set(seqs)
        random_word = random.choice(tuple(vocabulary))
        seqs.insert(j, random_word)
        edit = "insertion"
    else:
        # swap words
        # choose a random word from seq1 that isn't at position j
        indices = list(range(len(seqs)))
        indices.pop(j)
        i = random.choice(indices)
        seqs[i], seqs[j] = seqs[j], seqs[i]
        edit = "swap"
    seqs_noisy = ' '.join(seqs)
    return seqs_noisy, edit, j

def create_input_output_wrt_type(input_type, output_type, source_pairs):
    BOTH_CLEAN_tag = False
    NOISE_RATIO_tag = False
    if input_type == "clean" and output_type == "clean":
        BOTH_CLEAN_tag = True
    if input_type == "mixed" or output_type == "mixed": # mixed: both noisy and clean
        NOISE_RATIO_tag = True
    noisy_data = []
    for pair in source_pairs:
        seq1, seq2 = pair.split('\t')
        seq2 = seq2.strip('\n')
        seq1s = seq1.split()
        if len(seq1s) >= 5:
            # vocabulary = vocabulary | set(seq1s)
            # if random.random() < NOISE_RATIO: # make edit with prob < NOISE_RATIO
            if BOTH_CLEAN_tag == True:
                noisy_data.append([seq1, seq2.strip('\n'), "none", str(-1)])
            else:
                if NOISE_RATIO_tag == True:
                    if random.random() < NOISE_RATIO: # make edit with prob < NOISE_RATIO
                        seq1, edit, j = noice_introduce(seq1s)
                    else:
                        j = -1
                        edit = "none"
                else: seq1, edit, j = noice_introduce(seq1s)
                # noisy_data.append(seq1 + ',' + seq2 + ',' + edit + ',' + str(j))
                if input_type == "clean":
                    seq_input = seq2
                elif input_type in {"noisy", "mixed"} :
                    seq_input = seq1
                if output_type == "clean":
                    seq_output = seq2
                elif output_type in {"noisy", "mixed"} :
                    seq_output = seq1
                # noisy_data.append([seq1, seq2.strip('\n'), edit, str(j)])
                noisy_data.append([seq_input, seq_output, edit, str(j)])
    return noisy_data

# write to output
# for train
for (input_type, output_type) in [("clean", "clean"), ("noisy", "noisy"), \
                            ("noisy", "clean"), ("mixed", "clean"), ("clean", "noisy")]:
# for (input_type, output_type) in [("clean", "noisy")]:
    source_pairs = open("Stimuli/clean_" + args.language_file + "_train.txt", 'r').readlines()
    noisy_data = create_input_output_wrt_type(input_type, output_type, source_pairs)
    with open("Stimuli/" + args.language_file + "_train_" + input_type + "_" + output_type + ".csv", 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerows(noisy_data)

# for test
# for (input_type, output_type) in [("mixed", "clean")]:
source_pairs = open("Stimuli/clean_" + args.language_file + "_test.txt", 'r').readlines()
clean_clean_test = create_input_output_wrt_type("clean", "clean", source_pairs)
noisy_clean_test = create_input_output_wrt_type("noisy", "clean", source_pairs)
test_data = list(chain.from_iterable(zip(noisy_clean_test, clean_clean_test)))
with open("Stimuli/" + args.language_file + "_test.csv", 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerows(test_data)
