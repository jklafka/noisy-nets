import argparse

parser = argparse.ArgumentParser()
parser.add_argument("language_file", help="Name of the (noised) .tsv to use")
args = parser.parse_args()

text = open("Stimuli/" + args.language_file + ".txt", 'r').readlines()

clean_text = []
for pair in text:
    seq1, seq2 = pair.split('\t')
    seq1 = seq1.strip('.?!').lower()
    seq2 = seq2.strip('.?!\n').lower()
    clean_text.append(seq1 + '\t' + seq2 + '\n')

with open("Stimuli/clean_" + args.language_file + ".txt", 'w') as f:
    f.writelines(clean_text)
