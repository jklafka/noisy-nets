import argparse, random, csv
from collections import Counter
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

random.seed(2020)


parser = argparse.ArgumentParser()
parser.add_argument("language_file", help="Name of the (noised) .txt to use")
parser.add_argument("train_file", help="Where to print the training pairs")
parser.add_argument("test_file", help="Where to print the testing pairs")
parser.add_argument("vocab_file", help="Where to print the vocab file")
args = parser.parse_args()


def create_vocab(training_pairs, testing_pairs):
    '''
    Create a vocabulary mapping words to one-hot indices from the given training
    and testing sentence pairs.
    '''
    pairs = training_pairs + testing_pairs
    vocab = set()
    for pair in pairs:
        for line in pair[:2]:
            tokens = line.split(' ')
            vocab = vocab | set(tokens)
    vocab = {"SOS", "EOS"} | vocab # insert SOS and EOS tokens
    vocab = {word : index for index, word in enumerate(list(vocab))}
    return vocab

pairs = open("Stimuli/" + args.language_file + ".txt", 'r').readlines()

# pairs = [[line[0], line[1].strip('\n')] + line[2:] for line in pairs]
pairs = [(pair.split('\t')[0], pair.split('\t')[1].strip('\n')) for pair in pairs]
random.shuffle(pairs)

# get all pairs and lengths where target does not have a unique length
lengths = [len(pair[1].split()) for pair in pairs]
unique_lengths = [key for key, val in Counter(lengths).items() if val == 1]
pairs = [pairs[i] for i in range(len(pairs)) if lengths[i] not in unique_lengths]
lengths = [length for length in lengths if length not in unique_lengths]

# stratified train test split over all the pairs
pairs_train, pairs_test = train_test_split(pairs, test_size = 0.1, \
                                            stratify = lengths, random_state = 2020)
# write training and testing pairs to separate files
# with open("Stimuli/" + args.train_file + ".csv", 'w') as train_file:
#     writer = csv.writer(train_file)
#     writer.writerows(pairs_train)

# with open("Stimuli/" + args.test_file + ".csv", 'w') as test_file:
#     writer = csv.writer(test_file)
#     writer.writerows(pairs_test)

with open("Stimuli/" + args.train_file + ".txt", 'w') as filename:
        filename.writelines("%s\t%s\n" % (pair[0], pair[1]) for pair in pairs_train)

with open("Stimuli/" + args.test_file + ".txt", 'w') as filename:
        filename.writelines("%s\t%s\n" % (pair[0], pair[1]) for pair in pairs_test)


# get vocabulary and print to external file
vocab = create_vocab(pairs_train, pairs_test)

with open("Stimuli/" + args.vocab_file + ".txt", 'w') as filename:
        filename.writelines("%s\t%d\n" % (word, index) for index, word in enumerate(vocab))
