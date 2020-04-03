import argparse, random
from collections import Counter
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument("language_file", help="Name of the (noised) .txt to use")
parser.add_argument("train_file", help="Where to print the training pairs")
parser.add_argument("test_file", help="Where to print the testing pairs")
parser.add_argument("vocab_file", help="Where to print the vocab file")
args = parser.parse_args()


def get_BERT_vocab(tokenizer, pairs):
    '''
    Get one-hot indices for each BERT-token in language_file.
    '''
    vocab = set()
    for pair in pairs:
        for line in pair:
            tokens = tokenizer.tokenize(line)
            vocab = vocab | set(tokens)
    vocab = ["SOS", "EOS"] + list(vocab) # insert SOS and EOS tokens
    return vocab

# set up training and testing data
pairs = open("Stimuli/" + args.language_file + ".txt", 'r').readlines()
pairs = [line.strip('\n').split('\t') for line in pairs]
random.shuffle(pairs)

# get all pairs and lengths where target does not have a unique length
lengths = [len(pair[1].split()) for pair in pairs]
unique_lengths = [key for key, val in Counter(lengths).items() if val == 1]
pairs = [pairs[i] for i in range(len(pairs)) if lengths[i] not in unique_lengths]
lengths = [length for length in lengths if length not in unique_lengths]

# stratified train test split over all the pairs
pairs_train, pairs_test = train_test_split(pairs, test_size = 0.1, \
                                            stratify = lengths)
# write training and testing pairs to separate files
with open("Stimuli/" + args.train_file + ".txt", 'w') as filename:
        filename.writelines("%s\t%s\n" % (pair[0], pair[1]) for pair in pairs_train)

with open("Stimuli/" + args.test_file + ".txt", 'w') as filename:
        filename.writelines("%s\t%s\n" % (pair[0], pair[1]) for pair in pairs_test)


# get vocabulary and print to external file
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab = get_BERT_vocab(bert_tokenizer, pairs)

with open("Stimuli/" + args.vocab_file + ".txt", 'w') as filename:
        filename.writelines("%s\t%d\n" % (word, index) for index, word in enumerate(vocab))
