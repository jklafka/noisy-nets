import torch, random, logging, argparse, csv
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import pipeline

NUM_TRAINING = 50000
NUM_TESTING = 1000
MAX_LENGTH = 10
HIDDEN_SIZE = 768 # same as LM embedding
LEARNING_RATE = .01
SOS_token = 0
EOS_token = 1
TEACHER_FORCING_RATIO = 0.5

device = torch.device("cuda:0")
# device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("train_file", help="Where to read the training pairs")
parser.add_argument("test_file", help="Where to read the testing pairs")
parser.add_argument("vocab_file", help="Where to read the vocab file")
args = parser.parse_args()

logging.basicConfig(filename = "Results/noisy-test.csv", format="%(message)s", \
                    level=logging.INFO)




def sentence_to_tensor(vocab, sentence):
    '''
    Get list of vocabulary indices for each token in sentence from vocab.
    '''
    indexes = [] # [SOS_token]
    for token in sentence.split(' '):
        indexes.append(vocab[token])
    # indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = input.view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    '''
    Gated recurrent unit (GRU) neural network. Takes in a previous output
    or an SOS token and a previous hidden state or encoder hidden state.
    '''
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1).to(device)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden.to(device))
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(vocab, input_text, target_text, lm_encoder, encoder, decoder, \
            encoder_optimizer, decoder_optimizer, criterion):
    '''
    One training iteration for the LM-autoencoder.
    '''
    loss = 0
    decoder_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    encoder_hidden = encoder.initHidden()

    target_tensor = sentence_to_tensor(vocab, target_text)
    target_length = target_tensor.size(0)

    with torch.no_grad():
        encoded_layers = lm(input_text)
    encoded_layers = torch.tensor(encoded_layers).to(device)
    input_length = encoded_layers.size(1)

    encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

    ## Encoder half of the autoencoder
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            encoded_layers[:, ei, :], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    ## Decoder half of the autoencoder
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    ## handles variable-length input
    use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def test(vocab, input_text, target_text, lm_encoder, encoder, decoder):
    '''
    One testing iteration on LM-decoder.
    '''
    with torch.no_grad():
        encoder_hidden = encoder.initHidden()

        target_tensor = sentence_to_tensor(vocab, target_text)
        target_length = target_tensor.size(0)

        loss = 0
        predicted_string = ""
        target_string = ""

        encoded_layers = lm(input_text)
        encoded_layers = torch.tensor(encoded_layers).to(device)
        input_length = encoded_layers.size(1)
        encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

        ## Encoder half of the autoencoder
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                encoded_layers[:, ei, :], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        ## Decoder half of the autoencoder
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        ## handles variable-length input
        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            # get each predicted word and target word
            predicted_word = [key for key, value in vocab.items() \
                                if value == decoder_input.item()][0]
            predicted_string += predicted_word

            loss += int(decoder_input == target_tensor[i])

            if decoder_input.item() == EOS_token:
                break

            # add space if more words forthcoming
            predicted_string += ' '

    return loss / target_length, predicted_string


# read in training and testing pairs and vocab
training_pairs = []
testing_pairs = []
with open("Stimuli/" + args.train_file + ".csv", 'r') as train_file:
    reader = csv.reader(train_file)
    for row in reader:
        training_pairs.append(row)
training_pairs = random.sample(population = training_pairs, k = NUM_TRAINING)

with open("Stimuli/" + args.test_file + ".csv", 'r') as test_file:
    reader = csv.reader(test_file)
    for row in reader:
        testing_pairs.append(row)
testing_pairs = random.sample(population = testing_pairs, k = NUM_TESTING)

vocab = open("Stimuli/" + args.vocab_file + ".txt", 'r').readlines()
vocab = [line.strip('\n').split('\t') for line in vocab]
vocab = {word : int(index) for word, index in vocab}

# initialize lm encoder
lm = pipeline("feature-extraction", model = "bert-base-uncased",
                device=-1)

# initialize decoder, optimizer and loss function
encoder_rnn = EncoderRNN(HIDDEN_SIZE).to(device)
decoder_rnn = DecoderRNN(HIDDEN_SIZE, len(vocab)).to(device)
# decoder = AttnDecoderRNN(HIDDEN_SIZE, lang.n_words, dropout_p=0.1).to(device)
encoder_optimizer = optim.SGD(encoder_rnn.parameters(), lr=LEARNING_RATE)
decoder_optimizer = optim.SGD(decoder_rnn.parameters(), lr=LEARNING_RATE)
criterion = nn.NLLLoss()

# ------ training --------
for training_pair in training_pairs:
    input_text = training_pair[0]
    target_text = training_pair[1]

    loss = train(vocab, input_text, target_text, lm, encoder_rnn, decoder_rnn,
                    encoder_optimizer, decoder_optimizer, criterion)

# torch.save(decoder.state_dict(), "Models/current_decoder")

# ------ testing --------
for testing_pair in testing_pairs:
    input_text = testing_pair[0]
    target_text = testing_pair[1]

    loss, prediction = test(vocab, input_text, target_text, lm,
                            encoder_rnn, decoder_rnn)
    logging.info(';'.join([input_text, prediction, target_text, str(loss),
                str(testing_pair[2]), str(testing_pair[3])]))
    # logging.info(','.join([input_text, prediction, target_text, str(loss),
    #             str(testing_pair[2]), str(testing_pair[3])]))
