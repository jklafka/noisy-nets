# @Author: Josef Klafka <academic>
# @Date:   2020-03-27T17:41:16-04:00
# @Email:  jlklafka@gmail.com
# @Project: Noisy-nets
# @Last modified by:   academic
# @Last modified time: 2020-03-30T14:05:21-04:00



import torch, random, logging, argparse, statistics
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import BertTokenizer, BertModel

NUM_TRAINING = 100#120000
NUM_TESTING = 10#10000
MAX_LENGTH = 10
HIDDEN_SIZE = 768 # same as BERT embedding
BERT_LAYER = 11
LEARNING_RATE = .01
SOS_token = 0
EOS_token = 1
TEACHER_FORCING_RATIO = 0.5

# device = torch.device("cuda:0")
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("train_file", help="Where to read the training pairs")
parser.add_argument("test_file", help="Where to read the testing pairs")
parser.add_argument("vocab_file", help="Where to read the vocab file")
args = parser.parse_args()

# Load pre-trained model tokenizer (vocabulary)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.to(device)
bert_model.eval()

logging.basicConfig(filename = "Results/noisy-test.csv", format="%(message)s", \
                    level=logging.INFO)

# read in training and testing pairs and vocab
# logging.info("load files")
training_pairs = open("Stimuli/" + args.train_file + ".txt", 'r').readlines()
training_pairs = [line.strip('\n').split('\t') for line in training_pairs]
training_pairs = random.choices(training_pairs, k = NUM_TRAINING)

testing_pairs = open("Stimuli/" + args.test_file + ".txt", 'r').readlines()
testing_pairs = [line.strip('\n').split('\t') for line in testing_pairs]
testing_pairs = random.choices(testing_pairs, k = NUM_TESTING)

vocab = open("Stimuli/" + args.vocab_file + ".txt", 'r').readlines()
vocab = [line.strip('\n').split('\t') for line in vocab]
vocab = {word : int(index) for word, index in vocab}


def sentence_to_tensor(tokenizer, vocab, sentence):
    '''
    Get list of vocabulary indices for each token in sentence from vocab.
    '''
    indexes = [] # [SOS_token]
    for token in tokenizer.tokenize(sentence):
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
    Gated recurrent unit neural network. Takes in a BERT or GLoVe embedding and
    returns a vocabulary index and hidden state on its forward pass.
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


def train(vocab, input_text, target_text, model, tokenizer, encoder, decoder, \
            encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    '''
    One training iteration for the LM-autoencoder.
    '''
    loss = 0
    decoder_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    target_tensor = sentence_to_tensor(tokenizer, vocab, target_text)
    target_length = target_tensor.size(0)

    ## convert to vocabulary indices tensor
    tokenized_text = tokenizer.tokenize(input_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    input_length = tokens_tensor.size(0)

    tokens_tensor = tokens_tensor.to(device)
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            encoded_layers[:, ei, :], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            # decoder_output, decoder_hidden = decoder(
            #     decoder_input.unsqueeze(0), decoder_hidden)
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            # decoder_output, decoder_hidden = decoder(
            #     decoder_input.unsqueeze(0), decoder_hidden)
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


def test(vocab, input_text, target_text, model, tokenizer, encoder, decoder, \
            max_length=MAX_LENGTH):
    '''
    One testing iteration on LM-decoder.
    '''
    with torch.no_grad():
        target_tensor = sentence_to_tensor(tokenizer, vocab, target_text)

        loss = 0
        predicted_string = ""

        tokenized_text = tokenizer.tokenize(input_text)
        #convert to vocabulary indices tensor
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        tokens_tensor = tokens_tensor.to(device)
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor)
        encoder_outputs = encoded_layers
        target_length = min(encoder_outputs.size(1), target_tensor.size(0))

        # decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = decoder.initHidden()

        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(
                encoder_outputs[:,i,:].unsqueeze(0), decoder_hidden)
            # decoder_output, decoder_hidden, decoder_attention = decoder(
            #     decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.topk(1) #which
            decoder_input = topi.squeeze().detach()

            # loss += criterion(decoder_output, target_tensor[i])
            loss += int(decoder_input == target_tensor[i])

            # get each predicted word
            predicted_word = [key for key, value in vocab.items() \
                                if value == decoder_input.item()][0]
            predicted_string += ' ' + predicted_word

    # return loss.item() / target_length
    return loss / target_length, predicted_string, ' '.join(tokenized_text), \


# initialize decoder, optimizer and loss function
encoder_rnn = EncoderRNN(HIDDEN_SIZE).to(device)
decoder_rnn = DecoderRNN(HIDDEN_SIZE, len(vocab)).to(device)
# decoder = AttnDecoderRNN(HIDDEN_SIZE, lang.n_words, dropout_p=0.1).to(device)
encoder_optimizer = optim.SGD(encoder_rnn.parameters(), lr=LEARNING_RATE)
decoder_optimizer = optim.SGD(decoder_rnn.parameters(), lr=LEARNING_RATE)
criterion = nn.NLLLoss()

# training
# logging.info("start training")
training_losses = []
for training_pair in training_pairs:
    input_text = training_pair[0]
    target_text = training_pair[1]

    loss = train(vocab, input_text, target_text, bert_model, bert_tokenizer,
                 encoder_rnn, decoder_rnn, encoder_optimizer, decoder_optimizer,
                 criterion)
    training_losses.append(loss)

# torch.save(decoder.state_dict(), "Models/current_decoder")

# testing
# logging.info("start testing")
testing_accuracy = []
for testing_pair in testing_pairs:
    input_text = testing_pair[0]
    target_text = testing_pair[1]

    loss, prediction, input_tokens, target_tokens = test(vocab, input_text,
                target_text, bert_model, bert_tokenizer, encoder_rnn, decoder_rnn)
    testing_accuracy.append(loss)
    logging.info(input_tokens + ',' + prediction + ',' + target_tokens + ',' + str(loss))

total_accuracy = statistics.mean(testing_accuracy)
logging.info(str(total_accuracy) + ',' + str(NUM_TRAINING))