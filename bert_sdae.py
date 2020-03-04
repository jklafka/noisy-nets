import torch, random, logging, argparse
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import BertTokenizer, BertModel


MAX_LENGTH = 10
HIDDEN_SIZE = 768 # same as BERT embedding
NUM_ITERS = 7500
BERT_LAYER = 11
LEARNING_RATE = .01

SOS_token = 0
EOS_token = 1

# device = torch.device("cuda:0")
device = torch.device("cpu")

logging.basicConfig(filename = "noisy-results.csv", format="%(message)s", \
                    level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("train_file", help="Where to read the training pairs")
parser.add_argument("test_file", help="Where to read the testing pairs")
parser.add_argument("vocab_file", help="Where to read the vocab file")
args = parser.parse_args()

# Load pre-trained model tokenizer (vocabulary)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
bert_model = BertModel.from_pretrained('bert-base-uncased')
# bert_model.to('cuda')
bert_model.eval()

# read in training and testing pairs and vocab
training_pairs = open("Stimuli/" + args.train_file + ".txt", 'r').readlines()
training_pairs = [line.strip('\n').split('\t') for line in training_pairs]

test_pairs = open("Stimuli/" + args.test_file + ".txt", 'r').readlines()
test_pairs = [line.strip('\n').split('\t') for line in test_pairs]

vocab = open("Stimuli/" + args.vocab_file + ".txt", 'r').readlines()
vocab = [line.strip('\n').split('\t') for line in vocab]
vocab = {word : int(index) for word, index in vocab}


def sentence_to_tensor(tokenizer, vocab, sentence):
    '''
    Get list of vocabulary indices for each token in sentence from vocab.
    '''
    indexes = [SOS_token]
    indexes = [vocab[token] for token in tokenizer.tokenize(sentence)]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


class DecoderRNN(nn.Module):
    '''
    Gated recurrent unit neural network. Takes in a BERT or GLoVe embedding and
    returns a vocabulary index and hidden state on its forward pass.
    '''
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = F.relu(input)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(vocab, input_text, target_text, model, tokenizer, decoder, \
            decoder_optimizer, criterion, max_length=MAX_LENGTH):
    '''
    One training iteration for the decoder on BERT embeddings.
    '''
    decoder_optimizer.zero_grad()
    target_tensor = sentence_to_tensor(tokenizer, vocab, target_text)
    target_length = target_tensor.size(0)

    loss = 0

    tokenized_text = tokenizer.tokenize(input_text)
    #convert to vocabulary indices tensor
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    # tokens_tensor = tokens_tensor.to('cuda')

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor)

    encoder_outputs = encoded_layers[BERT_LAYER]

    # decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = decoder.initHidden()

    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    #
    # if use_teacher_forcing:
    # Teacher forcing: Feed the target as the next input
    for i in range(target_length):
        decoder_output, decoder_hidden = decoder(
            encoder_outputs[:,i].unsqueeze(0), decoder_hidden)
        # decoder_output, decoder_hidden, decoder_attention = decoder(
        #     decoder_input, decoder_hidden, encoder_outputs)

        topv, topi = decoder_output.topk(1) #which
        decoder_output = topi.squeeze().detach()

        # loss defined on one-hot vectors
        loss += criterion(decoder_output, target_tensor)
        decoder_input = decoder_output

    # else:
    # Without teacher forcing: use its own predictions as the next input
    # print(encoder_outputs)
    # print(decoder_hidden)
        # for di in range(target_length):
        #     decoder_output, decoder_hidden = decoder(
        #         encoder_outputs[:,di,:].unsqueeze(-3), decoder_hidden)
        #     # decoder_output, decoder_hidden, decoder_attention = decoder(
        #     #     decoder_input, decoder_hidden, encoder_outputs)
        #     topv, topi = decoder_output.topk(1)
        #     decoder_input = topi.squeeze().detach()  # detach from history as input
        #
        #     loss += criterion(decoder_output, target_tensor[di])
        # if decoder_input.item() == EOS_token:
        #     break

    loss.backward()

    decoder_optimizer.step()

    return loss.item() / target_length

def evaluate(lang, encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, \
                                        device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = \
                decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

# initialize decoder, optimizer and loss function
decoder = DecoderRNN(HIDDEN_SIZE, len(vocab))
decoder_optimizer = optim.SGD(decoder.parameters(), lr=LEARNING_RATE)
criterion = nn.NLLLoss()

training_losses = []
for iter in range(1, NUM_ITERS + 1):
    training_pair = training_pairs[iter - 1]
    input_text = training_pair[0]
    target_text = training_pair[1]

    loss = train(vocab, input_text, target_text, bert_model, bert_tokenizer,
                 decoder, decoder_optimizer, criterion)
    training_losses.append(loss)

### WRITE TESTING PAIRS FUNCTION
# testing_pairs = [random.choice(pairs) for _ in range(1000)]
# for pair in testing_pairs:
#     guess = evaluate(lang, encoder, decoder, pair[0])
#     guess = ' '.join(guess)
#     logging.info(pair[0] + ',' + guess + ',' + pair[1] + ',' + \
#                     str(int(guess == pair[1])))

# attn_decoder = AttnDecoderRNN(HIDDEN_SIZE, lang.n_words, dropout_p=0.1).to(device)
# trainIters(lang, bert_model, bert_tokenizer, attn_decoder, NUM_ITERS)
# testing_pairs = [random.choice(pairs) for _ in range(1000)]
# for pair in testing_pairs:
#     guess = evaluate(lang, encoder, attn_decoder, pair[0])
#     guess = ' '.join(guess)
#     logging.info(pair[0] + ',' + guess + ',' + pair[1] + ',' + \
#                     str(int(guess == pair[1])))
