import torch, random, logging, argparse
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import BertTokenizer, BertModel


MAX_LENGTH = 10
HIDDEN_SIZE = 768 # same as BERT embedding
NUM_ITERS = 7500
BERT_LAYER = 11

SOS_token = 0
EOS_token = 1

# device = torch.device("cuda:0")
device = torch.device("cpu")


logging.basicConfig(filename = "noisy-results.csv", format="%(message)s", \
                    level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("language_file", help="Name of the (noised) .txt to use")
args = parser.parse_args()


def getBERTvocab(tokenizer, language_file):
    '''
    Get one-hot indices for each BERT-token in language_file.
    '''
    text = language_file.readlines()
    vocab = set()
    for line in text:
        tokens = tokenizer.tokenize(line)
        vocab = vocab || set(tokens)
    vocab = ["SOS", "EOS"] + list(vocab) # insert SOS and EOS tokens
    vocab = {word, index for word, index in enumerate(vocab)}
    return vocab

def tensorFromSentence(vocab, sentence):
    indexes = [SOS_token]
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

class DecoderRNN(nn.Module):
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

# class AttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
#         super(AttnDecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length
#
#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward(self, input, hidden, encoder_outputs):
#         embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.dropout(embedded)
#
#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#         print(input)
#         print(embedded)
#         print(hidden)
#         print(attn_weights)
#         print(encoder_outputs)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                  encoder_outputs.unsqueeze(0)[0])
#
#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)
#
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#
#         output = F.log_softmax(self.out(output[0]), dim=1)
#         return output, hidden, attn_weights
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)


def train(input_text, target_tensor, model, tokenizer, decoder, \
            decoder_optimizer, criterion, max_length=MAX_LENGTH):

    decoder_optimizer.zero_grad()
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

        topv, topi = decoder_output.topk(1)
        decoder_output = topi.squeeze().detach()

        # loss defined on one-hot vectors
        loss += criterion(decoder_output, decoder_input)
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


def trainIters(lang, encoder, tokenizer, decoder, n_iters, learning_rate=0.01):
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    for i in range(n_iters):
        pair_i = random.choice(pairs)
        training_pairs = [(pair_i[0], tensorFromSentence(lang, pair_i[1]))]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_text = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_text, target_tensor, encoder, tokenizer,
                     decoder, decoder_optimizer, criterion)


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


# Load pre-trained model tokenizer (vocabulary)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
bert_model = BertModel.from_pretrained('bert-base-uncased')
# bert_model.to('cuda')
bert_model.eval()

lang, pairs = prepareData(args.language_file, True)
decoder = DecoderRNN(HIDDEN_SIZE, lang.n_words)
trainIters(lang, bert_model, bert_tokenizer, decoder, NUM_ITERS)
testing_pairs = [random.choice(pairs) for _ in range(1000)]
for pair in testing_pairs:
    guess = evaluate(lang, encoder, decoder, pair[0])
    guess = ' '.join(guess)
    logging.info(pair[0] + ',' + guess + ',' + pair[1] + ',' + \
                    str(int(guess == pair[1])))

# attn_decoder = AttnDecoderRNN(HIDDEN_SIZE, lang.n_words, dropout_p=0.1).to(device)
# trainIters(lang, bert_model, bert_tokenizer, attn_decoder, NUM_ITERS)
# testing_pairs = [random.choice(pairs) for _ in range(1000)]
# for pair in testing_pairs:
#     guess = evaluate(lang, encoder, attn_decoder, pair[0])
#     guess = ' '.join(guess)
#     logging.info(pair[0] + ',' + guess + ',' + pair[1] + ',' + \
#                     str(int(guess == pair[1])))
