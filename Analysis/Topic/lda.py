# @Author: Josef Klafka <jklafka>
# @Date:   2020-03-26T10:23:44-04:00
# @Email:  jlklafka@gmail.com
# @Project: Noisy-nets
# @Last modified by:   jklafka
# @Last modified time: 2020-03-26T10:44:14-04:00

from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary

texts = open("../Stimuli/eng-eng.txt").read().splitlines()
# take only the autoencoder target
texts = [text.split('\t')[1].split(' ') for text in texts]
text_dictionary = Dictionary(texts)
index_space = [text_dictionary.doc2bow(text) for text in texts]
lda = LdaModel(index_space, num_topics = 20)
