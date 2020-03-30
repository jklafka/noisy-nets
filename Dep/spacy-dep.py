# @Author: Josef Klafka <academic>
# @Date:   2020-03-30T11:59:48-04:00
# @Email:  jlklafka@gmail.com
# @Project: Noisy-nets
# @Last modified by:   academic
# @Last modified time: 2020-03-30T12:04:26-04:00

import spacy

# load model
nlp = spacy.load("en_core_web_sm")

doc1 = nlp("the cat sat on the mat")
doc2 = nlp("the the cat sat on the mat")

# print the children in each dep structure
print("Token", "Dep", "Head", "Children")
print("--------------------------------")
for token in doc1:
     print(token.text, token.dep_, token.head.text, [child for child in token.children])
for token in doc2:
     print(token.text, token.dep_, token.head.text, [child for child in token.children])

## visualize
from spacy import displacy
displacy.serve([doc1, doc2], style = "dep")
