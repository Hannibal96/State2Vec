import sys
import re
import numpy as np
import torch
from gensim.models import Word2Vec

def get_sentences_from_file(file_name):
    text = open(file_name)
    sentences = list()
    for line in text:
        #print(line),
        if line[0] == '#':
            continue

        line = line.replace(",","")
        line = line.replace(".","")
        line = line.replace(";","")
        line = line.replace("?","")
        line = line.replace("!","")
        line = line.replace("-","")

        split_line = line.split()
        split_line = [x.lower() for x in split_line]
        sentences.append(split_line)
    return sentences


corpus = get_sentences_from_file('reagan.txt')

model = Word2Vec(corpus, size=10, window=5, min_count=50, workers=4)

vocab = list(model.wv.vocab)

print((vocab))


print('Training model...')
model.train(corpus, total_examples=len(corpus), epochs=500)

#gov = (model.wv['program'])
#prg = (model.wv['government'])

#print(gov.dot(prg) )

#print(gov)
#print(prg)

print("---Most similar:")
print(model.most_similar(positive=['i', 'you'], negative=['my'], topn=1))

print("---Dosent match:")
print(model.doesnt_match(('goverment', 'federal', 'tax', 'but')))

print("---similarity:")
print(model.similarity('country', 'inflation'))
print(model.similarity('country', 'states'))
print(model.similarity('country', 'country'))


#print("=====================")
#print(model.accuracy('reagan_eval.txt'))


#print(get_actions_from_file(sys.argv[1]))




