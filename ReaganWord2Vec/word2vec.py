#!/usr/bin/python3.6

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

model = Word2Vec(corpus, size=5, window=5, min_count=50, workers=4)

vocab = list(model.wv.vocab)

print((vocab))


print('Training model...')
model.train(corpus, total_examples=len(corpus), epochs=5000)

#gov = (model.wv['program'])
#prg = (model.wv['government'])

#print(gov.dot(prg) )

#print(gov)
#print(prg)


print("---Most similar:")
print("---1: reagan, carter, peace")
print("---2: reagan, tax, carter")
print("---3: reagan, nation, carter")
print("---4: reagan, carter, peace")

print(model.most_similar(positive=['reagan', 'carter'], negative=['peace'], topn=5))
print(model.most_similar(positive=['reagan', 'tax'], negative=['carter'], topn=5))
print(model.most_similar(positive=['reagan', 'nation'], negative=['carter'], topn=5))
print(model.most_similar(positive=['reagan', 'carter'], negative=['peace'], topn=5))

print("---Dosent match:")
print("---1: smith, reagan, carter, but")
print("---2: smith, reagan, carter, energy")
print("---3: smith, reagan, carter, economic")
print("---4: smith, reagan, carter, program")
print("---5: military, reagan, policy, economic")

print(model.doesnt_match(('smith', 'reagan', 'carter', 'but')))
print(model.doesnt_match(('smith', 'reagan', 'carter', 'energy')))
print(model.doesnt_match(('smith', 'reagan', 'carter', 'economic')))
print(model.doesnt_match(('smith', 'reagan', 'carter', 'program')))
print(model.doesnt_match(('military', 'reagan', 'policy', 'economic')))

print("---similarity:")
print(model.similarity('country', 'inflation'))
print(model.similarity('country', 'states'))
print(model.similarity('country', 'country'))


#print("=====================")
#print(model.accuracy('reagan_eval.txt'))


#print(get_actions_from_file(sys.argv[1]))




