#========================================================================================
# Author:      thundercomb
# Description: Script to detect similarity between autoencoder model of a text and a text 
#              provided as input to the script. Provides a similarity score (percentage) 
#              indicating similarity of style and content.
#
#              As an example a sentence from the original modelled text should detect
#              high similarity, eg. 97% or 98%. A text in the same language, but written in
#              a very different style might score 92-95%. 
#              An input text written in a totally different language should score 
#              significantly lower, eg. 80-85%. 
#              And if the texts do not share many textual characters, the score will drop 
#              precipitously.
#
#              The higher the detected variance, the more like the original text the 
#              provided text is.
#             
#=========================================================================================

from torch.autograd import Variable
import torch.nn as nn
import torch

import numpy as np
import random
import os
import re
import pickle
import argparse

from rnn import *

parser = argparse.ArgumentParser(description='PyTorch char-rnn')
parser.add_argument('--text', type=str)
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--checkpoint', '-c', type=str)
parser.add_argument('--charfile', '-f', type=str)
parser.add_argument('--debug', '-d', default=False, action='store_true')
args = parser.parse_args()
debug = args.debug

with open(args.charfile, 'rb') as f:
    chars = pickle.load(f)

chars = sorted(list(set(chars)))
chars_len = len(chars)
char_to_index = {}
index_to_char = {}
for i, c in enumerate(chars):
    char_to_index[c] = i
    index_to_char[i] = c

random_state = np.random.RandomState(np.random.randint(1,9999))

def index_to_tensor(index):
    tensor = torch.zeros(1, 1).long()
    tensor[0,0] = index
    return Variable(tensor)

def check_probability(x, char, temperature):
    if char in char_to_index:
        cn = char_to_index[char]
    else:
        # if character not in our dict then assume it's a bad match
        return len(char_to_index) + 1

    if debug:
        print("Length of char_to_index: " + str(len(char_to_index)))
        print("Current char: " + char + " " + str(cn))
        print("Calculations x...")

    x = x.reshape(-1).astype(np.float)
    x /= temperature
    x = np.exp(x)
    x /= np.sum(x)

    if debug:
        print("Length of x: " + str(len(x)))
        print(x)
        for i, j in enumerate(x):
            print(i, j)

    xx = [sorted(x, reverse=True).index(i) for i in x]

    if debug:
        print(sorted(x, reverse=True))
        for i in sorted(x, reverse=True):
            xx.append(x.tolist().index(i))
        print("xx:" + str(xx))
        print("char: " + char + "  " + str(cn))
        print("PREDICTED PROBABILITY ORDER: " + str(xx[cn]))

    # xx contains the ordered values of char_to_index
    # i.e. a value '21' in first position means
    # the character at position 22 (21+1) in x was
    # the most probable, and so on

    xxx = x
    x = random_state.multinomial(1, x)

    if debug:
        print(x)

    x = np.argmax(x)

    if debug:
        print("Value: " + str(xxx[x]))
        print("Next char: " + index_to_char[x] + " " + str(x))
        print("Order of relevance: " + str(sorted(xxx, reverse=True).index(xxx[x])))
        print("End Manual Sample")

    #return x.astype(np.int64)
    return xx[cn]

def get_variance(model, text, temperature):
    with torch.no_grad():
        hidden = model.create_hidden(1)

    prime_tensors = [index_to_tensor(char_to_index[char]) for char in text[0]]

    for prime_tensor in prime_tensors[-2:]:
        _, hidden = model(prime_tensor, hidden)

    inp = prime_tensors[-1]
    probabilities = []
    for char in text[1:]:
        output, hidden = model(inp, hidden)
        probabilities.append(check_probability(output.data.numpy(), char, temperature))
        if char in char_to_index:
            inp = index_to_tensor(char_to_index[char])
        else:
            # if character not in char_to_index then it means it's a bad match for the model
            # yay! randomly pick the next seeding
            inp = index_to_tensor(char_to_index[random.choice(char_to_index.keys())])

    return probabilities

if os.path.exists(args.checkpoint):
    print('Parameters found at {}... loading'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
else:
    raise ValueError('File not found: {}'.format(args.checkpoint))

hidden_size = checkpoint['model']['encoder.weight'].size()[1]
n_layers = 0
for key in checkpoint['model'].keys():
    if 'cells.weight_hh' in key:
        n_layers = n_layers + 1

model = RNN(chars_len, hidden_size, chars_len, n_layers, 0.5)
model.load_state_dict(checkpoint['model'])
total_variance = get_variance(model, args.text, args.temperature)
variance = [ float(v) / len(char_to_index) for v in total_variance ]
average_similarity_percentage = 100 - (float(sum(variance)) / float(len(variance)) * 100)

if debug:
    print("List of individual variances: ")
    for v in variance:
        print(v)

print("Detected similarity: " + str(round(average_similarity_percentage,2)) + "%")