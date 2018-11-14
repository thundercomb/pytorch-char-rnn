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
#              Under the hood the script actually detects variance, and then converts it
#              to a similarity score for convenience. The lower the detected variance,
#              the more like the original text the provided text is.
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

def check_probability(x, cni):
    x_last = x.view(-1)
    reverse_sorted_x = torch.sort(x_last, descending=True)[0]
    cx = (reverse_sorted_x == x_last[cni]).nonzero()

    return cx

def get_variance(model, text):
    with torch.no_grad():
        hidden = model.create_hidden(1)
    last_char = "."
    inp = index_to_tensor(char_to_index[last_char])

    probabilities = torch.tensor([]).to(torch.long)
    if use_cuda:
        probabilities = probabilities.cuda()
    for char in text[1:]:
        if use_cuda:
            hidden = hidden.cuda()
            inp = inp.cuda()
        output, hidden = model(inp, hidden)

        if char in char_to_index:
            cni = torch.tensor([char_to_index[char]])
            output_data = output.data.to(torch.float64)
            if use_cuda:
                output_data = output_data.cuda()
            char_probability = check_probability(output_data, cni)
        else:
            char_probability = torch.tensor([[len(char_to_index) + 1]])

        if use_cuda:
            char_probability = char_probability.cuda()
        probabilities = torch.cat((probabilities, char_probability[0]))

        if char in char_to_index:
            inp = index_to_tensor(char_to_index[char])
        else:
            inp = inp

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

use_cuda = torch.cuda.is_available()

model = RNN(chars_len, hidden_size, chars_len, n_layers, 0.5)
model.load_state_dict(checkpoint['model'])
total_variance = get_variance(model, args.text)
variance = [ float(v) / len(char_to_index) for v in total_variance ]
average_similarity_percentage = 100 - (float(sum(variance)) / float(len(variance)) * 100)

if debug:
    print("List of individual variances: ")
    for v in variance:
        print(v)

print("Detected similarity: " + str(round(average_similarity_percentage,2)) + "%")
