from torch.autograd import Variable
import torch.nn as nn
import torch

import numpy as np
import os
import re
import pickle
import argparse

from rnn import *

parser = argparse.ArgumentParser(description='PyTorch char-rnn')
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--sample_len', type=int, default=500)
parser.add_argument('--checkpoint', '-c', type=str)
parser.add_argument('--seed', type=str, default='a')
parser.add_argument('--charfile', '-f', type=str)
parser.add_argument('--concatenate', type=int, default=0)
args = parser.parse_args()

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

def uppercase_sentences(match):
    return match.group(1) + ' ' + match.group(2).upper()

def index_to_tensor(index):
    tensor = torch.zeros(1, 1).long()
    tensor[0,0] = index
    return Variable(tensor)

def manual_sample(x, temperature):
    x = x.reshape(-1).astype(np.float)
    x /= temperature
    x = np.exp(x)
    x /= np.sum(x)
    x = random_state.multinomial(1, x)
    x = np.argmax(x)
    return x.astype(np.int64)

def sample(model, prime_str, predict_len, temperature, concatenate):
    with torch.no_grad():
        hidden = model.create_hidden(1)
    prime_tensors = [index_to_tensor(char_to_index[char]) for char in prime_str]

    for prime_tensor in prime_tensors[-2:]:
        _, hidden = model(prime_tensor, hidden)

    inp = prime_tensors[-1]
    predicted = prime_str
    for p in range(predict_len):
        output, hidden = model(inp, hidden)

        # Sample from the network as a multinomial distribution
        # output_dist = output.data.view(-1).div(temperature).exp()
        # top_i = torch.multinomial(output_dist, 1)[0]

        # Alternative: use numpy
        top_i = manual_sample(output.data.numpy(), temperature)

        # Add predicted character to string and use as next input
        predicted_char = index_to_char[top_i]
        predicted += predicted_char
        inp = index_to_tensor(char_to_index[predicted_char])

    predicted = predicted.split(' ', 1)[1].capitalize()
    predicted = re.sub(r'([.?!]) ([a-z])', uppercase_sentences, predicted)
    predicted = re.sub(r'([.?!]\n)([a-z])', uppercase_sentences, predicted)
    predicted = re.sub(r'([.?!]\n *\n)([a-z])', uppercase_sentences, predicted)
    if predicted.find('.'):
        predicted = predicted[:predicted.rfind('.')+1]
    if concatenate == -1:
        predicted = re.sub(r'\n', ' ', predicted)
    return predicted

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
print(sample(model, args.seed, args.sample_len, args.temperature, args.concatenate))
