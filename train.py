# Special thanks to Kyle McDonald, this is based on his example
# https://gist.github.com/kylemcdonald/2d06dc736789f0b329e11d504e8dee9f
# Thanks to Laurent Dinh for examples of parameter saving/loading in PyTorch
# Thanks to Sean Robertson for https://github.com/spro/practical-pytorch

from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
import torch

import numpy as np
import math
import os
import argparse
import pickle

from rnn import *

parser = argparse.ArgumentParser(description='PyTorch char-rnn')
parser.add_argument('--seq_length', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--rnn_size', type=int, default=128)
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=2e-3)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--input', '-i', type=str)
parser.add_argument('--output', '-o', type=str, default='checkpoints')
parser.add_argument('--seed', type=str, default='a')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

# randomise runs
torch.manual_seed(np.random.randint(1,9999))
random_state = np.random.RandomState(np.random.randint(1,9999))

seq_length = args.seq_length
batch_size = args.batch_size
hidden_size = args.rnn_size
epoch_count = args.max_epochs
n_layers = args.num_layers
lr = args.learning_rate
dropout = args.dropout
input_filename = args.input
checkpoint_prepend = os.path.join(args.output, 'checkpoint_')
final_checkpoint_prepend = os.path.join(args.output, 'final_checkpoint_')

with open(input_filename, 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
# Save chars to charfile for re-use in generate process
charfile = os.path.splitext(input_filename)[0] + '_chars.pkl'
with open(charfile, 'wb') as f:
    pickle.dump(chars, f)
chars_len = len(chars)
char_to_index = {}
index_to_char = {}
for i, c in enumerate(chars):
    char_to_index[c] = i
    index_to_char[i] = c

def chunks(l, n):
    for i in range(0, len(l) - n, n):
        yield l[i:i + n]

def index_to_tensor(index):
    tensor = torch.zeros(1, 1).long()
    tensor[0,0] = index
    return Variable(tensor)

def train():
    # convert all characters to indices
    batches = [char_to_index[char] for char in text]

    # chunk into sequences of length seq_length + 1
    batches = list(chunks(batches, seq_length + 1))

    # chunk sequences into batches
    batches = list(chunks(batches, batch_size))

    # convert batches to tensors and transpose
    # each batch is (sequence_length + 1) x batch_size
    batches = [torch.LongTensor(batch).transpose_(0, 1) for batch in batches]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    hidden = Variable(model.create_hidden(batch_size))

    if use_cuda:
        hidden = hidden.cuda()
        model.cuda()

    all_losses = []

    best_ep_loss = float('inf')
    try:
        epoch_progress = tqdm(range(1, epoch_count + 1))
        best_tl_loss = float('inf')
        for epoch in epoch_progress:
            random_state.shuffle(batches)

            batches_progress = tqdm(batches)
            best_loss = float('inf')
            for batch, batch_tensor in enumerate(batches_progress):
                if use_cuda:
                    batch_tensor = batch_tensor.cuda()

                # reset the model
                model.zero_grad()

                # everything except the last
                input_variable = Variable(batch_tensor[:-1])

                # everything except the first, flattened
                # what does this .contiguous() do?
                target_variable = Variable(batch_tensor[1:].contiguous().view(-1))

                # prediction and calculate loss
                output, _ = model(input_variable, hidden)
                loss = loss_function(output, target_variable)

                # backprop and optimize
                loss.backward()
                optimizer.step()

                loss = loss.data[0]
                best_tl_loss = min(best_tl_loss, loss)
                all_losses.append(loss)

                batches_progress.set_postfix(loss='{:.03f}'.format(loss))
                if loss < 1.3 and loss == best_tl_loss:
                    checkpoint_path = os.path.join(args.output, 'checkpoint_tl_')
                    checkpoint_path = checkpoint_path + str('{:.03f}'.format(loss)) + '.cp'
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, checkpoint_path)
   
            epoch_progress.set_postfix(loss='{:.03f}'.format(best_loss))
            best_ep_loss = min(best_ep_loss, loss)
            if loss == best_ep_loss:
                checkpoint_path = os.path.join(args.output, 'checkpoint_ep_')
                checkpoint_path = checkpoint_path + str('{:.03f}'.format(loss)) + '.cp'
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, checkpoint_path)

    except KeyboardInterrupt:
        pass

    # final save
    final_path = os.path.join(args.output, 'final_checkpoint_')
    final_path = final_path + str('{:.03f}'.format(loss)) + '.cp'
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, final_path)

model = RNN(chars_len, hidden_size, chars_len, n_layers, dropout)
train()
