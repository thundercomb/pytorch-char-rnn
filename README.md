# Syntax Char RNN

## Introduction

This version of Char RNN [as described by Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) builds on the [work done by Kyle Kastner and Kyle McDonald](https://gist.github.com/kastnerkyle/e7ca55807a7f4db811d830acf4ee75aa).

For an attempt to introduce context encodings into Char RNN data preparation see [Syntax Char RNN](https://github.com/thundercomb/pytorch-syntax-char-rnn) or its companion [blog post](https://thecombedthunderclap.blogspot.com/2018/02/syntax-char-rnn-for-context-encoding.html).

## Installation

Make sure you have a recent version of Python 2.7

```
git clone https://github.com/thundercomb/pytorch-char-rnn
cd pytorch-char-rnn
```

## Input

Provide a text file, eg. ```poetry.txt```. You might like to create a data directory.

```
data/poetry.txt
```

## Run

### Train

The train program creates the data structure and trains the neural network on chunked batches of the encoded text.

The train program takes ten arguments:  
```--seq_length``` : sequence length (default=50)  
```--batch_size``` : minibatch size (default=50)  
```--rnn_size``` : hidden state size (default=128)  
```--num_layers``` : number of rnn layers (default=2)  
```--max_epochs``` : maximum number of epochs (default=10)  
```--learning_rate``` : learning rate (default=2e-3)  
```--dropout``` : dropout (default=0.5)  
```--seed``` : seeding character(s) (default='a')  
```--input_dir``` : input directory (default='data/austen')  
```--output_dir``` : output directory (default='data/austen')  

The train program outputs checkpoints after each epoch, or whenever a better training loss under 1.3 is achieved.

```
python2.7 train.py --input data/poetry.txt --output checkpoints/ --max_epochs 200 --seq_length 135 --rnn_size 256
```

### Generate

The generate program samples a model loaded from a specified checkpoint and prints the results.

The generate program takes five arguments:  
```--temperature``` : number between 0 and 1; lower means more conservative predictions for sampling (default=0.8)  
```--sample_len``` : number of characters to sample (default=500)  
```--checkpoint``` : checkpoint model to load  
```--seed``` : seeding string to prime the sample
```--charfile``` : character encoding file
```--input_dir``` : input directory containing from which to load data (default='data/austen')  

The generate program outputs sampled text to the terminal.

```
python2.7 generate.py --checkpoint checkpoints/checkpoint_ep_0.789.cp --sample_len 2000 --charfile data/poetry_chars.pkl --temperature 1
```

## Contact

Feel free to comment, raise issues, or create pull requests. You can also reach out to me on Twitter [@thundercomb](https://twitter.com/thundercomb). 

## License

The software is licensed under the terms of the [GNU Public License v2](http://github.com/thundercomb/poetrydb/LICENSE.txt). It basically means you can reuse and modify this software as you please, as long as the resulting program(s) remain open and licensed in the same way.
