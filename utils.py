import os
import re
import csv
from collections import defaultdict
import numpy as np
from lstm import LSTM

def load_vocabulary(fpath = './runs/vocab'):
    vocabulary = {}
    with open(fpath, 'r') as f:
        for line in f:
            split = line.strip('\n').split(' ')
            vocabulary[split[0]] = int(split[1])
    return vocabulary

def context_win(words_index, wind_size):
    '''
    wind_size : int
      corresponding to the size of the window given a list of indexes composing a sentence
    words_index : list
      array containing words index

    Return a list of indexes corresponding to contex windows surrounding each word
    in the sentence
    '''
    assert (wind_size % 2) == 1
    assert wind_size >= 1
    words_index = list(words_index)

    lpadded = wind_size // 2 * [-1] + words_index + wind_size // 2 * [-1]
    out = [lpadded[i:(i+wind_size)] for i in range(len(words_index))]

    assert len(out) == len(words_index)
    return np.array(out, dtype=np.int32)

Status = ['B', 'M', 'E', 'S']

def load_data(fpath = './corpus/train.utf8', wind_size=7):

    X_train, Y_train = [], []
    vocabulary = load_vocabulary()
    with open(fpath) as f:
        for line in f:
            split = re.split(r'\s+', line.strip())
            y = []
            for word in split:
                length = len(word)
                if length == 1:
                    y.append(3)
                else:
                    y.extend([0] + [1]*(length-2) + [2])
            newline = ''.join(split)
            x = [vocabulary[char] if vocabulary.get(char) else 0 for char in newline]
            X_train.append(context_win(x, wind_size))
            Y_train.append(y)

    return X_train, Y_train, vocabulary

def train_with_sgd(model, X_train, y_train, learning_rate=0.001, \
    nepoch=20, callback_every=10000, callback=None):

    num_example_seen = 0
    for epoch in range(nepoch):
        # For each training example...
        for i in np.random.permutation(len(y_train)):
            # One SGD step
            if len(y_train[i]) < 3:
                continue;
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_example_seen += 1
            if (callback and callback_every and \
                num_example_seen % callback_every == 0):
                callback(model, num_example_seen)
    return model

def convert_predict_to_pos(predicts):
    pos_list = [Status[p] for p in predicts]
    return pos_list

def segment(predicts, sentence):
    pos_list = convert_predict_to_pos(predicts)
    assert len(pos_list) == len(sentence)
    words = []
    begin, nexti = 0, 0
    for i, char in enumerate(sentence):
        pos = pos_list[i]
        if pos == 'B':
            begin = i
        elif pos == 'E':
            words += [sentence[begin:i+1]]
            nexti = i + 1
        elif pos == 'S':
            words += [char]
            nexti = i + 1
    if nexti < len(sentence):
        words += [sentence[nexti:]]
    return words

def load_model(floder, modelClass, hyperparams):
    print("loading model from %s." % floder)
    print("...")

    E = np.load(os.path.join(floder, 'E.npy'))
    U = np.load(os.path.join(floder, 'U.npy'))
    W = np.load(os.path.join(floder, 'W.npy'))
    V = np.load(os.path.join(floder, 'V.npy'))
    b = np.load(os.path.join(floder, 'b.npy'))
    c = np.load(os.path.join(floder, 'c.npy'))

    hidden_dim = hyperparams['hidden_dim']
    embedding_dim = hyperparams['embedding_dim']
    vocab_size = hyperparams['vocab_size']
    num_clas = hyperparams['num_clas']
    wind_size = hyperparams['wind_size']

    model = modelClass(embedding_dim, hidden_dim, num_clas, wind_size, vocab_size)
    model.E.set_value(E)
    model.U.set_value(U)
    model.W.set_value(W)
    model.V.set_value(V)
    model.b.set_value(b)
    model.c.set_value(c)
    print("lstm model has been loaded.")
    return model
