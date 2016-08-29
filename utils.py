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

def context_win(l, win):
    '''
    win : int corresponding to the size of the window
    given a list of indexes composing a sentence
    l : array containing the word indexes

    it will return a list of indexes corresponding to contex windows surrounding
    each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i+win)] for i in range(len(l))]

    assert len(out) == len(l)
    return np.array(out, dtype=np.int32)

def load_data(fpath = './corpus/train.utf8', wind_size=7):
    '''
    0(B) : for a character located at the beginng of a word.
    1(I) : for a character inside of a word.
    2(E) : for a character at the end of a word.
    3(S) : for a character that is a word by itself.
    '''
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

def convert_predicts_to_segments(predicts, seq):
    assert len(predicts) == len(seq)
    i = 0
    segs = []
    while(i < len(seq)):
        if predicts[i] == 3:
            segs.append(seq[i])
            i = i + 1
        elif predicts[i] == 0:
            j = i + 1
            while(j < len(seq) and predicts[j] != 2):
                j += 1
            if j == len(seq):
                segs.append(seq[i:j])
            else:
                segs.append(seq[i:j+1])
            i = j + 1
    return segs

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
