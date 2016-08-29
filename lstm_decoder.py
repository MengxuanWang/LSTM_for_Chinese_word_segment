# -*- coding: utf8 -*-
import sys
import os
import time
import numpy as np
from utils import *
from lstm import LSTM

# Model Hpyerparameters
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "50"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "4"))
WIN_SIZE = int(os.environ.get("WIN_SIZE", 7))

vocab = load_vocabulary()
floder = os.path.abspath(os.path.join(os.path.curdir, "runs"))
hyperparams = {
    "embedding_dim" : EMBEDDING_DIM,
    "hidden_dim" : HIDDEN_DIM,
    "num_clas" : NUM_CLASSES,
    "wind_size" : WIN_SIZE,
    "vocab_size" : len(vocab)
}

model = load_model(floder, LSTM, hyperparams)

def seg(seq):
    word_to_index = [vocab[char] if vocab.get(char) else 0 for char in seq]
    idxs = context_win(word_to_index, WIN_SIZE)
    predicts = model.predict_class(idxs)
    segs = convert_predicts_to_segments(predicts, seq)
    return segs

def segment_from_console():
    sentence = sys.stdin.readline()
    while sentence:
        sentence = sentence.strip('\n')
        result = seg(sentence)
        print(result)
        sentence = sys.stdin.readline()

#segment_from_console()
