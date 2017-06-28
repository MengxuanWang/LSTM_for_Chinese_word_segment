import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
from utils import *
from lstm import LSTM

# Model Hyperparams
LEARNING_RATE =  float(os.environ.get("LEARNING_RATE", "0.001"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "50"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "4"))
WIND_SIZE = int(os.environ.get("WIND_SIZE", 7))

# Training parameters
NEPOCH = int(os.environ.get("NEPOCH", "20"))
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "15000"))

# load data
X_train, y_train, vocab = load_data('./corpus/msr_training.utf8')
X_test, y_test, vocab = load_data('./corpus/msr_test_gold.utf8')

# build model
model = LSTM(EMBEDDING_DIM, HIDDEN_DIM,\
    NUM_CLASSES, WIND_SIZE, len(vocab), bptt_truncate=-1)

# load trained model
# floder = os.path.abspath(os.path.join(os.path.curdir, "runs"))
# hyperparams = {
#     "embedding_dim" : EMBEDDING_DIM,
#     "hidden_dim" : HIDDEN_DIM,
#     "num_clas" : NUM_CLASSES,
#     "wind_size" : WIN_SIZE,
#     "vocab_size" : len(vocab)
# }
#
# model = load_model(floder, LSTM, hyperparams)

# callback
def sgd_callback(model, num_example_seen):
    loss = model.calculate_loss(X_test, y_test)
    print("\nnum_example_seen: %d" % (num_example_seen))
    print("-------------------------------------------")
    print("Loss: %f" % loss)
    print("\n")
    floder = os.path.abspath(os.path.join(os.path.curdir, "runs"))
    model.save_model(floder)
    sys.stdout.flush()

for epoch in range(NEPOCH):
    train_with_sgd(model, X_train, y_train, LEARNING_RATE, \
        nepoch=1, callback_every=PRINT_EVERY, callback=sgd_callback)
