#!/usr/bin/env python3
# Copyright Mycroft AI, Inc. 2017. All Rights Reserved.
import math
import sys
import numpy as np


def create_model():
    from keras.models import Sequential
    from keras.layers.core import Dense
    from keras.layers.recurrent import SimpleRNN, LSTM, GRU

    model = Sequential()
    eval('model.add(' + network_type + '(' + str(
        units) + ', stateful=True, batch_input_shape=(1, 1, 1)))')
    model.add(Dense(1, activation='linear'))
    model.compile("rmsprop", "mse")
    return model


def to_in(x):
    return np.array([x]).reshape(1, 1, 1)


def to_out(y):
    return np.array([y]).reshape(1, 1)


# Load arguments
def arg(default):
    arg.n += 1
    if len(sys.argv) > arg.n:
        return sys.argv[arg.n]
    else:
        return default
arg.n = 0

if len(sys.argv) == 1 or '-h' in sys.argv[1:]:
    print('Usage:', sys.argv[0], 'SimpleRNN|LSTM|GRU', '[units]', '[epochs]', '[resolution]', '[should_reset]')
    exit(0)

network_type = arg('LSTM')
units = int(arg('10'))
epochs = int(arg('40'))
resolution = int(arg('60'))
should_reset = bool(arg('False'))

# Generate training data
train_x = np.zeros(resolution)
train_y = np.zeros(resolution)

for i in range(resolution):
    train_x[i] = math.sin(2 * math.pi * (i - 1) / resolution)
    train_y[i] = math.sin(2 * math.pi * i / resolution)

# Create and train
model = create_model()
for epoch in range(epochs):
    print('Training epoch ' + str(epoch) + '...')
    if should_reset:
        model.reset_states()

    for tx, ty in zip(train_x, train_y):
        model.fit(to_in(tx), to_out(ty), verbose=0, batch_size=1, epochs=1)

if should_reset:
    model.reset_states()

# Test
print('\n=== Correct ===')
for i in train_y:
    print(i)

print('\n=== Prediction ===')
for i in train_x[:len(train_x) // 2]:
    prev = model.predict(to_in(i))[0][0]
    print(prev)

for _ in range(len(train_x) // 2, len(train_x)):
    prev = model.predict(to_in(prev))[0][0]
    print(prev)
