# Importing needed dependencies :
import re

import pandas as pd
import numpy as np
import string
from string import digits

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, CuDNNLSTM, Input, Embedding, TimeDistributed, Flatten, Dropout
from keras.callbacks import ModelCheckpoint

# -------------------------------------------------------------------------------------------------
# Character level mapping
# -------------------------------------------------------------------------------------------------
# 1. Read the text file into a list line by line as below
with open('tel.txt', 'r', encoding='utf-8-sig') as file:
    text = file.readlines()

# 2. Creating Dictionary

# Define empty list:
en_samples = []
de_samples = []

# Define empty sets to store the characters in them:
en_chars = set() # Holds English Character set
de_chars = set() # Holds French Character set

# Split the samples and get the character sets :
for line in text:
    en_, de_ = line.split('\t')
    de_ = '\t' + de_
    for char in de_:
        if char not in de_chars:
            de_chars.add(char)
    for char in en_:
        if char not in en_chars:
            en_chars.add(char)
    en_samples.append(en_)
    de_samples.append(de_)
# Add the chars \t and \n to the sets
de_chars.add('\n')
de_chars.add('\t')

en_chars.add('\n')
en_chars.add('\t')

# 3: Make the needed dictionaries to convert characters to integers and the opposite
de_char_to_int = dict()
de_int_to_char = dict()
en_char_to_int = dict()
en_int_to_char = dict()
for i, char in enumerate(de_chars):
    de_char_to_int[char] = i
    de_int_to_char[i] = char

for i, char in enumerate(en_chars):
    en_char_to_int[char] = i
    en_int_to_char[i] = char


#4.Compute the length of the longest sample and some other variables:
num_en_chars = len(en_chars)
num_de_chars = len(de_chars)

max_en_chars_per_sample = max([len(sample) for sample in en_samples])
max_de_chars_per_sample = max([len(sample) for sample in de_samples])

num_en_samples = len(en_samples)
num_de_samples = len(de_samples)

print(f'Number of D Samples  \t: {len(de_samples)}')
print(f'Number of E Samples \t: {len(en_samples)}')

print(f'Number of D Chars  \t: {len(de_chars)}')
print(f'Number of E Chars \t: {len(en_chars)}')

print(f'The Longest D Sample has {max_de_chars_per_sample} Chars')
print(f'The Longest E Sample has {max_en_chars_per_sample} Chars')

#5. initiate numpy arrays to hold the data that  seq2seq model will use:
encoder_input_data = np.zeros((num_en_samples,
                               max_en_chars_per_sample,
                               num_en_chars),
                              dtype='float32')

decoder_input_data = np.zeros((num_de_samples,
                               max_de_chars_per_sample,
                               num_de_chars),
                              dtype='float32')

target_data = np.zeros((num_de_samples,
                       max_de_chars_per_sample,
                       num_de_chars),
                      dtype='float32')

#6. One Hot Encode the samples by letter
for i, (en_sample, de_sample) in enumerate(zip(en_samples, de_samples)):
    for char, en_char in enumerate(en_sample):
        encoder_input_data[i, char, en_char_to_int[en_char]] = 1
    for char, de_char in enumerate(de_sample):
        decoder_input_data[i, char, de_char_to_int[de_char]] = 1
        if char > 0:
            target_data[i, char - 1, de_char_to_int[de_char]] = 1

print(f'Shape of encoder_input_data : {encoder_input_data.shape}')
print(f'Shape of decoder_input_data : {decoder_input_data.shape}')
print(f'Shape of target_data        : {target_data.shape}')

# ---------------------------------------------------------------------------------------------
# Word level mapping or dictionary creation called as embeding
# ---------------------------------------------------------------------------------------------
# 1. Read the text file into a list line by line as below
lines = pd.read_table('fra.txt',names=['en', 'de'], encoding='utf-8-sig')

# 2. Convert all lines into lower text
lines.en = lines.en.apply(lambda  x: x.lower())
lines.de = lines.de.apply(lambda  x: x.lower())

# 3. Try to handle comma's
lines.en=lines.en.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))
lines.de=lines.de.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))

# 4. handle punctuation by excluding it
exclude = set(string.punctuation)
lines.en=lines.en.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
lines.de=lines.de.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

# 5. handle the digits remove it
remove_digits = str.maketrans('', '', string.digits)
lines.en=lines.en.apply(lambda x: x.translate(remove_digits))
lines.de=lines.de.apply(lambda x: x.translate(remove_digits))

# 6. Append SOS and EOS to target data
lines.de = lines.de.apply(lambda x : 'SOS_ '+ x + ' _EOS')

# 7: Create word dictionaries :
en_words=set()
for line in lines.en:
    for word in line.split():
        if word not in en_words:
            en_words.add(word)

de_words=set()
for line in lines.de:
    for word in line.split():
        if word not in de_words:
            de_words.add(word)

# 8: find the lengths and sizes :
num_en_words = len(en_words)
num_de_words = len(de_words)

max_en_words_per_sample = max([len(sample.split()) for sample in lines.en])+5
max_de_words_per_sample = max([len(sample.split()) for sample in lines.de])+5

num_en_samples = len(lines.en)
num_de_samples = len(lines.de)

# 9: Get lists of words :
input_words = sorted(list(en_words))
target_words = sorted(list(de_words))

en_token_to_int = dict()
en_int_to_token = dict()

de_token_to_int = dict()
de_int_to_token = dict()

# 10: Tokenizing the words ( Convert them to numbers ) :
for i,token in enumerate(input_words):
    en_token_to_int[token]=i
    en_int_to_token[i]=token

for i,token in enumerate(target_words):
    de_token_to_int[token]=i
    de_int_to_token[i]=token

# 10: initiate numpy arrays to hold the data that our seq2seq model will use:
encoder_input_data = np.zeros((num_en_samples, max_en_words_per_sample), dtype='float32')
decoder_input_data = np.zeros((num_de_samples, max_de_words_per_sample), dtype='float32')
decoder_target_data = np.zeros((num_de_samples, max_de_words_per_sample, num_de_words),dtype='float32')

# 11: Go through the samples to obtain input, output and target data:
for i, (input_text, target_text) in enumerate(zip(lines.en, lines.de)):
    for t, word in enumerate(input_text.split()):
        encoder_input_data[i, t] = en_token_to_int[word]
    for t, word in enumerate(target_text.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = de_token_to_int[word]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, de_token_to_int[word]] = 1.

# -----------------------------------------------------------------------------------------------------------
# Defininig the Model
# -----------------------------------------------------------------------------------------------------------
# 1: Encoder Creation

# Defining some constants:
vec_len=300         # Length of the vector that we will get from the embedding layer
latent_dim=1024     # Hidden layers dimension
dropout_rate=0.2    # Rate of the dropout layers
batch_size=64       # Batch size
epochs=30           # Number of epochs

# Define an input sequence and process it.
# Input layer of the encoder :
encoder_input = Input(shape=(None,))

# Hidden layers of the encoder :
encoder_embedding=Embedding(input_dim = num_en_words, output_dim = vec_len)(encoder_input)
encoder_dropout=(TimeDistributed(Dropout(rate = dropout_rate)))(encoder_embedding)
encoder_LSTM=CuDNNLSTM(latent_dim, return_sequences=True)(encoder_dropout)

# Output layer of the encoder :
encoder_LSTM2_layer = CuDNNLSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_LSTM2_layer(encoder_LSTM)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# -----------------------------------------------------------------------------------------------------------
# Defininig the Decoder
# -----------------------------------------------------------------------------------------------------------
# Set up the decoder, using `encoder_states` as initial state.
# Input layer of the decoder :
decoder_input = Input(shape=(None,))

# Hidden layers of the decoder :
decoder_embedding_layer = Embedding(input_dim = num_de_words, output_dim = vec_len)
decoder_embedding = decoder_embedding_layer(decoder_input)

decoder_dropout_layer = (TimeDistributed(Dropout(rate = dropout_rate)))
decoder_dropout = decoder_dropout_layer(decoder_embedding)

decoder_LSTM_layer = CuDNNLSTM(latent_dim, return_sequences=True)
decoder_LSTM = decoder_LSTM_layer(decoder_dropout, initial_state = encoder_states)

decoder_LSTM_2_layer = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True)
decoder_LSTM_2,_,_ = decoder_LSTM_2_layer(decoder_LSTM)

# Output layer of the decoder :
decoder_dense = Dense(num_de_words, activation='softmax')
decoder_outputs = decoder_dense(decoder_LSTM_2)

# -----------------------------------------------------------------------------------------------------------
# Bringing Encoder and Decoder together
# -----------------------------------------------------------------------------------------------------------
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_input, decoder_input], decoder_outputs)

model.summary()

# Define a checkpoint callback :
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# Training the model

num_train_samples = 9000
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data[:num_train_samples,:],
               decoder_input_data[:num_train_samples,:]],
               decoder_target_data[:num_train_samples,:,:],
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.08,
          callbacks = callbacks_list)


