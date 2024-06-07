import pandas as pd
import numpy as np
import re
from unicodedata import normalize
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Masking
from keras_nlp.layers import TokenAndPositionEmbedding, TransformerEncoder, TransformerDecoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
import joblib

print("Initialising...")
print("This program is a test run of a transformer-based neural network used to translate English to Indonesian...\n")
print("Importing data\n")

# Load the dataset
df = pd.read_csv("english_indonesian_subtitles.csv")

# Drop the first row and reset index
df.drop([0], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop_duplicates(inplace=True)

# Cleaning text
print("Cleaning text...\n")

def clean_text(text):
    if isinstance(text, str):
        text = normalize('NFD', text.lower())
        text = re.sub('[^A-Za-z ]+', '', text)
    return text

def clean_and_prepare_text(text):
    text = '[start] ' + clean_text(text) + ' [end]'
    return text if isinstance(text, str) else ''  # Return an empty string if text is not a string

# Apply the cleaning functions to the DataFrame
df['id'] = df['id'].apply(clean_and_prepare_text)
df['en'] = df['en'].apply(clean_text)
df.head()

en = df['en']
indonesian = df['id']
print("\nCalculating maximum phrase length")

en_max_len = 106
id_max_len = 112
sequence_len = 112

print(f'Max phrase length (English): {en_max_len}')
print(f'Max phrase length (Indonesian): {id_max_len}')
print(f'Sequence length: {sequence_len}\n')

# Filter out non-string values and convert to lowercase
en_cleaned = [text.lower() for text in en if isinstance(text, str)]
id_cleaned = [text.lower() for text in indonesian if isinstance(text, str)]

# Tokenize and pad sequences
en_tokenizer = Tokenizer()
en_tokenizer.fit_on_texts(en_cleaned)
en_sequences = en_tokenizer.texts_to_sequences(en_cleaned)
en_x = pad_sequences(en_sequences, maxlen=sequence_len, padding='post')

id_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')
id_tokenizer.fit_on_texts(id_cleaned)
id_sequences = id_tokenizer.texts_to_sequences(id_cleaned)
id_y = pad_sequences(id_sequences, maxlen=sequence_len + 1, padding='post')

# Calculate the vocabulary sizes from the tokenizer instances
en_vocab_size = len(en_tokenizer.word_index) + 1
id_vocab_size = len(id_tokenizer.word_index) + 1

print("Calculating vocabulary size")
print(f'Vocabulary size (English): {en_vocab_size}')
print(f'Vocabulary size (Indonesian): {id_vocab_size}\n')

print("Quickly clearing RAM\n")
gc.collect()

# Define model parameters
num_heads = 1
embed_dim = 32

# Define the model architecture
encoder_input = Input(shape=(None,), dtype='int64', name='encoder_input')
decoder_input = Input(shape=(None,), dtype='int64', name='decoder_input')

# Token and Position Embedding layer should be defined elsewhere in the code
x_enc = TokenAndPositionEmbedding(en_vocab_size, sequence_len, embed_dim)(encoder_input)
encoder_output = TransformerEncoder(embed_dim, num_heads)(x_enc)

# Adding Masking to handle padded sequences
x_dec = TokenAndPositionEmbedding(id_vocab_size, sequence_len, embed_dim)(decoder_input)
masked_x_dec = Masking(mask_value=0)(x_dec)

x_dec = TransformerDecoder(embed_dim, num_heads)(masked_x_dec, encoder_output)
x_dec = Dropout(0.4)(x_dec)

decoder_output = Dense(id_vocab_size, activation='softmax')(x_dec)

decoder = Model([decoder_input, encoder_output], decoder_output)
decoder_output = decoder([decoder_input, encoder_output])

model = Model([encoder_input, decoder_input], decoder_output)

# Define the SGD optimizer with a reasonable learning rate
sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=sgd_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary(line_length=120)

# Assuming en_x and id_y are already defined
inputs = {'encoder_input': en_x, 'decoder_input': id_y[:, :-1]}
outputs = id_y[:, 1:]

batch_size = 32  # Choose an appropriate batch size
steps_per_epoch = len(en_x) // batch_size

# Fit the model with steps_per_epoch
callback = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
hist = model.fit(inputs, outputs, epochs=10, validation_split=0.2, callbacks=[callback], batch_size=batch_size)

# Saving the model
model.save("english_indonesian_translate_reduced.h5")

