import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

def sentences_to_indices(X, word_to_index, max_len):

    m = X.shape[0]

    X_indices = np.zeros((m, max_len))

    for i in range(m):

        sentence_words = X[i].lower().split()

        j = 0

        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j + 1

    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):

    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["cucumber"].shape[0]

    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(input_dim=vocab_len, output_dim=emb_dim, trainable=False)

    embedding_layer.build((None,))

    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def model_lstm(input_shape, word_to_vec_map, word_to_index, prob1=0.5, prob2=0.5):
    sentence_indices = Input(input_shape)

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    embeddings = embedding_layer(sentence_indices)

    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(prob1)(X)
    X = LSTM(128)(X)
    X = Dropout(prob2)(X)
    X = Dense(5)(X)
    X = Activation('softmax')(X)

    model = Model(inputs=sentence_indices, outputs=X)

    return model