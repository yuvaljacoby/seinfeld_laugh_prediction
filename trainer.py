from seinfeld_playground import load_corpus
from train_utils import *
from compare_models import *

import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

from sklearn.feature_selection import f_classif

from tensorflow.python.keras import models
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Concatenate

# module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# Import the Universal Sentence Encoder's TF Hub module
# embed = hub.Module(module_url)

def mlp_model(layers, units, dropout_rate, input_shape):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.

    # Returns
        An MLP model instance.
    """
    op_units, op_activation = 1, 'sigmoid'
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model

def train_ngram_model(train_df, test_df, train_texts, train_labels, val_texts, val_labels, learning_rate=1e-3, epochs=100,
                      batch_size=128, layers=2, units=64, dropout_rate=0.2):
    """Trains n-gram model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """

    # Verify that validation labels are in the same range as training labels.
    num_classes = 2

    # Vectorize texts.
    x_train, x_val = ngram_vectorize(train_texts, train_labels, val_texts)

    # Create model instance.
    model = mlp_model(layers=layers,
                      units=units,
                      dropout_rate=dropout_rate,
                      input_shape=x_train.shape[1:])

    # Compile model with learning parameters.
    loss = 'binary_crossentropy'

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(x_train, train_labels, epochs=epochs, callbacks=callbacks, validation_data=(x_val, val_labels),
                        verbose=2, batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('tfidf_mlp_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]

def sepcnn_model(blocks,
                 filters,
                 kernel_size,
                 embedding_dim,
                 dropout_rate,
                 pool_size,
                 input_shape,
                 num_features,
                 use_pretrained_embedding=False,
                 is_embedding_trainable=False,
                 embedding_matrix=None):
    """Creates an instance of a separable CNN model.

    # Arguments
        blocks: int, number of pairs of sepCNN and pooling blocks in the model.
        filters: int, output dimension of the layers.
        kernel_size: int, length of the convolution window.
        embedding_dim: int, dimension of the embedding vectors.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        pool_size: int, factor by which to downscale input at MaxPooling layer.
        input_shape: tuple, shape of input to the model.
        num_features: int, number of words (embedding input dimension).
        use_pretrained_embedding: bool, true if pre-trained embedding is on.
        is_embedding_trainable: bool, true if embedding layer is trainable.
        embedding_matrix: dict, dictionary with embedding coefficients.

    # Returns
        A sepCNN model instance.
    """
    op_units, op_activation = 1, 'sigmoid'
    model = models.Sequential()

    # Add embedding layer. If pre-trained embedding is used add weights to the
    # embeddings layer and set trainable to input is_embedding_trainable flag.
    if use_pretrained_embedding:
        model.add(Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0],
                            weights=[embedding_matrix],
                            trainable=is_embedding_trainable))
    else:
        model.add(Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0]))

    for _ in range(blocks-1):
        model.add(Dropout(rate=dropout_rate))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(MaxPooling1D(pool_size=pool_size))

    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(op_units, activation=op_activation))
    return model


def CNN_LSTM_model(embedding_dim,
                 dropout_rate,
                 input_shape,
                 num_features,
                 use_pretrained_embedding=False,
                 is_embedding_trainable=False,
                 embedding_matrix=None):

    # def UniversalEmbedding(x):
    #     return embed(tf.squeeze(tf.cast(x, tf.string)),
    #                  signature="default", as_dict=True)["default"]

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH-1,), dtype='int32')

    # embedded_sequences = tf.keras.layers.Lambda(UniversalEmbedding, output_shape=(512,))(sequence_input)

    embedded_sequences = tf.keras.layers.Embedding(num_features, embedding_dim, input_length=MAX_SEQUENCE_LENGTH-1)(sequence_input)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM
                                         (128,
                                          dropout=0.3,
                                          return_sequences=True,
                                          return_state=True,
                                          recurrent_activation='relu',
                                          recurrent_initializer='glorot_uniform'), name="bi_lstm_0")(embedded_sequences)

    lstm, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,
                                                                                      dropout=0.2,
                                                                                      return_sequences=True,
                                                                                      return_state=True,
                                                                                      recurrent_activation='relu',
                                                                                      recurrent_initializer='glorot_uniform'))(lstm)


    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])

    context_vector, attention_weights = Attention(64)(lstm, state_h)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(context_vector)

    model = tf.keras.Model(inputs=sequence_input, outputs=output)

    return model


def train_sequence_model(model, x_train, x_val, y_train, y_val, batch_size=32, learning_rate=1e-3, epochs=100):
    """Trains n-gram model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """

    # Compile model with learning parameters.
    loss = 'binary_crossentropy'

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(x_train, y_train, epochs=epochs, callbacks=callbacks, validation_data=(x_val, y_val),
                        verbose=2, batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    result = model.evaluate(x_val, y_val)
    print(result)
    # Save model.
    #model.save('tfidf_sepCNN_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1], model


def get_sequence_data(df, add_character=False):
    # split to train and test
    train_df = df.sample(frac=0.8, random_state=200)
    test_df = df.drop(train_df.index)

    # Vectorize texts.
    x_train, x_val, tokenizer_index = sequence_vectorize(train_df.txt, test_df.txt)
    y_train = train_df.is_funny
    y_val = test_df.is_funny

    if add_character:
        char_encoding_train = np.zeros(x_train.shape[0])
        char_encoding_train[train_df.character == 'JERRY'] = len(tokenizer_index) + 1
        char_encoding_train[train_df.character == 'KRAMER'] = len(tokenizer_index) + 2
        char_encoding_train[train_df.character == 'GEORGE'] = len(tokenizer_index) + 3
        char_encoding_train[train_df.character == 'ELAINE'] = len(tokenizer_index) + 4
        x_train = np.hstack([x_train, char_encoding_train.reshape((-1, 1))])

        char_encoding_test = np.zeros(x_val.shape[0])
        char_encoding_test[test_df.character == 'JERRY'] = len(tokenizer_index) + 1
        char_encoding_test[test_df.character == 'KRAMER'] = len(tokenizer_index) + 2
        char_encoding_test[test_df.character == 'GEORGE'] = len(tokenizer_index) + 3
        char_encoding_test[test_df.character == 'ELAINE'] = len(tokenizer_index) + 4
        x_val = np.hstack([x_val, char_encoding_test.reshape((-1, 1))])

    return tokenizer_index, x_train, x_val, y_train, y_val, train_df, test_df

if __name__ == "__main__":
    # load corpus
    df = load_corpus()

    tokenizer_index, x_train, x_val, y_train, y_val, train_df, test_df = get_sequence_data(df)

    # Create model instance.
    # model = sepcnn_model(blocks=3,
    #                      filters=32,
    #                      kernel_size=3,
    #                      embedding_dim=128,
    #                      dropout_rate=0.3,
    #                      pool_size=2,
    #                      input_shape=x_train.shape[1:],
    #                      num_features=len(tokenizer_index)+1)

    model = CNN_LSTM_model(embedding_dim=128,
                         dropout_rate=0.3,
                         input_shape=x_train.shape[1:],
                         num_features=len(tokenizer_index)+1)

    # history_val_acc, history_val_loss = train_ngram_model(train_df, test_df train_df.txt, train_df.is_funny.astype(np.float32), test_df.txt, test_df.is_funny.astype(np.float32))
    history_val_acc_sepCNN, history_val_loss_sepCNN, model_fit = train_sequence_model(model,
                                                                         x_train,
                                                                         x_val,
                                                                         y_train,
                                                                         y_val,
                                                                         batch_size=200)
    y_hat_val = model_fit.predict(x_val)


    compare_models_roc_curve(y_val, [y_hat_val], ['lstm'])
    plot_confusion_matrix(y_val, [y_hat_val], ['lstm'])
