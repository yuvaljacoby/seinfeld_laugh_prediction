import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import TimeDistributed
from sklearn.linear_model import LogisticRegression

from seinfeld_playground import *


def Model_OneHotEncoding(df_train, df_test):
    y_train = df_train.is_funny

    # encode the train text using OneHotEncoding
    cv, X_train = getOneHotEncoding(df_train.txt)
    X_test = cv.transform(df_test.txt)

    # Train LogisticRegression
    lr = LogisticRegression(solver='lbfgs', n_jobs=-1, max_iter=300)
    lr.fit(X_train, y_train)
    y_hat_lr = lr.predict_proba(X_test)
    return y_hat_lr[:, 1]


def mlp_model(input_shape, layers=2, units=64, dropout_rate=0.3):
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

    for _ in range(layers - 1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model

def sepcnn_model(blocks,
                 filters,
                 kernel_size,
                 embedding_dim,
                 dropout_rate,
                 pool_size,
                 input_shape,
                 num_features,
                 num_additional_features=None,
                 use_pretrained_embedding=False,
                 is_embedding_trainable=False,
                 embedding_matrix=None,
                 use_additional_features=True):
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

    sequence_input = Input(shape=(input_shape[0],), dtype='int32')

    # embedded_sequences = tf.keras.layers.Lambda(UniversalEmbedding, output_shape=(512,))(sequence_input)
    if use_pretrained_embedding is True:
        embedded_sequences = Embedding(num_features, embedding_matrix.shape[1], weights=[embedding_matrix],
                                                       input_length=input_shape[0], trainable=is_embedding_trainable)(sequence_input)
    else:
        embedded_sequences = Embedding(num_features, embedding_dim, input_length=input_shape[0])(sequence_input)

    print(embedded_sequences.shape)
    def conv_block(input_layer):
        dropout = Dropout(rate=dropout_rate)(input_layer)
        conv1 = SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same')(dropout)
        conv2 = SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same')(conv1)
        max_pool = MaxPooling1D(pool_size=pool_size)(conv2)

        return max_pool
    conv_blocks = [conv_block(embedded_sequences)]
    for i in range(blocks - 2):
        conv_blocks.append(conv_block(conv_blocks[i]))

    conv_a = SeparableConv1D(filters=filters * 2,
                          kernel_size=kernel_size,
                          activation='relu',
                          bias_initializer='random_uniform',
                          depthwise_initializer='random_uniform',
                          padding='same')(conv_blocks[-1])
    conv_b = SeparableConv1D(filters=filters * 2,
                          kernel_size=kernel_size,
                          activation='relu',
                          bias_initializer='random_uniform',
                          depthwise_initializer='random_uniform',
                          padding='same')(conv_a)
    avg_pool = GlobalAveragePooling1D()(conv_b)
    affine1 = Dense(64, activation='relu')(avg_pool)
    if use_additional_features:
        additional_features = Input(shape=(num_additional_features,), name='char_num')
        concat_layer = Concatenate()([affine1, additional_features])
        # affine2 = Dense(op_units, activation='relu')(concat_layer)
        output_layer = Dense(op_units, activation=op_activation, name='final_output')(concat_layer)
        model = tf.keras.Model(inputs=[sequence_input, additional_features], outputs=output_layer)
    else:
        output_layer = Dense(op_units, activation=op_activation)(affine1)
        model = tf.keras.Model(inputs=sequence_input, outputs=output_layer)
    return model


def LSTM_model(embedding_dim,
               dropout_rate,
               input_shape,
               num_features,
               num_additional_features=None,
               use_pretrained_embedding=False,
               is_embedding_trainable=False,
               embedding_matrix=None,
               use_additional_features=True):

    sequence_input = Input(shape=(input_shape[0],), dtype='int32')

    # embedded_sequences = tf.keras.layers.Lambda(UniversalEmbedding, output_shape=(512,))(sequence_input)
    if use_pretrained_embedding is True:
        embedded_sequences = Embedding(num_features, embedding_matrix.shape[1], weights=[embedding_matrix],
                                                       input_length=input_shape[0], trainable=is_embedding_trainable)(sequence_input)
    else:
        embedded_sequences = Embedding(num_features, embedding_dim, input_length=input_shape[0])(sequence_input)


    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM
                                         (128,
                                          dropout=dropout_rate,
                                          return_sequences=True,
                                          return_state=True,
                                          recurrent_activation='relu',
                                          recurrent_initializer='glorot_uniform'), name="bi_lstm_0")(embedded_sequences)

    lstm, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,
                                                                                                            dropout=dropout_rate,
                                                                                                            return_sequences=True,
                                                                                                            return_state=True,
                                                                                                            recurrent_activation='relu',
                                                                                                            recurrent_initializer='glorot_uniform'))(lstm)

    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])

    def attention_lambda(input_lambda):
        lstm_out = input_lambda[0]
        state = input_lambda[1]
        W1 = tf.keras.layers.Dense(64)
        W2 = tf.keras.layers.Dense(64)
        V = tf.keras.layers.Dense(1)

        hidden_with_time_axis = tf.keras.backend.expand_dims(state, 1)
        score = tf.keras.backend.tanh(W1(lstm_out) + W2(hidden_with_time_axis))
        attention_weights = tf.keras.layers.Softmax(axis=1)(V(score))
        context_vector = attention_weights * lstm_out
        context_vector = tf.keras.backend.sum(context_vector, axis=1)
        return context_vector, attention_weights

    context_vector, attention_weights = Lambda(attention_lambda)([lstm, state_h])

    lstm_pred = tf.keras.layers.Dense(1, activation='sigmoid')(context_vector)

    if use_additional_features:
        additional_features = Input(shape=(num_additional_features,), name='char_num')
        x = Concatenate()([context_vector, additional_features])
        affine1 = tf.keras.layers.Dense(64, activation='relu')(x)
    else:
        affine1 = tf.keras.layers.Dense(64, activation='relu')(context_vector)
    affine2 = tf.keras.layers.Dense(64, activation='relu')(affine1)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='final_output')(affine2)
    if use_additional_features:
        model = tf.keras.Model(inputs=[sequence_input, additional_features], outputs=[output, lstm_pred])
    else:
        model = tf.keras.Model(inputs=sequence_input, outputs=[output, lstm_pred])
    return model


def multiSentence_CNN_LSTM(blocks,
                          filters,
                          kernel_size,
                          embedding_dim,
                          dropout_rate,
                          pool_size,
                          input_shape,
                          num_features,
                          num_additional_features=None,
                          use_pretrained_embedding=False,
                          is_embedding_trainable=False,
                          embedding_matrix=None,
                          use_additional_features=True,
                          stateful=False):
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

    if stateful:
        sequence_input = Input(shape=(input_shape[0], input_shape[1],), batch_size=1, dtype='int32')
    else:
        sequence_input = Input(shape=(input_shape[0], input_shape[1],), dtype='int32')

    # embedded_sequences = tf.keras.layers.Lambda(UniversalEmbedding, output_shape=(512,))(sequence_input)
    if use_pretrained_embedding is True:
        embedded_sequences = TimeDistributed(Embedding(num_features, embedding_matrix.shape[1], weights=[embedding_matrix],
                                                            input_length=input_shape[1], trainable=is_embedding_trainable))(sequence_input)
    else:
        embedded_sequences = TimeDistributed(Embedding(num_features, embedding_dim, input_length=input_shape[1]))(sequence_input)

    print(embedded_sequences.shape)
    def conv_block(input_layer):
        dropout = Dropout(rate=dropout_rate)(input_layer)
        conv1 = TimeDistributed(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))(dropout)
        conv2 = TimeDistributed(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))(conv1)
        max_pool = TimeDistributed(MaxPooling1D(pool_size=pool_size))(conv2)

        return max_pool
    conv_blocks = [conv_block(embedded_sequences)]
    for i in range(blocks - 2):
        conv_blocks.append(conv_block(conv_blocks[i]))

    conv_a = TimeDistributed(SeparableConv1D(filters=filters * 2,
                          kernel_size=kernel_size,
                          activation='relu',
                          bias_initializer='random_uniform',
                          depthwise_initializer='random_uniform',
                          padding='same'))(conv_blocks[-1])
    conv_b = TimeDistributed(SeparableConv1D(filters=filters * 2,
                          kernel_size=kernel_size,
                          activation='relu',
                          bias_initializer='random_uniform',
                          depthwise_initializer='random_uniform',
                          padding='same'))(conv_a)
    avg_pool = TimeDistributed(GlobalAveragePooling1D())(conv_b)
    affine1 = TimeDistributed(Dense(64, activation='relu'))(avg_pool)

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM
                                         (128,
                                          dropout=dropout_rate,
                                          return_sequences=True,
                                          return_state=True,
                                          recurrent_activation='relu',
                                          recurrent_initializer='glorot_uniform',
                                          stateful=stateful), name="bi_lstm_0")(affine1)

    lstm, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,
                                                                                                            dropout=dropout_rate,
                                                                                                            return_sequences=True,
                                                                                                            return_state=True,
                                                                                                            recurrent_activation='relu',
                                                                                                            recurrent_initializer='glorot_uniform',
                                                                                                            stateful=stateful))(lstm)


    lstm_pred = TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))(lstm)

    if use_additional_features:
        # state_c = TimeDistributed(Concatenate())([forward_c, backward_c])
        # state_h = TimeDistributed(Concatenate())([forward_h, backward_h])
        # state_c = Concatenate()([forward_c, backward_c])
        # state_h = Concatenate()([forward_h, backward_h])
        # def attention_lambda(input_lambda):
        #     lstm_out = input_lambda[0]
        #     state = input_lambda[1]
        #     W1 = tf.keras.layers.Dense(64)
        #     W2 = tf.keras.layers.Dense(64)
        #     V = tf.keras.layers.Dense(1)
        #
        #     hidden_with_time_axis = tf.keras.backend.expand_dims(state, 1)
        #     score = tf.keras.backend.tanh(W1(lstm_out) + W2(hidden_with_time_axis))
        #     attention_weights = tf.keras.layers.Softmax(axis=1)(V(score))
        #     context_vector = attention_weights * lstm_out
        #     context_vector = tf.keras.backend.sum(context_vector, axis=1)
        #     return context_vector, attention_weights
        #
        # context_vector, attention_weights = Lambda(attention_lambda)([lstm, state_h])
        if stateful:
            additional_features = Input(shape=(input_shape[0], num_additional_features,), batch_size=1, name='additional_features')
        else:
            additional_features = Input(shape=(input_shape[0], num_additional_features,), name='additional_features')
        x = Concatenate(axis=2)([lstm, additional_features])
        affine1 = TimeDistributed(tf.keras.layers.Dense(64, activation='relu'))(x)
    else:
        affine1 = TimeDistributed(tf.keras.layers.Dense(64, activation='relu'))(lstm)
    affine2 = TimeDistributed(tf.keras.layers.Dense(64, activation='relu'))(affine1)
    output = TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid', name='final_output'))(affine2)
    if use_additional_features:
        model = tf.keras.Model(inputs=[sequence_input, additional_features], outputs=[output, lstm_pred])
    else:
        model = tf.keras.Model(inputs=sequence_input, outputs=[output, lstm_pred])
    return model

