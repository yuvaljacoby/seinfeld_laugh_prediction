from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import SeparableConv1D

from compare_models import *
from train_utils import *
from seinfeld_playground import *

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

    for _ in range(layers - 1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model


def train_ngram_model(train_df, test_df, train_texts, train_labels, val_texts, val_labels, learning_rate=1e-3,
                      epochs=100,
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
        print(conv1.shape)
        conv2 = SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same')(conv1)
        print(conv2.shape)
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
        char_one_hot = Input(shape=(num_additional_features,), name='char_num')
        concat_layer = Concatenate()([affine1, char_one_hot])
        # affine2 = Dense(op_units, activation='relu')(concat_layer)
        output_layer = Dense(op_units, activation=op_activation, name='final_output')(concat_layer)
        model = tf.keras.Model(inputs=[sequence_input, char_one_hot], outputs=output_layer)
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

    context_vector, attention_weights = Attention(64)(lstm, state_h)

    lstm_pred = tf.keras.layers.Dense(1, activation='sigmoid')(context_vector)

    if use_additional_features:
        char_one_hot = Input(shape=(num_additional_features,), name='char_num')
        x = Concatenate()([context_vector, char_one_hot])
        affine1 = tf.keras.layers.Dense(64, activation='relu')(x)
    else:
        affine1 = tf.keras.layers.Dense(64, activation='relu')(context_vector)
    affine2 = tf.keras.layers.Dense(64, activation='relu')(affine1)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='final_output')(affine2)
    if use_additional_features:
        model = tf.keras.Model(inputs=[sequence_input, char_one_hot], outputs=[output, lstm_pred])
    else:
        model = tf.keras.Model(inputs=sequence_input, outputs=[output, lstm_pred])
    return model


def train_sequence_model(model, x_train, x_val, y_train, y_val, additional_features_train=None,
                         additional_features_val=None, multiple_outputs=False, batch_size=32, learning_rate=1e-3, epochs=100):
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
    # loss = 'mean_squared_error'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    if multiple_outputs:
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'], loss_weights=[1, 0.2])
    else:
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'], loss_weights=[1])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    # Train and validate model.
    if additional_features_train is not None:
        if multiple_outputs:
            history = model.fit([x_train, additional_features_train], [y_train, y_train], epochs=epochs, callbacks=callbacks,
                                validation_data=([x_val, additional_features_val], [y_val, y_val]),
                                verbose=2, batch_size=batch_size)
        else:
            history = model.fit([x_train, additional_features_train], y_train, epochs=epochs, callbacks=callbacks,
                    validation_data=([x_val, additional_features_val], y_val),
                    verbose=2, batch_size=batch_size)
    else:
        if multiple_outputs:
            history = model.fit(x_train, [y_train, y_train], epochs=epochs, callbacks=callbacks,
                                validation_data=(x_val, [y_val, y_val]),
                                verbose=2, batch_size=batch_size)
        else:
            history = model.fit(x_train, y_train, epochs=epochs, callbacks=callbacks,
                                validation_data=(x_val, y_val),
                                verbose=2, batch_size=batch_size)
    # Print results.
    history = history.history
    if additional_features_train is not None:
        if multiple_outputs:
            print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['final_output_acc'][-1], loss=history['final_output_loss'][-1]))
            result = model.evaluate([x_val, additional_features_val], [y_val, y_val])
            print(result)
            return history['final_output_acc'][-1], history['final_output_loss'][-1], model
        else:
            print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_acc'][-1], loss=history['val_loss'][-1]))
            result = model.evaluate([x_val, additional_features_val], y_val)
            print(result)
            return history['val_acc'][-1], history['val_loss'][-1], model
    else:
        if multiple_outputs:
            print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['final_output_acc'][-1], loss=history['final_output_loss'][-1]))
            result = model.evaluate(x_val, [y_val, y_val])
            print(result)
            return history['final_output_acc'][-1], history['final_output_loss'][-1], model
        else:
            print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_acc'][-1], loss=history['val_loss'][-1]))
            result = model.evaluate(x_val, y_val)
            print(result)
            return history['val_acc'][-1], history['val_loss'][-1], model


def get_sequence_data(df_train, df_test, add_character=False):
    # split to train and test
    # train_df = df.sample(frac=0.8, random_state=200)
    # test_df = df.drop(train_df.index)

    # Vectorize texts.
    x_train, x_val, tokenizer_index = sequence_vectorize(df_train.txt, df_test.txt)
    y_train = df_train.is_funny
    y_val = df_test.is_funny

    if add_character:
        char_encoding_train = np.zeros(x_train.shape[0])
        char_encoding_train[df_train.character == 'JERRY'] = len(tokenizer_index) + 1
        char_encoding_train[df_train.character == 'KRAMER'] = len(tokenizer_index) + 2
        char_encoding_train[df_train.character == 'GEORGE'] = len(tokenizer_index) + 3
        char_encoding_train[df_train.character == 'ELAINE'] = len(tokenizer_index) + 4
        x_train = np.hstack([x_train, char_encoding_train.reshape((-1, 1))])

        char_encoding_test = np.zeros(x_val.shape[0])
        char_encoding_test[df_test.character == 'JERRY'] = len(tokenizer_index) + 1
        char_encoding_test[df_test.character == 'KRAMER'] = len(tokenizer_index) + 2
        char_encoding_test[df_test.character == 'GEORGE'] = len(tokenizer_index) + 3
        char_encoding_test[df_test.character == 'ELAINE'] = len(tokenizer_index) + 4
        x_val = np.hstack([x_val, char_encoding_test.reshape((-1, 1))])

    return tokenizer_index, x_train, x_val, y_train, y_val


if __name__ == "__main__":
    # load corpus
    show = False
    df = load_corpus()
    df_scene = getSceneData(df)
    df_train, df_test = split_train_test(df, 0.2)
    additional_features_train = np.zeros((df_train.shape[0], 6))
    additional_features_train[df_train.character == "JERRY", 0] = 1
    additional_features_train[df_train.character == "GEORGE", 1] = 1
    additional_features_train[df_train.character == "ELAINE", 2] = 1
    additional_features_train[df_train.character == "KRAMER", 3] = 1
    additional_features_train[:, 4] = df_train.start
    additional_features_train[:, 5] = df_train.length

    additional_features_val = np.zeros((df_test.shape[0], 6))
    additional_features_val[df_test.character == "JERRY", 0] = 1
    additional_features_val[df_test.character == "GEORGE", 1] = 1
    additional_features_val[df_test.character == "ELAINE", 2] = 1
    additional_features_val[df_test.character == "KRAMER", 3] = 1
    additional_features_val[:, 4] = df_test.start
    additional_features_val[:, 5] = df_test.length

    print("Preparing sequential data")
    tokenizer_index, x_train, x_val, y_train, y_val = get_sequence_data(df_train, df_test)
    print("Getting embedding")
    embedding_matrix = get_glove_embedding(len(tokenizer_index) + 1, tokenizer_index)
    print("Training cnn model")
    use_chars_CNN = True
    # Create model instance.
    model_cnn = sepcnn_model(blocks=3,
                             filters=32,
                             kernel_size=5,
                             embedding_dim=100,
                             dropout_rate=0.3,
                             pool_size=2,
                             input_shape=x_train.shape[1:],
                             num_features=len(tokenizer_index)+1,
                             num_additional_features=additional_features_train.shape[1],
                             embedding_matrix=embedding_matrix,
                             use_pretrained_embedding=True,
                             is_embedding_trainable=True,
                             use_additional_features=use_chars_CNN)
    history_val_acc_cnn, history_val_loss_cnn, model_cnn_fit = train_sequence_model(model_cnn,
                                                                                    x_train,
                                                                                    x_val,
                                                                                    y_train,
                                                                                    y_val,
                                                                                    batch_size=32,
                                                                                    epochs=10,
                                                                                    multiple_outputs=False,
                                                                                    additional_features_train=additional_features_train,
                                                                                    additional_features_val=additional_features_val)
    print("Finish training cnn model")
    if use_chars_CNN:
        y_hat_val_cnn = model_cnn_fit.predict([x_val,additional_features_val])
    else:
        y_hat_val_cnn = model_cnn_fit.predict(x_val)
    compare_models_roc_curve(y_val, [y_hat_val_cnn], ['CNN'], show)
    if show:
        plot_confusion_matrix(y_val, [y_hat_val_cnn], ['CNN'])

    from basic_trainers import Model_OneHotEncoding
    print("Training LSTM model")
    model_lstm = LSTM_model(embedding_dim=100,
                            dropout_rate=0.3,
                            input_shape=x_train.shape[1:],
                            num_features=len(tokenizer_index) + 1,
                            num_additional_features=additional_features_train.shape[1],
                            embedding_matrix=embedding_matrix,
                            use_pretrained_embedding=True,
                            is_embedding_trainable=True)

    # history_val_acc, history_val_loss = train_ngram_model(train_df, test_df train_df.txt, train_df.is_funny.astype(np.float32), test_df.txt, test_df.is_funny.astype(np.float32))
    history_val_acc_lstm, history_val_loss_lstm, model_lstm_fit = train_sequence_model(model_lstm,
                                                                                       x_train,
                                                                                       x_val,
                                                                                       y_train,
                                                                                       y_val,
                                                                                       additional_features_train=additional_features_train,
                                                                                       additional_features_val=additional_features_val,
                                                                                       batch_size=32,
                                                                                       epochs=10,
                                                                                       multiple_outputs=True)

    y_hat_val_lstm = model_lstm_fit.predict([x_val, additional_features_val])[0]
    compare_models_roc_curve(y_val, [y_hat_val_lstm], ['lstm'], show)
    if show:
        plot_confusion_matrix(y_val, [y_hat_val_lstm], ['lstm'])

    print("Finished training LSTM\n\n")


    from basic_trainers import Model_OneHotEncoding
    print("Training LSTM model")
    model_lstm_no_char = LSTM_model(embedding_dim=100,
                            dropout_rate=0.3,
                            input_shape=x_train.shape[1:],
                            num_features=len(tokenizer_index) + 1,
                            num_additional_features=additional_features_train.shape[1],
                            embedding_matrix=embedding_matrix,
                            use_pretrained_embedding=True,
                            is_embedding_trainable=True,
                            use_additional_features=False)

    # history_val_acc, history_val_loss = train_ngram_model(train_df, test_df train_df.txt, train_df.is_funny.astype(np.float32), test_df.txt, test_df.is_funny.astype(np.float32))
    history_val_acc_lstm_no_char, history_val_loss_lstm_no_char, model_lstm_no_char_fit = train_sequence_model(model_lstm_no_char,
                                                                                                               x_train,
                                                                                                               x_val,
                                                                                                               y_train,
                                                                                                               y_val,
                                                                                                               batch_size=32,
                                                                                                               epochs=10,
                                                                                                               multiple_outputs=True)

    y_hat_val_lstm_no_char = model_lstm_no_char_fit.predict(x_val)[0]
    compare_models_roc_curve(y_val, [y_hat_val_lstm_no_char], ['lstm_no_char'], show)
    if show:
        plot_confusion_matrix(y_val, [y_hat_val_lstm_no_char], ['lstm_no_char'])

    print("Finished training LSTM no char \n\n")

    from basic_trainers import Model_OneHotEncoding

    y_hats, labels = Model_OneHotEncoding(df_train, df_test)


    y_hats.append(y_hat_val_lstm)
    labels.append('LSTM')

    y_hats.append(y_hat_val_lstm_no_char)
    labels.append('LSTM_no_char')

    y_hats.append(y_hat_val_cnn)
    labels.append('CNN')
    print("Finished all training\n\n")

    auc = compare_models_roc_curve(y_val, y_hats, labels)
    print(auc)
    plot_confusion_matrix(y_val, y_hats, labels)

