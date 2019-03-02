import tensorflow as tf

def train_ngram_model(model, x_train, y_train, x_val, y_val, learning_rate=1e-3,
                      epochs=5, batch_size=128):
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
    print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

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
    try:
        if additional_features_train is not None:
            if multiple_outputs:
                try:
                    print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['final_output_acc'][-1], loss=history['final_output_loss'][-1]))
                    result = model.evaluate([x_val, additional_features_val], [y_val, y_val])
                    print(result)
                except KeyError:
                    print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_time_distributed_14_acc'][-1], loss=history['val_time_distributed_14_loss'][-1]))
                    result = model.evaluate([x_val, additional_features_val], [y_val, y_val])
                    print(result)
            else:
                print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_acc'][-1], loss=history['val_loss'][-1]))
                result = model.evaluate([x_val, additional_features_val], y_val)
                print(result)
        else:
            if multiple_outputs:
                try:
                    print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['final_output_acc'][-1], loss=history['final_output_loss'][-1]))
                    result = model.evaluate(x_val, [y_val, y_val])
                    print(result)
                except KeyError:
                    print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_time_distributed_14_acc'][-1], loss=history['val_time_distributed_14_loss'][-1]))
                    result = model.evaluate(x_val, [y_val, y_val])
                    print(result)
            else:
                print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_acc'][-1], loss=history['val_loss'][-1]))
                result = model.evaluate(x_val, y_val)
                print(result)
    except KeyError:
        pass
    return model

