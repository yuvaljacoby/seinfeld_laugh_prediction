import os
import argparse
import tensorflow as tf
from compare_models import *
from models import *
from trainers import *
from utils import *
from train_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path')
    parser.add_argument('--train_mlp', default=False, action='store_true')
    parser.add_argument('--train_CNN', default=False, action='store_true')
    parser.add_argument('--train_CNN_no_ftrs', default=False, action='store_true')
    parser.add_argument('--train_LSTM', default=False, action='store_true')
    parser.add_argument('--train_LSTM_no_ftrs', default=False, action='store_true')
    parser.add_argument('--train_Multi_LSTM', default=False, action='store_true')
    parser.add_argument('--load_models', default=False, action='store_true')
    parser.add_argument('--load_dfs', default=False, action='store_true')
    parser.add_argument('--run_predict', default=False, action='store_true')
    parser.add_argument('--plot_results', default=False, action='store_true')
    parser.add_argument('--save_predict', default=False, action='store_true')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    try:
        os.mkdir(args.out_path)
        os.mkdir("%s/figures"%args.out_path)
        os.mkdir("%s/model_predictions"%args.out_path)
        os.mkdir("%s/trained_models"%args.out_path)
    except OSError:
        print("Output directory already exists")

    if args.load_dfs:
        df_train = pd.read_csv('%s/df_train.csv'%args.out_path)
        df_test = pd.read_csv('%s/df_test.csv'%args.out_path)
        df = pd.concat(df_train, df_test)
    else:
        # load corpus
        df = load_corpus()
        # sort the df by scenes
        df_scene = getSceneData(df)
        # split the df to train and test (episode is either train or test)
        df_train, df_test = split_train_test(df_scene, 0.2)
        # save the dataframes for later analyzing
        df_train.to_csv('%s/df_train.csv'%args.out_path)
        df_test.to_csv('%s/df_test.csv'%args.out_path)

    if args.load_models:
        print("Loading models")
        model_cnn_lstm_multi_fit = tf.keras.models.load_model('%s/trained_models/lstm_multi_model.hdf5'%args.out_path)
        model_cnn_fit = tf.keras.models.load_model('%s/trained_models/cnn_model.hdf5'%args.out_path)
        model_cnn_no_ftrs_fit = tf.keras.models.load_model('%s/trained_models/cnn_model_no_ftrs.hdf5'%args.out_path)
        model_lstm_fit = tf.keras.models.load_model('%s/trained_models/lstm_model.hdf5'%args.out_path)
        model_lstm_no_ftrs_fit = tf.keras.models.load_model('%s/trained_models/lstm_no_ftrs_model.hdf5'%args.out_path)
        model_mlp_fit = tf.keras.models.load_model('%s/trained_models/mlp_model.hdf5'%args.out_path)
        args.train_CNN = True
        args.train_CNN_no_ftrs = True
        args.train_LSTM = True
        args.train_LSTM_no_ftrs = True
        args.train_mlp = True
        args.train_Multi_LSTM = True


    # prepare additional features for the models
    unique_chars = np.unique(df.character)
    additional_features_train = prepare_additional_ftrs(df_train, unique_chars)
    additional_features_val = prepare_additional_ftrs(df_test, unique_chars)

    print("Preparing sequential data")
    # prepare the data for sequence embedding
    tokenizer_index, x_train, x_val, y_train, y_val = get_sequence_data(df_train, df_test)
    print("Getting embedding")
    # load the glove embedding
    embedding_matrix = get_glove_embedding(len(tokenizer_index) + 1, tokenizer_index)

    y_s = []
    y_hats = []
    labels = []

    if args.train_mlp:
        print("Training MLP model")
        y_train_mlp = df_train.is_funny.astype(np.float32)
        y_val_mlp = df_test.is_funny.astype(np.float32)
        # Vectorize texts.
        x_train_mlp, x_val_mlp = ngram_vectorize(df_train.txt, y_train, df_test.txt)
        from scipy import sparse
        x_train_mlp = sparse.hstack((x_train_mlp, sparse.csr_matrix(additional_features_train))).A
        x_val_mlp = sparse.hstack((x_val_mlp, sparse.csr_matrix(additional_features_val))).A
        if not args.load_models:
            # Create model instance.
            mlp_model = mlp_model(input_shape=x_train_mlp.shape[1:])
            model_mlp_fit = train_ngram_model(mlp_model, x_train_mlp, y_train_mlp, x_val_mlp, y_val_mlp)
            print("Finish training mlp model")

            tf.keras.models.save_model(model_mlp_fit, '%s/trained_models/mlp_model.hdf5'%args.out_path, overwrite=True, include_optimizer=True)

        if args.run_predict:
            y_hat_val_MLP = model_mlp_fit.predict(x_val_mlp)
            y_hats.append(y_hat_val_MLP)
            y_s.append(y_val_mlp)
            labels.append('MLP')

    if args.train_Multi_LSTM:
        print("Training Multi Sentence LSTM model")
        # prepare data for the multi-sentence model
        num_sentences = 10
        x_train_multi, x_val_multi, y_train_multi, y_val_multi, \
        additional_features_train_multi, additional_features_val_multi = prepare_multi_sentence_data(x_train, x_val, y_train, y_val,
                                                                                                     additional_features_train, additional_features_val,
                                                                                                     num_sentences=num_sentences)
        if not args.load_models:
            model_cnn_lstm_multi = multiSentence_CNN_LSTM(blocks=3,
                                                          filters=64,
                                                          kernel_size=3,
                                                          embedding_dim=100,
                                                          dropout_rate=0.3,
                                                          pool_size=2,
                                                          input_shape=x_train_multi.shape[1:],
                                                          num_features=len(tokenizer_index)+1,
                                                          embedding_matrix=embedding_matrix,
                                                          use_pretrained_embedding=True,
                                                          is_embedding_trainable=True,
                                                          use_additional_features=True,
                                                          num_additional_features=additional_features_train.shape[1])

            model_cnn_lstm_multi_fit = train_sequence_model(model_cnn_lstm_multi,
                                                             x_train_multi,
                                                             x_val_multi,
                                                             y_train_multi,
                                                             y_val_multi,
                                                             batch_size=32,
                                                             epochs=10,
                                                             learning_rate=0.0001,
                                                             multiple_outputs=True,
                                                             additional_features_train=additional_features_train_multi,
                                                             additional_features_val=additional_features_val_multi)

            print("Finish training cnn lstm multi model")

            tf.keras.models.save_model(model_cnn_lstm_multi_fit, '%s/trained_models/lstm_multi_model.hdf5'%args.out_path, overwrite=True, include_optimizer=True)
        if args.run_predict:
            # for prediction here we make a "new" stateful model and copy the weights, this allows us to maintain history between predictions
            model_cnn_lstm_multi_stateful = multiSentence_CNN_LSTM(blocks=3, filters=64, kernel_size=3, embedding_dim=100, dropout_rate=0.5, pool_size=2, input_shape=(1, MAX_SEQUENCE_LENGTH), num_features=len(tokenizer_index)+1, embedding_matrix=embedding_matrix, use_pretrained_embedding=True, is_embedding_trainable=True, use_additional_features=True, num_additional_features=additional_features_train.shape[1], stateful=True)
            old_weights = model_cnn_lstm_multi_fit.get_weights()
            model_cnn_lstm_multi_stateful.set_weights(old_weights)
            # compile model
            model_cnn_lstm_multi_stateful.compile(loss='mean_squared_error', optimizer='adam')
            y_hat_val_cnn_lstm_multi = model_cnn_lstm_multi_stateful.predict([np.reshape(x_val, (-1, 1, MAX_SEQUENCE_LENGTH)), np.reshape(additional_features_val, (-1, 1, additional_features_val.shape[1]))], batch_size=1)[0][:, 0]
            y_hats.append(y_hat_val_cnn_lstm_multi)
            y_s.append(y_val)
            labels.append('LSTM_MULTI')

    if args.train_CNN:
        print("Training cnn model")
        if not args.load_models:
            # Create model instance.
            model_cnn = sepcnn_model(blocks=3,
                                     filters=32,
                                     kernel_size=5,
                                     embedding_dim=100,
                                     dropout_rate=0.5,
                                     pool_size=2,
                                     input_shape=x_train.shape[1:],
                                     num_features=len(tokenizer_index)+1,
                                     num_additional_features=additional_features_train.shape[1],
                                     embedding_matrix=embedding_matrix,
                                     use_pretrained_embedding=True,
                                     is_embedding_trainable=True,
                                     use_additional_features=True)
            model_cnn_fit = train_sequence_model(model_cnn,
                                                    x_train,
                                                    x_val,
                                                    y_train,
                                                    y_val,
                                                    batch_size=32,
                                                    epochs=5,
                                                    multiple_outputs=False,
                                                    additional_features_train=additional_features_train,
                                                    additional_features_val=additional_features_val)

            tf.keras.models.save_model(model_cnn_fit, '%s/trained_models/cnn_model.hdf5'%args.out_path, overwrite=True, include_optimizer=True)
        if args.run_predict:
            y_hat_val_cnn = model_cnn_fit.predict([x_val,additional_features_val])
            y_hats.append(y_hat_val_cnn)
            y_s.append(y_val)
            labels.append('CNN')
        print("Finish training cnn model")

    if args.train_CNN_no_ftrs:
        print("Training cnn no ftrs model")
        if not args.load_models:
            # Create model instance.
            model_cnn_no_ftrs = sepcnn_model(blocks=3,
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
                                             use_additional_features=False)
            model_cnn_no_ftrs_fit = train_sequence_model(model_cnn_no_ftrs,
                                                            x_train,
                                                            x_val,
                                                            y_train,
                                                            y_val,
                                                            batch_size=32,
                                                            epochs=5,
                                                            multiple_outputs=False)
            print("Finish training cnn no ftrs model")

            tf.keras.models.save_model(model_cnn_no_ftrs_fit, '%s/trained_models/cnn_model_no_ftrs.hdf5'%args.out_path, overwrite=True, include_optimizer=True)
        if args.run_predict:
            y_hat_val_cnn_no_ftrs = model_cnn_no_ftrs_fit.predict(x_val)
            y_hats.append(y_hat_val_cnn_no_ftrs)
            y_s.append(y_val)
            labels.append('CNN_no_ftrs')

    if args.train_LSTM:
        print("Training LSTM model")
        if not args.load_models:
            model_lstm = LSTM_model(embedding_dim=100,
                                    dropout_rate=0.3,
                                    input_shape=x_train.shape[1:],
                                    num_features=len(tokenizer_index) + 1,
                                    num_additional_features=additional_features_train.shape[1],
                                    embedding_matrix=embedding_matrix,
                                    use_pretrained_embedding=True,
                                    is_embedding_trainable=True,
                                    use_additional_features=True)

            model_lstm_fit = train_sequence_model(model_lstm,
                                                   x_train,
                                                   x_val,
                                                   y_train,
                                                   y_val,
                                                   additional_features_train=additional_features_train,
                                                   additional_features_val=additional_features_val,
                                                   batch_size=32,
                                                   epochs=5,
                                                   multiple_outputs=True)

            tf.keras.models.save_model(model_lstm_fit, '%s/trained_models/lstm_model.hdf5'%args.out_path, overwrite=True, include_optimizer=True)
        if args.run_predict:
            y_hat_val_lstm = model_lstm_fit.predict([x_val, additional_features_val])[0]
            y_hats.append(y_hat_val_lstm)
            y_s.append(y_val)
            labels.append('LSTM')

        print("Finished training LSTM\n\n")

    if args.train_LSTM_no_ftrs:
        print("Training LSTM no features model")
        if not args.load_models:
            model_lstm_no_ftrs = LSTM_model(embedding_dim=100,
                                            dropout_rate=0.3,
                                            input_shape=x_train.shape[1:],
                                            num_features=len(tokenizer_index) + 1,
                                            num_additional_features=additional_features_train.shape[1],
                                            embedding_matrix=embedding_matrix,
                                            use_pretrained_embedding=True,
                                            is_embedding_trainable=True,
                                            use_additional_features=False)

            model_lstm_no_ftrs_fit = train_sequence_model(model_lstm_no_ftrs,
                                                           x_train,
                                                           x_val,
                                                           y_train,
                                                           y_val,
                                                           batch_size=32,
                                                           epochs=5,
                                                           multiple_outputs=True)
            tf.keras.models.save_model(model_lstm_no_ftrs_fit,'%s/trained_models/lstm_no_ftrs_model.hdf5'%args.out_path, overwrite=True, include_optimizer=True)
        if args.run_predict:
            y_hat_val_lstm_no_ftrs = model_lstm_no_ftrs_fit.predict(x_val)[0]
            y_hats.append(y_hat_val_lstm_no_ftrs)
            labels.append('LSTM_no_ftrs')
            y_s.append(y_val)

        print("Finished training LSTM no ftrs \n\n")
    print("Finished all training\n\n")
    if args.run_predict:
        y_hat_val_logistic_regression = logisticRegressionModel(df_train, df_test)
        y_hats.append(y_hat_val_logistic_regression)
        labels.append('logistic regression')
        y_s.append(y_val)


    if args.save_predict:
        for i in np.arange(len(y_hats)):
            np.savetxt("%s/model_predictions/%s_labels.csv"%(args.out_path, labels[i]), y_s[i], delimiter=",")
            np.savetxt("%s/model_predictions/%s_predictions.csv"%(args.out_path, labels[i]), y_hats[i], delimiter=",")

    if args.plot_results:
        auc = compare_models_roc_curve(y_s, y_hats, labels, out_dir=args.out_path)
        print(auc)
        plot_confusion_matrix(y_s, y_hats, labels, out_dir=args.out_path)
