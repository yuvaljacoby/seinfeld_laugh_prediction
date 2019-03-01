import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text

# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 40K features.
TOP_K = 40000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2

# Limit on the length of text sequences. Sequences longer than this
# will be truncated. Sequences shorter will be zero padded.
MAX_SEQUENCE_LENGTH = 20


def scene_permutation(df):
    scene_groups = df.groupby('global_scene_number').groups
    order = np.random.permutation(list(scene_groups.keys()))
    order_idx = []

    for i in order:
        order_idx = np.hstack((order_idx, scene_groups[i].values))

    return df.loc[order_idx, :].reset_index()


def split_train_test(df, test_ratio=0.2, seed=42):
    '''
    Split data to train and test based on episode --> episodes will be fully in train / test
    Then shuffle the scenes inside each split --> each scene will stay in order (but the scene after can from different time)
    Uses global_episode_num and global_scene_number, start
    :param df: df with features (using global_episode_num)
    :param test_ratio: float [0,1] ratio of samples to keep in test
    :param seed: int seed for randomness
    :return: df_train, df_test
    '''

    df = df.sort_values(by=['global_episode_num', 'global_scene_number', 'start'])
    np.random.seed(seed=seed)
    test_episode = np.random.choice(df.global_episode_num,
                                    size=int(len(df.global_episode_num.unique()) * test_ratio),
                                    replace=False)
    train_episode = set(df.global_episode_num) - set(test_episode)

    df_train = df[df.global_episode_num.isin(train_episode)]
    df_test = df[df.global_episode_num.isin(test_episode)]

    # df_train = scene_permutation(df_train)
    # df_test = scene_permutation(df_test)
    return df_train, df_test


def ngram_vectorize(train_texts, train_labels, val_texts):
    """Vectorizes texts as n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    return x_train, x_val


def sequence_vectorize(train_texts, val_texts):
    """Vectorizes texts as sequence vectors.

    1 text = 1 sequence vector with fixed length.

    # Arguments
        train_texts: list, training text strings.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val, word_index: vectorized training and validation
            texts and word index dictionary.
    """
    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    max_length = MAX_SEQUENCE_LENGTH

    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer.word_index


def get_glove_embedding(num_features, tokenizer_index):
        EMBEDDING_DIM=100
        embeddings_index = {}
        with open('glove.6B.100d.txt', 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        embedding_matrix = np.zeros((num_features, EMBEDDING_DIM))
        for word, i in tokenizer_index.items():
            embedding_vector = embeddings_index.get(word)
            # TODO: stem words
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix


def get_sequence_data(df_train, df_test):
    # Vectorize texts.
    x_train, x_val, tokenizer_index = sequence_vectorize(df_train.txt, df_test.txt)
    y_train = df_train.is_funny
    y_val = df_test.is_funny
    return tokenizer_index, x_train, x_val, y_train, y_val

def prepare_multi_sentence_data(x_train, x_val, y_train, y_val,
                                additional_ftrs_train, additional_ftrs_val, num_sentences=5):
    # prepare the data by taking each sentence and adding num_sentences after it
    x_train_multi = np.zeros((x_train.shape[0] - num_sentences, num_sentences, x_train.shape[1]))
    x_val_multi = np.zeros((x_val.shape[0] - num_sentences, num_sentences, x_val.shape[1]))

    y_train_multi = np.zeros((x_train.shape[0] - num_sentences, num_sentences, 1))
    y_val_multi = np.zeros((x_val.shape[0] - num_sentences, num_sentences, 1))

    additional_features_train_multi = np.zeros((additional_ftrs_train.shape[0] - num_sentences, num_sentences, additional_ftrs_train.shape[1]))
    additional_features_val_multi = np.zeros((additional_ftrs_val.shape[0] - num_sentences, num_sentences, additional_ftrs_val.shape[1]))

    for i in np.arange(x_train.shape[0]-num_sentences):
        x_train_multi[i] = x_train[i:i+num_sentences]
        y_train_multi[i] = np.array(y_train[i:i+num_sentences]).reshape((num_sentences,1))
        additional_features_train_multi[i] = additional_ftrs_train[i:i+num_sentences]
    for i in np.arange(x_val.shape[0]-num_sentences):
        x_val_multi[i] = x_val[i:i+num_sentences]
        y_val_multi[i] = np.array(y_val[i:i+num_sentences]).reshape((num_sentences,1))
        additional_features_val_multi[i] = additional_ftrs_val[i:i+num_sentences]
    return x_train_multi, x_val_multi , y_train_multi, y_val_multi, \
           additional_features_train_multi, additional_features_val_multi

def prepare_additional_ftrs(df):
    # add additional high level features to be concatenated in the model
    num_ftrs = 14
    additional_features = np.zeros((df.shape[0], num_ftrs))
    additional_features[df.character == "JERRY", 0] = 1
    additional_features[df.character == "GEORGE", 1] = 1
    additional_features[df.character == "ELAINE", 2] = 1
    additional_features[df.character == "KRAMER", 3] = 1
    additional_features[df.character == "NEWMAN", 4] = 1
    additional_features[df.character == "MORTY", 5] = 1
    additional_features[df.character == "FRANK", 6] = 1
    additional_features[df.character == "ESTELLE", 7] = 1
    additional_features[df.character == "HELEN", 8] = 1
    additional_features[:, 9] = df.start
    additional_features[:, 10] = df.length
    additional_features[:, 11] = df.num_words
    additional_features[:, 12] = df.length / df.num_words
    additional_features[:, 13] = df.avg_word_length
    # additional_features[:, 9] = df.n_scene_characters
    return additional_features
