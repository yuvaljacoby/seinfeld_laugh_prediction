import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest

from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
import tensorflow as tf

# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 20K features.
TOP_K = 40000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2

# Limit on the length of text sequences. Sequences longer than this
# will be truncated.
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

    df_train = scene_permutation(df_train)
    df_test = scene_permutation(df_test)
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

class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
