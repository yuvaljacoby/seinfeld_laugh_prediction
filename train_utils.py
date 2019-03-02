import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer

import gensim

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


def getWord2Vec(text_array, min_count=5, window_size=5, model_size=250):
    """
    Method that handles the cleaning, tokenizing of the corpus and training of the model on that corpus.
    :param text_array: The corpus, as an array of sentences
    :param min_count: How many times a word must appear to be included
    :param window_size: `window` is the maximum distance between the current and predicted word within a sentence.
    :param model_size: the dimensionality of the feature vectors.
    :param clean: Whether to clean the corpus
    :return:
    """
    corpus_for_word2vec = text_array
    corpus_for_word2vec = [sentence.split() for sentence in corpus_for_word2vec]
    print('Starting to train model')
    try:
        model = gensim.models.Word2Vec(corpus_for_word2vec, min_count=min_count,
                                            window=window_size, size=model_size, iter=50)
    except RuntimeError:
        print('No word appeared %d times, reran with min_count=1' % min_count)
        model = gensim.models.Word2Vec(corpus_for_word2vec, min_count=1,
                                            window=window_size, size=model_size, iter=50)
    return model


def getTrigramEncoding(text_array):
    freq = CountVectorizer(ngram_range=(3, 3), analyzer='char_wb') # trigram
    corpus_trigrams = freq.fit_transform(text_array)

    onehot = Binarizer()
    corpus_trigrams_one_hot = onehot.fit_transform(corpus_trigrams.toarray())

    return freq, corpus_trigrams_one_hot


def getOneHotEncoding(text_array, is_binary=True):
    '''
    Creatres a one hot encoding / binary encoding for a given array
    :param text_array: Array to encode
    :param is_binary: to return a OneHotEncoding or BinaryEncoding
    :return: vectorizer, array shape (text_array.shape[0], # unique words(text_array)
    '''
    cv = CountVectorizer(binary=is_binary)
    freq = cv.fit_transform(text_array)

    return cv, freq


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

def prepare_additional_ftrs(df, unique_chars):
    # add additional high level features to be concatenated in the model
    num_ftrs = unique_chars.shape[0] + 5
    additional_features = np.zeros((df.shape[0], num_ftrs))
    for i, char in enumerate(unique_chars):
        additional_features[df.character == char, i] = 1
    additional_features[:, 9] = df.start
    additional_features[:, 10] = df.length
    additional_features[:, 11] = df.num_words
    additional_features[:, 12] = df.length / df.num_words
    additional_features[:, 13] = df.avg_word_length
    return additional_features
