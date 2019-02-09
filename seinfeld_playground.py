from seinfeld_laugh_corpus import corpus
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer
import gensim


def create_index_usingduplicated(df, grouping_cols):
    df.sort_values(grouping_cols, inplace=True)
    duplicated = df.duplicated(subset=grouping_cols, keep='first')
    new_group = ~duplicated
    return new_group.cumsum()

def load_corpus():
    seinfeld = corpus.load(fold_laughs=True)
    num_episodes = len(seinfeld.screenplays)
    episodes_len = [len(seinfeld.screenplays[i].lines) for i in np.arange(num_episodes)]

    characters = np.array([seinfeld.screenplays[i][j].character for i in np.arange(num_episodes) for j in np.arange(episodes_len[i])])
    txt = np.array([seinfeld.screenplays[i][j].txt for i in np.arange(num_episodes) for j in np.arange(episodes_len[i])])
    num_words = np.array([len(seinfeld.screenplays[i][j].txt.split()) for i in np.arange(num_episodes) for j in np.arange(episodes_len[i])])
    start = np.array([seinfeld.screenplays[i][j].start for i in np.arange(num_episodes) for j in np.arange(episodes_len[i])])
    end = np.array([seinfeld.screenplays[i][j].end for i in np.arange(num_episodes) for j in np.arange(episodes_len[i])])
    is_funny = np.array([seinfeld.screenplays[i][j].is_funny for i in np.arange(num_episodes) for j in np.arange(episodes_len[i])])
    laugh_time = np.array([seinfeld.screenplays[i][j].laugh_time for i in np.arange(num_episodes) for j in np.arange(episodes_len[i])])
    episode_name = np.array([seinfeld.screenplays[i].episode_name for i in np.arange(num_episodes) for j in np.arange(episodes_len[i])])
    episode_num = np.array([seinfeld.screenplays[i].episode for i in np.arange(num_episodes) for j in np.arange(episodes_len[i])])
    season = np.array([seinfeld.screenplays[i].season for i in np.arange(num_episodes) for j in np.arange(episodes_len[i])])
    line_num = np.array([j for i in np.arange(num_episodes) for j in np.arange(episodes_len[i])])
    total_lines_in_episode = np.array([episodes_len[i] for i in np.arange(num_episodes) for j in np.arange(episodes_len[i])])

    df = pd.DataFrame({'character': characters,
                       'txt': txt,
                       'num_words': num_words,
                       'start': start.astype(np.float),
                       'end': end.astype(np.float),
                       'length': (end.astype(np.float) - start.astype(np.float)),
                       'is_funny': is_funny.astype(np.bool),
                       'laugh_time': laugh_time.astype(np.float),
                       'episode_num': episode_num.astype(np.int),
                       'line_num': line_num.astype(np.int),
                       'episode_name': episode_name,
                       'season': season.astype(np.int),
                       'total_lines': total_lines_in_episode.astype(np.int)})

    df = df.sort_values(by=['season', 'episode_num', 'start']).reset_index(drop=True)

    df['global_episode_num'] = create_index_usingduplicated(df, ['season', 'episode_num'])
    return df


def plotStuff(df_to_plot):
    # sentence time length by character and by funniness
    g = sns.FacetGrid(df_to_plot, col='character', row='is_funny')
    g.map(sns.distplot, "length", bins=50, color='b')

    # sentence length by character and by funniness
    g2 = sns.FacetGrid(df_to_plot, col='character', row='is_funny')
    g2.map(sns.distplot, "num_words", bins=50, color='b')

    # sentence time length vs laugh_time dist by character
    g3 = sns.FacetGrid(df_to_plot, col='character')
    g3.map(sns.kdeplot, "length", "laugh_time", shaded=True)

    # sentence length vs laugh_time dist by character
    g3 = sns.FacetGrid(df_to_plot, col='character')
    g3.map(sns.kdeplot, "num_words", "laugh_time", shaded=True)
    plt.show()


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


def getTrigramEncoding(text_array):
    freq = CountVectorizer(ngram_range=(3, 3), analyzer='char_wb') # trigram
    corpus_trigrams = freq.fit_transform(text_array)

    onehot = Binarizer()
    corpus_trigrams_one_hot = onehot.fit_transform(corpus_trigrams.toarray())

    return freq, corpus_trigrams_one_hot


def getWord2Vec(text_array, min_count=5, window_size=5, model_size=250, clean=False):
    """
    Method that handles the cleaning, tokenizing of the corpus and training of the model on that corpus.
    :param text_array: The corpus, as an array of sentences
    :param min_count: How many times a word must appear to be included
    :param window_size: `window` is the maximum distance between the current and predicted word within a sentence.
    :param model_size: the dimensionality of the feature vectors.
    :param clean: Whether to clean the corpus
    :return:
    """
    if clean:
        corpus_for_word2vec = text_array
        # TODO
        # corpus_for_word2vec = clean_corpus(text_array)
    else:
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


def getSceneData(df):
    df.loc[:, 'time_from_prev'] = np.array([0] + [df.start[i] - df.end[i - 1]
                                           if df.episode_num[i] == df.episode_num[i - 1] else 0
                                           for i in range(1, len(df.start))])

    # notebook with histogram that "explains" why 1.8
    df.loc[:, 'new_scene'] = df['time_from_prev'] > 1.8
    # Other than that, if it's a new episode we want new scene, used heuristic for that
    df.loc[(df.time_from_prev == 0) & (df.start <= 2), 'new_scene'] = True

    # Calculate features
    text_for_scene = []
    charcteres_scene = []
    number_rows_scene = []
    scene_number_in_episode = []
    scene_counter = 0
    prev_episode = None
    for i, row in df.iterrows():
        if row['new_scene']:
            text_for_scene.append(row.txt)
            charcteres_scene.append(set([row.character]))
            number_rows_scene.append(1)

            if prev_episode != row.episode_num:
                scene_counter = 0

            scene_counter += 1
            scene_number_in_episode.append(scene_counter)
            prev_episode = row.episode_num
        else:
            text_for_scene[-1] += "\n" + row.txt
            charcteres_scene[-1].add(row.character)
            number_rows_scene[-1] += 1
            scene_number_in_episode.append(scene_counter)

    # append for each row the scene properties that we have
    df.loc[:, 'scene_text'] = [text_for_scene[i] for i in range(len(number_rows_scene)) for _ in range(number_rows_scene[i])]
    df.loc[:, 'scene_characters'] = [charcteres_scene[i] for i in range(len(number_rows_scene))
                              for _ in range(number_rows_scene[i])]
    df.loc[:, 'n_scene_characters'] = df.scene_characters.str.len()
    df.loc[:, 'scene_number_in_episode'] = scene_number_in_episode
    df.loc[:, 'global_scene_number'] = create_index_usingduplicated(df, ['global_episode_num', 'scene_number_in_episode'])
    return df


if __name__ == "__main__":
    df = load_corpus()
    df_main = df[df['character'].isin(["JERRY", "ELAINE", "KRAMER", "GEORGE"])]
    # model = getWord2Vec(df.txt)
    print('here')
    # plotStuff(df_main)
    freq, corpus_one_hot = getOneHotEncoding(df.txt)
    print(corpus_one_hot.shape)
    # freq_trigrams, corpus_trigrams_one_hot = getTrigramEncoding(df.txt)




"""
Implementation ideas:
1. Implement simple tfIdf model and classify sentence on its own
2. Use word2vec instead and a CNN only.
3. Use word2vec and CNN + LSTM model to use history of sequential sentences
4. Use word2vec + other handcrafter features and CNN + LSTM
5. Use transformer

Visualization ideas:
1. Show different insights with simple plots of funniness vs time/character etc
2. Show TSNE of corpus
3. Try and find different clusters depending on character or subject
4. Try and visualize attention mechanism in LSTM
"""