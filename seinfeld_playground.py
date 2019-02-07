from seinfeld_laugh_corpus import corpus
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer

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


def getOneHotEncoding(text_array):
    freq = CountVectorizer()
    corpus_freq = freq.fit_transform(text_array)

    onehot = Binarizer()
    corpus_one_hot = onehot.fit_transform(corpus_freq.toarray())
    return freq, corpus_one_hot


def getTrigramEncoding(text_array):
    freq = CountVectorizer(ngram_range=(3, 3), analyzer='char_wb') # trigram
    corpus_trigrams = freq.fit_transform(text_array)

    onehot = Binarizer()
    corpus_trigrams_one_hot = onehot.fit_transform(corpus_trigrams.toarray())

    return freq, corpus_trigrams_one_hot

def getSceneData(df):
    df['time_from_prev'] = np.array(
        [0] + [df.start[i] - df.end[i - 1] if df.episode_num[i] == df.episode_num[i - 1] else 0
               for i in range(1, len(df.start))])

    # notebook with histogram that "explains" why 1.8
    df['new_scene'] = df['time_from_prev'] > 1.8
    # Other than that, if it's a new episode we want new scene, used heuristic for that
    df.loc[(df.time_from_prev == 0) & (df.start <= 2), 'new_scene'] = True

    # Calculate features
    text_for_scene = []
    charcteres_scene = []
    number_rows_scene = []
    for i, row in df.iterrows():
        if row['new_scene']:
            text_for_scene.append(row.txt)
            charcteres_scene.append(set([row.character]))
            number_rows_scene.append(1)
        else:
            text_for_scene[-1] += "\n" + row.txt
            charcteres_scene[-1].add(row.character)
            number_rows_scene[-1] += 1

    # append for each row the scene properties that we have
    df['scene_text'] = [text_for_scene[i] for i in range(len(number_rows_scene)) for _ in range(number_rows_scene[i])]
    df['scene_characters'] = [charcteres_scene[i] for i in range(len(number_rows_scene))
                              for _ in range(number_rows_scene[i])]
    df['n_scene_characters'] = df.scene_characters.str.len()

    return df

if __name__ == "__main__":
    df = load_corpus()
    df_main = df[df['character'].isin(["JERRY", "ELAINE", "KRAMER", "GEORGE"])]
    # plotStuff(df_main)
    # freq, corpus_one_hot = getOneHotEncoding(df.txt)
    # freq_trigrams, corpus_trigrams_one_hot = getTrigramEncoding(df.txt)

    print(df_main.head())




