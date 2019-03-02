import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from seinfeld_laugh_corpus import corpus

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

#TODO maybe remove this
def clean_characters(df):
    max_in_a_row = 10
    curr_character = df['character'][0]  # The current character speaking.
    curr_count = 0  # The count for the current character speaking.
    character_count = list()  # Will containg the rows we will collect...


    for i, row in df.iterrows():
        if curr_character != row['character']:
            character_count.append({'character': curr_character,
                                    'count': curr_count,
                                    'season': row['season'],
                                    'episode_num': row['episode_num'],
                                    'last_index': i - 1,
                                    'start': row['start'],
                                    'end': row['end'],
                                    'length': row['length']})
            curr_character = row['character']
            curr_count = 1
        else:
            curr_count += 1

    repeating_characters = pd.DataFrame(character_count)
    indices_to_remove = []
    for _, row in repeating_characters.iterrows():
        if row['count'] >= max_in_a_row:
            indices_to_remove += [row['last_index'] - j for j in range(row['count'])]

    return indices_to_remove

def create_index_using_duplicated(df, grouping_cols):
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
    txt_split = np.chararray.split(txt, ' ')
    avg_word_size = []
    for sentence_array in txt_split:
        sum = 0.0
        for word in sentence_array:
            sum += len(word)
        avg_word_size.append(sum / len(sentence_array))
    avg_word_size = np.array(avg_word_size)
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
                       'num_words': num_words.astype(np.float),
                       'avg_word_length': avg_word_size,
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

    # char_idx_remove = clean_characters(df)
    # df.loc[char_idx_remove, ['character']] = np.nan
    df['global_episode_num'] = create_index_using_duplicated(df, ['season', 'episode_num'])
    return df

#TODO maybe remove this
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
    df.loc[:, 'global_scene_number'] = create_index_using_duplicated(df, ['global_episode_num', 'scene_number_in_episode'])
    return df



