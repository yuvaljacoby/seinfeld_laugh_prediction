from seinfeld_laugh_corpus import corpus
import numpy as np
import pandas as pd
import seaborn as sns

seinfeld = corpus.load(fold_laughs=True)

num_episodes = len(seinfeld.screenplays)
episodes_len = [len(seinfeld.screenplays[i].lines) for i in np.arange(num_episodes)]

characters = np.array([seinfeld.screenplays[i][j].character for i in np.arange(num_episodes) for j in np.arange(episodes_len[i])])
txt = np.array([seinfeld.screenplays[i][j].txt for i in np.arange(num_episodes) for j in np.arange(episodes_len[i])])
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

df_main = df[df['character'].isin(["JERRY", "ELAINE", "KRAMER", "GEORGE"])]

# sentence length by character and by funniness
g = sns.FacetGrid(df_main, col='character', row='is_funny')
g.map(sns.distplot, "length", bins=50, color='b')
plt.show()

print(df_main.head())




