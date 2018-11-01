import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np


# control of importance
importance_of_director = 0.5
importance_of_actor = 0.5
importance_of_plot_keywords = 2
importance_of_genres = 0.5
importance_of_imdb_score = 1

df = pd.read_csv("movie.csv")

df = df[["director_name", "genres", "actor_1_name", "movie_title", "imdb_score", "plot_keywords"]]
print(df)

# convert all datas to lowercase
df['movie_title'] = df['movie_title'].str.lower()
df['plot_keywords'] = df['plot_keywords'].str.lower()
df['director_name'] = df['director_name'].str.lower()
df['genres'] = df['genres'].str.lower()
df['actor_1_name'] = df['actor_1_name'].str.lower()

# content based recommendation engine
# plot_keywords + genres + director + actor + imdb_score
# tf-idf vector for plot_keywords and genres

from sklearn.feature_extraction.text import TfidfVectorizer

# check Null values
print(df['plot_keywords'].isnull().sum())

# remove null values
df.dropna(how='any',axis=0,inplace=True)


# replace | with space (to separate keywords in plot keywords)
def replace_str(x):
    return x.replace("|"," ")


df['plot_keywords'] = df.apply(lambda row:replace_str(row['plot_keywords']),axis=1)
df['genres'] = df.apply(lambda row:replace_str(row['genres']),axis=1)

# trim all data in movie title
def strip(x):
    return x.strip()

df['movie_title'] = df.apply(lambda row:strip(row['movie_title']),axis=1)

# tf-idf with n-gram
vectorizer_plot_keywords = TfidfVectorizer(ngram_range=(1,1))
vectorizer_genres = TfidfVectorizer(ngram_range=(1,1))
# tokenize and build vocab
#print("list of strings",list(df['plot_keywords'].values))
vectorizer_plot_keywords.fit(list(df['plot_keywords'].values))
vectorizer_genres.fit(list(df['genres'].values))
#summarize
#print(vectorizer.vocabulary_)


# input movie to recommend other movies
selected_movie_title = input("input movie title(all lower case)")
print("selected title: ",selected_movie_title)
df_movie = df[df['movie_title'] == selected_movie_title]
pd.set_option('expand_frame_repr', False)
print(df_movie)

selected_plot_keywords = df_movie.loc[:,"plot_keywords"].iloc[0]
selected_genres = df_movie.loc[:,"genres"].iloc[0]
selected_director_name = df_movie.loc[:,"director_name"].iloc[0]
selected_actor_1_name = df_movie.loc[:,"actor_1_name"].iloc[0]
selected_imdb_score = df_movie.loc[:,"imdb_score"].iloc[0]

vector_plot_keywords = vectorizer_plot_keywords.transform([selected_plot_keywords])
vector_genres = vectorizer_genres.transform([selected_genres])

#print("vector shape: ",vector.shape)
#print(vector.toarray())

from sklearn.metrics.pairwise import cosine_similarity


# put cosine score in each row
def cosine_score_plot_keywords(x):
    vector_row = vectorizer_plot_keywords.transform([x])
    cosine_score = cosine_similarity(vector_plot_keywords,vector_row)
    return cosine_score[0][0]

# put cosine score in each row
def cosine_score_genres(x):
    vector_row = vectorizer_genres.transform([x])
    cosine_score = cosine_similarity(vector_genres,vector_row)
    return cosine_score[0][0]

# put squared imdb score difference
def imdb_score_difference_square(x):
    #imdb_score_row = x['imdb_score']
    return np.square(selected_imdb_score.astype(np.float32)-x)

df['cosine_score_plot_keywords'] = df.apply(lambda row:cosine_score_plot_keywords(row['plot_keywords']),axis=1)

df['cosine_score_genres'] = df.apply(lambda row:cosine_score_genres(row['genres']),axis=1)

df['imdb_score_difference_square'] = df.apply(lambda row:imdb_score_difference_square(row['imdb_score']),axis=1)

#print(df)

# scaling min-max scaling the imdb_score
from sklearn.preprocessing import MinMaxScaler

df['imdb_score_difference_square_scaled'] = MinMaxScaler().fit_transform(df[['imdb_score_difference_square']])
#print(df)

def convert_distance_to_similarity(x):
    return (1-x)


df['imdb_score_similarity'] = df.apply(lambda row:convert_distance_to_similarity(row['imdb_score_difference_square_scaled']),axis=1)
#print(df)
#df = df.rename(columns={"imdb_score_difference_square": "imdb_score_similarity"})

# put 0.5 in case same director/actor, otherwise put 0

def checkDirector(x):
    if selected_director_name == x:
        return 0.5
    else:
        return 0.0

def checkActor(x):
    if selected_actor_1_name == x:
        return 0.5
    else:
        return 0.0

df['director_name_similarity'] = df.apply(lambda row: checkDirector(row['director_name']), axis=1)
df['actor_1_name_similarity'] = df.apply(lambda row: checkActor(row['actor_1_name']), axis=1)

# now count total similarity score and sort data by this score

def countTotalSimilarityScore(x):
    actor_1_name_similarity = importance_of_actor*x['actor_1_name_similarity']
    director_name_similarity = importance_of_director*x['director_name_similarity']
    imdb_score_similarity = importance_of_imdb_score*x['imdb_score_similarity']
    cosine_score_genres = importance_of_genres*x['cosine_score_genres']

    # give priority to plot_keywords
    cosine_score_plot_keywords = importance_of_plot_keywords*x['cosine_score_plot_keywords']

    total_score = np.square(actor_1_name_similarity)+np.square(director_name_similarity)+np.square(imdb_score_similarity)+np.square(cosine_score_genres)+np.square(cosine_score_plot_keywords)
    return total_score


df['total_score'] = df.apply(lambda row: countTotalSimilarityScore(row), axis=1)

# sort w.r.t cosine score
df.sort_values("total_score",ascending=False,inplace=True)
print(df[['movie_title','director_name','director_name_similarity','actor_1_name','actor_1_name_similarity','plot_keywords','genres','imdb_score','total_score']].head(10))