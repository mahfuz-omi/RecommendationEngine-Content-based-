import warnings

warnings.filterwarnings('ignore')

import pandas as pd

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
# plot_keywords only
# tf-idf vector for plot_keywords

from sklearn.feature_extraction.text import TfidfVectorizer

# check Null values
print(df['plot_keywords'].isnull().sum())

# remove null values
df.dropna(how='any',axis=0,inplace=True)


# replace | with space (to separate keywords in plot keywords)
def replace_str(x):
    return x.replace("|"," ")


df['plot_keywords'] = df.apply(lambda row:replace_str(row['plot_keywords']),axis=1)

# trim all data in movie title
def strip(x):
    return x.strip()

df['movie_title'] = df.apply(lambda row:strip(row['movie_title']),axis=1)

# tf-idf with n-gram
vectorizer = TfidfVectorizer(ngram_range=(1,1))
# tokenize and build vocab
#print("list of strings",list(df['plot_keywords'].values))
vectorizer.fit(list(df['plot_keywords'].values))
#summarize
#print(vectorizer.vocabulary_)


# input movie to recommend other movies
movie_title = input("input movie title(all lower case)")
print("selected title: ",movie_title)
df_movie = df[df['movie_title'] == movie_title]
pd.set_option('expand_frame_repr', False)
print(df_movie)

print("selected movie: ",df_movie.loc[:,"plot_keywords"].values)
vector = vectorizer.transform(df_movie.loc[:,"plot_keywords"])
#print("vector shape: ",vector.shape)
#print(vector.toarray())

from sklearn.metrics.pairwise import cosine_similarity


# put cosine score in each row
def cosine_put(x):
    vector_row = vectorizer.transform([x])
    cosine_score = cosine_similarity(vector,vector_row)
    return cosine_score[0][0]

df['cosine_score'] = df.apply(lambda row:cosine_put(row['plot_keywords']),axis=1)


# sort w.r.t cosine score
df.sort_values("cosine_score",ascending=False,inplace=True)
print(df[['movie_title','cosine_score','plot_keywords']].head(10))