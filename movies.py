# pip install pandas 
# pip install torch-hd

# to run: python movies.py

# import necessary packages
import sys
import pandas as pd
import torchhd as hd
import torch as tc
import numpy as np

# read movies csv into a data frame (in-memory table)
df_movies = pd.read_csv("movies.csv")
print(df_movies)

# number of dimensions
d = 10000
n = 1

# get the list of columns for parameterization
vars = df_movies.columns

# name of movie
var_movie = vars[0]
df_movie = df_movies[var_movie].drop_duplicates()

# name of Lead Studio of the Movie
var_studio = vars[1]
df_studio = df_movies[var_studio].drop_duplicates().dropna().sort_values().reset_index(drop=True)

# value of Rotten Tomatoes score(s)
var_rotten_tomatoes = vars[2]
df_rotten_tomatoes = df_movies[var_rotten_tomatoes].drop_duplicates().dropna().sort_values().reset_index(drop=True)

# value of audience score(s)
var_audience_score = vars[3]
df_audience_score = df_movies[var_audience_score].drop_duplicates().dropna().sort_values().reset_index(drop=True)

# story theme or storyline of the movie
var_story = vars[4]
df_story = df_movies[var_story].drop_duplicates().dropna().sort_values().reset_index(drop=True)

# genre of the movie
var_genre = vars[5]
df_genre = df_movies[var_genre].drop_duplicates().dropna().sort_values().reset_index(drop=True)

# Year the movie was released
var_year = vars[15]
df_year = df_movies[var_year].drop_duplicates().dropna().sort_values().reset_index(drop=True)

# aggregate the parameters for the queries
keys = hd.random(7 ,d)
k_movie, k_studio, k_rotten_tomatoes, k_audience_score, k_story, k_genre, k_year = keys

# get the base line or denominator for normalization of the hash values
max_size = sys.maxsize * 2 + 1

# number of results for closest match of query using Hamming Distance
k = 4

# read my movie preference csv into a Dataframe (in-memory table)
df_my_movie = pd.read_csv("my_movie.csv")

# Query to get the index of the closest match using Hamming Distance or Hamming Similarity
def get_index(d, var_parameter, df_parameter, max_size, df_my, hash=True):
    s = df_parameter.size
    hd_lib_parameters = hd.random(s, d)

    if(not hash):
       hd_lib_parameter_vals = [dfs/max_size for dfs in df_parameter.values]
    else: 
       hd_lib_parameter_vals = [dfs/max_size for dfs in pd.util.hash_array(df_parameter.values)]


    df_my_parameter_val = df_my[var_parameter].drop_duplicates().dropna().sort_values().reset_index(drop=True)

    if(not hash):
       hd_my_parameter_vals = [dfs/max_size for dfs in df_my_parameter_val.values]
    else:
       hd_my_parameter_vals = [dfs/max_size for dfs in pd.util.hash_array(df_my_parameter_val.values)]

    # create HyperVectors for my movie preferences 
    hd_my_parameters_vals = hd.bind(hd_lib_parameters, hd.ensure_vsa_tensor(hd_my_parameter_vals, vsa="MAP"))[:s]

    # create HyperVectors for the movies in the Library
    hd_lib_parameters_vals = hd.bind(hd_lib_parameters.transpose(0, -1), hd.ensure_vsa_tensor(hd_lib_parameter_vals, vsa="MAP"))
    hd_lib_parameters_vals_mem = hd.ensure_vsa_tensor(hd_lib_parameters_vals, vsa="MAP").transpose(0, -1)[:s]

    # will do a Hamming Similarity to find the closest matches based on Hamming Distance or Edit Distance
    hd_like_parameters = hd.hamming_similarity(hd_lib_parameters_vals_mem, hd_my_parameters_vals)
    idx_parameter = np.argmax(hd_like_parameters) % s
    hd_like_parameters_array = np.array(hd_like_parameters[idx_parameter])
    idx_parameters = np.argpartition(hd_like_parameters_array, kth=-k)[-k:]
    idx_parameters_sorted = idx_parameters[np.argsort(hd_like_parameters_array[idx_parameters])]

    # will return exact or closest k matches for the query based on Hamming Distance of Hashed Parameter Values
    return idx_parameters_sorted

# get the indices for the parameters for exact closest match
idx_studios = get_index(d, var_studio, df_studio, max_size, df_my_movie)
idx_stories = get_index(d, var_story, df_story, max_size, df_my_movie)
idx_genres = get_index(d, var_genre, df_genre, max_size, df_my_movie)
idx_years = get_index(d, var_year, df_year, max_size, df_my_movie)
idx_rotten_tomatoes = get_index(d, var_rotten_tomatoes, df_rotten_tomatoes, max_size, df_my_movie, False)
idx_audience_scores = get_index(d, var_audience_score, df_audience_score, max_size, df_my_movie, False)

# print out the result
print(df_studio.iloc[idx_studios])
print(df_story.iloc[idx_stories])
print(df_genre.iloc[idx_genres])
print(df_year.iloc[idx_years])
print(df_rotten_tomatoes.iloc[idx_rotten_tomatoes])
print(df_audience_score.iloc[idx_audience_scores])

# Will now try to do a recommendation based query

# read my liked movies list from a csv into a Dataframe (in-memory table)
df_liked_movies = pd.read_csv("liked_movies.csv")
print(df_liked_movies)

# get the counts of liked movies, their columns or parameters & the library of movies list length
lm = len(df_liked_movies)
km = len(df_liked_movies.columns)
mm = len(df_movies)

# the number of recommendations of suggested movies based on my preferences
kk = 9

# weights on some important parameters
w_story = 0.6
w_genre = 0.5
w_studio = 0.3
w_audience_score = 0.2
w_rotten_tomatoes = 0.1
w_year = 0.05

# total weight for normalization purposes
w_total = w_story + w_genre + w_studio + w_audience_score + w_rotten_tomatoes

# flag to indicate whether weights need to be applied
apply_weights = True

# apply normalized hashing to the parameters
def apply_normalized_hashing(r):
   ra = np.array(r)
   rah = pd.util.hash_array(ra)
   return rah/max_size

# apply normalized weighting to the parameters
def apply_normalized_weights(r):
   ra = r
   raw = np.array([ra[0]*w_studio, ra[1]*w_rotten_tomatoes, ra[2]*w_audience_score, ra[3]*w_story, ra[4]*w_genre, r[5]*w_year])
   return raw/w_total

# get and apply weights & hashes to the dataframes
def get_hash(df_params: pd.DataFrame, apply_weights=True):
   df_hashes = df_params.drop(columns=['Movie','TheatersOpenWeek','OpeningWeekend','BOAvgOpenWeekend','DomesticGross','ForeignGross','WorldGross','Budget','Profitability','OpenProfit'], errors='ignore').apply(lambda r: apply_normalized_hashing(r), axis=1)
   if apply_weights:
      df_weights = df_hashes.apply(lambda r: apply_normalized_weights(r))
      return df_weights
   else:
      return df_hashes

# apply hashes & weights to the list of my liked movies, my preferences & the library of movies
df_liked_movies_hash = get_hash(df_liked_movies, apply_weights=apply_weights)
df_my_movies_hash = get_hash(df_my_movie, apply_weights=apply_weights)
df_movies_hash = get_hash(df_movies, apply_weights=apply_weights)

# create a MAP Tensor with HyperVectors for my liked movies
hd_liked_movies = hd.random(lm, d).transpose(0, -1)

# create a MAP Tensor for my liked movies, my preferences & the library of movies
hd_liked_movies_mem = hd.ensure_vsa_tensor(df_liked_movies_hash)
hd_my_movies_mem = hd.ensure_vsa_tensor(df_my_movies_hash)
hd_movies_mem = hd.ensure_vsa_tensor(df_movies_hash)

# create a binding of my preferences & my likings
# binding gives the product operation which measures dissimilarity just like people with different views co-author and contribute to compiling a book
# on the other hand bundling is like aggregating or collecting your favorite books based on your preference or liking and hence is additive and measures similarity
# permute is shifting left or right the components of the hypervectors which change the orientation of them in N-D space
hd_my_liked_mem = hd.bind(hd_my_movies_mem, hd_liked_movies_mem)

# apply cosine similarity between the MAP Tensor for my likings & the movies in the library
# This will give the similarity based on how aligned the hypervectors are ... i.e. the more aligned the closer to +1 or -1
hd_my_liked_movies_mem = hd.cosine_similarity(hd_my_liked_mem, hd_movies_mem)

# fetch the list of top kk indices for movie recommendations
hd_my_liked_movies_mem_idx = int(np.argmax(hd_my_liked_movies_mem)) % mm
hd_my_liked_movies_mem_idxs = np.argpartition(hd_my_liked_movies_mem, kth=-kk)[-kk:].transpose(0, -1)
hd_my_liked_movies_mem_idxs_max = [hd_my_liked_movies_mem_idxs[i] for i in range(0, mm - 1) if hd_my_liked_movies_mem_idxs[i, 0] == hd_my_liked_movies_mem_idx]
hd_my_liked_movies_mem_idxs_max_arr = [hd_my_liked_movies_mem_idxs_max[0][i] for i in range(0, kk - 1)]

# print the top kk movie recommendations based on my input preferences
print(df_movies.iloc[hd_my_liked_movies_mem_idxs_max_arr].drop_duplicates())
