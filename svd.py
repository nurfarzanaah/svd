'''
This is a SVD practive based on Movie Recommendation.

In this practice, the MovieLens 1M Dataset is used. 
This dataset is chosen because it does not require any preprocessing
as the main focus is on SVD and recommender systems.

1. Ratings.dat:
Columns : [user_id, movie_id, ratings, time]
Rows    : 6040 users

2. Movies.dat:
Columns : ['movie_id', 'title', 'genre']
Rows    : 3952 movies
'''

# Import the required python libraries
from pandas.core.arrays.sparse import dtype
import numpy as np
import pandas as pd

# Read the dataset. It consists of two files ('rating.dat', and 'movies.dat')
data = pd.io.parsers.read_csv('data/ratings.dat',
            names = ['user_id', 'movie_id', 'rating', 'time'],
            engine = 'python', delimiter = '::',
            encoding = "windows-1251")
movie_data = pd.io.parsers.read_csv('data/movies.dat',
    names=['movie_id', 'title', 'genre'],
    engine='python', delimiter='::', encoding = "windows-1251")

# Create the rating matrix with rows as movies and columns as users
ratings_mat = np.ndarray(
    shape = (np.max(data.movie_id.values), np.max(data.user_id.values)),
    dtype = np.uint8)
ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values

# # Normalize the matrix
normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T

# Compute the singular value decomposition (SVD)
A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
U, S, V = np.linalg.svd(A)

# Defina a function to calculate the cosine similarity.
## Sort by the most similiar and return the top N results
def top_cosine_similarity(data, movie_id, top_n = 10):
    index = movie_id -1 ## Movie id starts from 1 in the dataset
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# Define a function to print N similar movies
def print_similar_movies(movie_data, movie_id, top_indexes):
    print('\nRecommendations for {0}: \n'.format(
        movie_data[movie_data.movie_id == movie_id].title.values[0]))
    for id in top_indexes + 1:
        print(movie_data[movie_data.movie_id == id].title.values[0])

# Initialise the value of k principal components, id of the movie
# as given in the dataset, and number of top elements to be printed.
k = 50
movie_id = 2808 # (getting an id from movie.dat)
top_n = 10
sliced = V.T[:, :k] # representative data
indexes = top_cosine_similarity(sliced, movie_id, top_n)

# Print the top N similar movies
print_similar_movies(movie_data, movie_id, indexes)
