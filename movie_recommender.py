import pandas as pd
import numpy as np
import ast
import pickle

# Load datasets
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/credits.csv")

# Merge datasets
movies = movies.merge(credits, on="title")

# Select important columns
movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Remove null values
movies.dropna(inplace=True)

# Convert string into list
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Take top 3 cast members only
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])

# Fetch director name
def fetch_director(text):
    L = []

    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])

    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# Remove spaces between words
def collapse(L):
    return [i.replace(" ", "") for i in L]

movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)

# Create tags column
movies['tags'] = (
    movies['overview'] + " " +
    movies['genres'].astype(str) + " " +
    movies['keywords'].astype(str) + " " +
    movies['cast'].astype(str) + " " +
    movies['crew'].astype(str)
)

# Convert to lowercase
movies['tags'] = movies['tags'].apply(lambda x: x.lower())

# Vectorization
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(movies['tags']).toarray()

# ML Model
from sklearn.neighbors import NearestNeighbors

print("Training ML model...")

model = NearestNeighbors(metric='cosine', algorithm='brute')

model.fit(vectors)

print("Model trained successfully ✅")

# Recommendation Function
def recommend(movie):

    if movie not in movies['title'].values:
        print("❌ Movie not found! Try exact name.")
        return

    movie_index = movies[movies['title'] == movie].index[0]

    distances, indices = model.kneighbors(
        [vectors[movie_index]],
        n_neighbors=6
    )

    print("\nRecommended Movies:\n")

    for i in indices[0][1:]:
        print("👉", movies.iloc[i].title)

# User Input
movie_name = input("Enter movie name: ")

recommend(movie_name)

# Save files
pickle.dump(model, open('models/model.pkl', 'wb'))
pickle.dump(movies, open('models/movies.pkl', 'wb'))
pickle.dump(vectors, open('models/vectors.pkl', 'wb'))

print("\nFiles saved successfully ✅")
