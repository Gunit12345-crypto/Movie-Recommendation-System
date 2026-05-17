import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("tmdb_5000_movies.csv")

movies = movies[['title', 'overview']]
movies.dropna(inplace=True)

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(movies['overview']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]

    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select Movie",
    movies['title'].values
)

if st.button("Recommend"):

    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies")

    for movie in recommendations:
        st.write("👉", movie)
