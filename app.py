import streamlit as st
import pickle

# PAGE CONFIG
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# LOAD FILES
movies = pickle.load(open('movies.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
vectors = pickle.load(open('vectors.pkl', 'rb'))

# RECOMMEND FUNCTION
def recommend(movie):

    movie_index = movies[movies['title'] == movie].index[0]

    distances, indices = model.kneighbors(
        [vectors[movie_index]],
        n_neighbors=6
    )

    recommended_movies = []

    for i in indices[0][1:]:
        recommended_movies.append(movies.iloc[i].title)

    return recommended_movies

# UI
st.title("🎬 Movie Recommendation System")

st.write("AI/ML based movie recommendation using NLP and KNN")

selected_movie = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button("Recommend Movies"):

    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies")

    for movie in recommendations:
        st.write("👉", movie)
