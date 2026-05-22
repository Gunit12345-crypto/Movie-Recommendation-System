
# PAGE CONFIG
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# CUSTOM CSS
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    color: #E50914;
    font-size: 50px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #BBBBBB;
    font-size: 20px;
    margin-bottom: 30px;
}

.movie-card {
    background-color: #1e1e1e;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
    color: white;
    font-size: 18px;
}

.stButton>button {
    background-color: #E50914;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    border: none;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# LOAD DATA
movies = pd.read_csv("tmdb_5000_movies.csv")

movies = movies[['title', 'overview']]

movies.dropna(inplace=True)

# VECTORIZATION
cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = cv.fit_transform(
    movies['overview']
).toarray()

# SIMILARITY
similarity = cosine_similarity(vectors)

# RECOMMEND FUNCTION
def recommend(movie):

    movie_index = movies[
        movies['title'] == movie
    ].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(
            movies.iloc[i[0]].title
        )

    return recommended_movies

# SIDEBAR
st.sidebar.title("📌 About Project")

st.sidebar.write("""
This is a Machine Learning based
Movie Recommendation System.

Technologies Used:
- Python
- Streamlit
- Scikit-learn
- NLP
- Cosine Similarity
""")

# MAIN TITLE
st.markdown(
    "<div class='title'>🎬 Movie Recommendation System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI/ML based movie recommendation using NLP</div>",
    unsafe_allow_html=True
)

# SELECT MOVIE
selected_movie = st.selectbox(
    "🎥 Select a Movie",
    movies['title'].values
)

# BUTTON
if st.button("🚀 Recommend Movies"):

    recommendations = recommend(selected_movie)

    st.subheader("🔥 Recommended Movies")

    for movie in recommendations:

        st.markdown(
            f"<div class='movie-card'>👉 {movie}</div>",
            unsafe_allow_html=True
        )

# FOOTER
st.write("---")

st.caption("Built with ❤️ using Python and Machine Learning")
