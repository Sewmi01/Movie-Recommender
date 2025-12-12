import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# 1. Load data
@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return movies, ratings

# 2. Build content-based similarity matrix using genres
@st.cache_data
def build_similarity(movies):
    # Handle missing genres
    movies['genres'] = movies['genres'].fillna('')

    # Genres are like "Action|Adventure|Sci-Fi"
    # We'll split by '|' and treat each as a token
    cv = CountVectorizer(tokenizer=lambda x: x.split('|'))
    genre_matrix = cv.fit_transform(movies['genres'])

    similarity = cosine_similarity(genre_matrix)

    return similarity


# 3. Recommend similar movies based on a given movie title
def recommend_by_movie(title, movies, similarity, n=5):
    title = title.lower()

    # Find the index of the movie
    indices = movies[movies['title'].str.lower() == title].index

    if len(indices) == 0:
        return []

    idx = indices[0]

    # Get similarity scores for this movie
    sim_scores = list(enumerate(similarity[idx]))

    # Sort movies by similarity score (high to low)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Skip the first one because it's the same movie (similarity = 1.0)
    sim_scores = sim_scores[1:n+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the titles
    return movies.iloc[movie_indices][['movieId', 'title', 'genres']]


# 4. Recommend movies for a user based on their past high ratings
def recommend_for_user(user_id, movies, ratings, similarity, n=10, min_rating=4.0):
    # Get ratings of the user
    user_ratings = ratings[ratings['userId'] == user_id]

    if user_ratings.empty:
        return pd.DataFrame(columns=['title', 'estimated_score'])

    # Consider only movies they rated >= min_rating
    liked_movies = user_ratings[user_ratings['rating'] >= min_rating]

    if liked_movies.empty:
        return pd.DataFrame(columns=['title', 'estimated_score'])

    # Map movieIds to indices in movies dataframe
    movie_id_to_index = pd.Series(data=movies.index, index=movies['movieId'])

    # Indices of liked movies
    liked_indices = []
    for mid in liked_movies['movieId']:
        if mid in movie_id_to_index:
            liked_indices.append(movie_id_to_index[mid])

    if not liked_indices:
        return pd.DataFrame(columns=['title', 'estimated_score'])

    # Compute a recommendation score for all movies
    # Score = sum of similarities to liked movies
    sim_scores = similarity[liked_indices].sum(axis=0)

    # Do not recommend movies the user has already rated
    already_rated_indices = [movie_id_to_index[mid] for mid in user_ratings['movieId'] if mid in movie_id_to_index]
    sim_scores[already_rated_indices] = -1  # set to -1 so they go to bottom

    # Get top N indices
    top_indices = sim_scores.argsort()[::-1][:n]

    recommended = movies.iloc[top_indices][['movieId', 'title', 'genres']].copy()
    recommended['estimated_score'] = sim_scores[top_indices]

    return recommended


# 5. Streamlit App UI
def main():
    st.title("🎬 Movie Recommendation System")
    st.write("Content-based recommendations using genres + basic user preference suggestions.")

    # Load data and similarity matrix
    movies, ratings = load_data()
    similarity = build_similarity(movies)

    # Sidebar options
    st.sidebar.header("Choose Recommendation Type")
    option = st.sidebar.radio(
        "Select mode:",
        ("🔎 Movie-based recommendations", "👤 User-based recommendations")
    )

    if option == "🔎 Movie-based recommendations":
        st.subheader("Find similar movies to a given movie")

        movie_titles = movies['title'].values
        selected_movie = st.selectbox("Select a movie:", movie_titles)

        if st.button("Recommend similar movies"):
            results = recommend_by_movie(selected_movie, movies, similarity, n=10)

            if results.empty:
                st.warning("No similar movies found.")
            else:
                st.write("### Similar Movies:")
                for _, row in results.iterrows():
                    st.write(f"**{row['title']}**")
                    st.write(f"Genres: {row['genres']}")
                    st.write("---")

    elif option == "👤 User-based recommendations":
        st.subheader("Recommend movies for a user")

        min_user_id = int(ratings['userId'].min())
        max_user_id = int(ratings['userId'].max())

        user_id = st.number_input("Enter a user ID:", min_value=min_user_id, max_value=max_user_id, value=min_user_id, step=1)

        if st.button("Recommend for this user"):
            results = recommend_for_user(int(user_id), movies, ratings, similarity, n=10, min_rating=4.0)

            if results.empty:
                st.warning("No recommendations (maybe this user has no high ratings). Try another user ID.")
            else:
                st.write(f"### Top recommendations for user {user_id}:")
                for _, row in results.iterrows():
                    st.write(f"**{row['title']}**  (Estimated Score: {row['estimated_score']:.2f})")
                    st.write(f"Genres: {row['genres']}")
                    st.write("---")


if __name__ == "__main__":
    main()
