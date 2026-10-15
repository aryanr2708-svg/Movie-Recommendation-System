import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
folder = Path(__file__).resolve().parent
ratings_df=pd.read_csv(folder/"ratings.csv")
movies_df = pd.read_csv(folder/"movies.csv")
ratings_df["liked"] = (ratings_df["rating"] >= 4).astype(int)
train_df, test_df = train_test_split(
    ratings_df,
    test_size=0.2,
    random_state=42
)
user_ids = ratings_df["userId"].unique()
movie_ids = ratings_df["movieId"].unique()
user_map = {uid: i for i, uid in enumerate(user_ids)}
movie_map = {mid: i for i, mid in enumerate(movie_ids)}
users = train_df["userId"].map(user_map).to_numpy()
movies = train_df["movieId"].map(movie_map).to_numpy()
likes = train_df["liked"].to_numpy()
n_users=len(user_ids)
n_movies=len(movie_ids)
r=np.random.default_rng(1)
U= r.normal(0,0.1,(n_users,10))
V= r.normal(0,0.1,(n_movies,10))
user_bias = np.zeros(n_users)
movie_bias = np.zeros(n_movies)

like_rate = np.clip(likes.mean(), 1e-6, 1 - 1e-6)
global_bias = np.log(like_rate / (1 - like_rate))
learning_rate = 0.05
reg = 0.1
for step in range(100):
    learning_rate = 0.05

counts_U = np.bincount(users, minlength=n_users)
counts_V = np.bincount(movies, minlength=n_movies)

for step in range(100):
    scores = (
        np.sum(U[users] * V[movies], axis=1)
        + user_bias[users]
        + movie_bias[movies]
        + global_bias
    )

    p = 1 / (1 + np.exp(-np.clip(scores, -30, 30)))
    d = p - likes

    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)

    np.add.at(grad_U, users, d[:, None] * V[movies])
    np.add.at(grad_V, movies, d[:, None] * U[users])

    grad_U = grad_U / np.maximum(counts_U[:, None], 1) + reg * U
    grad_V = grad_V / np.maximum(counts_V[:, None], 1) + reg * V

    grad_user_bias = np.bincount(
        users, weights=d, minlength=n_users
    ) / np.maximum(counts_U, 1) + reg * user_bias

    grad_movie_bias = np.bincount(
        movies, weights=d, minlength=n_movies
    ) / np.maximum(counts_V, 1) + reg * movie_bias

    U -= learning_rate * grad_U
    V -= learning_rate * grad_V
    user_bias -= learning_rate * grad_user_bias
    movie_bias -= learning_rate * grad_movie_bias
train_scores = (np.sum(U[users] * V[movies], axis=1)+ user_bias[users]+ movie_bias[movies]+ global_bias)

train_p = 1 / (1 + np.exp(-np.clip(train_scores, -30, 30)))

train_predictions = (train_p >= 0.5).astype(int)

train_accuracy = np.mean(train_predictions == likes)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
test_users = test_df["userId"].map(user_map).to_numpy()
test_movies = test_df["movieId"].map(movie_map).to_numpy()
test_likes = test_df["liked"].to_numpy()
test_scores = np.sum(
    U[test_users] * V[test_movies],
    axis=1
)

test_p = 1 / (1 + np.exp(-np.clip(test_scores, -30, 30)))
test_predictions = (test_p >= 0.5).astype(int)
test_accuracy = np.mean(test_predictions == test_likes)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

def recommend(user_id):
    user = user_map[user_id]
    scores = (U[user] @ V.T+ user_bias[user]+ movie_bias+ global_bias)
    watched = ratings_df.loc[ratings_df["userId"] == user_id, "movieId"]
    watched_indices = [movie_map[mid] for mid in watched]
    scores[watched_indices] = -np.inf
    top5 = np.argsort(scores)[-5:][::-1]
    recommended_ids = movie_ids[top5]
    recommendations = movies_df[ movies_df["movieId"].isin(recommended_ids)].copy()
    recommendations["rank"] = recommendations["movieId"].map({mid: i + 1 for i, mid in enumerate(recommended_ids)})
    recommendations = recommendations.sort_values("rank")
    return recommendations[["title", "genres"]]


def recommend_new_user(selected_movie_ids, top_n=5):
    available = movies_df[
        movies_df["movieId"].isin(movie_map)
    ].copy().reset_index(drop=True)
    available["genre_text"] = (available["genres"].fillna("").str.replace("|", " ", regex=False))
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b[\w-]+\b",binary=True)
    genre_matrix = vectorizer.fit_transform(
        available["genre_text"]
    )
    selected_mask = available["movieId"].isin(selected_movie_ids)
    selected_indices = np.flatnonzero(selected_mask.to_numpy())
    if len(selected_indices) == 0:
        return available.iloc[0:0][["title", "genres"]]
    profile = np.asarray(
        genre_matrix[selected_indices].mean(axis=0)
    )
    scores = cosine_similarity(profile,genre_matrix).ravel()
    scores[selected_indices] = -np.inf
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return available.iloc[top_indices][["title", "genres"]]