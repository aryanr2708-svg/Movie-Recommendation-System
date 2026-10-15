import streamlit as st
from videorec import recommend_new_user, movies_df, movie_map
st.title("🎬 Movie Recommendation System")
st.write("Select movies you like to get personalized recommendations!")
available_movies = movies_df[movies_df["movieId"].isin(movie_map)].sort_values("title")
selected_titles = st.multiselect(
    "Choose movies you like:",
    options=available_movies["title"].tolist(),
    max_selections=10
)
if st.button("Get Recommendations"):
    if not selected_titles:
        st.warning("Please select at least one movie!")
    else:
        selected_ids = available_movies.loc[available_movies["title"].isin(selected_titles),"movieId"].tolist()
        results = recommend_new_user(selected_ids)
        st.subheader("Recommended Movies")
        st.dataframe(results, hide_index=True)