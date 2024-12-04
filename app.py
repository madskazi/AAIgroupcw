import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Add this at the very top of the file, before any other Streamlit commands
st.set_page_config(layout="wide")

# Add custom CSS after the st.set_page_config
st.markdown("""
<style>
.small-select {
    max-width: 200px !important;
}
.stButton > button {
    background-color: pink;
    color: black !important;
    border: none;
}
.stButton > button:hover {
    background-color: pink !important;
    color: white !important;
    border: 2px solid white !important;
}
/* Style for selected items in multiselect */
.stMultiSelect [data-baseweb="tag"] {
    background-color: pink !important;
    color: black !important;
}
/* Add focus/click style for multiselect and select */
.stMultiSelect [data-baseweb="select"] > div:first-child,
.stSelectbox [data-baseweb="select"] > div:first-child {
    border-color: white !important;
    box-shadow: 0 0 0 1px white !important;
}
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    df = pd.read_csv('song_dataset.csv')
    return df

# Cache matrix computations
@st.cache_data
def compute_matrices(df_songsDB):
    user_item_matrix = df_songsDB.pivot_table(index='user', columns='song', values='play_count', fill_value=0)
    svd = TruncatedSVD(n_components=20, random_state=20)
    svd_matrix = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_
    return user_item_matrix, svd_matrix, item_factors

# Load data and compute matrices once
df_songsDB = load_data()
user_item_matrix, svd_matrix, item_factors = compute_matrices(df_songsDB)

# Cache the TF-IDF computation
@st.cache_data
def compute_tfidf(df_songsDB):
    df_songsDB['combined_features'] = (
        df_songsDB['artist_name'] + " " +
        df_songsDB['release'] + " " +
        df_songsDB['title']
    )
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_songsDB['combined_features'])
    return tfidf, tfidf_matrix

# Helper functions
def content_score_calculator(selected_songs, unlistened_songs):
    df_songsDB['combined_features'] = (
        df_songsDB['artist_name'] + " " +
        df_songsDB['release'] + " " +
        df_songsDB['title']
    )

    selected_song_features = df_songsDB[df_songsDB['title'].isin(selected_songs)]['combined_features']
    unlistened_song_features = df_songsDB[df_songsDB['song'].isin(unlistened_songs)]['combined_features']

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_songsDB['combined_features'])

    selected_matrix = tfidf.transform(selected_song_features)
    unlistened_matrix = tfidf.transform(unlistened_song_features)
    similarity_scores = cosine_similarity(selected_matrix, unlistened_matrix)

    avg_similarity = similarity_scores.mean(axis=0)

    return dict(zip(unlistened_songs, avg_similarity))

def collaborative_score_calculator(user_id, unlistened_songs):
    user_idx = user_item_matrix.index.get_loc(user_id)
    user_vector = svd_matrix[user_idx]
    cf_scores = {}

    for song_id in unlistened_songs:
        if (song_id in user_item_matrix.columns):
            song_idx = user_item_matrix.columns.get_loc(song_id)
            song_vector = item_factors[:, song_idx]
            cf_scores[song_id] = np.dot(user_vector, song_vector)
        else:
            cf_scores[song_id] = 0
    return cf_scores

def hybridRecommendationEngine(user_id, selected_songs):
    alpha = 0.5

    listened_songs = df_songsDB[df_songsDB['user'] == user_id]['song'].unique()
    all_songs = df_songsDB['song'].unique()
    unlistened_songs = set(all_songs) - set(listened_songs)

    cf_scores = collaborative_score_calculator(user_id, unlistened_songs)
    content_scores = content_score_calculator(selected_songs, unlistened_songs)

    final_scores = {}
    for song_id in unlistened_songs:
        cf_score = cf_scores.get(song_id, 0)
        content_score = content_scores.get(song_id, 0)
        final_scores[song_id] = alpha * cf_score + (1 - alpha) * content_score

    scores = list(final_scores.values())
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 1

    if max_score > min_score:
        normalized_scores = {
            song_id: (score - min_score) / (max_score - min_score)
            for song_id, score in final_scores.items()
        }
    else:
        normalized_scores = {song_id: 0.5 for song_id in final_scores}

    sorted_songs = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
    recommended_song_ids = [song_id for song_id, _ in sorted_songs[:10]]

    recommended_songs = (
        pd.DataFrame(recommended_song_ids, columns=['song'])
        .merge(df_songsDB[['song', 'title', 'release', 'artist_name']].drop_duplicates(), on='song', how='left')
        .assign(recommendation=lambda x: x['title'] + ' by ' + x['artist_name'])
    )
    return recommended_songs['recommendation'].tolist()

# Streamlit app
st.title("Delta Melody Match 🎶")

# Make columns take more width
col1, col2 = st.columns([2, 4])

with col1:
    with st.container():
        user_id = st.selectbox(
            "👤 Select User ID",
            options=df_songsDB['user'].unique().tolist(),
            key="small_select"
        )
        st.markdown('<style>div[data-testid="stSelectbox"] > div:first-child {max-width: 200px;}</style>', unsafe_allow_html=True)

songs_selectable = df_songsDB[df_songsDB['user'] == user_id]['title'].unique()

with col2:
    song_titles = st.multiselect(
        "🎵 Select Songs You Like",
        options=songs_selectable,
        default=songs_selectable[:1]
    )

# Make the recommendations table wider
if st.button("Get Recommendations"):
    st.subheader("Recommended Songs")
    recommendations = hybridRecommendationEngine(user_id, song_titles)
    for i, rec in enumerate(recommendations, 1):
        # Split the recommendation into title and artist
        title, artist = rec.split(' by ')
        st.write(f"{i}. ***{title}*** by {artist}")
