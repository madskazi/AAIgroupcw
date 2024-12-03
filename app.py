import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# Load the dataset
# df_songsDB = pd.read_csv('C://Users//Asus//OneDrive//Desktop//Grishma_AI//Applied_AI//Assessment_1//API//song_dataset.csv')
df_songsDB = pd.read_csv('/Users/Matka/Desktop/uni assignments/sem 1 - II/Applied AI/asmnt 1/song_dataset.csv')

# Build a dropdown for user selection
st.title("Song Recommendation System")
user_id = st.selectbox("Select User", options=df_songsDB['user'].unique())

# Dynamically update the dropdown based on user selection
if user_id:
    listened_songs = df_songsDB[df_songsDB['user'] == user_id]['title'].unique()
    selected_songs = st.multiselect("Select Songs You've Liked", options=listened_songs)

# Prepare the dataset for collaborative filtering
reader = Reader(rating_scale=(1, df_songsDB['play_count'].max()))
surpriseData = Dataset.load_from_df(df_songsDB[['user', 'song', 'play_count']], reader)
trainset = surpriseData.build_full_trainset()

# Train the collaborative filtering model
algo_SVD = SVD()
algo_SVD.fit(trainset)

# Define the hybrid recommendation engine functions
def content_score_calculator(selected_songs, unlistened_songs):
    df_songsDB['combined_features'] = (
        df_songsDB['artist_name'] + " " +
        df_songsDB['release'] + " " +
        df_songsDB['title']
    )
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_songsDB['combined_features'])

    selected_features = df_songsDB[df_songsDB['title'].isin(selected_songs)]['combined_features']
    unlistened_features = df_songsDB[df_songsDB['song'].isin(unlistened_songs)]['combined_features']

    selected_matrix = tfidf.transform(selected_features)
    unlistened_matrix = tfidf.transform(unlistened_features)

    similarity_scores = cosine_similarity(selected_matrix, unlistened_matrix)
    avg_similarity = similarity_scores.mean(axis=0)

    return dict(zip(unlistened_songs, avg_similarity))

def collaborative_score_calculator(user_id, unlistened_songs):
    cf_scores = {}
    for song_id in unlistened_songs:
        prediction = algo_SVD.predict(user_id, song_id).est
        cf_scores[song_id] = prediction
    return cf_scores

def hybridRecommendationEngine(user_id, selected_songs):
    alpha = 0.5  # Blending parameter

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

    sorted_songs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    recommended_song_ids = [song_id for song_id, _ in sorted_songs[:10]]

    recommended_songs = (
        pd.DataFrame(recommended_song_ids, columns=['song'])
        .merge(df_songsDB[['song', 'title', 'release', 'artist_name', 'year']].drop_duplicates(), on='song', how='left')
    )
    return recommended_songs

# Recommendation button
if st.button("Get Recommendations"):
    if selected_songs:
        recommendations = hybridRecommendationEngine(user_id, selected_songs)
        st.write("Recommended Songs:")
        st.dataframe(recommendations)
    else:
        st.warning("Please select at least one song!")

