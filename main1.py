import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time

#Basic
import pandas as pd
import numpy as np

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

#Clsutering and Recommendations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('genres_v2.csv')

#dropping rows with no song names and no uri and then dropping duplicates as well

# Drop rows with no song names and no URI
df = df.dropna(subset=['song_name', 'uri'])

# Remove duplicate rows based on 'song_name' and 'uri'
df = df.drop_duplicates(subset=['song_name', 'uri'])

# Get the count of final rows
final_row_count = len(df)

#print("Count of final rows:", final_row_count)

import matplotlib.pyplot as plt
import seaborn as sns

cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo','uri','genre','song_name']
filtered_df = df[cols]

#print(filtered_df.info())

num_cols = [i for i in filtered_df.columns if filtered_df[i].dtype != 'object']
scaler = StandardScaler()

filtered_df[num_cols] = scaler.fit_transform(filtered_df[num_cols])

X = filtered_df.drop(['uri','genre','song_name'], axis=1)  # Drop non-numeric columns if any

# Choose the number of clusters (you mentioned 5 clusters)
n_clusters = 7

# Initialize KMeans model
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

filtered_df_no_pca = filtered_df
# Fit the KMeans model to your data
filtered_df_no_pca['cluster'] = kmeans.fit_predict(X)

# Perform PCA
n_components = 10  # Adjust the value based on the number of clusters
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(filtered_df[num_cols])

# Scree plot
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

# 4. Perform PCA
n_components = 6  # Adjust the number of components as needed
pca = PCA(n_components=n_components)
pca_df = pca.fit_transform(filtered_df[num_cols])
# print(pca_df)

# 5. K-Means Clustering on PCA Results
n_clusters = 7 # Number of clusters (adjust as needed)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
filtered_df_pca = filtered_df
filtered_df_pca['cluster'] = kmeans.fit_predict(pca_result)

#print(filtered_df_pca['cluster'].value_counts())

#Label each cluster with specific mood

filtered_df_pca['mood'] = np.where(filtered_df_pca['cluster'] == 0, 'Surprised', np.nan)
filtered_df_pca['mood'] = np.where(filtered_df_pca['cluster'] == 1, 'Angry', filtered_df_pca['mood'])
filtered_df_pca['mood'] = np.where(filtered_df_pca['cluster'] == 2, 'Happy', filtered_df_pca['mood'])
filtered_df_pca['mood'] = np.where(filtered_df_pca['cluster'] == 3, 'Disgust', filtered_df_pca['mood'])
filtered_df_pca['mood'] = np.where(filtered_df_pca['cluster'] == 4, 'Sad', filtered_df_pca['mood'])
filtered_df_pca['mood'] = np.where(filtered_df_pca['cluster'] == 5, 'Fear', filtered_df_pca['mood'])
filtered_df_pca['mood'] = np.where(filtered_df_pca['cluster'] == 6, 'Neutral', filtered_df_pca['mood'])

#print(filtered_df_pca)



face_classifier = cv2.CascadeClassifier(r'C:\Users\hi\Documents\sem 6\proj\emotionDetection\haarcascade_frontalface_default.xml')
emotion_classifier = load_model(r'C:\Users\hi\Documents\sem 6\proj\emotionDetection\model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            return label, (x, y, w, h)
    return None, None

def main():
    cap = cv2.VideoCapture(0)

    start_time = time.time()
    detected_emotion = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_emotion, face_coords = detect_emotion(frame)
        
        if detected_emotion:
            if face_coords:
                x, y, w, h = face_coords
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, detected_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            #print("Detected Emotion:", detected_emotion)

        cv2.imshow('Emotion Detector', frame)


        if time.time() - start_time >= 5:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return detected_emotion

main()



import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import ipywidgets as widgets
from sklearn.metrics.pairwise import cosine_similarity
import webbrowser

# Initialize Spotify client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="8259db9b0e8d4ac89d8d84833e6516ac",
                                               client_secret="009fcd47eca64b3cb5d993e3cf94b422",
                                               redirect_uri="http://localhost:8000",
                                               scope="user-library-read playlist-modify-public user-modify-playback-state"), requests_timeout=10, retries=10)

# Define the filtered_df_pca DataFrame or ensure it is accessible here

# Function to recommend songs based on detected mood
def recommend_songs_by_mood(detected_mood, filtered_df_pca):
    # Filter DataFrame based on detected mood
    filtered_by_mood = filtered_df_pca[filtered_df_pca['mood'] == detected_mood]
    # Select 10 random songs from the filtered DataFrame
    random_songs = filtered_by_mood.sample(26)
    return random_songs

# Detected mood (for testing purpose)
detected_mood = main()

# Get recommended songs
recommended_songs = recommend_songs_by_mood(detected_mood, filtered_df_pca)

# Function to create playlist and add recommended songs
def create_and_add_to_playlist(songs_df, playlist_name):
    # Create a new playlist
    playlist = sp.user_playlist_create(sp.current_user()['id'], playlist_name, public=True)
    # Extract URIs of recommended songs
    uris = songs_df['uri'].tolist()
    # Add songs to the playlist
    sp.playlist_add_items(playlist['id'], uris)
    # Get Spotify URI of the created playlist
    playlist_uri = sp.user_playlist(sp.current_user()['id'], playlist['id'])['uri']
    return f"spotify:playlist:{playlist_uri.split(':')[-1]}"

# Create a playlist and add recommended songs
playlist_name = f"Recommended Songs for {detected_mood}"
playlist_uri = create_and_add_to_playlist(recommended_songs, playlist_name)

# Start playback of the playlist on the specified device
device_id = "d50f61d10230b0d4627f7d0ba4b050203a75e2be"
sp.start_playback(device_id=device_id, context_uri=playlist_uri)

print("Playlist created successfully.")
print("Spotify URI for the playlist:", playlist_uri)

# if __name__ == '__main__':
#     main()