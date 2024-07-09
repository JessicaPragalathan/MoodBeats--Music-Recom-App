# from flask import Flask, request, jsonify, render_template
# import cv2
# import numpy as np
# from keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# import base64


# app = Flask(__name__, static_folder='static')



# face_classifier = cv2.CascadeClassifier(r'C:\Users\hi\Documents\sem 6\proj\emotionDetection\haarcascade_frontalface_default.xml')
# emotion_classifier = load_model(r'C:\Users\hi\Documents\sem 6\proj\emotionDetection\model.h5')
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# def detect_emotion(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

#         if np.sum([roi_gray]) != 0:
#             roi = roi_gray.astype('float') / 255.0
#             roi = img_to_array(roi)
#             roi = np.expand_dims(roi, axis=0)

#             print("Emotion Detection: Image data processed successfully.")
#             prediction = emotion_classifier.predict(roi)[0]
#             label = emotion_labels[prediction.argmax()]
#             return label
#     return None
# @app.route('/')
# def login():
#     return render_template('login.html')
# @app.route('/index')
# def index():
#     return render_template('index.html')

# @app.route('/detect_emotion', methods=['POST'])
# def process_image():
#     try:
#         data = request.get_json()
#         image_data = data.get('image_data')

#         # Print the first 100 characters of the image data
#         print("Received Image Data:", image_data[:100])

#         # Convert base64 image data to numpy array
#         nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         detected_emotion = detect_emotion(frame)
#         if detected_emotion:
#             print("Detected Emotion:", detected_emotion)
#             return jsonify({'emotion': detected_emotion})
#         else:
#             print("No emotion detected.")
#             return jsonify({'error': 'No emotion detected.'})
#     except Exception as e:
#         print('Error:', str(e))
#         return jsonify({'error': 'Error processing image.'})




# if __name__ == '__main__':
#     # Load emotion detection model
#     emotion_classifier = load_model(r'C:\Users\hi\Documents\sem 6\proj\emotionDetection\model.h5')
#     print('Emotion detection model loaded successfully.')
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template, redirect, url_for
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

app = Flask(__name__, static_folder='static')

# Load models
face_classifier = cv2.CascadeClassifier(r'C:\Users\hi\Documents\sem 6\proj\emotionDetection\haarcascade_frontalface_default.xml')
emotion_classifier = load_model(r'C:\Users\hi\Documents\sem 6\proj\emotionDetection\model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load and preprocess the dataset
df = pd.read_csv('genres_v2.csv')
df = df.dropna(subset=['song_name', 'uri']).drop_duplicates(subset=['song_name', 'uri'])
cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'uri', 'genre', 'song_name']
filtered_df = df[cols]
num_cols = [col for col in filtered_df.columns if filtered_df[col].dtype != 'object']
scaler = StandardScaler()
filtered_df[num_cols] = scaler.fit_transform(filtered_df[num_cols])
X = filtered_df.drop(['uri', 'genre', 'song_name'], axis=1)
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
filtered_df['cluster'] = kmeans.fit_predict(X)
n_components = 6
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(filtered_df[num_cols])
filtered_df['mood'] = np.select(
    [filtered_df['cluster'] == i for i in range(n_clusters)],
    ['Surprised', 'Angry', 'Happy', 'Disgust', 'Sad', 'Fear', 'Neutral']
)

# Initialize Spotify client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="8259db9b0e8d4ac89d8d84833e6516ac",
                                               client_secret="009fcd47eca64b3cb5d993e3cf94b422",
                                               redirect_uri="http://localhost:8000",
                                               scope="user-library-read playlist-modify-public user-modify-playback-state"))

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
            return label
    return None

def recommend_songs_by_mood(detected_mood, filtered_df):
    filtered_by_mood = filtered_df[filtered_df['mood'] == detected_mood]
    random_songs = filtered_by_mood.sample(26)
    return random_songs

def create_and_add_to_playlist(songs_df, playlist_name):
    playlist = sp.user_playlist_create(sp.current_user()['id'], playlist_name, public=True)
    uris = songs_df['uri'].tolist()
    sp.playlist_add_items(playlist['id'], uris)
    playlist_uri = sp.user_playlist(sp.current_user()['id'], playlist['id'])['uri']
    return f"spotify:playlist:{playlist_uri.split(':')[-1]}"

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        image_data = data.get('image_data')
        nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        detected_emotion = detect_emotion(frame)
        if detected_emotion:
            recommended_songs = recommend_songs_by_mood(detected_emotion, filtered_df)
            playlist_name = f"Moodlist: {detected_emotion}"
            playlist_uri = create_and_add_to_playlist(recommended_songs, playlist_name)
            return jsonify({'emotion': detected_emotion, 'playlist_uri': playlist_uri})
        else:
            return jsonify({'error': 'No emotion detected.'})
    except Exception as e:
        print('Error:', str(e))
        return jsonify({'error': 'Error processing image.'})

@app.route('/play_playlist/<playlist_uri>')
def play_playlist(playlist_uri):
    device_id = "d50f61d10230b0d4627f7d0ba4b050203a75e2be"
    sp.start_playback(device_id=device_id, context_uri=playlist_uri)
    return redirect(f"https://open.spotify.com/playlist/{playlist_uri.split(':')[-1]}")

if __name__ == '__main__':
    app.run(debug=True)
