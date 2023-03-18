import time
import numpy as np
import pyaudio
import librosa
import keras.models
from keras.applications.vgg16 import preprocess_input

# Set the sampling frequency and duration of the recording
fs = 44100
duration = 1

# Set the distance between the two microphones in meters
mic_distance = 0.1

# Load the pre-trained VGG16 model for audio classification
model = keras.models.load_model('audio_classifier.h5')

# Define the class labels for the model output
labels = ['Human speech', 'Animal sound', 'Other sound']

# Record audio from two microphones
def record_audio():
    p = pyaudio.PyAudio()
    stream1 = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)
    stream2 = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)
    print('Recording...')
    frames1 = []
    frames2 = []
    for i in range(0, int(fs / 1024 * duration)):
        data1 = stream1.read(1024)
        data2 = stream2.read(1024)
        frames1.append(data1)
        frames2.append(data2)
    print('Recording finished')
    stream1.stop_stream()
    stream2.stop_stream()
    stream1.close()
    stream2.close()
    p.terminate()
    return b''.join(frames1), b''.join(frames2)

# Calculate the time delay between the arrival of the sound at the two microphones
def calculate_delay(data1, data2):
    corr = np.correlate(np.frombuffer(data1, dtype=np.int16), np.frombuffer(data2, dtype=np.int16), mode='full')
    delay_idx = np.argmax(corr) - (len(data1) - 1)
    delay = delay_idx / fs
    return delay

# Calculate the distance between the two microphones using the time delay and the speed of sound
def calculate_distance(delay):
    speed_of_sound = 343 # meters per second
    distance = speed_of_sound * delay / 2
    return distance

# Extract relevant features from the audio data
def extract_features(audio_data):
    X, sample_rate = librosa.load(audio_data, sr=fs, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0)
    return features

# Classify the sound using the pre-trained VGG16 model
def classify_sound(features):
    x = np.expand_dims(features, axis=0)
    x = preprocess_input(x)
    y = model.predict(x)
    return labels[np.argmax(y)]

# Main function
if __name__ == '__main__':
    while True:
        input('Press Enter to record audio...')
        data1, data2 = record_audio()
        delay = calculate_delay(data1, data2)
        distance = calculate_distance(delay)
        print(f'Distance: {distance - mic_distance:.2f} meters')
        features = extract_features(data1)
        sound_type = classify_sound(features)
        print(f'Sound type: {sound_type}')
       time.sleep(1)
