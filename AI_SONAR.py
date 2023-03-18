import time
import numpy as np
import pyaudio
import wave
import struct
import math
import librosa

# Set the sampling frequency and duration of the recording
fs = 44100
duration = 1

# Set the distance between the two microphones in meters
mic_distance = 0.1

# Record audio from the built-in microphone of the web camera
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)
    print('Recording...')
    frames = []
    for i in range(0, int(fs / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    print('Recording finished')
    stream.stop_stream()
    stream.close()
    p.terminate()
    return b''.join(frames)

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
def extract_features(data):
    X, sample_rate = librosa.load(np.frombuffer(data, dtype=np.int16), sr=fs, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0)
    return features

# Classify the sound using a simple rule-based method
def classify_sound(features):
    threshold = 0.5
    if features[0] > threshold:
        return 'Human speech'
    elif features[1] > threshold:
        return 'Animal sound'
    else:
        return 'Other sound'

# Main function
if __name__ == '__main__':
    with wave.open('microphone1.wav', 'wb') as wave_file1, wave.open('microphone2.wav', 'wb') as wave_file2:
        wave_file1.setnchannels(1)
        wave_file1.setsampwidth(2)
        wave_file1.setframerate(fs)
        wave_file2.setnchannels(1)
        wave_file2.setsampwidth(2)
        wave_file2.setframerate(fs)
        while True:
            input('Press Enter to record audio...')
            data = record_audio()
            wave_file1.writeframes(data)
            wave_file2.writeframes(data)
            data1 = wave_file1.readframes(fs * duration)
            data2 = wave_file2.readframes(fs * duration)
            delay = calculate_delay(data1, data2)
            distance = calculate_distance(delay)
            print(f'Distance: {distance - mic_distance:.2f} meters')
            features = extract_features(data)
            sound_type = classify_sound(features)
            print(f'Sound type: {sound_type}')
            time.sleep(1)
