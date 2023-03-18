import time
import numpy as np
import pyaudio
import speech_recognition as sr

# Set the sampling frequency and duration of the recording
fs = 44100
duration = 1

# Set the distance between the two microphones in meters
mic_distance = 0.1

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

# Recognize speech from the audio data using the Google Speech Recognition API
def recognize_speech(audio_data):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_data) as source:
        audio = recognizer.record(source)
    try:
        words = recognizer.recognize_google(audio)
        return words
    except sr.UnknownValueError:
        return None

# Main function
if __name__ == '__main__':
    while True:
        input('Press Enter to record audio...')
        data1, data2 = record_audio()
        delay = calculate_delay(data1, data2)
        distance = calculate_distance(delay)
        print(f'Distance: {distance - mic_distance:.2f} meters')
        words = recognize_speech(data1)
        if words:
            print(f'Detected words: {words}')
        else:
            print('No words detected')
        time.sleep(1)
