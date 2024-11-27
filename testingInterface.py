import pyaudio
import wave
import numpy as np
import librosa
import soundfile
from sklearn.neural_network import MLPClassifier
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox

# Feature extraction function
def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

# Record audio function
def record_audio(file_name, duration=5, sample_rate=44100, channels=1):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=sample_rate,
                    input=True, frames_per_buffer=1024)
    frames = []
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

# Load the model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Predict emotion function
def predict_emotion(file_name, model_path):
    model = load_model(model_path)
    feature = extract_feature(file_name, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    prediction = model.predict(feature)
    return prediction[0]

# GUI Application
class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mental State Detector")
        self.root.geometry("400x300")
        
        # File path for saved recording
        self.live_audio_path = "live_audio.wav"
        self.model_path = "D:\SpeechEmotionRecogniser\Speech_Emotion_Detection_Model.pkl"

        # Heading
        tk.Label(root, text="Emotion Recogniser", font=("Helvetica", 16, "bold")).pack(pady=10)

        # Record button
        tk.Button(root, text="Record Audio", font=("Helvetica", 12), command=self.record_audio).pack(pady=10)
        
        # Predict button
        tk.Button(root, text="Predict Emotion", font=("Helvetica", 12), command=self.predict_emotion).pack(pady=10)

        # Result Label
        self.result_label = tk.Label(root, text="", font=("Helvetica", 14, "italic"))
        self.result_label.pack(pady=20)

    def record_audio(self):
        try:
            record_audio(self.live_audio_path, duration=5)
            messagebox.showinfo("Success", "Audio Recorded Successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to record audio: {e}")

    def predict_emotion(self):
        try:
            emotion = predict_emotion(self.live_audio_path, self.model_path)
            self.result_label.config(text=f"Predicted Emotion: {emotion}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict Emotion: {e}")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()
