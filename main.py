import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import librosa
import soundfile
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import pyaudio
import wave
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Emotions to observe
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Initialize the Multi Layer Perceptron Classifier
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08,
                      hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

# Status update function
def update_status(message):
    status_label.config(text=message)

# Record audio function
def record_audio():
    """Record audio from the microphone and save to a WAV file."""
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 48000
    seconds = 5
    filename = "Predict-Record-Audio.wav"

    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
    frames = []

    update_status("Recording audio...")
    for _ in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    update_status("Recording complete.")
    return filename

# Feature extraction function
def extract_feature(file_name, mfcc, chroma, mel):
    """Extract features (mfcc, chroma, mel) from a sound file."""
    try:
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            result = np.array([])

            if chroma:
                stft = np.abs(librosa.stft(X))
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
    except Exception as e:
        messagebox.showerror("Error", f"Error extracting features: {e}")
        return None

# Data loading function
def load_data(test_size=0.2):
    """Load the data and extract features for each sound file."""
    x, y = [], []
    for file in glob.glob("Dataset/Actor_*/*.wav"):
        try:
            file_name = os.path.basename(file)
            emotion = emotions[file_name.split("-")[2]]

            if emotion not in observed_emotions:
                continue
            feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
            if feature is not None:
                x.append(feature)
                y.append(emotion)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    if len(x) == 0:
        messagebox.showerror("Error", "No data found. Please check your dataset path and files.")
        return None, None, None, None
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Training function
def train_model():
    """Train the emotion recognition model."""
    x_train, x_test, y_train, y_test = load_data(test_size=0.25)
    if x_train is None or len(x_train) == 0:
        messagebox.showerror("Error", "Insufficient data to train the model.")
        return

    # Perform cross-validation
    update_status("Performing cross-validation...")
    cross_validate_model(model, x_train, y_train)

    update_status("Training model...")
    model.fit(x_train, y_train)

    # Ask user to save the trained model
    save_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle Files", "*.pkl")])
    if save_path:
        save_model(model, save_path)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    update_status(f"Training complete. Model Accuracy: {accuracy*100:.2f}%")
    messagebox.showinfo("Training Complete", f"Model Accuracy: {accuracy*100:.2f}%")

# Cross-validation function
def cross_validate_model(model, x, y):
    """Perform cross-validation on the model."""
    scores = cross_val_score(model, x, y, cv=5)
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean()}")

# Model saving function
def save_model(model, filename):
    """Save the trained model to a file."""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# Model loading function
def load_model(filename):
    """Load a saved model from a file."""
    if os.path.exists(filename):
        print(f"Loading model from {filename}")
        return joblib.load(filename)
    else:
        print(f"Model file {filename} not found.")
        return None

# Prediction function
def predict_emotion(file):
    """Predict the emotion from an audio file."""
    model_path = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
    if not model_path:
        messagebox.showerror("Error", "No model selected. Please select a model file.")
        return None

    model = load_model(model_path)
    if model is None:
        messagebox.showerror("Error", "Failed to load the model. Please ensure the file is valid.")
        return None

    feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
    if feature is not None:
        prediction = model.predict([feature])[0]
        return prediction
    return None

# Record and predict function
def record_and_predict():
    """Record audio and predict emotion."""
    file = record_audio()
    prediction = predict_emotion(file)
    if prediction:
        display_prediction(prediction)

# Predict from file function
def predict_from_file():
    """Select an audio file and predict emotion."""
    file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if not file:
        return
    prediction = predict_emotion(file)
    if prediction:
        display_prediction(prediction)

# Display prediction function
def display_prediction(emotion):
    emoji_map = {
        'calm': "😌",
        'happy': "😊",
        'fearful': "😱",
        'disgust': "🤬"
    }
    emoji = emoji_map.get(emotion, "")
    messagebox.showinfo("Prediction", f"Emotion Predicted: {emotion} {emoji}")

# Plot waveform function
def plot_waveform(file):
    y, sr = librosa.load(file)
    fig = Figure(figsize=(5, 2))
    ax = fig.add_subplot(111)
    ax.plot(y)
    ax.set_title("Waveform")
    canvas = FigureCanvasTkAgg(fig, master=app)
    canvas.get_tk_widget().pack()

# GUI Setup
app = tk.Tk()
app.title("Speech Emotion Recognition")
app.geometry("500x600")

logo = tk.PhotoImage(file="C:\\Users\\hp\\shivank\\final project Major\\logo.png")
logo_label = tk.Label(app, image=logo)
logo_label.pack(pady=10)

status_label = tk.Label(app, text="Welcome to Speech Emotion Recognition", font=("Arial", 12))
status_label.pack(pady=10)

train_button = tk.Button(app, text="Train Model", font=("Arial", 14), command=train_model)
train_button.pack(pady=10)

record_button = tk.Button(app, text="Record & Predict", font=("Arial", 14), command=record_and_predict)
record_button.pack(pady=10)

predict_button = tk.Button(app, text="Predict from File", font=("Arial", 14), command=predict_from_file)
predict_button.pack(pady=10)

quit_button = tk.Button(app, text="Quit", font=("Arial", 14), command=app.quit)
quit_button.pack(pady=20)

app.mainloop()
