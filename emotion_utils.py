import torch
import numpy as np
import librosa
from transformers import Wav2Vec2FeatureExtractor
from hubert_model import HubertForSER
from collections import Counter
import os
MODEL_PATH = os.path.join('static', 'models', 'hubert_full_model.pt')

EMOTION_NAMES = {
    0: "happy",
    1: "fear",
    2: "surprise",
    3: "sadness",
    4: "neutral",
    5: "anger",
    6: "disgust"
}

def load_audio(file_path, sample_rate=16000, max_length=250000):
    """Load and preprocess an audio file for emotion recognition."""
    waveform, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    if len(waveform) > max_length:
        waveform = waveform[:max_length]
    else:
        waveform = np.pad(waveform, (0, max_length - len(waveform)), 'constant')
    return torch.from_numpy(waveform).float()

def predict_emotion(audio_path, model_path=MODEL_PATH, fine_tuning_type="full", num_emotions=7, hidden_size=256):
    """
    Predict emotion from an audio file using a trained HuBERT model.
    Returns (emotion_index, confidence, emotion_name)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model (do not reload if already loaded)
    model = HubertForSER(num_emotions=num_emotions, fine_tuning_type=fine_tuning_type, hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    waveform = load_audio(audio_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    inputs = feature_extractor(waveform.numpy().reshape(1, -1), sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = torch.ones(input_values.shape, device=device)
    with torch.no_grad():
        logits = model(input_values, attention_mask)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    emotion_name = EMOTION_NAMES.get(predicted_class, f"Unknown ({predicted_class})")
    return predicted_class, confidence, emotion_name

def get_most_frequent_emotion(emotion_list):
    """Return the most frequent emotion name from a list of emotion names."""
    if not emotion_list:
        return "N/A"
    counter = Counter(emotion_list)
    return counter.most_common(1)[0][0]

def detect_vocal_emotion(audio_path):
    """
    Detects the vocal emotion from an audio file.
    Returns (emotion_name, confidence)
    """
    _, confidence, emotion_name = predict_emotion(audio_path)
    return emotion_name, confidence

def detect_phone_in_frame(phone_model, frame):
    """
    Detects if a phone is present in the given frame.
    Returns (detected: bool, confidence: float)
    """
    # Placeholder: always returns False, 0.0
    return False, 0.0 