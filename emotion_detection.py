# emotion_utils.py""
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import datetime
import os
# Mappage des émotions détectées vers états personnalisés
emotion_mapping = {
    'anger': 'stressed',
    'fear': 'anxious',
    'happiness': 'confident',
    'sadness': 'unconfident',
    'neutral': 'relaxed',
    'surprise': 'enthusiastic'
}

# Index des classes d’émotion
class_indices = {
    0: 'anger',
    1: 'fear',
    2: 'happiness',
    3: 'sadness',
    4: 'neutral',
    5: 'surprise'
}

EMOTION_DIR = "emotion_detections"
if not os.path.exists(EMOTION_DIR):
    os.makedirs(EMOTION_DIR)

def save_emotion_detection_image(frame, emotion, mapped_emotion, confidence):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{EMOTION_DIR}/emotion_{emotion}_{timestamp}_{int(confidence*100)}.jpg"
    cv2.imwrite(filename, frame)
    return filename

def load_emotion_model():
    return load_model('vgg_high.h5')

def detect_emotion_in_frame(model, frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None, 0.0
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.astype('float32') / 255.0
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.repeat(face_roi, 3, axis=-1)
        predictions = model.predict(face_roi)[0]
        emotion_idx = np.argmax(predictions)
        emotion = class_indices[emotion_idx]
        mapped_emotion = emotion_mapping[emotion]
        confidence = np.max(predictions)
        return emotion, mapped_emotion, confidence
    return None, None, 0.0

