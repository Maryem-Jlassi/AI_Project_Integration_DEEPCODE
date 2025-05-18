import speech_recognition as sr
import pyttsx3
import tempfile
import pygame
import os
import logging

def init_speech_modules():
    """Initializes the speech synthesis and recognition modules."""
    logging.info("Initializing voice modules...")

    # Initialize pygame for audio playback
    pygame.mixer.init()

    # Initialize the speech recognizer
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8

    logging.info("Voice modules initialized.")
    return recognizer

def speak(text: str) -> str:
    """Synthesizes speech from text using pyttsx3 with David voice and returns audio path."""
    try:
        # Initialize the TTS engine
        engine = pyttsx3.init()
        
        # Set David voice
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'David' in voice.name:
                engine.setProperty('voice', voice.id)
                break
        
        # Create temp file
        temp_path = tempfile.mktemp(suffix='.wav')
        
        # Save to file
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        
        return temp_path
    except Exception as e:
        logging.error(f"pyttsx3 synthesis error: {e}")
        raise

def listen_for_response(prompt: str = None, timeout: int = 30) -> str:
    """
    Listens for the candidate's voice response with improved detection.
    """
    recognizer = sr.Recognizer()
    
    if prompt:
        print(prompt)

    print("I'm listening! (it's your turn to speak 🎤)")

    try:
        with sr.Microphone() as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            recognizer.pause_threshold = 2.2  # Longer delay for relaxed conversation

            try:
                audio = recognizer.listen(source, timeout=timeout)
            except sr.WaitTimeoutError:
                print("No worries, you can take your time to answer.")
                return None

            try:
                response = recognizer.recognize_google(audio, language='en-US')
                print(f"Great! I heard: {response}")
                return response
            except sr.UnknownValueError:
                print("Sorry, I didn't quite catch what you said.")
                return None
            except sr.RequestError:
                print("Oops, small technical problem with voice recognition.")
                return None
    except Exception as e:
        print(f"Technical issue: {e}")
        print("No worries, you can type your answer instead!")
        return None
    