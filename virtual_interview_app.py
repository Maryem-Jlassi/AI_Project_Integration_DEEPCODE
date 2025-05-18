from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import uuid
import time
import json
import pyttsx3
import threading
import random
import speech_recognition as sr
import tempfile
import wave
import pyaudio
import numpy as np
import io

# Create the Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'virtual_interview_secret_key'

# Constants
AUDIO_OUTPUT_FOLDER = os.path.join('static', 'audio')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(AUDIO_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global state
is_interview_active = False
current_app_id = None
current_question_index = 0

# Dictionary to store active interviews
active_interviews = {}

# Sample interview questions
INTERVIEW_QUESTIONS = [
    "Can you tell me about yourself and your background in programming?",
    "What programming languages and frameworks are you most comfortable with?",
    "Could you describe a challenging project you've worked on and how you overcame obstacles?",
    "How do you approach debugging a complex issue in your code?",
    "What's your experience with version control systems like Git?",
    "How do you stay updated with the latest technologies and industry trends?",
    "Can you explain how you would design a scalable web application?",
    "What's your approach to writing clean, maintainable code?",
    "How do you handle feedback on your code during code reviews?",
    "Do you have any questions for me about the position or company?"
]

# Sample follow-up responses
FOLLOW_UP_RESPONSES = [
    "Thank you for sharing that. Let's move on to the next question.",
    "That's interesting. I'd like to ask you about something else now.",
    "I appreciate your detailed response. Let's continue with another topic.",
    "Great, that gives me a good understanding. Moving forward...",
    "Thank you for your insights. Let's explore another area."
]

# Initialize text-to-speech engine with David voice
def initialize_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    
    # Set voice to David (for Windows)
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'david' in voice.name.lower():
            print(f"Found David voice: {voice.name}")
            engine.setProperty('voice', voice.id)
            break
    
    return engine

# Generate speech audio file
def generate_speech_audio(text, filename):
    engine = initialize_tts_engine()
    
    # Generate unique audio file path
    audio_path = os.path.join(AUDIO_OUTPUT_FOLDER, filename)
    
    # Save speech to file
    print(f"Generating speech audio: '{text[:30]}...' to {audio_path}")
    engine.save_to_file(text, audio_path)
    engine.runAndWait()
    
    return f"/static/audio/{filename}"

# Generate interviewer response based on candidate's input
def generate_interviewer_response(transcription, interview_id):
    # Get the current question index for this interview
    if interview_id not in active_interviews:
        return "I'm sorry, but this interview session has expired."
    
    interview_data = active_interviews[interview_id]
    question_index = interview_data.get('question_index', 0)
    
    # Generate a follow-up response
    follow_up = random.choice(FOLLOW_UP_RESPONSES)
    
    # Move to the next question
    new_index = (question_index + 1) % len(INTERVIEW_QUESTIONS)
    active_interviews[interview_id]['question_index'] = new_index
    
    # If we've reached the end, the next call will finish the interview
    if new_index == 0:
        return "Thank you for your responses. That concludes our interview questions. Do you have any questions for me?"
    
    # Otherwise, return a follow-up and the next question
    return f"{follow_up} {INTERVIEW_QUESTIONS[new_index]}"

# Check if the interview should be finished
def check_if_interview_finished(interview_id):
    if interview_id not in active_interviews:
        return True
    
    interview_data = active_interviews[interview_id]
    question_index = interview_data.get('question_index', 0)
    
    # If we've gone through all questions and are back at the beginning
    if question_index == 0:
        # Mark this interview as completed
        active_interviews[interview_id]['completed'] = True
        return True
    
    return False

# Transcribe audio to text using speech recognition
def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    
    # Check file extension
    file_ext = os.path.splitext(audio_file_path)[1].lower()
    wav_file_path = audio_file_path
    
    # If not already a WAV file, convert it
    if file_ext != '.wav':
        try:
            print(f"Converting {file_ext} to wav format using PyAudio")
            
            # Create a temporary file for the converted wav
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                wav_file_path = temp_wav.name
            
            # Read the input file as binary
            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()
            
            # For webm files, we need to handle them specially
            # Since we're getting the audio directly from the browser via Web Speech API,
            # we'll rely on the client-side transcription instead of trying to process webm files
            if file_ext == '.webm':
                print("Webm format detected - using client-side transcription instead")
                return "Using client-side transcription"
            
            # For other formats, we can try to convert them to WAV using PyAudio
            # This is a simplified approach and may not work for all formats
            # In a production environment, you might want to use a more robust solution
            try:
                # Create a WAV file with standard parameters
                with wave.open(wav_file_path, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(16000)  # 16kHz
                    wf.writeframes(audio_data)
                print(f"Converted audio to {wav_file_path}")
            except Exception as e:
                print(f"Error converting to WAV: {e}")
                return "Sorry, I couldn't process your audio format."
        except Exception as e:
            print(f"Error handling audio file: {e}")
            return "Sorry, I couldn't process your audio format."
    
    try:
        with sr.AudioFile(wav_file_path) as source:
            # Adjust for ambient noise and record
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
            
            # Use Google's speech recognition service
            # You can also use other services like recognizer.recognize_sphinx() for offline recognition
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
        return "I couldn't understand what you said. Could you please repeat?"
    except sr.RequestError as e:
        print(f"Could not request results from Speech Recognition service; {e}")
        return "Sorry, the speech recognition service is currently unavailable."
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        return "There was an error processing your speech."
    finally:
        # Clean up the temporary wav file if it was created
        if file_ext != '.wav' and os.path.exists(wav_file_path):
            try:
                os.remove(wav_file_path)
                print(f"Removed temporary wav file: {wav_file_path}")
            except Exception as e:
                print(f"Error removing temporary wav file: {e}")
                # Continue processing even if cleanup fails

# Routes
@app.route('/')
def index():
    """Serves the interview HTML page."""
    return render_template('virtual_interview.html')

@app.route('/api/start_interview', methods=['POST'])
def start_interview():
    try:
        # Generate a new interview ID
        interview_id = str(uuid.uuid4())
        
        # Get candidate name from request
        data = request.json
        candidate_name = data.get('name', 'Candidate')
        
        print(f"Starting new interview for {candidate_name} with ID: {interview_id}")
        
        # Initialize the interview data
        active_interviews[interview_id] = {
            'candidate_name': candidate_name,
            'question_index': 0,
            'start_time': time.time(),
            'completed': False
        }
        
        # Generate the first question
        first_question = INTERVIEW_QUESTIONS[0]
        greeting = f"Hello {candidate_name}, I'm David, your virtual interviewer today. I'll be asking you a series of questions to learn more about your skills and experience. Let's begin with the first question. {first_question}"
        
        # Generate audio for the greeting
        audio_filename = f"greeting_{uuid.uuid4()}.wav"
        audio_url = generate_speech_audio(greeting, audio_filename)
        
        return jsonify({
            'interview_id': interview_id,
            'text': greeting,
            'audio_url': audio_url
        })
    except Exception as e:
        print(f"Error starting interview: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_audio_message', methods=['POST'])
def process_audio_message():
    """Processes audio data from the candidate and returns an AI response."""
    global current_app_id, is_interview_active, current_question_index
    
    try:
        # Check if there's an audio file in the request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file received'}), 400
        
        audio_file = request.files['audio']
        print(f"Received audio file: {audio_file.filename}, size: {request.content_length} bytes")
        
        # Determine the file extension from the content type
        content_type = audio_file.content_type
        file_ext = '.webm'  # Default extension
        
        if content_type == 'audio/wav' or content_type == 'audio/x-wav':
            file_ext = '.wav'
        elif content_type == 'audio/mp3' or content_type == 'audio/mpeg':
            file_ext = '.mp3'
        elif content_type == 'audio/ogg':
            file_ext = '.ogg'
        
        print(f"Audio content type: {content_type}, using extension: {file_ext}")
        client_transcription = request.form.get('transcription', '')
        
        # Save the audio file temporarily
        temp_audio_path = os.path.join(AUDIO_OUTPUT_FOLDER, f"temp_{uuid.uuid4()}{os.path.splitext(audio_file.filename)[1]}")
        audio_file.save(temp_audio_path)
        
        # Use client-side transcription if available, otherwise transcribe server-side
        transcription = client_transcription
        if not transcription or transcription.strip() == '':
            # Transcribe the audio to text
            transcription = transcribe_audio(temp_audio_path)
        
        print(f"Transcription: {transcription}")
        
        # Clean up the temporary file
        try:
            os.remove(temp_audio_path)
        except Exception as e:
            print(f"Error removing temporary file: {e}")
        
        # Generate a follow-up response and next question
        follow_up = random.choice(FOLLOW_UP_RESPONSES)
        
        # Move to the next question
        current_question_index = (current_question_index + 1) % len(INTERVIEW_QUESTIONS)
        
        # Check if we've reached the end of the interview
        if current_question_index == 0:
            # Last question reached, end the interview
            response_text = "Thank you for participating in this virtual interview. We've completed all the questions. I hope you enjoyed this experience. Have a great day!"
            audio_filename = f"final_{uuid.uuid4()}.wav"
            audio_url = generate_speech_audio(response_text, audio_filename)
            
            return jsonify({
                'text': response_text,
                'audio_url': audio_url,
                'transcription': transcription,
                'interview_finished': True
            })
        
        # Generate AI response with David voice
        response_text = f"{follow_up} {INTERVIEW_QUESTIONS[current_question_index]}"
        audio_filename = f"response_{uuid.uuid4()}.wav"
        
        # Generate audio file
        audio_url = generate_speech_audio(response_text, audio_filename)
        
        # Return response
        return jsonify({
            'text': response_text,
            'audio_url': audio_url,
            'transcription': transcription,
            'interview_finished': False
        })
    except Exception as e:
        print(f"Error processing audio message: {e}")
        return jsonify({'error': 'Failed to process audio message'}), 500

@app.route('/api/process_text_message', methods=['POST'])
def process_text_message():
    """Processes text messages from the candidate and returns an AI response."""
    global current_app_id, is_interview_active, current_question_index
    
    try:
        # Get the text message from the request
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'No text message received'}), 400
        
        message = data['text']
        print(f"Received text message: {message}")
        
        # Generate a follow-up response and next question
        follow_up = random.choice(FOLLOW_UP_RESPONSES)
        
        # Move to the next question
        current_question_index = (current_question_index + 1) % len(INTERVIEW_QUESTIONS)
        
        # Check if we've reached the end of the interview
        if current_question_index == 0:
            # Last question reached, end the interview
            response_text = "Thank you for participating in this virtual interview. We've completed all the questions. I hope you enjoyed this experience. Have a great day!"
            audio_filename = f"final_{uuid.uuid4()}.mp3"
            audio_url = generate_speech_audio(response_text, audio_filename)
            
            return jsonify({
                'text': response_text,
                'audio_url': audio_url,
                'interview_finished': True
            })
        
        # Generate AI response with David voice
        response_text = f"{follow_up} {INTERVIEW_QUESTIONS[current_question_index]}"
        audio_filename = f"response_{uuid.uuid4()}.mp3"
        
        # Generate audio file
        audio_url = generate_speech_audio(response_text, audio_filename)
        
        # Return response
        return jsonify({
            'text': response_text,
            'audio_url': audio_url
        })
    except Exception as e:
        print(f"Error processing text: {e}")
        return jsonify({'error': str(e)}), 500

# Add a route to serve static files directly
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    print("Starting the virtual interview Flask server...")
    # Create audio output directory if it doesn't exist
    os.makedirs(AUDIO_OUTPUT_FOLDER, exist_ok=True)
    # Use a different port to avoid conflicts
    app.run(debug=True, host='127.0.0.1', port=5000)
