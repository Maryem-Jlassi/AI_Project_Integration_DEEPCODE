import sys
import time
import random
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import speech_recognition as sr
import os
import pyttsx3
import uuid
from datetime import datetime
import tempfile
import json

# Fix Windows path issues
if sys.platform == "win32":
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath

# Constants
AUDIO_OUTPUT_FOLDER = os.path.join('static', 'audio')
os.makedirs(AUDIO_OUTPUT_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Utility functions
def log_interview_event(application_id: str, event_type: str, description: str) -> None:
    """Log an interview event to the application's JSON file"""
    app_data_dir = os.path.join("application_data", application_id)
    os.makedirs(app_data_dir, exist_ok=True)
    application_path = os.path.join(app_data_dir, "application_details.json")
    try:
        app_data = {}
        if os.path.exists(application_path):
            with open(application_path, 'r', encoding='utf-8') as f:
                try: app_data = json.load(f)
                except json.JSONDecodeError: app_data = {}
        if 'interview_logs' not in app_data: app_data['interview_logs'] = []
        event = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), 'type': event_type, 'description': description}
        app_data['interview_logs'].append(event)
        with open(application_path, 'w', encoding='utf-8') as f:
            json.dump(app_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Error logging interview event for app {application_id}: {e}")

# Dummy functions for model loading
def load_models_dummy():
    logger.info("Using dummy model loader.")
    return None, None, None, None, None, None, None

def generate_question_dummy(domain, difficulty):
    logger.info(f"Using dummy question generator for {domain} ({difficulty}).")
    return f"This is a dummy {difficulty} question about {domain}."

def evaluate_response_dummy(question, response):
    logger.info(f"Using dummy response evaluator for: {response[:30]}...")
    return {"score": 0.6, "feedback": "This is dummy feedback. Your answer was noted."}


class AITechnicalInterviewer:
    def __init__(self, config_path: str = "interview_config.json"):
        logger.info("Initializing AI Technical Interview System...")
        self.application_id = str(uuid.uuid4())  # Internal APP ID
        log_interview_event(self.application_id, "system_init", f"Interviewer initialized with AppID: {self.application_id}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default config.")
            self.config = {
                "tts_rate": 150, 
                "tts_volume": 0.9, 
                "tts_voice_name": "David",
                "speech_recognition_language": "en-US",
                "num_technical_questions": 3
            }
        except Exception as e:
            logger.error(f"Failed to load config: {e}. Using default config.")
            self.config = {
                "tts_rate": 150, 
                "tts_volume": 0.9, 
                "tts_voice_name": "David", 
                "speech_recognition_language": "en-US", 
                "num_technical_questions": 3
            }

        # Using dummy versions for model loading
        try:
            (self.tokenizer, self.language_model, self.tech_eval_model, self.question_generator_tokenizer,
            self.question_generator, self.tfidf_vectorizer, self.answering_pipeline) = load_models_dummy()
            logger.info("Models loaded (or dummied).")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

        self.recognizer = sr.Recognizer()
        self.knowledge_base = self._load_knowledge_base_internal()

        self.candidate_name = None
        self.position = None
        self.domains = None
        self.interview_history = []
        self.last_transcription = None
        self.interview_stage = "initial"  # initial -> greeting -> background -> technical -> project -> closing -> completed
        self.current_question = None
        self.technical_questions_asked = 0
        self.is_finished = False
        
        # Simple evaluation metrics
        self.evaluation_metrics = {
            "technical_knowledge": 0,
            "technical_knowledge_count": 0,
            "communication": 0,
            "communication_count": 0,
            "problem_solving": 0,
            "problem_solving_count": 0
        }
        
        logger.info("AI Technical Interview System ready!")

    def _load_knowledge_base_internal(self) -> Dict[str, Dict[str, Any]]:
        """Load a simplified knowledge base for question generation"""
        return {
            "python": {
                "concepts": ["lists", "decorators", "generators", "context managers", "async/await"],
                "difficulty_levels": {
                    "easy": ["lists", "tuples", "dictionaries"],
                    "medium": ["decorators", "generators", "context managers"],
                    "hard": ["metaclasses", "async/await", "descriptors"]
                }
            },
            "machine learning": {
                "concepts": ["linear regression", "CNN", "RNN", "random forest", "clustering"],
                "difficulty_levels": {
                    "easy": ["linear regression", "decision trees", "clustering"],
                    "medium": ["CNN", "RNN", "random forest"],
                    "hard": ["transformer models", "reinforcement learning", "GANs"]
                }
            },
            "algorithms": {
                "concepts": ["sorting", "graphs", "dynamic programming", "greedy algorithms"],
                "difficulty_levels": {
                    "easy": ["sorting", "searching", "linked lists"],
                    "medium": ["graphs", "dynamic programming", "greedy algorithms"],
                    "hard": ["NP-hard problems", "approximation algorithms", "amortized analysis"]
                }
            }
        }

    def start_interview(self, candidate_name: str, position: str, domains: List[str], application_id: str = None) -> Dict[str, Any]:
        """
        Sets up the interview with candidate details.
        Returns first AI message with audio.
        """
        if application_id:
            self.application_id = application_id
            logger.info(f"Using external application ID: {self.application_id}")
        
        self.candidate_name = candidate_name
        self.position = position
        self.domains = domains
        self.interview_stage = "greeting"
        self.is_finished = False
        self.interview_history = []
        self.last_transcription = None
        self.technical_questions_asked = 0
        
        logger.info(f"Interview setup for {candidate_name}, Pos: {position}, Domains: {domains} (AppID: {self.application_id})")
        log_interview_event(self.application_id, "interview_setup", f"Setup for {candidate_name}, Pos: {position}, Domains: {domains}")
        
        # Generate first greeting message
        greeting = f"Hello {candidate_name}! I'm Alex, your virtual interviewer for the {position} position. Thanks for joining me today. To start, could you tell me a bit about your background and experience relevant to this role?"
        
        # Add to interview history
        self.interview_history.append({
            "speaker": "ai", 
            "message": greeting, 
            "timestamp": datetime.now().isoformat()
        })
        
        # Set current question and transition to background stage
        self.current_question = "Tell me about your background and experience."
        self.interview_stage = "background"
        
        # Return message with audio
        return self.get_message_with_audio(greeting)

    def process_audio_response(self, audio_data: bytes) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Processes audio from the candidate, transcribes it, and generates AI response.
        Returns (transcription, ai_response_dict)
        """
        logger.info(f"Processing audio data (AppID: {self.application_id})")
        self.last_transcription = None  # Reset before new transcription
        temp_file_path = None

        try:
            # Save received audio data to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', mode='wb') as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            with sr.AudioFile(temp_file_path) as source:
                audio = self.recognizer.record(source)
            
            try:
                # Use Google Speech Recognition API
                transcription = self.recognizer.recognize_google(
                    audio, 
                    language=self.config.get("speech_recognition_language", "en-US")
                )
                
                self.last_transcription = transcription
                logger.info(f"Transcription (AppID: {self.application_id}): {transcription}")
                
                # Add to interview history
                self.interview_history.append({
                    "speaker": "candidate", 
                    "message": transcription, 
                    "type": "audio_transcribed", 
                    "timestamp": datetime.now().isoformat()
                })
                
                log_interview_event(self.application_id, "candidate_audio_response", transcription)
                
                # Get AI response based on transcription
                ai_text_reply = self._get_next_ai_response(transcription)
                ai_response_dict = self.get_message_with_audio(ai_text_reply)
                
                return transcription, ai_response_dict

            except sr.UnknownValueError:
                logger.warning(f"Could not understand audio (AppID: {self.application_id})")
                self.last_transcription = None
                
                ai_text_reply = "I'm sorry, I couldn't quite catch what you said. Could you please repeat that or try typing your response?"
                ai_response_dict = self.get_message_with_audio(ai_text_reply)
                
                log_interview_event(self.application_id, "transcription_failed", "UnknownValueError")
                return None, ai_response_dict
                
            except sr.RequestError as e:
                logger.error(f"Speech recognition API error (AppID: {self.application_id}): {e}")
                self.last_transcription = None
                
                ai_text_reply = "There seems to be an issue with the speech recognition service. Please try typing your response instead."
                ai_response_dict = self.get_message_with_audio(ai_text_reply)
                
                log_interview_event(self.application_id, "transcription_failed", f"RequestError: {e}")
                return None, ai_response_dict
        
        except Exception as e:
            logger.error(f"Error processing audio (AppID: {self.application_id}): {e}")
            self.last_transcription = None
            
            ai_text_reply = "An unexpected error occurred while processing your audio. Please try typing your response."
            ai_response_dict = self.get_message_with_audio(ai_text_reply)
            
            log_interview_event(self.application_id, "audio_processing_error", str(e))
            return None, ai_response_dict
            
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def process_text_response(self, message: str) -> Dict[str, Any]:
        """
        Processes text message from candidate and generates AI response.
        Returns AI's response dict with text and audio.
        """
        logger.info(f"Processing text message (AppID: {self.application_id}): '{message}'")
        
        # Add to interview history
        self.interview_history.append({
            "speaker": "candidate", 
            "message": message, 
            "type": "text", 
            "timestamp": datetime.now().isoformat()
        })
        
        log_interview_event(self.application_id, "candidate_text_response", message)

        # Get AI response
        ai_text_reply = self._get_next_ai_response(message)
        
        # Add to interview history
        self.interview_history.append({
            "speaker": "ai", 
            "message": ai_text_reply, 
            "timestamp": datetime.now().isoformat()
        })
        
        log_interview_event(self.application_id, "ai_response_generated", ai_text_reply)
        
        # Return response dict with audio
        return self.get_message_with_audio(ai_text_reply)

    def _get_next_ai_response(self, candidate_input: str) -> str:
        """Determine AI's response based on current stage and candidate input."""
        ai_response = "I'm sorry, I'm not sure how to respond to that."  # Default fallback
        
        # If interview is already finished
        if self.is_finished:
            return "The interview has already concluded. Thank you for your time!"

        # GREETING STAGE: Send initial introduction and question
        if self.interview_stage == "greeting":
            ai_response = f"Hello {self.candidate_name}! I'm Alex, your virtual interviewer. Thanks for joining. To start, could you tell me a bit about your background and experience relevant to the {self.position} position?"
            self.current_question = "Tell me about your background and experience."
            self.interview_stage = "background"
            
        # BACKGROUND STAGE: Evaluate background response and move to technical
        elif self.interview_stage == "background":
            logger.info(f"Background response (AppID: {self.application_id}): {candidate_input[:50]}...")
            
            # Simplified evaluation for background question
            evaluation = evaluate_response_dummy(self.current_question, candidate_input)
            self._update_metrics("communication", evaluation['score'])

            ai_response = f"Thank you for sharing your background. I'd like to explore your technical knowledge a bit more. Let's move on to some technical questions related to {', '.join(self.domains)} for the {self.position} position."
            self.interview_stage = "technical"
            # Will fall through to ask first technical question
        
        # TECHNICAL STAGE: Ask technical questions and evaluate responses
        if self.interview_stage == "technical":
            if self.technical_questions_asked < self.config.get("num_technical_questions", 3):
                # If not the first technical question, evaluate previous answer
                if self.current_question and self.technical_questions_asked > 0:
                    logger.info(f"Tech response to '{self.current_question}' (AppID: {self.application_id}): {candidate_input[:50]}...")
                    evaluation = evaluate_response_dummy(self.current_question, candidate_input)
                    self._update_metrics("technical_knowledge", evaluation['score'])
                    
                    # Give some acknowledgment before asking next question
                    ai_response = f"Thank you for that answer. {evaluation['feedback']} "
                
                # Ask a new technical question
                domain_to_ask = random.choice(self.domains) if self.domains else "general"
                difficulty = random.choice(["easy", "medium"])
                new_question = generate_question_dummy(domain_to_ask, difficulty)
                
                self.current_question = new_question
                ai_response += new_question
                self.technical_questions_asked += 1
                
                # Transition to project stage after last technical question
                if self.technical_questions_asked >= self.config.get("num_technical_questions", 3):
                    self.interview_stage = "project"
            else:
                # Should have transitioned already
                self.interview_stage = "project"
                # Fall through to project stage

        # PROJECT STAGE: Ask about challenging project and evaluate response
        if self.interview_stage == "project":
            # If we already asked project question and now received answer
            if self.current_question and "challenging project" in self.current_question:
                logger.info(f"Project response (AppID: {self.application_id}): {candidate_input[:50]}...")
                evaluation = evaluate_response_dummy(self.current_question, candidate_input)
                self._update_metrics("problem_solving", evaluation['score'])
                
                ai_response = f"Thank you for sharing that project experience. {evaluation.get('feedback', '')} Do you have any questions for me about the role or the company?"
                self.current_question = None  # Candidate asks now
                self.interview_stage = "closing"
            else:
                # Ask project question
                ai_response = "Now I'd like to hear about your practical experience. Could you describe a challenging project you've worked on, your role in the project, and how you overcame any obstacles you encountered?"
                self.current_question = "Describe a challenging project you've worked on."
                # Stays in 'project' stage until answered

        # CLOSING STAGE: Answer candidate questions and conclude
        elif self.interview_stage == "closing":
            # Check if candidate asked a question
            if '?' in candidate_input and len(candidate_input) > 5:
                answer_to_candidate = self._answer_candidate_question_internal(candidate_input)
                ai_response = f"{answer_to_candidate} Do you have any other questions?"
                # Stays in closing stage
            else:
                # No more questions from candidate, conclude interview
                ai_response = f"Thank you for your time today, {self.candidate_name}! We've covered your background, technical skills, and project experience. The hiring team will review the interview and be in touch regarding next steps. Have a great day!"
                self.interview_stage = "completed"
                self.is_finished = True
                self.current_question = None

        return ai_response

    def _answer_candidate_question_internal(self, question: str) -> str:
        """Generate answers to candidate questions about the role/company"""
        logger.info(f"Answering candidate question (AppID: {self.application_id}): {question[:50]}...")
        
        # Simple pattern matching for common questions
        question_lower = question.lower()
        
        if "salary" in question_lower or "compensation" in question_lower:
            return "The compensation package is competitive and based on experience and qualifications. The HR team would discuss the specific details during the next stage of the hiring process."
            
        elif "remote" in question_lower or "work from home" in question_lower:
            return "The company offers flexible work arrangements including hybrid and remote options depending on the team and role requirements."
            
        elif "team size" in question_lower or "team structure" in question_lower:
            return "The team consists of about 8-10 members including developers, designers, and product managers, following an agile methodology."
            
        elif "tech stack" in question_lower or "technologies" in question_lower:
            return f"For this {self.position} role, the main technologies used include Python, JavaScript, and cloud services like AWS. The specific frameworks depend on the project requirements."
            
        elif "interview process" in question_lower or "next steps" in question_lower:
            return "After this technical interview, there would typically be a final round with the team lead and possibly other team members. The entire process usually takes 2-3 weeks."
            
        elif "company culture" in question_lower:
            return "The company values innovation, collaboration, and work-life balance. Regular team activities and professional development opportunities are encouraged."
            
        elif "position" in question_lower or "role" in question_lower or "job" in question_lower:
            return f"This {self.position} role focuses on developing key features for our main product, collaborating with cross-functional teams, and implementing best practices in software development."
            
        elif "company" in question_lower:
            return "The company is a growing technology firm specializing in innovative solutions using AI and cloud technologies. It has a strong market presence and a collaborative work environment."
            
        else:
            return "That's a good question. The hiring manager would be the best person to provide more specific details on that during the next stage of the interview process."

    def _update_metrics(self, metric_name: str, score: float):
        """Update evaluation metrics with new score"""
        current_total = self.evaluation_metrics.get(metric_name, 0) * self.evaluation_metrics.get(f"{metric_name}_count", 0)
        new_count = self.evaluation_metrics.get(f"{metric_name}_count", 0) + 1
        
        self.evaluation_metrics[metric_name] = (current_total + score) / new_count
        self.evaluation_metrics[f"{metric_name}_count"] = new_count
        
        logger.info(f"Metric '{metric_name}' updated to {self.evaluation_metrics[metric_name]:.2f} (AppID: {self.application_id})")

    def get_message_with_audio(self, message: str) -> Dict[str, Any]:
        """
        Generates audio for an AI message using pyttsx3.
        Returns dict with text, audio URL, and request ID.
        """
        if not message:
            return {'text': "", 'audio_url': None, 'request_id': str(uuid.uuid4())}
        
        try:
            # Try to find preferred voice or use default
            selected_voice_id = None
            for voice in voices:
                if preferred_voice_name in voice.name:
                    selected_voice_id = voice.id
                    break
            
            # If no specific voice found, use the first available
            if not selected_voice_id and voices:
                selected_voice_id = voices[0].id
            
            # Set voice if found
            if selected_voice_id:
                engine.setProperty('voice', selected_voice_id)
            else:
                logger.warning(f"No TTS voices found/available (AppID: {self.application_id}).")

            # Generate audio file
            engine.save_to_file(message, audio_path)
            engine.runAndWait()

            # Create relative URL path for the audio file
            audio_url = f"/static/audio/{audio_filename}"
            logger.info(f"Generated audio for message at {audio_url} (AppID: {self.application_id})")
            
            return {
                'text': message, 
                'audio_url': audio_url, 
                'request_id': str(uuid.uuid4())
            }
            
        except Exception as e:
            logger.error(f"Error generating audio for message (AppID: {self.application_id}): {e}")
            
            # Return message without audio on failure
            return {
                'text': message, 
                'audio_url': None, 
                'request_id': str(uuid.uuid4())
            }

    def generate_detailed_report(self) -> None:
        """Generate and save a detailed interview report."""
        if not self.candidate_name:
            logger.warning(f"Cannot generate report: candidate_name not set (AppID: {self.application_id}).")
            return

        logger.info(f"Generating detailed report for AppID: {self.application_id}...")
        
        # Calculate overall score based on metrics
        tech_score = self.evaluation_metrics.get("technical_knowledge", 0)
        comm_score = self.evaluation_metrics.get("communication", 0)
        prob_score = self.evaluation_metrics.get("problem_solving", 0)
        
        # Simple weighted average
        overall_score = (tech_score * 0.5) + (comm_score * 0.25) + (prob_score * 0.25)
        
        # Generate recommendations based on score
        recommendation = "Highly Recommended" if overall_score >= 0.8 else \
                         "Recommended" if overall_score >= 0.6 else \
                         "Consider with Reservations" if overall_score >= 0.4 else \
                         "Not Recommended"

        # Create report data
        report_data = {
            "application_id": self.application_id,
            "candidate_name": self.candidate_name,
            "position": self.position,
            "interview_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "domains_evaluated": self.domains,
            "evaluation_metrics": {
                "technical_knowledge": tech_score,
                "communication": comm_score,
                "problem_solving": prob_score,
                "overall_score": overall_score
            },
            "recommendation": recommendation,
            "interview_history": self.interview_history,
        }
        
        # Create reports directory if it doesn't exist
        reports_dir = "interview_reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate filename based on candidate name and application ID
        safe_name = re.sub(r'[^\w\s]', '', self.candidate_name).replace(' ', '_')
        report_filename = os.path.join(reports_dir, f"report_{safe_name}_{self.application_id}.json")
        
        try:
            # Save report to file
            with open(report_filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=4, ensure_ascii=False)
                
            logger.info(f"Detailed report saved to: {report_filename} (AppID: {self.application_id})")
            log_interview_event(self.application_id, "report_generated", f"Report saved: {report_filename}")
        except Exception as e:
            logger.error(f"Error saving report (AppID: {self.application_id}): {e}")