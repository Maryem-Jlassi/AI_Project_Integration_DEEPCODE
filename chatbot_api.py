import ollama
import os
import pandas as pd
import sounddevice as sd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import librosa
import soundfile as sf
import whisper
import base64
import io
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import logging
from pydub import AudioSegment
import functools
import asyncio
from threading import Lock
from cachetools import LRUCache
import librosa
from scipy.io import wavfile
from scipy import signal  # Ajout de l'import correct pour butter
from typing import Dict
import noisereduce as nr

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun  # Updated import
from typing import Any, Optional
import langdetect
import concurrent.futures




# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemins des fichiers
CSV_PATH = "files.csv"
DOCS_DIR = "actia_docs/"
MODEL_NAME = "bert-base-uncased"
FAISS_INDEX_PATH = "faiss_index.index"
EMBEDDINGS_CACHE_PATH = "embeddings_cache.npy"
FINE_TUNED_MODEL_PATH = "fine_tuned_model"

# Initialisation de FastAPI
app = FastAPI(title="ACTIA Chatbot API")

# Configuration CORS pour Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales avec chargement paresseux
_model_whisper = None
_tokenizer = None
_embedder = None
_bert_model = None
_faiss_index = None
_all_documents = []

# Cache pour stocker les embeddings de requêtes récentes
query_embeddings_cache = LRUCache(maxsize=100)
# Verrou pour les opérations d'indexation
index_lock = Lock()

# Modèles Pydantic pour les requêtes/réponses
class QueryRequest(BaseModel):
    query: str

class AudioQueryRequest(BaseModel):
    audio_data: str  # Base64 encoded audio

class RagResponse(BaseModel):
    answer: str
    sources: List[str]
    processing_time: Optional[str] = None
class AgentInfo(BaseModel):
    id: str
    name: str
    description: str

class AgentRequest(BaseModel):
    agent_id: str
    query: str
class SearchTool(BaseTool):
    name: str = "DuckDuckGoSearch"
    description: str = "Useful for searching the internet for information"
    
    def _run(self, query: str) -> str:
        search = DuckDuckGoSearchRun()
        return search.run(query)
        
    async def _arun(self, query: str) -> str:
        # Implement async version if needed
        search = DuckDuckGoSearchRun()
        return search.run(query)
# Fonctions d'accès aux modèles avec chargement paresseux
def get_embedder():
    global _embedder
    if _embedder is None:
        logger.info("Chargement du modèle d'embedding...")
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder
# Définition des agents
def setup_agents():
    # Create the tool instance
    search_tool = SearchTool()
    
    # Définition des agents
    engineering_interview_agent = Agent(
        role="Engineering Interview Preparer",
        goal="Préparer les candidats aux entretiens techniques d'ingénierie",
        backstory="Je suis un expert en entretiens techniques avec une vaste expérience dans le domaine de l'ingénierie. J'aide les candidats à se préparer aux questions techniques, à comprendre les algorithmes et structures de données, et à améliorer leurs compétences en résolution de problèmes.",
        verbose=False,
        allow_delegation=True,
        tools=[search_tool]
    )
    
    resume_agent = Agent(
        role="Resume Strategist",
        goal="Optimiser les CV pour maximiser l'impact et les chances de décrocher un entretien",
        backstory="Je suis un expert en création et optimisation de CV avec une profonde connaissance des attentes des recruteurs dans le domaine technique. J'aide les candidats à mettre en valeur leurs compétences et expériences pertinentes.",
        verbose=False,
        allow_delegation=True,
        tools=[search_tool]
    )
    
    hr_policies_agent = Agent(
        role="HR Policies Expert",
        goal="Fournir des informations précises sur les politiques RH d'ACTIA",
        backstory="Je suis un expert des politiques RH d'ACTIA avec une connaissance approfondie du manuel de l'employé, des avantages sociaux et des procédures internes.",
        verbose=False,
        allow_delegation=True,
        tools=[search_tool]
    )
    
    career_path_agent = Agent(
        role="Career Path Advisor",
        goal="Conseiller sur les parcours professionnels et opportunités d'évolution chez ACTIA",
        backstory="Je suis un conseiller en développement de carrière spécialisé dans l'industrie automobile et les technologies embarquées. J'aide les employés à identifier les opportunités d'évolution au sein d'ACTIA.",
        verbose=False,
        allow_delegation=True,
        tools=[search_tool]
    )
    
    return {
        "engineering_interview": engineering_interview_agent,
        "resume": resume_agent,
        "hr_policies": hr_policies_agent,
        "career_path": career_path_agent
    }

# Fonctions pour créer des tâches spécifiques
def create_engineering_interview_task(agent, query):
    return Task(
        description=f"Analyser la demande suivante concernant un entretien d'ingénierie et préparer une réponse détaillée: {query}",
        agent=agent,
        expected_output="Une réponse détaillée pour aider à préparer un entretien technique d'ingénierie"
    )

def create_resume_task(agent, query):
    return Task(
        description=f"Analyser la demande suivante concernant un CV et préparer des conseils d'optimisation: {query}",
        agent=agent,
        expected_output="Des conseils détaillés pour optimiser un CV technique"
    )

def create_hr_policies_task(agent, query):
    return Task(
        description=f"Répondre à la question suivante sur les politiques RH d'ACTIA: {query}",
        agent=agent,
        expected_output="Une explication claire des politiques RH d'ACTIA concernant cette question"
    )

def create_career_path_task(agent, query):
    return Task(
        description=f"Analyser la demande suivante concernant l'évolution de carrière chez ACTIA: {query}",
        agent=agent,
        expected_output="Des conseils et informations sur les opportunités d'évolution de carrière chez ACTIA"
    )

# Fonctions pour gérer les requêtes d'agents
def classify_query_intent(query):
    """Classifier l'intention de la requête pour déterminer quel agent utiliser"""
    query_lower = query.lower()
    
    # Logique simple de classification - à améliorer avec un modèle ML plus tard
    if any(keyword in query_lower for keyword in ["entretien", "interview", "technique", "question technique", "algorithme"]):
        return "engineering_interview"
    elif any(keyword in query_lower for keyword in ["cv", "resume", "curriculum", "candidature"]):
        return "resume"
    elif any(keyword in query_lower for keyword in ["politique", "règle", "procédure", "avantage", "congé"]):
        return "hr_policies"
    elif any(keyword in query_lower for keyword in ["carrière", "évolution", "promotion", "formation", "compétence"]):
        return "career_path"
    else:
        return "general"  # Pour les requêtes générales, utilisez votre système RAG existant

# Variables globales pour les agents
_agents = None

# Fonction pour accéder aux agents avec chargement paresseux
def get_agents():
    global _agents
    if _agents is None:
        logger.info("Initialisation des agents Crew AI...")
        _agents = setup_agents()
    return _agents

# Fonction pour traiter les requêtes via Crew AI
async def process_with_crew_ai(query):
    """Traiter une requête en utilisant l'agent Crew AI approprié"""
    start_time = time.time()
    
    # Classifier l'intention et obtenir l'agent approprié
    intent = classify_query_intent(query)
    agents = get_agents()
    
    if intent == "general":
        # Utiliser le système RAG existant pour les requêtes générales
        return None
    
    try:
        agent = agents[intent]
        
        # Créer la tâche appropriée
        if intent == "engineering_interview":
            task = create_engineering_interview_task(agent, query)
        elif intent == "resume":
            task = create_resume_task(agent, query)
        elif intent == "hr_policies":
            task = create_hr_policies_task(agent, query)
        elif intent == "career_path":
            task = create_career_path_task(agent, query)
        
        # Créer et exécuter le crew avec un seul agent pour cette tâche
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
            process=Process.sequential  # ou Process.hierarchical selon le besoin
        )
        
        result = crew.kickoff()
        elapsed_time = time.time() - start_time
        
        return {
            "answer": result,
            "sources": [f"Agent: {agent.role}"],
            "processing_time": f"{elapsed_time:.2f}s"
        }
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement avec Crew AI: {e}")
        return None
def get_whisper_model():
    global _model_whisper
    if _model_whisper is None:
        logger.info("Chargement du modèle Whisper...")
        _model_whisper = whisper.load_model("medium")
    return _model_whisper

def get_bert_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        logger.info("Chargement du tokenizer BERT...")
        _tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer

def get_bert_model():
    global _bert_model
    if _bert_model is None:
        logger.info("Chargement du modèle BERT...")
        if os.path.exists(FINE_TUNED_MODEL_PATH):
            _bert_model = BertForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_PATH)
        else:
            # Fallback à un modèle pré-entraîné si pas de modèle fine-tuné
            _bert_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    return _bert_model

def get_documents_and_faiss_index():
    global _all_documents, _faiss_index
    if not _all_documents:
        logger.info("Chargement des documents...")
        _all_documents, _ = load_all_documents()
        
    if _faiss_index is None:
        logger.info("Chargement de l'index FAISS...")
        if os.path.exists(FAISS_INDEX_PATH):
            _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            _faiss_index, _ = create_faiss_index(_all_documents)
            save_faiss_index(_faiss_index)
    
    return _all_documents, _faiss_index

# Cache pour les fonctions coûteuses
def memoize(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

# Extraction du texte optimisée
@memoize
def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

@memoize
def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()

# Transcription audio optimisée
async def transcribe_audio_async(file_path):
    model = get_whisper_model()
    loop = asyncio.get_event_loop()
    # Exécuter la transcription dans un thread séparé pour ne pas bloquer
    return await loop.run_in_executor(
        None, 
        lambda: model.transcribe(file_path, language="fr")["text"]
    )

# Création de l'index FAISS optimisé
def create_faiss_index(chunks):
    embedder = get_embedder()
    # Vérifier si on a un cache d'embeddings
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        logger.info("Chargement des embeddings depuis le cache")
        embeddings = np.load(EMBEDDINGS_CACHE_PATH)
    else:
        logger.info("Calcul des embeddings")
        # Traiter les chunks par lots pour économiser la mémoire
        batch_size = 64
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_embeddings = embedder.encode(batch, convert_to_tensor=False)
            embeddings.append(batch_embeddings)
        embeddings = np.vstack(embeddings)
        np.save(EMBEDDINGS_CACHE_PATH, embeddings)
    
    dimension = embeddings.shape[1]
    
    # Utiliser un index plus rapide pour les grands ensembles de données
    if len(chunks) > 10000:
        # Index IVF pour une recherche rapide
        quantizer = faiss.IndexFlatL2(dimension)
        nlist = min(4096, int(len(chunks) / 10))  # nombre de cellules Voronoi
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train(embeddings)
        index.add(embeddings)
    else:
        # Index simple pour les petits ensembles
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype='float32'))
    
    return index, chunks

# Fonction de récupération optimisée
def retrieve_relevant_chunks(query, k=3):  # Augmenter le nombre de chunks de 3 à 5
    docs, index = get_documents_and_faiss_index()
    embedder = get_embedder()
    
    # Utiliser le cache si la requête a déjà été traitée
    if query in query_embeddings_cache:
        query_embedding = query_embeddings_cache[query]
    else:
        query_embedding = embedder.encode([query], convert_to_tensor=False)
        query_embeddings_cache[query] = query_embedding
    
    distances, indices = index.search(np.array(query_embedding, dtype='float32'), k=k)
    return [docs[idx] for idx in indices[0] if idx < len(docs)]

# Fonction de mise à jour incrémentale de l'index
def update_faiss_index(new_chunks):
    global _faiss_index
    with index_lock:
        docs, index = get_documents_and_faiss_index()
        embedder = get_embedder()
        
        # Calculer les embeddings pour les nouveaux chunks
        new_embeddings = embedder.encode(new_chunks, convert_to_tensor=False)
        
        # Ajouter à l'index existant
        index.add(np.array(new_embeddings, dtype='float32'))
        
        # Mettre à jour le cache d'embeddings si nécessaire
        if os.path.exists(EMBEDDINGS_CACHE_PATH):
            current_embeddings = np.load(EMBEDDINGS_CACHE_PATH)
            updated_embeddings = np.vstack([current_embeddings, new_embeddings])
            np.save(EMBEDDINGS_CACHE_PATH, updated_embeddings)
        
        # Sauvegarder l'index mis à jour
        save_faiss_index(index)
        
        return index

def build_rag_prompt(query, relevant_chunks):
    context = "\n\n".join(relevant_chunks[:1])  # 1 chunk seulement
    model_name =   "llama3:latest"
    
    lang = detect_language(query)
    if lang == "en":
        system_prompt = "You are a friendly HR assistant for ACTIA. Always answer in English, directly and concisely."
    else:
        system_prompt = """Vous êtes un assistant RH convivial pour ACTIA. 
        
        Directives pour vos réponses:
        1. Répondez DIRECTEMENT et PRÉCISÉMENT à la question posée
        2. Adressez-vous à l'utilisateur de façon chaleureuse et directe
        3. Ne déviez pas vers des sujets connexes mais différents
        4. Évitez les formulations qui sonnent comme des listes ou des énumérations
        5. Ne dites pas "Selon le contexte" ou "D'après les informations"
        6. Ne commencez pas par des phrases comme "Voici les informations" ou "Je peux vous dire que"
        7. Répondez de manière concise (2-4 phrases) mais conversationnelle
        8. Si le contexte ne contient pas la réponse à la question spécifique posée, indiquez-le clairement
        9. Restez strictement focalisé sur la question posée
        10. Intégrez naturellement l'information dans une réponse fluide
        11. si une question est posée en anglais, répondez en anglais
        
        Utilisez le contexte fourni pour répondre précisément, mais formulez votre réponse comme si vous parliez directement à un collègue."""
    return {
        "model": model_name,
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Contexte: {context}\n\nQuestion: {query}"}
        ]
    }
# Créez une fonction de prétraitement audio
async def preprocess_audio(filename):
    """
    Prétraite l'audio pour réduire le bruit et améliorer la qualité pour Whisper
    """
    # Charger l'audio avec librosa (préserve l'audio comme série temporelle)
    audio_data, sample_rate = librosa.load(filename, sr=16000)
    
    # Réduction du bruit
    reduced_noise = nr.reduce_noise(
        y=audio_data, 
        sr=sample_rate,
        stationary=False,  # Pour le bruit ambiant non-stationnaire comme dans une classe
        prop_decrease=0.75  # Niveau de réduction du bruit (ajustable)
    )
    
    # Normalisation du volume
    normalized_audio = librosa.util.normalize(reduced_noise)
    
    # Correction: Utilisation de scipy.signal.butter au lieu de librosa.filters.butter
    b, a = signal.butter(
        N=5,  # Ordre du filtre
        Wn=[300/8000, 3400/8000],  # Fréquences normalisées (Wn = freq/(sample_rate/2))
        btype='band'
    )
    filtered_audio = signal.filtfilt(b, a, normalized_audio)
    
    # Sauvegarde du fichier prétraité
    preprocessed_filename = f"preprocessed_{filename}"
    wavfile.write(preprocessed_filename, sample_rate, (filtered_audio * 32767).astype(np.int16))
    
    return preprocessed_filename
# Routes FastAPI optimisées
@app.on_event("startup")
async def startup_event():
    # Initialiser uniquement l'index FAISS et les documents au démarrage
    # Les autres modèles seront chargés à la demande
    try:
        get_agents()  # Force la création des agents CrewAI
        get_documents_and_faiss_index()
        get_embedder()  # Charger le modèle d'embedding
        get_whisper_model()  # Charger Whisper
        get_bert_model()  # Charger BERT
    except Exception as e:
        logger.error(f"Erreur d'initialisation: {e}")

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API du Chatbot RH ACTIA"}

@app.post("/query", response_model=RagResponse)
async def process_query(request: QueryRequest):
    start_time = time.time()
    query = request.query.strip()

    if query.strip().lower() in ["bonjour", "salut", "hello", "hi", "actibot", "hola"]:
        elapsed_time = time.time() - start_time
        return RagResponse(
            answer="Bonjour c'est actibot ! Comment puis-je vous aider ?",
            sources=[],
            processing_time=f"{elapsed_time:.2f}s"
        )

    if not query:
        raise HTTPException(status_code=400, detail="La requête ne peut pas être vide")

    # 1. CrewAI
    crew_ai_response = await process_with_crew_ai(query)
    if crew_ai_response:
        return RagResponse(**crew_ai_response)

    # 2. RAG
    relevant_chunks = retrieve_relevant_chunks(query, k=1)
    if not relevant_chunks:
        elapsed_time = time.time() - start_time
        return RagResponse(
            answer="Aucune information pertinente trouvée.",
            sources=[],
            processing_time=f"{elapsed_time:.2f}s"
        )

    prompt = build_rag_prompt(query, relevant_chunks)

    # 3. Appel à Ollama dans un thread pour ne pas bloquer
    try:
        ollama_start = time.time()

        if prompt.get("stream", False):
            def run_stream():
                return list(ollama.chat(**prompt))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_stream)
                response_chunks = future.result()
                answer = "".join(chunk["message"]["content"] for chunk in response_chunks if "message" in chunk)
        else:
            response = ollama.chat(**prompt)
            answer = response["message"]["content"]

        elapsed_time = time.time() - start_time
        ollama_duration = time.time() - ollama_start

        logger.info(f"⏱ Temps total: {elapsed_time:.2f}s (ollama: {ollama_duration:.2f}s)")

        return RagResponse(
            answer=answer,
            sources=relevant_chunks[:3],
            processing_time=f"{elapsed_time:.2f}s"
        )

    except Exception as e:
        logger.error(f"Erreur ollama: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Ajoutez également une modification similaire à la route process_audio_query
@app.post("/query/audio", response_model=RagResponse)
async def process_audio_query(file: UploadFile = File(...)):
    start_time = time.time()
    temp_filename = None
    converted_filename = None
    preprocessed_filename = None
    
    try:
        # Lire le contenu du fichier audio
        audio_bytes = await file.read()
        
        # Sauvegarder temporairement le fichier audio
        temp_filename = f"temp_audio_{int(time.time())}.wav"
        with open(temp_filename, "wb") as f:
            f.write(audio_bytes)
        
        try:
            # Convertir explicitement en format WAV compatible avec Whisper
            audio = AudioSegment.from_file(temp_filename)
            # Exporter en WAV 16kHz mono (format idéal pour Whisper)
            converted_filename = f"converted_audio_{int(time.time())}.wav"
            audio.export(
                converted_filename, 
                format="wav", 
                parameters=["-ac", "1", "-ar", "16000"]
            )
            
            # NOUVELLE ÉTAPE: Prétraitement audio pour réduire le bruit
            preprocessed_filename = await preprocess_audio(converted_filename)
            
            # Transcription asynchrone
            transcription = await transcribe_audio_async(preprocessed_filename)
            
        except Exception as audio_error:
            logger.error(f"Erreur spécifique à la transcription: {audio_error}")
            raise HTTPException(status_code=400, detail=f"Erreur de transcription: {str(audio_error)}")
        finally:
            # Nettoyer les fichiers temporaires quoi qu'il arrive
            for filename in [temp_filename, converted_filename, preprocessed_filename]:
                if filename and os.path.exists(filename):
                    os.remove(filename)

        if not transcription:
            raise HTTPException(status_code=400, detail="Impossible de transcrire l'audio")
        
        logger.info(f"Transcription réussie: {transcription}")
        
        # Essayer d'abord de traiter avec Crew AI
        crew_ai_response = await process_with_crew_ai(transcription)
        
        # Si Crew AI a pu traiter la requête, retourner sa réponse
        if crew_ai_response:
            return RagResponse(**crew_ai_response)
        
        # Sinon, continuer avec le système RAG existant
        # Récupération des chunks pertinents
        relevant_chunks = retrieve_relevant_chunks(transcription)

        if not relevant_chunks:
            elapsed_time = time.time() - start_time
            return RagResponse(
                answer=f"J'ai transcrit votre question comme: '{transcription}'. Cependant, je n'ai pas trouvé d'informations pertinentes pour y répondre.",
                sources=[],
                processing_time=f"{elapsed_time:.2f}s"
            )
        
        # Construction du prompt RAG
        prompt = build_rag_prompt(transcription, relevant_chunks)

        # Appel à Ollama
        response = ollama.chat(**prompt)
        answer = response['message']['content']

        elapsed_time = time.time() - start_time
        logger.info(f"Temps de traitement audio: {elapsed_time:.2f} secondes")
        
        return RagResponse(
            answer=answer,
            sources=relevant_chunks[:3],
            processing_time=f"{elapsed_time:.2f}s"
        )
    except Exception as e:
        logger.error(f"Erreur lors du traitement de l'audio: {e}")
        # Nettoyage des fichiers en cas d'erreur générale
        for filename in [temp_filename, converted_filename, preprocessed_filename]:
            if filename and os.path.exists(filename):
                try:
                    os.remove(filename)
                except:
                    pass
        raise HTTPException(status_code=500, detail=f"Erreur de traitement: {str(e)}")

@app.post("/upload/document")
async def upload_document(file: UploadFile = File(...)):
    start_time = time.time()
    
    try:
        # Vérifier le type de fichier
        if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
            raise HTTPException(status_code=400, detail="Seuls les fichiers PDF et TXT sont acceptés")
        
        # Créer le répertoire de documents s'il n'existe pas
        if not os.path.exists(DOCS_DIR):
            os.makedirs(DOCS_DIR)
        
        # Sauvegarder le fichier
        file_path = os.path.join(DOCS_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extraire et indexer le contenu
        if file.filename.endswith('.pdf'):
            chunks = chunk_text(extract_text_from_pdf(file_path))
        else:
            chunks = chunk_text(extract_text_from_txt(file_path))
        
        # Mettre à jour les documents
        global _all_documents
        _all_documents, _ = get_documents_and_faiss_index()
        _all_documents.extend(chunks)
        
        # Mise à jour incrémentale de l'index FAISS (plus rapide que recréer)
        update_faiss_index(chunks)
        
        elapsed_time = time.time() - start_time
        
        return {
            "message": f"Document {file.filename} uploadé et indexé avec succès", 
            "chunks_added": len(chunks),
            "processing_time": f"{elapsed_time:.2f}s"
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'upload du document: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur d'upload: {str(e)}")
@app.get("/agents", response_model=List[AgentInfo])
async def list_agents():
    """Obtenir la liste des agents disponibles"""
    agents = get_agents()
    
    agent_info_list = [
        AgentInfo(
            id="engineering_interview",
            name="Engineering Interview Preparer",
            description="Aide à la préparation d'entretiens techniques d'ingénierie"
        ),
        AgentInfo(
            id="resume",
            name="Resume Strategist",
            description="Conseils d'optimisation de CV pour maximiser vos chances"
        ),
        AgentInfo(
            id="hr_policies",
            name="HR Policies Expert",
            description="Expert des politiques RH d'ACTIA"
        ),
        AgentInfo(
            id="career_path",
            name="Career Path Advisor",
            description="Conseiller en évolution professionnelle chez ACTIA"
        )
    ]
    
    return agent_info_list

@app.post("/query/agent", response_model=RagResponse)
async def query_specific_agent(request: AgentRequest):
    """Interroger un agent spécifique"""
    start_time = time.time()
    
    if not request.query:
        raise HTTPException(status_code=400, detail="La requête ne peut pas être vide")
    
    if not request.agent_id:
        raise HTTPException(status_code=400, detail="Veuillez spécifier un agent")
    
    agents = get_agents()
    if request.agent_id not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{request.agent_id}' non trouvé")
    
    try:
        agent = agents[request.agent_id]
        
        # Créer la tâche appropriée selon l'agent
        if request.agent_id == "engineering_interview":
            task = create_engineering_interview_task(agent, request.query)
        elif request.agent_id == "resume":
            task = create_resume_task(agent, request.query)
        elif request.agent_id == "hr_policies":
            task = create_hr_policies_task(agent, request.query)
        elif request.agent_id == "career_path":
            task = create_career_path_task(agent, request.query)
        
        # Créer et exécuter le crew avec un seul agent pour cette tâche
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
            process=Process.sequential
        )
        
        result = crew.kickoff()
        elapsed_time = time.time() - start_time
        
        return RagResponse(
            answer=result,
            sources=[f"Agent: {agent.role}"],
            processing_time=f"{elapsed_time:.2f}s"
        )
    
    except Exception as e:
        logger.error(f"Erreur lors de la consultation de l'agent: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de traitement: {str(e)}")
# Fonctions utilitaires maintenues de l'original
def load_all_documents():
    all_chunks, labels = [], []
    
    try:
        df = pd.read_csv(CSV_PATH, delimiter=",", encoding="ISO-8859-1")
        for _, row in df.iterrows():
            question, answer = row.iloc[1], row.iloc[3]
            all_chunks.append(f"Question: {question} Réponse: {answer}")
            labels.append(1)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du CSV: {e}")

    if os.path.exists(DOCS_DIR):
        for filename in os.listdir(DOCS_DIR):
            filepath = os.path.join(DOCS_DIR, filename)
            try:
                if filename.lower().endswith('.pdf'):
                    text_chunks = chunk_text(extract_text_from_pdf(filepath))
                elif filename.lower().endswith('.txt'):
                    text_chunks = chunk_text(extract_text_from_txt(filepath))
                else:
                    continue
                all_chunks.extend(text_chunks)
                labels.extend([0] * len(text_chunks))
            except Exception as e:
                logger.error(f"Erreur avec {filename}: {e}")
    return all_chunks, labels

def chunk_text(text, max_length=500):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def save_faiss_index(index, filename=FAISS_INDEX_PATH):
    with index_lock:
        faiss.write_index(index, filename)

def detect_language(text):
    try:
        return langdetect.detect(text)
    except:
        return "fr"

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)