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
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import logging
from pydub import AudioSegment

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemins des fichiers
CSV_PATH = "files.csv"
DOCS_DIR = "actia_docs/"
MODEL_NAME = "bert-base-uncased"
FAISS_INDEX_PATH = "faiss_index.index"
FINE_TUNED_MODEL_PATH = "fine_tuned_model"

# Initialisation de FastAPI
app = FastAPI(title="ACTIA Chatbot API")

# Configuration CORS pour Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # URL du frontend Angular
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Autorise tous les headers
)

# Chargement global des modèles
model_whisper = None
tokenizer = None
embedder = None
bert_model = None
faiss_index = None
all_documents = []

# Modèles Pydantic pour les requêtes/réponses
class QueryRequest(BaseModel):
    query: str

class AudioQueryRequest(BaseModel):
    audio_data: str  # Base64 encoded audio

class RagResponse(BaseModel):
    answer: str
    sources: List[str]

# Initialisation du tokenizer et du modèle d'embedding
def init_models():
    global model_whisper, tokenizer, embedder, bert_model, faiss_index, all_documents
    
    logger.info("Initialisation des modèles...")
    
    # Chargement du modèle Whisper
    model_whisper = whisper.load_model("medium")
    
    # Initialisation du tokenizer BERT
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Initialisation du modèle d'embedding
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Chargement des documents
    all_documents, labels = load_all_documents()
    logger.info(f"Chargement de {len(all_documents)} segments de texte")
    
    # Chargement ou création de l'index FAISS
    if os.path.exists(FAISS_INDEX_PATH):
        logger.info("Chargement de l'index FAISS existant")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        logger.info("Création d'un nouvel index FAISS")
        faiss_index, _ = create_faiss_index(all_documents)
        save_faiss_index(faiss_index)
    
    # Chargement ou fine-tuning du modèle BERT
    if os.path.exists(FINE_TUNED_MODEL_PATH):
        logger.info("Chargement du modèle BERT fine-tuné")
        bert_model = BertForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_PATH)
    else:
        logger.info("Fine-tuning du modèle BERT")
        bert_model = fine_tune_bert(all_documents, labels)
    
    logger.info("Initialisation terminée")

# Extraction du texte des documents
def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()

# Transcription audio
def transcribe_audio(audio_data, sample_rate=16000):
    # Enregistrer l'audio temporairement
    temp_filename = "temp_audio.wav"
    sf.write(temp_filename, audio_data, sample_rate)
    
    # Transcrire avec Whisper
    result = model_whisper.transcribe(temp_filename, language="fr")
    
    # Suppression du fichier temporaire
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    return result["text"]

# Fonction pour découper un texte en chunks
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

def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_tensor=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))
    return index, chunks

def retrieve_relevant_chunks(query, k=3):
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    distances, indices = faiss_index.search(np.array(query_embedding, dtype='float32'), k=k)
    return [all_documents[idx] for idx in indices[0] if idx < len(all_documents)]

class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

def fine_tune_bert(texts, labels):
    dataset = CustomDataset(texts, labels)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    training_args = TrainingArguments(
        output_dir="./results", 
        num_train_epochs=3, 
        per_device_train_batch_size=4, 
        save_strategy="epoch"
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(FINE_TUNED_MODEL_PATH)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_PATH)
    return model

def save_faiss_index(index, filename=FAISS_INDEX_PATH):
    faiss.write_index(index, filename)

def build_rag_prompt(query, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    return {
        "model": "llama3:latest",
        "messages": [
            {"role": "system", "content": "Vous êtes un assistant RH pour ACTIA. Répondez précisément (2-3 phrases maximum) à la question de l'utilisateur en vous basant sur le contexte fourni."},
            {"role": "user", "content": f"Contexte: {context}\n\nQuestion: {query}"}
        ]
    }

# Routes FastAPI
@app.on_event("startup")
async def startup_event():
    init_models()

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API du Chatbot RH ACTIA"}

@app.post("/query", response_model=RagResponse)
async def process_query(request: QueryRequest):
    query = request.query
    
    if not query:
        raise HTTPException(status_code=400, detail="La requête ne peut pas être vide")
    
    # Récupération des chunks pertinents
    relevant_chunks = retrieve_relevant_chunks(query)
    
    if not relevant_chunks:
        return RagResponse(
            answer="Désolé, je n'ai pas trouvé d'informations pertinentes. Pouvez-vous reformuler votre question ?",
            sources=[]
        )
    
    # Construction du prompt RAG
    prompt = build_rag_prompt(query, relevant_chunks)
    
    try:
        # Appel à Ollama
        response = ollama.chat(**prompt)
        answer = response['message']['content']
        
        return RagResponse(
            answer=answer,
            sources=relevant_chunks[:3]  # Retourner les sources utilisées
        )
    except Exception as e:
        logger.error(f"Erreur lors de l'appel à Ollama: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de génération: {str(e)}")

@app.post("/query/audio", response_model=RagResponse)
async def process_audio_query(file: UploadFile = File(...)):
    try:
        # Lire le contenu du fichier audio
        audio_bytes = await file.read()
        
        # Sauvegarder temporairement le fichier audio
        temp_filename = "temp_audio.wav"
        with open(temp_filename, "wb") as f:
            f.write(audio_bytes)
        
        # Vérifier que le fichier existe et n'est pas vide
        if not os.path.exists(temp_filename) or os.path.getsize(temp_filename) == 0:
            raise HTTPException(status_code=400, detail="Fichier audio invalide ou vide")
            
        try:
            # Convertir explicitement en format WAV compatible avec Whisper
            audio = AudioSegment.from_file(temp_filename)
            # Exporter en WAV 16kHz mono (format idéal pour Whisper)
            converted_filename = "converted_audio.wav"
            audio.export(
                converted_filename, 
                format="wav", 
                parameters=["-ac", "1", "-ar", "16000"]
            )
            
            # Transcrire avec Whisper depuis le fichier converti
            result = model_whisper.transcribe(converted_filename, language="fr")
            transcription = result["text"]
            
            # Nettoyer les fichiers temporaires
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            if os.path.exists(converted_filename):
                os.remove(converted_filename)
                
        except Exception as audio_error:
            logger.error(f"Erreur spécifique à la transcription: {audio_error}")
            raise HTTPException(status_code=400, detail=f"Erreur de transcription: {str(audio_error)}")

        if not transcription:
            raise HTTPException(status_code=400, detail="Impossible de transcrire l'audio")
        
        logger.info(f"Transcription réussie: {transcription}")
        
        # Récupération des chunks pertinents
        relevant_chunks = retrieve_relevant_chunks(transcription)

        if not relevant_chunks:
            return RagResponse(
                answer=f"J'ai transcrit votre question comme: '{transcription}'. Cependant, je n'ai pas trouvé d'informations pertinentes pour y répondre.",
                sources=[]
            )
        
        # Construction du prompt RAG
        prompt = build_rag_prompt(transcription, relevant_chunks)

        # Appel à Ollama
        response = ollama.chat(**prompt)
        answer = response['message']['content']

        return RagResponse(
            answer=answer,
            sources=relevant_chunks[:3]
        )
    except Exception as e:
        logger.error(f"Erreur lors du traitement de l'audio: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de traitement: {str(e)}")

@app.post("/upload/document")
async def upload_document(file: UploadFile = File(...)):
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
        
        # Mettre à jour les documents et l'index
        global all_documents, faiss_index
        all_documents.extend(chunks)
        
        # Recréer l'index FAISS
        faiss_index, _ = create_faiss_index(all_documents)
        save_faiss_index(faiss_index)
        
        return {"message": f"Document {file.filename} uploadé et indexé avec succès", "chunks_added": len(chunks)}
    
    except Exception as e:
        logger.error(f"Erreur lors de l'upload du document: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur d'upload: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)