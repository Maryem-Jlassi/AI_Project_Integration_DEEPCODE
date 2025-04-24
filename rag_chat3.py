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
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import soundfile as sf

# Chemins des fichiers
CSV_PATH = "files.csv"
DOCS_DIR = "actia_docs/"
MODEL_NAME = "bert-base-uncased"
FAISS_INDEX_PATH = "faiss_index.index"
FINE_TUNED_MODEL_PATH = "fine_tuned_model"
# Charger le modèle et le processeur pré-entraîné
import whisper
model = whisper.load_model("medium")

# Initialisation du tokenizer et du modèle d'embedding
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

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
    
# Function to record audio
def record_audio(duration=5, sample_rate=16000):
    print("🎤 Enregistrement de l'audio...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to finish
    print("🎧 Audio enregistré.")
    return audio.flatten()

def transcribe_audio(audio, sample_rate=16000, temp_filename="temp_audio.wav"):
    # Enregistrer l'audio temporairement
    sf.write(temp_filename, audio, sample_rate)
    # Transcrire avec Whisper
    result = model.transcribe(temp_filename, language="fr")
    return result["text"]


# Combine recording and transcription into one function
def record_and_transcribe_audio(duration=5):
    audio = record_audio(duration=duration)  # Record the audio
    transcription = transcribe_audio(audio)  # Transcribe the recorded audio
    return transcription
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
        print(f"Erreur lors du chargement du CSV: {e}")

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
                print(f"Erreur avec {filename}: {e}")
    return all_chunks, labels

def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_tensor=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))
    return index, chunks

def retrieve_relevant_chunks(query, index, chunks, k=3):
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding, dtype='float32'), k=k)
    return [chunks[idx] for idx in indices[0] if idx < len(chunks)]





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
        output_dir="./results", num_train_epochs=3, per_device_train_batch_size=4, save_strategy="epoch"
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
        "model": "llama3",
    "messages": [
        {"role": "system", "content": "Vous êtes un assistant RH pour ACTIA. Répondez précisément à la question de l'utilisateur."},
        {"role": "user", "content": query}
    ]
    }

def main():
    print("📢 Chargement de la base de connaissances ACTIA...")
    all_chunks, labels = load_all_documents()
    if not all_chunks:
        print("Aucune donnée trouvée.")
        return
    print(f"✅ {len(all_chunks)} segments de texte chargés.")

    if os.path.exists(FINE_TUNED_MODEL_PATH):
        print("🔄 Chargement du modèle BERT fine-tuné...")
        model = BertForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_PATH)
        tokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
    else:
        print("🚀 Entraînement du modèle BERT...")
        model = fine_tune_bert(all_chunks, labels)
        print("✅ Modèle BERT entraîné et sauvegardé !")
    
    if os.path.exists(FAISS_INDEX_PATH):
        print("🔄 Chargement de l'index FAISS...")
        index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        print("🚀 Création de l'index FAISS...")
        index, _ = create_faiss_index(all_chunks)
        save_faiss_index(index)
        print("✅ Index FAISS sauvegardé.")
    
    print("🤖 Chatbot RH ACTIA - Posez vos questions (tapez 'exit' pour quitter)")
    while True:
        query = input("\nVotre question (ou appuyez sur '1' pour poser une question oralement) : ").strip()
        
        if query == "1":
            print("📢 Vous pouvez poser votre question oralement...")
            query = record_and_transcribe_audio(duration=5)  # Enregistrer pendant 5 secondes
            print(f"🎤 Question transcrite : {query}")
        if query.lower() == "exit":
            print("👋 Au revoir !")
            break
        relevant_chunks = retrieve_relevant_chunks(query, index, all_chunks)
        print("\n🔍 Chunks avant filtrage :", relevant_chunks[:5])  # Affiche les 5 premiers résultats
        if not relevant_chunks:
            print("⚠️ Désolé, je n'ai pas trouvé d'informations pertinentes. Pouvez-vous reformuler votre question ?")
            continue



        prompt = build_rag_prompt(query, relevant_chunks)
        try:
            response = ollama.chat(**prompt)
            print("\n💬 Réponse de l'IA :", response['message']['content'])
        except Exception as e:
            print(f"⚠️ Erreur Ollama : {e}")

if __name__ == "__main__":
    main()