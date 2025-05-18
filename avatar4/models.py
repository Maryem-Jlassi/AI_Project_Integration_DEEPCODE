from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def load_models():
    """Loads the different AI models needed for the interview."""
    print("Loading AI models...")

    # Model for language understanding
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    language_model = AutoModel.from_pretrained("xlm-roberta-base")

    # Model for evaluating technical responses
    tech_eval_model = AutoModelForSequenceClassification.from_pretrained(
        "textattack/roberta-base-SST-2"
    )

    # Model for generating questions (GPT-2 or equivalent)
    try:
        question_generator_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        question_generator = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    except Exception as e:
        print(f"Error loading generation model: {e}")
        print("Using fallback question generator...")
        question_generator = None
        question_generator_tokenizer = None

    # Model for analyzing the relevance of responses
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Model for answering candidate questions (English)
    try:
        # Try to use a French-speaking model (Flan-T5 or CamemBERT)
        answering_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",  # Can be replaced by a more French-speaking model if needed
            tokenizer="google/flan-t5-base"
        )
    except Exception as e:
        print("Error loading question answering model (en):", e)
        try:
            answering_pipeline = pipeline(
                "text-generation",
                model="t5-small"
            )
        except Exception as e2:
            print("Error falling back on t5-small:", e2)
            answering_pipeline = None

    print("Models loaded successfully.")
    return tokenizer, language_model, tech_eval_model, question_generator_tokenizer, question_generator, tfidf_vectorizer, answering_pipeline