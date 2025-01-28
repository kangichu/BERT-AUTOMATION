from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import logging

def generate_embeddings(narratives):
    try:
        logging.info("Loading BERT model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")

        vector_data = []
        for id, narrative in narratives:
            tokens = tokenizer(narrative, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                output = model(**tokens)
            embeddings = output.last_hidden_state.mean(dim=1).squeeze().numpy()
            vector_data.append((id, narrative, embeddings))
        logging.info(f"Generated embeddings for {len(vector_data)} narratives.")
        return vector_data
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        raise

def generate_single_embedding(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(text).tolist()