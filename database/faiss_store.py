# faiss_manager.py

import faiss
import numpy as np
import logging
import os

class FAISSManager:
    def __init__(self, dimension, index_file="faiss_index.idx", index_dir="faiss_index"):
        self.dimension = dimension
        self.index_dir = index_dir
        self.index_file = os.path.join(index_dir, index_file)

        # Create directory for the index file if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)

        self.index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean distance)

        # Load existing index if available
        if os.path.exists(self.index_file):
            self.load_index()
        else:
            logging.info("No existing FAISS index found. Creating a new one.")

    def save_index(self):
        """Save the FAISS index to disk."""
        faiss.write_index(self.index, self.index_file)
        logging.info(f"FAISS index saved to {self.index_file}.")

    def load_index(self):
        """Load the FAISS index from disk."""
        self.index = faiss.read_index(self.index_file)
        logging.info(f"FAISS index loaded from {self.index_file}.")

    def add_vectors(self, vector_data):
        """
        Add vectors to the FAISS index.
        :param vector_data: List of tuples [(ref_id, narrative, embedding), ...]
        """
        embeddings = np.array([item[2] for item in vector_data]).astype('float32')
        self.index.add(embeddings)

        # Save the index after adding vectors
        self.save_index()
        logging.info(f"Added {len(vector_data)} vectors to the FAISS index.")

    def search(self, query_vector, k=5):
        """
        Search the FAISS index for the k nearest neighbors.
        :param query_vector: A single query vector (numpy array).
        :param k: Number of nearest neighbors to return.
        :return: List of (indices, distances).
        """
        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)
        return indices[0], distances[0]


def store_embeddings_in_faiss(vector_data, dimension=768, index_dir="faiss_index"):
    try:
        logging.info("Initializing FAISS Manager...")
        faiss_manager = FAISSManager(dimension, index_dir=index_dir)

        logging.info("Adding embeddings to FAISS index...")
        faiss_manager.add_vectors(vector_data)

        logging.info(f"Successfully stored {len(vector_data)} embeddings in FAISS index.")
    except Exception as e:
        logging.error(f"Error storing embeddings in FAISS: {e}")
        raise
