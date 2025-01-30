import faiss
import numpy as np
import logging
import os

class FAISSManager:
    def __init__(self, dimension, index_file="faiss_index.idx", index_dir="faiss_index", narratives=None):
        self.dimension = dimension
        self.index_dir = index_dir
        self.index_file = os.path.join(index_dir, index_file)
        self.narratives = narratives if narratives is not None else []  # Store narratives in a list

        # Create directory for the index file if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)

        self.index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean distance)

        # Load existing index if available
        if os.path.exists(self.index_file):
            self.load_index()
        else:
            logging.info(f"No existing FAISS index found at {self.index_file}. Creating a new one.")

    def save_index(self):
        """Save the FAISS index to disk."""
        try:
            faiss.write_index(self.index, self.index_file)
            logging.info(f"FAISS index successfully saved to {self.index_file}.")
        except Exception as e:
            logging.error(f"Failed to save FAISS index to {self.index_file}: {str(e)}", exc_info=True)
            raise

    def load_index(self):
        """Load the FAISS index from disk."""
        try:
            logging.info(f"Attempting to load FAISS index from: {self.index_file}")

            # Check if the FAISS index file exists before loading
            if not os.path.exists(self.index_file):
                logging.warning(f"FAISS index file not found at {self.index_file}. Creating a new index.")
                self.index = faiss.IndexFlatL2(self.dimension)  # Create a new empty FAISS index
                return self.index  # Return empty index so caller can proceed

            # Load the existing FAISS index
            self.index = faiss.read_index(self.index_file)
            logging.info(f"FAISS index successfully loaded from {self.index_file}.")
            return self.index

        except Exception as e:
            logging.error(f"Failed to load FAISS index from {self.index_file}: {str(e)}", exc_info=True)
            raise  # Re-raise the exception to be handled by the caller

    def reconstruct(self, index):
        """
        Reconstruct the narrative for a given index.
        :param index: The index of the vector to reconstruct.
        :return: The narrative corresponding to the index or None if not found.
        """
        try:
            if index < len(self.narratives):
                return self.narratives[index]
            else:
                logging.warning(f"Index {index} out of bounds for narratives.")
                return None
        except Exception as e:
            logging.error(f"Error reconstructing narrative at index {index}: {e}")
            return None

    def search(self, query_vector, k=5):
        """
        Search the FAISS index for the k nearest neighbors.
        :param query_vector: A single query vector (numpy array).
        :param k: Number of nearest neighbors to return.
        :return: Tuple of (indices, distances).
        """
        try:
            logging.info(query_vector)

            if self.index is None or self.index.ntotal == 0:
                logging.warning("FAISS index is empty. No results can be returned.")
                return [], []

            # Convert query_vector to a NumPy array if it's a list
            if isinstance(query_vector, list):
                query_vector = np.array(query_vector)

            # Ensure query_vector is a 2D array (batch of vectors)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)  # Reshape to (1, n_features)

            # Normalize the query vector
            query_vector = normalize_vectors(query_vector)

            # Check the dimension of the query vector against the index
            query_dim = query_vector.shape[1]
            index_dim = self.index.d
            if query_dim != index_dim:
                logging.warning(f"Query vector dimension ({query_dim}) does not match index dimension ({index_dim}). Resizing query vector.")
                if query_dim < index_dim:
                    # If query vector is smaller, pad it with zeros (or any other padding strategy)
                    padding = np.zeros((1, index_dim - query_dim), dtype=np.float32)
                    query_vector = np.concatenate([query_vector, padding], axis=1)
                elif query_dim > index_dim:
                    # If query vector is larger, trim it (or any other truncation strategy)
                    query_vector = query_vector[:, :index_dim]

            # Perform the search on the FAISS index
            distances, indices = self.index.search(query_vector, k)

            logging.info(f"FAISS search completed. Top {k} indices: {indices[0]}, Distances: {distances[0]}")
            return indices[0], distances[0]

        except Exception as e:
            logging.error(f"Error searching FAISS index: {e}", exc_info=True)
            raise

    def reconstruct(self, index):
        try:
            if index < len(self.narratives):
                return self.narratives[index]
            else:
                logging.warning(f"Index {index} out of bounds for narratives.")
                return None
        except Exception as e:
            logging.error(f"Error reconstructing narrative at index {index}: {e}")
            return None

def store_embeddings_in_faiss(vector_data, dimension=768, index_dir="faiss_index"):
    """
    Store embeddings in a FAISS index.
    """
    try:
        logging.info("Initializing FAISS Manager...")
        faiss_manager = FAISSManager(dimension, index_dir=index_dir)

        logging.info("Adding embeddings to FAISS index...")
        faiss_manager.add_vectors(vector_data)

        logging.info(f"Successfully stored {len(vector_data)} embeddings in FAISS index.")
    except Exception as e:
        logging.error(f"Error storing embeddings in FAISS: {e}", exc_info=True)
        raise


def normalize_vectors(vectors):
    """Normalize vectors to unit length."""
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norm