import os
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector
import logging

def create_projector_config(log_dir="logs/embedding_logs", embeddings=None, metadata=None):
    """
    Create projector config for visualizing embeddings in TensorBoard.
    """
    logging.info(f"Using log directory: {log_dir}")

    # Ensure embeddings and metadata are provided
    if embeddings is None or metadata is None:
        logging.error("Embeddings and metadata must be provided.")
        return

    logging.info(f"Found embeddings with shape: {len(embeddings)} and metadata with length: {len(metadata)}.")
    
    # Convert embeddings into a TensorFlow tensor
    embeddings_tensor = tf.constant(embeddings, dtype=tf.float32)
    
    # Ensure that embeddings and metadata files exist in the log_dir
    embeddings_file = os.path.join(log_dir, "embeddings.tsv")
    metadata_file = os.path.join(log_dir, "metadata.tsv")
    
    # Save embeddings to embeddings.tsv
    with open(embeddings_file, "w") as f:
        for emb in embeddings:
            f.write("\t".join(map(str, emb)) + "\n")

    # Save metadata to metadata.tsv
    with open(metadata_file, "w") as f:
        for meta in metadata:
            f.write(meta + "\n")

    logging.info(f"Saved embeddings to {embeddings_file} and metadata to {metadata_file}.")
    
    # Initialize the summary writer
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    # Write histogram instead of tensor for visualization
    with summary_writer.as_default():
        tf.summary.histogram("embeddings", embeddings_tensor, step=0)
    
    logging.info(f"Embeddings histogram written to event files in {log_dir}.")
    
    # Initialize projector config
    config = projector.ProjectorConfig()
    
    # Add embedding configuration
    embedding = config.embeddings.add()
    embedding.tensor_name = "embeddings"
    embedding.metadata_path = os.path.relpath(metadata_file, log_dir)
    embedding.tensor_path = os.path.relpath(embeddings_file, log_dir)
    
    # Save config file to the log directory
    config_path = os.path.join(log_dir, "projector_config.pbtxt")
    with open(config_path, "w") as f:
        f.write(str(config))
    
    logging.info(f"Saved projector config to {config_path}")

    # Visualize embeddings within the context of the summary writer
    with summary_writer.as_default():
        projector.visualize_embeddings(log_dir, config)
    
    logging.info(f"TensorBoard embeddings setup complete at {log_dir}.")


def export_to_tensorboard(embeddings, metadata, log_dir="logs/embedding_logs"):
    """
    Export embeddings and metadata to files for TensorBoard visualization.
    :param embeddings: Numpy array of shape (n_samples, embedding_dim).
    :param metadata: List of metadata corresponding to the embeddings.
    :param log_dir: Directory where the logs will be saved.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Save embeddings to TSV
    embeddings_file = os.path.join(log_dir, "embeddings.tsv")
    metadata_file = os.path.join(log_dir, "metadata.tsv")

    try:
        np.savetxt(embeddings_file, embeddings, delimiter="\t")
        with open(metadata_file, "w") as f:
            f.write("\n".join(metadata))
        
        logging.info(f"Exported embeddings to TensorBoard format at {log_dir}.")
    
    except Exception as e:
        logging.error(f"Failed to export embeddings or metadata: {e}")
