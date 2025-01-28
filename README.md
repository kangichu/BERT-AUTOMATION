
## Comprehensive Documentation for Real Estate Data Pipeline

### Table of Contents
1. Overview  
2. Key Concepts  
3. Execution Flow (Detailed Pipeline Steps)  
   - Data Retrieval  
   - Narrative Generation  
   - Embedding Generation  
   - FAISS Indexing and Storage  
   - Visualization with TensorBoard  
   - Updating the FAISS Index with New Listings  
4. Code Walkthrough  
   - File Structure  
   - Function Descriptions  
5. Logging and Error Handling  
6. FAQs and Troubleshooting  

---

### 1. Overview  
This real estate data pipeline processes property listings, generates narratives, and indexes them for efficient similarity-based searches. It handles:  
- Fetching data from a MySQL database.  
- Generating narratives using AI models or manual logic.  
- Creating vector embeddings for listings.  
- Indexing embeddings in FAISS for similarity searches.  
- Visualizing results in TensorBoard.  

The pipeline also supports periodic updates for new listings and exposes endpoints via a Flask API for searching and updating.  

---

### 2. Key Concepts  

#### **Transformers**  
- **Models Used**:  
  - GPT-2, BLOOM: Generate property descriptions.  
  - BERT: Convert text into vector embeddings.  

#### **FAISS**  
- **Definition**: A similarity search library that stores and retrieves embeddings efficiently.  

#### **TensorBoard**  
- **Purpose**: Visualize embeddings and metadata for insights.  

#### **Flask API**  
- **Endpoints**:  
  - `/search`: Search for similar properties based on text queries.  
  - `/update_pipeline`: Add new listings to the FAISS index.  

---

### 3. Execution Flow  

#### **Pipeline Execution Flow in `main.py`**  

The pipeline orchestrates the following steps:  

---

#### **Step 1: Data Retrieval**  

**Description:**  
Fetches property data from the MySQL database, including related details like amenities and complex information.  

**Key Function:**  
```python
def fetch_data_from_mysql():
    """
    Retrieves all published listings and their associated complex details from MySQL.
    """
    query = """
    SELECT listings.*, complexes.title AS complex_title
    FROM listings
    LEFT JOIN complexes ON listings.complex_id = complexes.id
    WHERE listings.status = 'Published';
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    return rows
```

**Pipeline Code:**  
```python
rows = fetch_data_from_mysql()
logging.info(f"Retrieved {len(rows)} listings from MySQL.")
```

---

#### **Step 2: Narrative Generation**  

**Description:**  
Generates property descriptions using AI models (GPT-2, BLOOM) or manual logic if preferred.  

**Key Function (Manual):**  
```python
def generate_narrative_manually(rows):
    """
    Creates property descriptions manually from listing data.
    """
    narratives = []
    for row in rows:
        narrative = f"{row['name']} is a {row['listing_type']} in {row['county_specific']}."
        narratives.append((row['id'], narrative))
    return narratives
```

**Pipeline Code:**  
```python
narratives = generate_narrative_manually(rows)
logging.info(f"Generated {len(narratives)} property narratives.")
```

---

#### **Step 3: Embedding Generation**  

**Description:**  
Converts generated narratives into dense vector representations using BERT.  

**Key Function:**  
```python
def generate_embeddings(narratives):
    """
    Converts narratives into dense embeddings using BERT.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    vector_data = []
    for id, narrative in narratives:
        tokens = tokenizer(narrative, return_tensors="pt")
        output = model(**tokens)
        embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
        vector_data.append((id, narrative, embedding))
    return vector_data
```

**Pipeline Code:**  
```python
vector_data = generate_embeddings(narratives)
logging.info(f"Generated embeddings for {len(vector_data)} narratives.")
```

---

#### **Step 4: FAISS Indexing and Storage**  

**Description:**  
Embeddings are stored in FAISS, enabling efficient similarity searches.  

**Key Function:**  
```python
def store_embeddings_in_faiss(vector_data, dimension=768):
    """
    Stores vector embeddings in a FAISS index for similarity-based searches.
    """
    faiss_manager = FAISSManager(dimension)
    faiss_manager.add_vectors(vector_data)
    logging.info(f"Stored {len(vector_data)} embeddings in the FAISS index.")
```

**Pipeline Code:**  
```python
store_embeddings_in_faiss(vector_data)
logging.info("Embeddings stored in FAISS successfully.")
```

---

#### **Step 5: Visualization with TensorBoard**  

**Description:**  
Exports embeddings and metadata for TensorBoard visualization.  

**Key Function:**  
```python
def export_to_tensorboard(embeddings, metadata, log_dir="logs/embedding_logs"):
    """
    Exports embeddings and metadata to TensorBoard for visualization.
    """
    os.makedirs(log_dir, exist_ok=True)
    np.savetxt(os.path.join(log_dir, "embeddings.tsv"), embeddings, delimiter="\t")
    with open(os.path.join(log_dir, "metadata.tsv"), "w") as f:
        f.write("\n".join(metadata))
    logging.info("Embeddings exported for TensorBoard.")
```

**Pipeline Code:**  
```python
export_to_tensorboard(
    [item[2] for item in vector_data],
    [f"ID: {item[0]} | Narrative: {item[1]}" for item in vector_data]
)
```

---

#### **Step 6: Updating the FAISS Index with New Listings**  

**Description:**  
Updates the FAISS index by processing only new listings from the database.  

**Key Function:**  
```python
def update_pipeline_with_new_listings(index_path):
    """
    Fetches new listings and adds their embeddings to the existing FAISS index.
    """
    last_processed_id = load_last_processed_id()
    new_rows, new_last_processed_id = fetch_new_listings_from_mysql(last_processed_id)
    if new_rows:
        narratives = generate_narrative_manually(new_rows)
        vector_data = generate_embeddings(narratives)
        faiss_manager = FAISSManager()
        faiss_index = faiss_manager.load_index(index_path)
        for item in vector_data:
            faiss_index.add(np.array([item[2]]))
        faiss_manager.save_index(faiss_index, index_path)
        save_last_processed_id(new_last_processed_id)
        logging.info(f"Updated FAISS index with {len(new_rows)} new listings.")
```

**Pipeline Code:**  
```python
update_pipeline_with_new_listings('faiss_index')
```

---

### 4. Code Walkthrough  

#### **File Structure**  
```plaintext
project/
├── main.py                   # Main pipeline orchestration
├── data_fetcher/
│   ├── mysql_fetcher.py      # Fetch data from MySQL
│   └── last_processed.py     # Track last processed listing ID
├── narrative_generator/
│   ├── gpt_narrative.py      # AI-powered narrative generation
│   └── manual_narrative.py   # Manual narrative logic
├── vectorizer/
│   └── bert_vectorizer.py    # Embedding generation
├── database/
│   └── faiss_store.py        # FAISS index management
├── data_handler/
│   └── tensorboard_exporter.py  # TensorBoard export
├── utils/
│   └── logger.py             # Logging setup
└── logs/                     # Log files
```

---

### 5. Logging and Error Handling  

**Logging Setup:**  
```python
def setup_logging():
    """
    Initializes daily rotating logs.
    """
    handler = logging.handlers.TimedRotatingFileHandler("logs/pipeline.log", when="midnight")
    logging.basicConfig(handlers=[handler], level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Logging initialized.")
```

**Error Handling Example:**  
```python
try:
    rows = fetch_data_from_mysql()
except Exception as e:
    logging.error(f"Error during data retrieval: {e}")
```

---

### 6. FAQs and Troubleshooting  

**Q: How do I handle new listings?**  
- Use `update_pipeline_with_new_listings('faiss_index')`.  

**Q: Why are embeddings not visible in TensorBoard?**  
- Ensure embeddings and metadata are exported to `logs/embedding_logs`.  

**Q: How do I add new models?**  
- Extend the `narrative_generator` module to integrate new AI models.  

