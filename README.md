## Comprehensive Documentation for Real Estate Data Pipeline

### Table of Contents
1. Overview  
2. Key Concepts  
3. Execution Flow  
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
This real estate data pipeline automates the processing, indexing, and searching of property listings. It performs several functions:  

- Fetching listings from a MySQL database.
- Generating human-readable narratives for listings using AI or manual approaches.
- Converting narratives into embeddings using BERT for similarity searches.
- Storing these embeddings in a FAISS index for high-performance queries.
- Visualizing embeddings with TensorBoard.
- Periodically updating the pipeline with new data.

---

### 2. Key Concepts  

#### **Transformers**:
- Models like GPT-2, BLOOM, and BERT are utilized for narrative generation and embedding creation.
- **BERT** transforms text into vector embeddings for mathematical representation of meaning.

#### **FAISS**:
- A library optimized for similarity search. It indexes embeddings and enables efficient retrieval of similar items.

#### **TensorBoard**:
- Used for visualizing embeddings and their metadata, offering insights into relationships between items.

#### **Flask API**:
- Exposes endpoints for searching and updating the FAISS index:
  - `/search`: Returns search results based on user queries.
  - `/update_pipeline`: Updates the FAISS index with new listings.

---

### 3. Execution Flow  

#### **Pipeline Execution Flow in `main.py`**

Below is the complete pipeline, step by step:

---

#### **Step 1: Data Retrieval**

- **Purpose**: Fetches property listings from MySQL, including complex details and amenities.  

**Key Function**:  
```python
def fetch_data_from_mysql():
    """
    Query MySQL to get all published listings with complex details.
    """
    query = """
    SELECT listings.*, complexes.title AS complex_title, 
           GROUP_CONCAT(CONCAT(amenities.type, ': ', amenities.amenity) SEPARATOR '; ') as amenities
    FROM listings
    LEFT JOIN complexes ON listings.complex_id = complexes.id
    LEFT JOIN amenities ON listings.id = amenities.listing_id
    WHERE listings.status = 'Published'
    GROUP BY listings.id;
    """
    cursor.execute(query)
    return cursor.fetchall()
```

**Pipeline Code**:  
```python
rows = fetch_data_from_mysql()
logging.info(f"Retrieved {len(rows)} listings from MySQL.")
```

---

#### **Step 2: Narrative Generation**

- **Purpose**: Generates descriptive texts for each property listing, summarizing key details like location, price, and amenities.

**Key Function (Manual Approach)**:  
```python
def generate_narrative_manually(rows):
    """
    Generates simple property descriptions based on data fields.
    """
    narratives = []
    for row in rows:
        narrative = f"{row['name']} is a {row['listing_type']} in {row['county_specific']}."
        narratives.append((row['id'], narrative))
    return narratives
```

**Pipeline Code**:  
```python
narratives = generate_narrative_manually(rows)
logging.info(f"Generated {len(narratives)} property narratives.")
```

---

#### **Step 3: Embedding Generation**

- **Purpose**: Converts narrative texts into vector embeddings for similarity-based searches.

**Key Function**:  
```python
def generate_embeddings(narratives):
    """
    Uses BERT to generate embeddings from narrative texts.
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

**Pipeline Code**:  
```python
vector_data = generate_embeddings(narratives)
logging.info(f"Generated embeddings for {len(vector_data)} narratives.")
```

---

#### **Step 4: FAISS Indexing and Storage**

- **Purpose**: Stores embeddings in a FAISS index, enabling efficient similarity searches.

**Key Function**:  
```python
def store_embeddings_in_faiss(vector_data):
    """
    Adds embeddings to a FAISS index, initializing it if it doesn't exist.
    """
    faiss_manager = FAISSManager(dimension=768)
    faiss_manager.add_vectors(vector_data)
    logging.info(f"Stored {len(vector_data)} embeddings in the FAISS index.")
```

**Pipeline Code**:  
```python
store_embeddings_in_faiss(vector_data)
logging.info("Embeddings stored in FAISS successfully.")
```

---

#### **Step 5: Visualization with TensorBoard**

- **Purpose**: Exports embeddings for TensorBoard visualization to analyze their relationships.

**Key Function**:  
```python
def export_to_tensorboard(embeddings, metadata, log_dir="logs/embedding_logs"):
    """
    Prepares embeddings and metadata for TensorBoard visualization.
    """
    os.makedirs(log_dir, exist_ok=True)
    np.savetxt(os.path.join(log_dir, "embeddings.tsv"), embeddings, delimiter="\t")
    with open(os.path.join(log_dir, "metadata.tsv"), "w") as f:
        f.write("\n".join(metadata))
    logging.info("Embeddings exported for TensorBoard.")
```

**Pipeline Code**:  
```python
export_to_tensorboard(
    [item[2] for item in vector_data],
    [f"ID: {item[0]} | Narrative: {item[1]}" for item in vector_data]
)
```

---

#### **Step 6: Updating the FAISS Index with New Listings**

- **Purpose**: Updates the FAISS index by processing newly added listings.

**Key Function**:  
```python
def update_pipeline_with_new_listings(index_path):
    """
    Retrieves new listings, generates their embeddings, and updates the FAISS index.
    """
    last_processed_id = load_last_processed_id()
    new_rows, new_last_processed_id = fetch_new_listings_from_mysql(last_processed_id)
    if new_rows:
        narratives = generate_narrative_manually(new_rows)
        vector_data = generate_embeddings(narratives)
        faiss_manager = FAISSManager(dimension=768)
        faiss_index = faiss_manager.load_index(index_path)
        faiss_manager.add_vectors(vector_data)
        save_last_processed_id(new_last_processed_id)
        logging.info(f"Updated FAISS index with {len(new_rows)} new listings.")
```

**Pipeline Code**:  
```python
update_pipeline_with_new_listings('faiss_index')
```

---

### 4. Code Walkthrough  

**File Structure**:  
```
project/
├── main.py                   # Main pipeline orchestration
├── config.py                 # Connection Details to MySQL Database
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
├── logs/                     # Log files
└── faiss_index/              # FAISS Index Files
```

---

### 5. Logging and Error Handling  

**Logging Setup**:  
```python
def setup_logging():
    """
    Configures daily rotating log files.
    """
    handler = logging.handlers.TimedRotatingFileHandler("logs/pipeline.log", when="midnight")
    logging.basicConfig(handlers=[handler], level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Logging initialized.")
```

---

### 6. FAQs and Troubleshooting  

Q: **How do I handle new listings?**  
A: Use `update_pipeline_with_new_listings('faiss_index')` to process new data.  

Q: **Why are embeddings not visible in TensorBoard?**  
A: Verify that embeddings are exported to `logs/embedding_logs`.  

Q: **How do I add new models?**  
A: Extend the `narrative_generator` module with additional AI models.

---
