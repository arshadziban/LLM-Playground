## Project Structure

- `scripts/scrape.ipynb`  
  Scrape paragraphs about Bangladesh from Wikipedia and save raw data.

- `scripts/clean_chunk_embed.ipynb`  
  Clean the scraped text, split it into chunks, and create embeddings.

- `scripts/build_faiss_index.py`  
  Build a FAISS index from the embeddings for similarity search.

- `scripts/query_rag.ipynb`  
  Query the FAISS index and generate answers using the RAG model.

---

## How to Run

1. **Scrape Data**  
   Run `scripts/scrape.ipynb` to collect raw paragraphs.

2. **Clean, Chunk & Embed**  
   Run `scripts/clean_chunk_embed.ipynb` to preprocess the data and generate embeddings.

3. **Build FAISS Index**  
   Run `scripts/build_faiss_index.py` to create the vector search index.

4. **Query the RAG Model**  
   Run `scripts/query_rag.ipynb` to interactively ask questions and get generated answers.

---

## Requirements

- Python 3.7+
- Install dependencies with:  
  `pip install -r requirements.txt`  
  (Include packages like `requests`, `beautifulsoup4`, `sentence-transformers`, `faiss-cpu`, `transformers`, `torch`)
