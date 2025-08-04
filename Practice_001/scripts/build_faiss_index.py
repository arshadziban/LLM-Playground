import os
import faiss
import numpy as np
import pickle

def main():
    # Use relative paths based on the project structure
    cleaned_dir = os.path.join('data', 'cleaned', 'bangladesh.text')
    index_dir = os.path.join('data', 'index')
    os.makedirs(index_dir, exist_ok=True)

    # Load embeddings and chunks from relative paths
    embeddings_path = os.path.join(cleaned_dir, 'embeddings.npy')
    chunks_path = os.path.join(cleaned_dir, 'chunks.pkl')

    print(f"Loading embeddings from: {embeddings_path}")
    print(f"Loading chunks from: {chunks_path}")

    embeddings = np.load(embeddings_path)
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)

    print(f"Loaded {len(chunks)} chunks with embeddings shape: {embeddings.shape}")

    dim = embeddings.shape[1]

    # Build FAISS index (L2 distance)
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save FAISS index and chunks for retrieval at relative paths
    index_path = os.path.join(index_dir, 'faiss.index')
    chunks_save_path = os.path.join(index_dir, 'chunks.pkl')

    faiss.write_index(index, index_path)
    with open(chunks_save_path, 'wb') as f:
        pickle.dump(chunks, f)

    print(f"FAISS index saved at: {index_path}")
    print(f"Chunks saved at: {chunks_save_path}")

if __name__ == "__main__":
    main()
