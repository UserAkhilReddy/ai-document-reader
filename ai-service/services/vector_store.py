import faiss
import numpy as np

from services.embeddings import embed_texts

def create_vector_store(chunks: list[str]):

    if not chunks:
        return []
    
    embeddings= embed_texts(chunks) #converting the chunks into embeddings

    vectors = np.array(embeddings).astype("float32") #to numpy array

    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(vectors)

    return index, chunks

def search(index, stored_chunks : list[str], query_embedding: list[float], top_k=3):
    """search faiss index for most top 3 relevant chunks"""

    if index is None or not stored_chunks: 
        return []
    
    query_vector= np.array([query_embedding]).astype("float32")

    distances, indices = index.search( query_vector, top_k)
    
    results =[]
    for idx in indices[0]:
        if idx < len(stored_chunks): 
            results.append(stored_chunks[idx])

            return results