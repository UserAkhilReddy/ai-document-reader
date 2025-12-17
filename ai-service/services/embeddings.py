from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts : list[str]) -> list[list[float]]:
    """conversion """
    if not texts: 
        return []
    
    embeddings = model.encode(texts , convert_to_numpy = True)  
    return embeddings.tolist()

def embed_query (query: str) -> list[float]:
    """conversion of question"""
    if not query.strip():
        return []
    
    embedding = model.encode(query, convert_to_numpy=True)
    return embedding.tolist()
    
               
