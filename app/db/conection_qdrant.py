from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import logging


logging.basicConfig(level=logging.DEBUG)
qdrant_client = QdrantClient(url="http://qdrant:6333")

COLLECTION_NAME = "documents" 

def create_collection_if_not_exists():
    """
    Verifica se a coleção existe no Qdrant. Caso não exista, cria uma nova.
    """
    collections = qdrant_client.get_collections().collections
    existing_collections = [col.name for col in collections]

    if COLLECTION_NAME not in existing_collections:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Collection '{COLLECTION_NAME}' criada com sucesso.")
    else:
        print(f"Collection '{COLLECTION_NAME}' já existe.")

def save_embeddings(chunks, embeddings):
    """
    Salva os embeddings gerados no Qdrant.
    """
    points = [
        PointStruct(id=np.random.randint(1, 1e9), vector=embedding, payload={"text": chunk})
        for chunk, embedding in zip(chunks, embeddings)
    ]

    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Inserido {len(chunks)} chunks no banco vetorial.")

def search_similar_documents(query_text, top_k=5):
    from app.embeddings.embeddings import embed_text 
    query_embedding = embed_text(query_text)

    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k
    )

    return [result.payload["text"] for result in search_results]

# Garante que a coleção existe ao iniciar o módulo
create_collection_if_not_exists()