import json
import os
import logging
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from db.conection_qdrant import save_embeddings 

logging.basicConfig(level=logging.DEBUG)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text: str):
    """Gera embeddings para um texto usando um modelo de transformação."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings.flatten().tolist()

def generate_embeddings_from_context_file(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            logging.warning(f"Formato inesperado no arquivo {file_path}. Esperado uma lista de objetos JSON.")
            return []

        embeddings = []
        texts = []

        for item in data:
            if isinstance(item, dict):
                text = " ".join(str(value) for value in item.values())  # Junta os valores dos campos do JSON
                texts.append(text)
                embeddings.append(embed_text(text))

        # Salvar os embeddings no banco vetorial
        save_embeddings(texts, embeddings)
        logging.info(f"Embeddings gerados e salvos com sucesso para o arquivo: {file_path}")

        return texts

    except FileNotFoundError:
        logging.error(f"Arquivo de contexto não encontrado: {file_path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Erro ao decodificar JSON no arquivo: {file_path}")
        return []