from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
from db.conection_qdrant import create_collection_if_not_exists, save_embeddings, search_similar_documents
from rag.rag import get_rag_response
from embeddings.embeddings import embed_text
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

CORS(app, supports_credentials=True, origins="*", methods=["POST", "GET", "OPTIONS"], allow_headers=["*"])

create_collection_if_not_exists()

class ChatRequest(BaseModel):
    prompt: str

@app.route('/chat/', methods=['POST'])
def chat():
    try:
        data = request.get_json()

        if not data or 'prompt' not in data:
            return jsonify({"error": "Prompt não fornecido."}), 400

        prompt = data['prompt']
        logging.debug(f"Recebido o prompt: {prompt}")

        # Gerar o embedding do prompt
        prompt_embedding = embed_text(prompt)

        # Salvar os embeddings gerados no banco vetorial
        save_embeddings([prompt], [prompt_embedding])

        # Buscar documentos similares com base no embedding do prompt
        similar_documents = search_similar_documents(prompt)

        # Se não houver documentos similares, retornar uma resposta padrão
        if not similar_documents:
            logging.debug("Nenhum documento similar encontrado.")
            return jsonify({"response": "Ainda não tenho informações sobre isso. Pode fornecer mais detalhes?"})

        # Obter a resposta do RAG com base nos documentos similares
        response = get_rag_response(prompt, similar_documents)

        return jsonify({"response": response})

    except Exception as e:
        logging.error(f"Erro no processamento do chat: {str(e)}")
        return jsonify({"error": "Ocorreu um erro ao processar a sua solicitação."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
