import logging
import os
from dotenv import load_dotenv
from groq import Groq
from db.conection_qdrant import search_similar_documents, create_collection_if_not_exists
from cache.cag import get_cached_response, set_cached_response
from embeddings.embeddings import generate_embeddings_from_context_file
from config.config import (
    CONTEXT_DADOS_ESCOLA, CONTEXT_DATAS_VACINAS, CONTEXT_DICIONARIO_VACINAS,
    CONTEXT_ESCOLAS_MUNICIPAIS, CONTEXT_FAIXAS_TRANSPORTE, CONTEXT_LOCAIS_POSTOS,
    CONTEXT_POSTOS_VACINA, CONTEXT_TRANSPORTE, CONTEXT_NOVA_BASE
)

logging.basicConfig(level=logging.DEBUG)

load_dotenv()

api_key = os.environ.get('GROQ_API_KEY')
if not api_key:
    raise ValueError("Erro: 'GROQ_API_KEY' não encontrada no ambiente.")
client = Groq(api_key=api_key)

create_collection_if_not_exists()

PROMPT_TEMPLATE = """
Você é a Aurora, uma IA especializada em fornecer informações sobre a cidade de Recife em educação, saúde e transporte. Seu conhecimento se limita exclusivamente aos dados contidos nos arquivos JSON fornecidos...

Contexto relevante:
{context}

Regras para resposta:
1- Responda apenas com base nos dados disponíveis nos arquivos JSON. Se a informação solicitada não estiver presente, diga algo como: ‘Desculpe, não encontrei essa informação nos meus arquivos.’

2- Se não houver informações suficientes no contexto, responda:
"Desculpe, não encontrei informações suficientes para responder à sua pergunta. Mas estou aqui para ajudar no que for possível!"

3- Nunca invente ou forneça informações externas. NÃO mencione explicitamente que está seguindo um contexto na resposta final.

4- Seja objetiva e clara, apresentando as informações de forma acessível para qualquer usuário.

5- Mantenha a formalidade e a precisão, especialmente nos tópicos de saúde, educação e transporte.

### Pergunta:
{question}
"""

def generate_embeddings():
    contexts = [
        CONTEXT_DADOS_ESCOLA, CONTEXT_DATAS_VACINAS, CONTEXT_DICIONARIO_VACINAS,
        CONTEXT_ESCOLAS_MUNICIPAIS, CONTEXT_FAIXAS_TRANSPORTE, CONTEXT_LOCAIS_POSTOS,
        CONTEXT_POSTOS_VACINA, CONTEXT_TRANSPORTE, CONTEXT_NOVA_BASE
    ]
    embeddings = []
    for context in contexts:
        try:
            embeddings_from_context = generate_embeddings_from_context_file(context)
            embeddings.extend(embeddings_from_context)
            logging.debug(f"Embeddings gerados para {context}: {len(embeddings_from_context)} registros")
        except Exception as e:
            logging.error(f"Erro ao gerar embeddings para {context}: {e}")
    return embeddings

def search_similar_documents(prompt, top_k=5):
    """
    Função para buscar documentos similares no banco vetorial.
    """
    try:
        similar_docs = search_similar_documents(prompt, top_k=top_k)
        logging.debug(f"{len(similar_docs)} documentos semelhantes encontrados.")
        return similar_docs
    except Exception as e:
        logging.error(f"Erro ao buscar documentos semelhantes: {e}")
        return []

def select_top_documents(similar_documents, max_documents=10):
    if not isinstance(similar_documents, list):
        logging.error(f"Formato inesperado para similar_documents: {type(similar_documents)}")
        return []
    valid_documents = [doc for doc in similar_documents if isinstance(doc, dict)]
    sorted_documents = sorted(valid_documents, key=lambda x: x.get('score', 0), reverse=True)
    return sorted_documents[:max_documents]

def get_rag_response(prompt: str, similar_documents=None):
    cached_response = get_cached_response(prompt)
    if cached_response:
        return cached_response
    
    # Se similar_documents não for fornecido, realiza a busca
    if not similar_documents:
        similar_documents = search_similar_documents(prompt)
    
    top_documents = select_top_documents(similar_documents)
    
    embeddings = generate_embeddings()  # Continue gerando embeddings aqui, ou passe como argumento
    context = "\n\n".join(embeddings)
    
    if top_documents:
        context += "\n\nInformações adicionais:\n"
        for doc in top_documents:
            content = doc.get('content', 'Sem conteúdo disponível')
            context += f"• {content}\n"
    
    context = context[:2000] if len(context) > 2000 else context
    if not context.strip():
        return "Desculpe, não encontrei informações suficientes para responder sua pergunta."
    
    messages = [
        {"role": "system", "content": "Você é Aurora, uma IA especializada em informações sobre Recife."},
        {"role": "user", "content": PROMPT_TEMPLATE.format(context=context, question=prompt)}
    ]
    
    try:
        chat_completion = client.chat.completions.create(messages=messages, model="llama-3.3-70b-versatile")
        response = chat_completion.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Erro ao chamar API do Groq: {e}")
        return "Erro ao gerar a resposta. Por favor, tente novamente."
    
    set_cached_response(prompt, response)
    return response
