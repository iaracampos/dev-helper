import os
import json
import logging
import redis
import numpy as np
from sentence_transformers import SentenceTransformer
import hnswlib
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("retriever")

def get_redis_connection():
    redis_host = os.getenv("REDIS_HOST", "redis")
    try:
        conn = redis.Redis(host=redis_host, port=6379, db=0,
                           socket_connect_timeout=5, socket_keepalive=True)
        if conn.ping():
            logger.info(f"Conectado ao Redis em: {redis_host}")
            return conn
    except redis.ConnectionError as e:
        logger.error(f"Erro de conexão Redis: {e}")
    return None

r = get_redis_connection()
if not r:
    exit(1)

pubsub = r.pubsub()
pubsub.subscribe("questions_channel")

logger.info("Carregando modelo de embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Modelo de embeddings carregado")

logger.info("Inicializando índice vetorial...")
index = hnswlib.Index(space='cosine', dim=384)
index.init_index(max_elements=1000, ef_construction=200, M=16)

documents = [
    "Python é uma linguagem de programação interpretada de alto nível",
    "Herança permite que uma classe herde atributos e métodos de outra",
    "Docker é uma plataforma para criar e gerenciar containers",
    "FastAPI é um framework web moderno para construir APIs em Python",
    "Um decorador em Python modifica o comportamento de funções"
]

embeddings = model.encode(documents)
index.add_items(embeddings, np.arange(len(documents)))
logger.info(f"Índice vetorial pronto com {len(documents)} documentos")

def retrieve_contexts(query: str, k: int = 3):
    query_embedding = model.encode([query])[0]
    labels, _ = index.knn_query(query_embedding, k=k)
    return [documents[label] for label in labels[0]]

def main():
    logger.info("Serviço Retriever pronto. Aguardando perguntas...")
    for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                data = json.loads(message['data'])
                request_id = data['id']
                question = data['question']
                k = data.get('k', 3)

                logger.info(f"Processando pergunta (ID: {request_id}): {question}")

                contexts = retrieve_contexts(question, k)
                logger.info(f"Contextos encontrados: {len(contexts)}")

                response = {
                    "id": request_id,
                    "question": question,
                    "contexts": contexts,
                    "answer": "Resposta fictícia gerada pelo generator (substituir aqui se quiser)"
                }

                r.set(f"response:{request_id}", json.dumps(response), ex=300)
                logger.info(f"Resposta publicada no Redis para (ID: {request_id})")

            except Exception as e:
                logger.error(f"Erro ao processar mensagem: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
