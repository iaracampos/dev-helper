import os
import json
import logging
import redis
import numpy as np
import socket
import time
from sentence_transformers import SentenceTransformer
import hnswlib
from dotenv import load_dotenv

# Configuração inicial
load_dotenv()

# Configuração de logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("retriever")

# --------------------------------------------------
# Configuração Redis 
# --------------------------------------------------
def get_redis_connection():
    """Estabelece conexão com Redis com fallback automático"""
    hosts_to_try = [
        'redis',        # Nome do serviço no Docker
        'localhost',    # Para execução sem Docker
        '127.0.0.1',    # Fallback local
        'host.docker.internal'  # Para Docker no Windows/Mac
    ]
    
    for host in hosts_to_try:
        try:
            conn = redis.Redis(
                host=host,
                port=6379,
                db=0,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30
            )
            if conn.ping():
                logger.info(f"Conectado ao Redis em: {host}")
                return conn
        except redis.ConnectionError:
            logger.warning(f"Falha ao conectar em: {host}")
            continue
    
    logger.error("Não foi possível conectar a nenhum host Redis")
    return None

r = get_redis_connection()
if not r:
    exit(1)

pubsub = r.pubsub()
pubsub.subscribe("questions_channel")

# --------------------------------------------------
# Configuração do Modelo de Embeddings
# --------------------------------------------------
logger.info("Carregando modelo de embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Modelo de embeddings carregado")

# --------------------------------------------------
# Banco de Dados Vetorial (HNSW)
# --------------------------------------------------
logger.info("Inicializando índice vetorial...")
index = hnswlib.Index(space='cosine', dim=384)
index.init_index(max_elements=1000, ef_construction=200, M=16)

# Dados de exemplo 
documents = [
    "Python é uma linguagem de programação interpretada de alto nível",
    "Herança permite que uma classe herde atributos e métodos de outra",
    "Docker é uma plataforma para criar e gerenciar containers",
    "FastAPI é um framework web moderno para construir APIs em Python",
    "Um decorador em Python modifica o comportamento de funções"
]

# Gerar embeddings e adicionar ao índice
embeddings = model.encode(documents)
index.add_items(embeddings, np.arange(len(documents)))
logger.info(f"Índice vetorial pronto com {len(documents)} documentos")

# --------------------------------------------------
# Função Principal
# --------------------------------------------------
def retrieve_contexts(query: str, k: int = 3) -> list:
    """Busca os k contextos mais relevantes para a consulta"""
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
                
                logger.info(f"Processando pergunta (ID: {request_id}): {question}")
                
                # Busca contextos
                contexts = retrieve_contexts(question)
                logger.info(f"Contextos encontrados: {len(contexts)}")
                
                # Prepara dados para o Generator
                response = {
                    "id": request_id,
                    "question": question,
                    "contexts": contexts
                }
                
                # Publica no canal do Generator
                r.publish("generator_channel", json.dumps(response))
                logger.info(f"Resposta publicada (ID: {request_id})")
                
            except json.JSONDecodeError:
                logger.error("Erro ao decodificar mensagem JSON")
            except Exception as e:
                logger.error(f"Erro inesperado: {str(e)}", exc_info=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Serviço Retriever encerrado")
    except Exception as e:
        logger.error(f"Erro fatal: {str(e)}", exc_info=True)