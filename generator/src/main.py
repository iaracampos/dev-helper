import os
import json
import logging
import redis
import socket
import time
from llama_cpp import Llama
from dotenv import load_dotenv

# Configuração inicial
load_dotenv()

# Configuração de logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("generator")

# --------------------------------------------------
# Conexão Redis 
# --------------------------------------------------
def get_redis_connection():
    """Estabelece conexão com Redis com fallback automático"""
    hosts_to_try = [
        'redis',
        'localhost',
        '127.0.0.1',
        'host.docker.internal'
    ]
    
    for host in hosts_to_try:
        try:
            conn = redis.Redis(
                host=host,
                port=6379,
                db=0,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            if conn.ping():
                logger.info(f" Conectado ao Redis em: {host}")
                return conn
        except (redis.ConnectionError, socket.gaierror) as e:
            logger.warning(f"  Falha ao conectar em {host}: {str(e)}")
            continue
    
    logger.error(" Não foi possível conectar a nenhum host Redis")
    return None

r = get_redis_connection()
if not r:
    exit(1)

pubsub = r.pubsub()
pubsub.subscribe("generator_channel")

# --------------------------------------------------
# Configuração do Modelo LLM
# --------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "generator/models/mistral.gguf")

def load_llm_model():
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {MODEL_PATH}")

        logger.info(f"Carregando modelo de: {MODEL_PATH}")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False
        )
        logger.info("\ Modelo carregado com sucesso!")
        return llm
    except Exception as e:
        logger.error(f" Falha ao carregar modelo: {str(e)}")
        exit(1)

llm = load_llm_model()

# --------------------------------------------------
# Funções Principais
# --------------------------------------------------
def build_prompt(question: str, contexts: list[str]) -> str:
    """Constrói o prompt para o LLM"""
    context_str = "\n".join(contexts)
    return f"""<s>[INST] <<SYS>>
Você é um assistente técnico especializado em desenvolvimento de software.
Responda com base APENAS no contexto fornecido.
Seja conciso e preciso. Se não souber, diga "Não posso ajudar com isso".
<</SYS>>

Contexto:
{context_str}

Pergunta: {question} [/INST]"""

def generate_response(prompt: str) -> str:
    """Gera resposta com tratamento de erros"""
    try:
        output = llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop=["</s>", "[INST]"],
            echo=False
        )
        return output['choices'][0]['text'].strip()
    except Exception as e:
        logger.error(f"Erro na geração: {str(e)}")
        return "Erro ao gerar resposta"

# --------------------------------------------------
# Loop Principal
# --------------------------------------------------
logger.info("Generator pronto. Aguardando requisições...")

for message in pubsub.listen():
    if message['type'] == 'message':
        try:
            start_time = time.time()
            data = json.loads(message['data'])
            request_id = data['id']
            question = data['question']
            contexts = data['contexts']

            logger.info(f"Processando (ID: {request_id}): {question[:50]}...")

            # Gerar resposta
            prompt = build_prompt(question, contexts)
            answer = generate_response(prompt)
            elapsed = time.time() - start_time

            logger.info(f"Resposta gerada em {elapsed:.2f}s (ID: {request_id})")

            # Publicar resposta
            r.set(f"response:{request_id}", json.dumps({
                "id": request_id,
                "question": question,
                "answer": answer,
                "status": "completed"
            }), ex=300)

        except json.JSONDecodeError:
            logger.error(" Mensagem JSON inválida")
        except Exception as e:
            logger.error(f" Erro inesperado: {str(e)}", exc_info=True)