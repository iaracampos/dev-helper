import json
import logging
import os
import socket
import time
import uuid
from pathlib import Path
import redis
from dotenv import load_dotenv
from llama_cpp import Llama

# Configuração inicial
load_dotenv()

# Configuração de logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("generator")

# Conexão com Redis
def connect_redis() -> redis.Redis:
    redis_hosts = ["redis", "localhost", "127.0.0.1", "host.docker.internal"]
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    
    for host in redis_hosts:
        try:
            r = redis.Redis(
                host=host,
                port=redis_port,
                db=0,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            if r.ping():
                logger.info(f"Conectado ao Redis em {host}:{redis_port}")
                return r
        except (redis.ConnectionError, socket.gaierror) as e:
            logger.debug(f"Falha ao conectar em {host}: {str(e)}")
    
    logger.critical("Não foi possível conectar a nenhum host Redis")
    raise RuntimeError("Conexão com Redis falhou")

# Carregamento do Modelo
def load_model() -> Llama:
    # Configuração de paths do modelo
    docker_model = Path("/app/models/mistral.gguf")
    local_model = Path(__file__).parent.parent / "models" / "mistral.gguf"
    model_path = Path(os.getenv("MODEL_PATH", docker_model))
    
    if not model_path.exists():
        model_path = local_model
    
    if not model_path.exists():
        logger.critical(f"Arquivo do modelo não encontrado: {model_path}")
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}")

    logger.info(f"Carregando modelo: {model_path}")
    
    return Llama(
        model_path=str(model_path),
        n_ctx=int(os.getenv("N_CTX", 2048)),
        n_threads=int(os.getenv("N_THREADS", max(os.cpu_count() or 4, 4))),
        n_gpu_layers=int(os.getenv("N_GPU_LAYERS", 0)),
        verbose=False
    )

# Geração de Resposta
def generate_response(llm: Llama, prompt: str) -> str:
    try:
        output = llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop=["~~", "[INST]"],
            echo=False
        )
        return output["choices"][0]["text"].strip()
    except Exception as e:
        logger.error(f"Erro na geração: {str(e)}")
        return f"Erro ao gerar resposta: {str(e)}"

# Construção do Prompt
def build_prompt(question: str, contexts: list[str] = None) -> str:
    context = "\n".join(contexts) if contexts else "Nenhum contexto fornecido"
    return f"""~~[INST] <> Você é um assistente técnico especializado em desenvolvimento de software.
Responda apenas com base no CONTEXTO abaixo. Seja conciso e, se não souber, diga "Não posso ajudar com isso".
<> CONTEXTO:
{context}

PERGUNTA: {question}
[/INST]"""

# Processamento Principal
def main():
    # Inicializa conexões
    redis_conn = connect_redis()
    llm = load_model()
    
    # Configura subscription
    pubsub = redis_conn.pubsub()
    pubsub.subscribe("questions_channel")
    logger.info("Generator pronto - ouvindo no canal 'questions_channel'")

    # Loop principal
    for message in pubsub.listen():
        if message["type"] != "message":
            continue

        try:
            start_time = time.time()
            payload = json.loads(message["data"])
            
            # Extrai dados da mensagem
            request_id = payload.get("id", str(uuid.uuid4()))
            question = payload["question"]
            k = payload.get("k", 3)
            contexts = payload.get("contexts", [])
            
            logger.info(f"Processando ID {request_id}: {question[:50]}...")

            # Gera resposta
            prompt = build_prompt(question, contexts)
            answer = generate_response(llm, prompt)
            processing_time = time.time() - start_time

            # Envia resposta
            response_data = {
                "id": request_id,
                "question": question,
                "contexts": contexts,
                "answer": answer,
                "processing_time": round(processing_time, 2),
                "status": "completed"
            }
            
            redis_conn.setex(
                f"response:{request_id}",
                time=3600,  
                value=json.dumps(response_data)
            )
            
            logger.info(f"Resposta para ID {request_id} pronta em {processing_time:.2f}s")

        except json.JSONDecodeError:
            logger.error("Mensagem JSON inválida recebida")
        except KeyError as e:
            logger.error(f"Campo obrigatório faltando: {str(e)}")
            redis_conn.setex(
                f"response:{request_id}",
                time=3600,
                value=json.dumps({
                    "id": request_id,
                    "error": f"Campo obrigatório faltando: {str(e)}",
                    "status": "failed"
                })
            )
        except Exception as e:
            logger.error(f"Erro inesperado: {str(e)}", exc_info=True)
            redis_conn.setex(
                f"response:{request_id}",
                time=3600,
                value=json.dumps({
                    "id": request_id,
                    "error": str(e),
                    "status": "failed"
                })
            )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Generator encerrado pelo usuário")
    except Exception as e:
        logger.critical(f"Falha crítica: {str(e)}", exc_info=True)