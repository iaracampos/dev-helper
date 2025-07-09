
import json
import logging
import os
import socket
import time
from pathlib import Path
import redis
from dotenv import load_dotenv
from llama_cpp import Llama


load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("generator")

# --------------------------------------------------------------------------- #
# Redis                                                                       #
# --------------------------------------------------------------------------- #
def connect_redis() -> redis.Redis:
    """Tenta vários hosts até conseguir uma conexão Redis válida."""
    for host in ("redis", "localhost", "127.0.0.1", "host.docker.internal"):
        try:
            r = redis.Redis(
                host=host,
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=0,
                socket_connect_timeout=5,
            )
            if r.ping():
                logger.info("Conectado ao Redis em %s", host)
                return r
        except (redis.ConnectionError, socket.gaierror):
            logger.debug("Falha ao conectar em %s", host, exc_info=True)
    logger.critical("Não foi possível conectar a nenhum host Redis.")
    raise SystemExit(1)


redis_conn = connect_redis()
pubsub = redis_conn.pubsub()
pubsub.subscribe("generator_channel")

# --------------------------------------------------------------------------- #
# Modelo Llama‑cpp                                                            #
# --------------------------------------------------------------------------- #

docker_model = Path("/app/models/mistral.gguf")
repo_model = Path(__file__).resolve().parent.parent / "models" / "mistral.gguf"

MODEL_PATH = Path(os.getenv("MODEL_PATH", docker_model))
if not MODEL_PATH.exists():
    MODEL_PATH = repo_model  

N_CTX = int(os.getenv("N_CTX", 2_048))
N_THREADS = int(os.getenv("N_THREADS", os.cpu_count() or 4))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", 0))


def load_model() -> Llama:
    if not MODEL_PATH.exists():
        logger.critical("Arquivo do modelo não encontrado: %s", MODEL_PATH)
        raise SystemExit(1)

    logger.info("Carregando modelo: %s", MODEL_PATH)
    return Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )


llm = load_model()

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
STOP_TOKENS = ["~~", "[INST]"]


def build_prompt(question: str, contexts: list[str]) -> str:
    ctx = "\n".join(contexts)
    return (
        "~~[INST] <> Você é um assistente técnico especializado em desenvolvimento de software.\n"
        "Responda apenas com base no CONTEXTO abaixo. Seja conciso e, se não souber, diga "
        '"Não posso ajudar com isso".\n'
        f"<> CONTEXTO:\n{ctx}\n\nPERGUNTA: {question}\n[/INST]"
    )


def generate(prompt: str) -> str:
    try:
        out = llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop=STOP_TOKENS,
            echo=False,
        )
        return out["choices"][0]["text"].strip()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro na geração: %s", exc)
        return "Erro ao gerar resposta."


# --------------------------------------------------------------------------- #
# Loop principal                                                              #
# --------------------------------------------------------------------------- #
logger.info("Generator iniciado ― aguardando mensagens…")

for msg in pubsub.listen():
    if msg["type"] != "message":
        continue

    try:
        start = time.perf_counter()
        payload = json.loads(msg["data"])

        req_id: str = payload["id"]
        question: str = payload["question"]
        contexts: list[str] = payload["contexts"]

        logger.info("ID %s ─ processando ‘%s…’", req_id, question[:60])

        answer = generate(build_prompt(question, contexts))
        elapsed = time.perf_counter() - start

        redis_conn.set(
            f"response:{req_id}",
            json.dumps(
                {
                    "id": req_id,
                    "question": question,
                    "answer": answer,
                    "status": "completed",
                    "elapsed": elapsed,
                }
            ),
            ex=300,
        )
        logger.info("ID %s ─ resposta pronta em %.2fs", req_id, elapsed)

    except json.JSONDecodeError:
        logger.error("Mensagem JSON inválida recebida: %s", msg)
    except Exception: 
        logger.exception("Erro inesperado no loop principal")
