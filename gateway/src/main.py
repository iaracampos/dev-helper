import os
import json
import uuid
import logging
import redis
import asyncio
import socket
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gateway")

def get_redis_connection():
    redis_host = os.getenv("REDIS_HOST", "redis")
    try:
        conn = redis.Redis(host=redis_host, port=6379, db=0,
                           socket_connect_timeout=5, socket_keepalive=True)
        if conn.ping():
            logger.info(f"Conectado ao Redis em: {redis_host}")
            return conn
    except (redis.ConnectionError, socket.gaierror) as e:
        logger.error(f"Erro de conexão Redis: {e}")
    return None

r = get_redis_connection()
if not r:
    logger.error("Não foi possível conectar ao Redis")
    exit(1)

class Question(BaseModel):
    question: str
    k: int = 3

@app.post("/ask")
async def ask_question(q: Question):
    request_id = str(uuid.uuid4())
    logger.info(f"Nova pergunta (ID: {request_id}): {q.question}")

    request_data = {
        "id": request_id,
        "question": q.question,
        "k": q.k,
        "status": "pending"
    }
    # Publica a pergunta no canal
    r.publish("questions_channel", json.dumps(request_data))
    # Salva a requisição para controle
    r.set(f"request:{request_id}", json.dumps(request_data), ex=300)

    # Espera resposta com timeout (até 30 segundos)
    for _ in range(30):
        await asyncio.sleep(1)
        response = r.get(f"response:{request_id}")
        if response:
            response_data = json.loads(response)
            logger.info(f"Resposta recebida (ID: {request_id})")
            return {
                "id": request_id,
                "question": response_data.get("question"),
                "contexts": response_data.get("contexts"),
                "answer": response_data.get("answer", "Sem resposta gerada")
            }

    raise HTTPException(status_code=504, detail="Timeout aguardando resposta")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "redis": "active" if r.ping() else "inactive"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
