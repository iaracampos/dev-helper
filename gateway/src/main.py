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

# Configuração inicial
load_dotenv()

app = FastAPI()

# Configuração de logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gateway")

# --------------------------------------------------
# Conexão Redis 
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
                socket_keepalive=True
            )
            if conn.ping():
                logger.info(f" Conectado ao Redis em: {host}")
                return conn
        except (redis.ConnectionError, socket.gaierror) as e:
            logger.warning(f" Falha ao conectar em {host}: {str(e)}")
            continue
    
    logger.error("Não foi possível conectar a nenhum host Redis")
    return None

r = get_redis_connection()
if not r:
    exit(1)

# --------------------------------------------------
# Modelos e Endpoints
# --------------------------------------------------
class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(question: Question):
    try:
        request_id = str(uuid.uuid4())
        logger.info(f" Nova pergunta (ID: {request_id}): {question.question}")

        # Publicar pergunta no Redis
        request_data = {
            "id": request_id,
            "question": question.question,
            "status": "pending"
        }
        r.publish("questions_channel", json.dumps(request_data))
        r.set(f"request:{request_id}", json.dumps(request_data), ex=300)

        # Aguardar resposta com timeout
        for _ in range(60):  # 60 tentativas (1s cada)
            await asyncio.sleep(1)
            response = r.get(f"response:{request_id}")
            if response:
                response_data = json.loads(response)
                logger.info(f" Resposta enviada (ID: {request_id})")
                return {"answer": response_data["answer"]}

        raise HTTPException(status_code=504, detail="Timeout aguardando resposta")

    except Exception as e:
        logger.error(f"Erro ao processar pergunta: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "redis": "active" if r.ping() else "inactive"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )