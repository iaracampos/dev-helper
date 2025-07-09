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
   
    r.publish("questions_channel", json.dumps(request_data))

    r.set(f"request:{request_id}", json.dumps(request_data), ex=3600)

   
    max_attempts = 360  
    for attempt in range(max_attempts):
        await asyncio.sleep(5)  
        response = r.get(f"response:{request_id}")
        if response:
            response_data = json.loads(response)
            logger.info(f"Resposta recebida (ID: {request_id}) após {attempt * 5} segundos")
            return {
                "id": request_id,
                "question": response_data.get("question"),
                "contexts": response_data.get("contexts"),
                "answer": response_data.get("answer", "Sem resposta gerada"),
                "processing_time": f"{attempt * 5} segundos"
            }
        
        
        if attempt % 12 == 0: 
            logger.info(f"Aguardando resposta (ID: {request_id}) - {attempt // 12} minutos decorridos")

    raise HTTPException(
        status_code=504, 
        detail="Timeout após 30 minutos aguardando resposta. O processamento pode estar demorando mais que o normal."
    )

@app.get("/status/{request_id}")
async def check_status(request_id: str):
    request_data = r.get(f"request:{request_id}")
    if not request_data:
        raise HTTPException(status_code=404, detail="Requisição não encontrada ou expirada")
    
    request_data = json.loads(request_data)
    response_data = r.get(f"response:{request_id}")
    
    if response_data:
        response_data = json.loads(response_data)
        return {
            "status": "completed",
            "response": response_data
        }
    else:
        return {
            "status": request_data.get("status", "pending"),
            "message": "A requisição ainda está sendo processada"
        }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "redis": "active" if r.ping() else "inactive"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)