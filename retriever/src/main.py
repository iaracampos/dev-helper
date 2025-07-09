
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
import hnswlib
import numpy as np
import redis
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()  

EMB_MODEL_NAME: str = os.getenv("EMB_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_PATH: Path = Path(os.getenv("INDEX_PATH", "index/hnswlib_index.bin"))
META_PATH: Path = Path(os.getenv("META_PATH", "index/meta.json"))
REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
TOP_K_DEFAULT: int = int(os.getenv("TOP_K", 5))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("retriever")


def _connect_redis() -> redis.Redis | None:
    """Tenta conectar no Redis; devolve None se não conseguir."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        # faz um ping 
        r.ping()
        logger.info("Conectado ao Redis em %s:%s (db=%s)", REDIS_HOST, REDIS_PORT, REDIS_DB)
        return r
    except redis.exceptions.ConnectionError as exc:
        logger.warning("Redis indisponível: %s", exc)
        return None


def _load_metadata() -> Dict[int, Dict]:
    if not META_PATH.exists():
        logger.warning("Arquivo de metadados %s não encontrado; retornando dicionário vazio.", META_PATH)
        return {}
    with META_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    # converte chaves para int (hnswlib retorna int IDs)
    return {int(k): v for k, v in data.items()}


def _save_metadata(meta: Dict[int, Dict]) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with META_PATH.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)




class Retriever:

    def __init__(self, rebuild_if_missing: bool = False):
        # modelo de embeddings
        self.model = SentenceTransformer(EMB_MODEL_NAME)

        # emb_dim é inferido pelo modelo
        self.emb_dim: int = self.model.get_sentence_embedding_dimension()

        # tenta conectar ao Redis 
        self.redis = _connect_redis()

        # metadados {internal_id: {"text": ..., "source": ...}}
        self.meta: Dict[int, Dict] = _load_metadata()

        # índice vetorial
        self.index = hnswlib.Index(space="cosine", dim=self.emb_dim)

        if INDEX_PATH.exists():
            logger.info("Carregando índice de %s ...", INDEX_PATH)
            self.index.load_index(str(INDEX_PATH))
            self.index.set_ef(64)
        elif rebuild_if_missing:
            logger.info("Índice não encontrado. Reconstruindo do zero ...")
            # não temos embeddings salvos; gere a partir dos metadados
            self._rebuild_index()
        else:
            raise FileNotFoundError(
                f"Índice {INDEX_PATH} não encontrado. "
                "Passe rebuild_if_missing=True para reconstruir ou execute o script de ingestão."
            )
    def search(self, query: str, top_k: int = TOP_K_DEFAULT) -> List[Tuple[float, Dict]]:
        """
        Devolve uma lista de (score, metadata) ordenada pelo melhor score.
        """
        if not query.strip():
            return []

        # Embedding da pergunta (usa Redis como cache se possível)
        qkey = f"emb:{query}"
        if self.redis is not None and self.redis.exists(qkey):
            query_emb = np.array(json.loads(self.redis.get(qkey)), dtype=np.float32)
            logger.debug("Embedding de consulta recuperado do Redis.")
        else:
            query_emb = self.model.encode(query, convert_to_numpy=True)
            if self.redis is not None:

                self.redis.set(qkey, json.dumps(query_emb.tolist()))

        # busca no índice
        labels, distances = self.index.knn_query(query_emb, k=top_k)
        
        labels, distances = labels[0], distances[0]

        results: List[Tuple[float, Dict]] = []
        for label, dist in zip(labels, distances):
            score = 1 - dist
            meta = self.meta.get(int(label), {})
            results.append((score, meta))

        return results

    def get_contexts(self, query: str, top_k: int = TOP_K_DEFAULT) -> List[str]:
        """Retorna somente os textos (campo 'text') dos documentos mais próximos."""
        return [m.get("text", "") for _, m in self.search(query, top_k)]


    def _rebuild_index(self) -> None:
        """Reconstrói o índice a partir de `self.meta`."""
        if not self.meta:
            raise RuntimeError(
                "Não há metadados para reconstruir o índice. "
                "Execute o script de ingestão primeiro."
            )

        embeddings = []
        ids = []
        for internal_id, m in self.meta.items():
            text = m.get("text", "")
            emb = self.model.encode(text, convert_to_numpy=True)
            embeddings.append(emb)
            ids.append(internal_id)

        emb_matrix = np.vstack(embeddings).astype(np.float32)

        # inicializa e treina o índice
        self.index.init_index(max_elements=len(ids), ef_construction=200, M=16)
        self.index.add_items(emb_matrix, ids)
        self.index.set_ef(64)

        # salva no disco
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.index.save_index(str(INDEX_PATH))
        logger.info("Índice salvo em %s", INDEX_PATH)

    
        if self.redis is not None:
            pipe = self.redis.pipeline(True)
            for text, emb in zip((m["text"] for m in self.meta.values()), embeddings):
                pipe.set(f"emb:{text}", json.dumps(emb.tolist()))
            pipe.execute()
            logger.info("Embeddings salvos em cache Redis.")



    