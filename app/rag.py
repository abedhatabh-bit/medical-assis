import os, json
import numpy as np
from typing import List, Dict
from app.config import EMBED_MODEL, get_openai_client

STORE_DIR = 'store'
CHUNKS_PATH = os.path.join(STORE_DIR, 'chunks.jsonl')
EMB_PATH = os.path.join(STORE_DIR, 'embeddings.npy')
IDMAP_PATH = os.path.join(STORE_DIR, 'id_map.json')

def load_chunks() -> List[Dict]:
    if not os.path.exists(CHUNKS_PATH): return []
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_id_map() -> list:
    return json.load(open(IDMAP_PATH, 'r', encoding='utf-8'))

def embed_query(q: str) -> np.ndarray:
    client = get_openai_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    v = np.array(resp.data[0].embedding, dtype='float32')
    v = v / (np.linalg.norm(v) + 1e-8)
    return v

def retrieve(query: str, k=8) -> List[Dict]:
    if not os.path.exists(EMB_PATH):
        raise RuntimeError('No index yet. Ingest sources first.')
    # Memory-map embeddings to reduce load time and memory footprint
    X = np.load(EMB_PATH, mmap_mode='r')
    id_map = load_id_map()
    qv = embed_query(query)
    # Compute similarities lazily without materializing full sims for huge matrices
    sims = X @ qv  # numpy handles mmap transparently
    # Get top-k indices efficiently
    if k <= 0:
        return []
    if k >= sims.shape[0]:
        top_idx = np.argsort(-sims)
    else:
        # argpartition for partial top-k, then sort those
        part = np.argpartition(-sims, k-1)[:k]
        order = np.argsort(-sims[part])
        top_idx = part[order]
    # Load only once and map by id
    rows = load_chunks()
    id_to_row = {r['id']: r for r in rows}
    return [id_to_row[id_map[i]] for i in top_idx]
