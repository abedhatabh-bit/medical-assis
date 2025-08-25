import os, json
import numpy as np
from typing import List, Dict
from openai import OpenAI
from app.config import OPENAI_API_KEY, EMBED_MODEL, OFFLINE_MODE, LOCAL_EMBED_DIM

STORE_DIR = 'store'
CHUNKS_PATH = os.path.join(STORE_DIR, 'chunks.jsonl')
EMB_PATH = os.path.join(STORE_DIR, 'embeddings.npy')
IDMAP_PATH = os.path.join(STORE_DIR, 'id_map.json')

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

def load_chunks() -> List[Dict]:
    if not os.path.exists(CHUNKS_PATH): return []
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_id_map() -> list:
    return json.load(open(IDMAP_PATH, 'r', encoding='utf-8'))

def embed_query(q: str) -> np.ndarray:
    if OFFLINE_MODE:
        rng = np.random.default_rng(abs(hash(q)) % (2**32))
        v = rng.standard_normal(LOCAL_EMBED_DIM).astype('float32')
    else:
        client = get_client()
        resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
        v = np.array(resp.data[0].embedding, dtype='float32')
    v = v / (np.linalg.norm(v) + 1e-8)
    return v

def retrieve(query: str, k: int = 8) -> List[Dict]:
    if not os.path.exists(EMB_PATH):
        raise RuntimeError('No index yet. Ingest sources first.')

    # Memory-map embeddings to avoid loading full matrix into RAM
    X = np.load(EMB_PATH, mmap_mode='r')
    id_map = load_id_map()
    qv = embed_query(query)

    # Compute similarities and select top-k efficiently
    sims = X @ qv
    if k <= 0:
        return []
    if k >= sims.shape[0]:
        top_idx = np.argsort(-sims)
    else:
        part = np.argpartition(-sims, k - 1)[:k]
        top_idx = part[np.argsort(-sims[part])]

    # Read only needed rows from chunks file
    target_ids = {id_map[i] for i in top_idx}
    results_by_id = {}
    if not os.path.exists(CHUNKS_PATH):
        raise RuntimeError('Missing chunks store. Ingest sources first.')
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            rid = row.get('id')
            if rid in target_ids:
                results_by_id[rid] = row
                if len(results_by_id) == len(target_ids):
                    break

    return [results_by_id[id_map[i]] for i in top_idx]
