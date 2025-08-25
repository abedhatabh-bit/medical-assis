import os, json
import numpy as np
from typing import List, Dict
from app.embedding import embed_query
from app.storage import CHUNKS_PATH, EMB_PATH, load_id_map


def retrieve(query: str, k: int = 8) -> List[Dict]:
    if not os.path.exists(EMB_PATH):
        raise RuntimeError('No index yet. Ingest sources first.')

    X = np.load(EMB_PATH, mmap_mode='r')
    id_map = load_id_map()
    qv = embed_query(query)

    sims = X @ qv
    if k <= 0:
        return []
    if k >= sims.shape[0]:
        top_idx = np.argsort(-sims)
    else:
        part = np.argpartition(-sims, k - 1)[:k]
        top_idx = part[np.argsort(-sims[part])]

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
