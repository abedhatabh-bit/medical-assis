import os, json
from typing import List, Dict

STORE_DIR = 'store'
CHUNKS_PATH = os.path.join(STORE_DIR, 'chunks.jsonl')
EMB_PATH = os.path.join(STORE_DIR, 'embeddings.npy')
IDMAP_PATH = os.path.join(STORE_DIR, 'id_map.json')

os.makedirs(STORE_DIR, exist_ok=True)

def save_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, 'a', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def load_chunks() -> List[Dict]:
    if not os.path.exists(CHUNKS_PATH): return []
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_id_map() -> list:
    if os.path.exists(IDMAP_PATH):
        return json.load(open(IDMAP_PATH, 'r', encoding='utf-8'))
    return []

def save_id_map(m: list) -> None:
    json.dump(m, open(IDMAP_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)