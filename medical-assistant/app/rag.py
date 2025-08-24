
    import os, json 
    import numpy as np 
    from typing import List, Dict 
    from openai import OpenAI 
    from app.config import OPENAI_API_KEY, EMBED_MODEL 
     
    STORE_DIR = "store" 
    CHUNKS_PATH = os.path.join(STORE_DIR, "chunks.jsonl") 
    EMB_PATH = os.path.join(STORE_DIR, "embeddings.npy") 
    IDMAP_PATH = os.path.join(STORE_DIR, "id_map.json") 
     
    client = OpenAI(api_key=OPENAI_API_KEY) 
     
    def load_chunks() -> List[Dict]: 
    if not os.path.exists(CHUNKS_PATH): return [] 
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f: 
    return [json.loads(line) for line in f] 
     
    def load_id_map() -> list: 
    return json.load(open(IDMAP_PATH, "r", encoding="utf-8")) 
     
    def embed_query(q: str) -> np.ndarray: 
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q]) 
    v = np.array(resp.data[0].embedding, dtype="float32") 
    v = v / (np.linalg.norm(v) + 1e-8) 
    return v 
     
    def retrieve(query: str, k=8) -> List[Dict]: 
    if not os.path.exists(EMB_PATH): 
    raise RuntimeError("No index yet. Ingest sources first.") 
    X = np.load(EMB_PATH) # normalized row vectors 
    id_map = load_id_map() 
    qv = embed_query(query) # normalized 
    sims = X @ qv 
    idx = np.argsort(-sims)[:k] 
    rows = load_chunks() 
    id_to_row = {r["id"]: r for r in rows} 
    return [id_to_row[id_map[i]] for i in idx] 
     