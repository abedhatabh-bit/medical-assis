import os, json, uuid
import numpy as np
from typing import List, Dict
from openai import OpenAI
from app.config import OPENAI_API_KEY, EMBED_MODEL
import requests
from bs4 import BeautifulSoup

STORE_DIR = 'store'
CHUNKS_PATH = os.path.join(STORE_DIR, 'chunks.jsonl')
EMB_PATH = os.path.join(STORE_DIR, 'embeddings.npy')
IDMAP_PATH = os.path.join(STORE_DIR, 'id_map.json')
os.makedirs(STORE_DIR, exist_ok=True)

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

def clean_text(t: str) -> str:
    return ' '.join(t.split()).strip()

def chunk_text(text: str, max_chars: int = 900, overlap: int = 100) -> List[str]:
    text = clean_text(text)
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        cut = text.rfind('.', start, end)
        if cut == -1 or (cut - start) < int(max_chars * 0.6):
            cut = end
        chunk = text[start:cut].strip()
        if len(chunk) > 100:
            chunks.append(chunk)
        start = max(0, cut - overlap)
    return chunks

def save_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, 'a', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def load_id_map() -> list:
    if os.path.exists(IDMAP_PATH):
        return json.load(open(IDMAP_PATH, 'r', encoding='utf-8'))
    return []

def save_id_map(m: list) -> None:
    json.dump(m, open(IDMAP_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

def embed_texts(texts: List[str], batch_size: int = 128) -> np.ndarray:
    client = get_client()
    all_vecs: List[list] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_vecs.extend([d.embedding for d in resp.data])
    X = np.array(all_vecs, dtype='float32')
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / norms

def chunk_and_index(text: str, meta: Dict) -> Dict:
    chunks = chunk_text(text)
    if not chunks:
        raise RuntimeError('No chunks generated')
    emb = embed_texts(chunks)
    rows = []
    id_map = load_id_map()
    for ch in chunks:
        cid = str(uuid.uuid4())
        rows.append({'id': cid, 'text': ch, 'meta': meta})
        id_map.append(cid)
    save_jsonl(CHUNKS_PATH, rows)
    save_id_map(id_map)
    if os.path.exists(EMB_PATH):
        old = np.load(EMB_PATH)
        new = np.vstack([old, emb])
    else:
        new = emb
    np.save(EMB_PATH, new)
    return {'added_chunks': len(chunks), 'meta': meta}

def ingest_pdf(path: str, meta: Dict) -> Dict:
    import fitz  # Lazy import heavy dependency
    doc = fitz.open(path)
    pages = [page.get_text('text') for page in doc]
    text = '\n'.join(pages)
    return chunk_and_index(text, meta)

def ingest_web(url: str, meta: Dict) -> Dict:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        resp = requests.get(url, timeout=30, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        text = clean_text(soup.get_text(' '))
        return chunk_and_index(text, meta | {'source_type': 'web', 'url': url})
    except Exception as e:
        raise RuntimeError(f"Failed to ingest web page: {str(e)}")

if __name__ == '__main__':
    # Example usage - update paths as needed
    # pdf_sources = [
    #     {'path': '/path/to/your/document.pdf',
    #         'title': 'Example Document', 'year': 2024, 'publisher': 'Publisher'}
    # ]
    # for src in pdf_sources:
    #     print(ingest_pdf(src['path'], {'title': src['title'], 'year': src['year'], 'publisher': src['publisher']}))
    print("Ingest module loaded. Use CLI commands to ingest documents.")