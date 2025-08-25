import os, json, uuid
import numpy as np
from typing import List, Dict
from app.config import EMBED_MODEL, get_openai_client

STORE_DIR = 'store'
CHUNKS_PATH = os.path.join(STORE_DIR, 'chunks.jsonl')
EMB_PATH = os.path.join(STORE_DIR, 'embeddings.npy')
IDMAP_PATH = os.path.join(STORE_DIR, 'id_map.json')
CHUNK_OFFSETS_PATH = os.path.join(STORE_DIR, 'chunk_offsets.npy')
os.makedirs(STORE_DIR, exist_ok=True)


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


def save_jsonl(path: str, rows: List[Dict]) -> list:
    """Append rows to JSONL and return byte offsets for each appended line."""
    offsets: list = []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'ab') as f:
        for r in rows:
            line = (json.dumps(r, ensure_ascii=False) + '\n').encode('utf-8')
            offset = f.tell()
            f.write(line)
            offsets.append(offset)
    return offsets


def load_id_map() -> list:
    if os.path.exists(IDMAP_PATH):
        return json.load(open(IDMAP_PATH, 'r', encoding='utf-8'))
    return []


def save_id_map(m: list) -> None:
    json.dump(m, open(IDMAP_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)


def embed_texts(texts: List[str], batch_size: int = 128) -> np.ndarray:
    client = get_openai_client()
    all_vecs: list = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_vecs.extend(d.embedding for d in resp.data)
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
    offsets = save_jsonl(CHUNKS_PATH, rows)
    save_id_map(id_map)
    if os.path.exists(EMB_PATH):
        old = np.load(EMB_PATH)
        new = np.vstack([old, emb])
    else:
        new = emb
    np.save(EMB_PATH, new)
    off_arr = np.array(offsets, dtype='int64')
    if os.path.exists(CHUNK_OFFSETS_PATH):
        old_off = np.load(CHUNK_OFFSETS_PATH)
        new_off = np.concatenate([old_off, off_arr])
    else:
        new_off = off_arr
    np.save(CHUNK_OFFSETS_PATH, new_off)
    return {'added_chunks': len(chunks), 'meta': meta}


def ingest_pdf(path: str, meta: Dict) -> Dict:
    import fitz  # Lazy import heavy dep
    doc = fitz.open(path)
    pages = [page.get_text('text') for page in doc]
    text = '\n'.join(pages)
    meta_with_src = dict(meta)
    meta_with_src.update({'source_type': 'pdf', 'path': path})
    return chunk_and_index(text, meta_with_src)


def ingest_web(url: str, meta: Dict) -> Dict:
    import trafilatura  # Lazy import
    import requests
    from bs4 import BeautifulSoup

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise RuntimeError('Failed to fetch URL')
    text = trafilatura.extract(downloaded, include_comments=False, include_formatting=False)
    if not text or len(text) < 500:
        html = requests.get(url, timeout=30).text
        soup = BeautifulSoup(html, 'html.parser')
        text = clean_text(soup.get_text(' '))
    meta_with_src = dict(meta)
    meta_with_src.update({'source_type': 'web', 'url': url})
    return chunk_and_index(text, meta_with_src)


if __name__ == '__main__':
    print('Use the CLI: python -m app.cli --help')