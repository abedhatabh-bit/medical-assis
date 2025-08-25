import os, json, uuid
import numpy as np
from typing import List, Dict
from app.embedding import embed_texts
from app.storage import STORE_DIR, CHUNKS_PATH, EMB_PATH, IDMAP_PATH, save_jsonl, load_id_map, save_id_map
import requests
from bs4 import BeautifulSoup

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
    return chunk_and_index(text, meta | {'source_type': 'pdf', 'path': path})

def ingest_web(url: str, meta: Dict) -> Dict:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    text = clean_text(soup.get_text(' '))
    return chunk_and_index(text, meta | {'source_type': 'web', 'url': url})

if __name__ == '__main__':
    pdf_sources = [
        {'path': r'C:\\Users\\Admin\\Downloads\\First Aid for the USMLE Step 1 2025 35th Edition.pdf',
            'title': 'First Aid Step 1', 'year': 2025, 'publisher': 'McGraw-Hill'}
    ]
    for src in pdf_sources:
        print(ingest_pdf(src['path'], {'title': src['title'], 'year': src['year'], 'publisher': src['publisher']}))