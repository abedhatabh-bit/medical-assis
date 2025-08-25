from typing import List
import numpy as np
from app.clients import get_openai_client
from app.config import EMBED_MODEL


def normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / norms


def embed_texts(texts: List[str], batch_size: int = 128) -> np.ndarray:
    client = get_openai_client()
    all_vecs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_vecs.extend([d.embedding for d in resp.data])
    X = np.array(all_vecs, dtype='float32')
    return normalize_rows(X)


def embed_query(text: str) -> np.ndarray:
    client = get_openai_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    v = np.array(resp.data[0].embedding, dtype='float32')
    return v / (np.linalg.norm(v) + 1e-8)