from typing import Optional
from openai import OpenAI
from app.config import OPENAI_API_KEY

_client: Optional[OpenAI] = None

def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set in environment")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client