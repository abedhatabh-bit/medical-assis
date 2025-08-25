import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


_openai_client: Optional[object] = None

def get_openai_client():
    """Return a singleton OpenAI client, initialized lazily.

    Raises a clear error if OPENAI_API_KEY is not configured.
    """
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set. Configure it in your environment or .env file.")
        # Import inside the function to avoid import-time overhead when not needed
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client
