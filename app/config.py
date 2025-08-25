import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "false").lower() in {"1","true","yes","on"}
LOCAL_EMBED_DIM = int(os.getenv("LOCAL_EMBED_DIM", "384"))
