### Medical Assistant CLI

Setup

- python3 -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt
- cp .env.example .env && edit OPENAI_API_KEY

CLI usage

- python -m app ingest-pdf --path "/path/to/file.pdf" --title "Title" --year 2024 --publisher "Org"
- python -m app ingest-web --url "https://example.com" --title "Example" --year 2024 --publisher "Org"
- python -m app generate --topic "Hypertension" --audience "3rd-year medical students" --out store/outputs/lesson.json

Notes

- Embeddings are stored under `store/` as `chunks.jsonl`, `embeddings.npy`, and `id_map.json`.
- Retrieval uses memory-mapped arrays and partial sorting for speed.