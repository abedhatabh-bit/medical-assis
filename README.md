# Medical Assistant CLI

A powerful CLI tool for ingesting medical documents and generating educational content using AI.

## Features

- **PDF Ingestion**: Extract and process text from medical PDF documents
- **Web Scraping**: Ingest content from medical websites and articles
- **AI-Powered Content Generation**: Create lessons, flashcards, and quizzes
- **Efficient Storage**: Uses memory-mapped arrays for fast retrieval
- **Semantic Search**: Find relevant content using embeddings

## Quick Setup

1. **Create virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

4. **Verify installation**:
   ```bash
   python -m app --help
   ```

## Usage

### Ingest PDF Documents
```bash
python -m app ingest-pdf \
  --path "/path/to/medical-document.pdf" \
  --title "First Aid for USMLE Step 1" \
  --year 2024 \
  --publisher "McGraw-Hill"
```

### Ingest Web Content
```bash
python -m app ingest-web \
  --url "https://example.com/medical-article" \
  --title "Diabetes Management Guidelines" \
  --year 2024 \
  --publisher "Medical Journal"
```

### Generate Educational Content
```bash
python -m app generate \
  --topic "Hypertension management" \
  --audience "3rd-year medical students" \
  --out store/outputs/hypertension-lesson.json
```

## Configuration

### Environment Variables (.env)
```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-small
```

### Storage Structure
- `store/chunks.jsonl` - Text chunks with metadata
- `store/embeddings.npy` - Vector embeddings (memory-mapped)
- `store/id_map.json` - Chunk ID mapping
- `store/outputs/` - Generated lessons and content

## Requirements

- Python 3.8+
- OpenAI API key
- 500MB+ free disk space for embeddings
- Internet connection for web ingestion

## Troubleshooting

### Common Issues

1. **OpenAI API Error**: Ensure your API key is valid and has credits
2. **PDF Processing Error**: Verify the PDF file exists and is readable
3. **Web Scraping Blocked**: Some sites block automated requests
4. **Memory Issues**: Large documents may require more RAM for processing

### Performance Tips

- Process documents in batches for large collections
- Use SSD storage for better embedding retrieval speed
- Monitor API usage to avoid rate limits

## Notes

- Embeddings use cosine similarity for semantic search
- Text chunking preserves sentence boundaries when possible
- Generated content is for educational purposes only
- Always verify medical information from authoritative sources