from typing import List, Dict
from openai import OpenAI
from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.rag import retrieve
import json, os

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

EXPLANATION_PROMPT = '''You are a medical education expert. Write a concise lesson in English for the specified audience about the requested topic. Use only the quoted passages below. Do not add information from outside them. Return a JSON object with keys: lesson_title, learning_objectives[], key_points[], explanation_sections[{title, content, citations[]}], safety_notes[], glossary[{term, ar, en}], references[{title, year, url_or_doi}] Source excerpts: {context} Topic: {topic} Safety note: This content is for education only and is not medical advice.'''

FLASHCARDS_PROMPT = '''You are an Anki flashcard creator. Extract up to 12 concise English flashcards (definition, etiology, diagnosis, treatment, complications). Return JSON: cards[{type, question, answer, cloze_optional, source_citation_id, tags[]}]. Use only the provided excerpts. Avoid repetition.'''

QUIZ_PROMPT = '''You are a medical examiner. Create 6 single-best-answer MCQs in English. Return JSON: questions[{stem, options[4-5], correct_index, rationale_per_option[], difficulty(1-5), objective_ref, citations[]}]. Base all facts only on the provided excerpts.'''

def build_context(chunks: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(chunks):
        meta = c.get('meta', {})
        src = meta.get('path') or meta.get('title', '')
        lines.append(f"[{i}] {c['text']}\n(source: {src}, publisher: {meta.get('publisher','')}, year: {meta.get('year','')})")
    return '\n\n'.join(lines)

def generate_json(prompt: str) -> Dict:
    client = get_client()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={'type': 'json_object'},
        messages=[{'role': 'user', 'content': prompt}]
    )
    return json.loads(resp.choices[0].message.content)

def make_lesson(topic: str, audience: str = '3rd-year medical students') -> Dict:
    chunks = retrieve(topic, k=10)
    if not chunks: raise RuntimeError('No chunks found. Ingest sources first.')
    context = build_context(chunks)
    explanation = generate_json(EXPLANATION_PROMPT.format(context=context, topic=topic, audience=audience))
    flashcards = generate_json(FLASHCARDS_PROMPT + '\n\n' + context)
    quiz = generate_json(QUIZ_PROMPT + '\n\n' + context)
    return {'explanation': explanation, 'flashcards': flashcards, 'quiz': quiz, 'used_chunks': chunks}

if __name__ == '__main__':
    # Example usage
    # topic = 'Medical Topic Example'
    # out = make_lesson(topic)
    # os.makedirs('store/outputs', exist_ok=True)
    # with open('store/outputs/lesson_example.json', 'w', encoding='utf-8') as f:
    #     json.dump(out, f, ensure_ascii=False, indent=2)
    # print('Created: store/outputs/lesson_example.json')
    print("Generate module loaded. Use CLI commands to generate lessons.")
