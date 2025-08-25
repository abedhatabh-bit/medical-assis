import argparse
import json
import os
from typing import Dict

from app import ingest as ingest_mod
from app import generate as generate_mod


def cmd_ingest_pdf(args: argparse.Namespace) -> None:
    meta = {}
    if args.title:
        meta['title'] = args.title
    if args.year:
        meta['year'] = args.year
    if args.publisher:
        meta['publisher'] = args.publisher
    out = ingest_mod.ingest_pdf(args.path, meta)
    print(json.dumps(out, ensure_ascii=False))


def cmd_ingest_web(args: argparse.Namespace) -> None:
    meta = {}
    if args.title:
        meta['title'] = args.title
    if args.year:
        meta['year'] = args.year
    if args.publisher:
        meta['publisher'] = args.publisher
    out = ingest_mod.ingest_web(args.url, meta)
    print(json.dumps(out, ensure_ascii=False))


def cmd_generate(args: argparse.Namespace) -> None:
    out: Dict = generate_mod.make_lesson(args.topic, audience=args.audience)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Created: {args.output}")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='rag-plus', description='RAG pipeline CLI')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_pdf = sub.add_parser('ingest-pdf', help='Ingest a PDF file')
    p_pdf.add_argument('path', help='Path to PDF')
    p_pdf.add_argument('--title', help='Title metadata')
    p_pdf.add_argument('--year', type=int, help='Year metadata')
    p_pdf.add_argument('--publisher', help='Publisher metadata')
    p_pdf.set_defaults(func=cmd_ingest_pdf)

    p_web = sub.add_parser('ingest-web', help='Ingest a web URL')
    p_web.add_argument('url', help='URL to ingest')
    p_web.add_argument('--title', help='Title metadata')
    p_web.add_argument('--year', type=int, help='Year metadata')
    p_web.add_argument('--publisher', help='Publisher metadata')
    p_web.set_defaults(func=cmd_ingest_web)

    p_gen = sub.add_parser('generate', help='Generate lesson, flashcards, quiz')
    p_gen.add_argument('topic', help='Topic to query the RAG index with')
    p_gen.add_argument('--audience', default='3rd-year medical students')
    p_gen.add_argument('-o', '--output', help='Output JSON path (e.g., store/outputs/lesson.json)')
    p_gen.set_defaults(func=cmd_generate)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()

