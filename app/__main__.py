import argparse
import os
from app.ingest import ingest_pdf, ingest_web
from app.generate import make_lesson

def main():
    parser = argparse.ArgumentParser(prog='medical-assistant')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_pdf = sub.add_parser('ingest-pdf', help='Ingest a PDF into the store')
    p_pdf.add_argument('--path', required=True)
    p_pdf.add_argument('--title', required=True)
    p_pdf.add_argument('--year', type=int, default=0)
    p_pdf.add_argument('--publisher', default='')

    p_web = sub.add_parser('ingest-web', help='Ingest a web page into the store')
    p_web.add_argument('--url', required=True)
    p_web.add_argument('--title', required=True)
    p_web.add_argument('--year', type=int, default=0)
    p_web.add_argument('--publisher', default='')

    p_gen = sub.add_parser('generate', help='Generate lesson JSON from store')
    p_gen.add_argument('--topic', required=True)
    p_gen.add_argument('--audience', default='3rd-year medical students')
    p_gen.add_argument('--out', default='store/outputs/lesson.json')

    args = parser.parse_args()
    if args.cmd == 'ingest-pdf':
        meta = {'title': args.title, 'year': args.year, 'publisher': args.publisher}
        print(ingest_pdf(args.path, meta))
    elif args.cmd == 'ingest-web':
        meta = {'title': args.title, 'year': args.year, 'publisher': args.publisher}
        print(ingest_web(args.url, meta))
    elif args.cmd == 'generate':
        data = make_lesson(args.topic, audience=args.audience)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        import json
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f'Wrote {args.out}')

if __name__ == '__main__':
    main()

