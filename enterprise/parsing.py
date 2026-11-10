"""Structure-aware ingestion. No network access or execution of document content."""
import csv
import io
import re
from pathlib import Path

SUPPORTED = {'.txt', '.md', '.csv', '.pdf', '.docx'}
MAX_BYTES = 12 * 1024 * 1024


def parse_file(filename: str, data: bytes) -> list[dict]:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError('Supported files: PDF, DOCX, TXT, Markdown and CSV.')
    if not data or len(data) > MAX_BYTES:
        raise ValueError('File must contain data and be at most 12 MiB.')
    if ext == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ValueError('Encrypted PDFs are unsupported. Upload an unlocked copy.')
        sections = [{'text': p.extract_text() or '', 'page': i + 1} for i, p in enumerate(reader.pages)]
    elif ext == '.docx':
        from docx import Document
        import zipfile
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if sum(i.file_size for i in z.infolist()) > 40 * 1024 * 1024:
                raise ValueError('Expanded DOCX exceeds the 40 MiB limit.')
        doc = Document(io.BytesIO(data))
        # iter_inner_content retains paragraph/table order.
        lines = []
        for item in doc.iter_inner_content():
            if hasattr(item, 'rows'):
                lines.extend(' | '.join(c.text for c in row.cells) for row in item.rows)
            else:
                prefix = '# ' if item.style and item.style.name.startswith('Heading') else ''
                lines.append(prefix + item.text)
        sections = [{'text': '\n\n'.join(lines), 'page': None}]
    else:
        try:
            text = data.decode('utf-8-sig')
        except UnicodeDecodeError as exc:
            raise ValueError('Text files must be UTF-8 encoded.') from exc
        if ext == '.csv':
            rows = list(csv.reader(io.StringIO(text)))
            header = rows[0] if rows else []
            text = '\n\n'.join(' | '.join(f'{k}: {v}' for k, v in zip(header, row)) for row in rows[1:])
        sections = [{'text': text, 'page': None}]
    if not any(s['text'].strip() for s in sections):
        raise ValueError('No readable text. Scanned PDFs need OCR before upload.')
    return sections


def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')[:100] or 'general'


def dynamic_chunks(sections: list[dict], max_words: int = 180) -> list[dict]:
    """Keep clauses/sections intact; pack related paragraphs and split oversized units.

    Each chunk retains parent section, clause label, page and part number. Oversized
    clauses share one rule key so resolution operates on complete parent groups.
    """
    if max_words < 20:
        raise ValueError('max_words must be at least 20')
    result = []
    heading = 'General'
    for section in sections:
        text = section['text'].replace('\r\n', '\n')
        if re.search(r'\[C\d+\]', text):
            # The supplied controlled corpus embeds its metadata in a preamble.
            # Metadata is already stored structurally; do not index it as a rule.
            text = re.sub(r'(?mi)^(?:Document ID|Version|Department|Publication Date|Effective Date|Authority|Scope):[^\n]*\n?', '', text)
        # Preserve explicit [C1] boundaries even when several occur on one line.
        text = re.sub(r'\s*(\[C\d+\])', r'\n\n\1', text)
        blocks = re.split(r'\n\s*\n', text)
        pending = []
        label = None

        def flush():
            nonlocal pending
            if not pending:
                return
            body = '\n\n'.join(pending).strip()
            words = body.split()
            pieces = []
            if len(words) <= max_words:
                pieces = [body]
            else:
                # Sentence boundaries first; only very long sentences split by words.
                buf = []
                for sentence in re.split(r'(?<=[.!?])\s+', body):
                    sw = sentence.split()
                    if buf and len(buf) + len(sw) > max_words:
                        pieces.append(' '.join(buf)); buf = []
                    while len(sw) > max_words:
                        pieces.append(' '.join(sw[:max_words])); sw = sw[max_words:]
                    buf.extend(sw)
                if buf:
                    pieces.append(' '.join(buf))
            for part, piece in enumerate(pieces, 1):
                result.append({'text': piece, 'section': heading, 'clause': label,
                               'page': section.get('page'), 'part': part,
                               'rule_key': label or slug(heading)})
            pending = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue
            # Heading followed by content in the same block.
            hm = re.match(r'^#{1,6}\s+([^\n]+)(?:\n(.*))?$', block, re.S)
            cm = re.match(r'^\[(C\d+)\]\s*(.*)$', block, re.S)
            if hm:
                flush(); heading = hm[1].strip(); label = None
                if hm[2]: pending.append(hm[2])
            elif cm:
                flush(); label = cm[1]; pending.append(cm[2])
            else:
                if pending and sum(len(p.split()) for p in pending) + len(block.split()) > max_words:
                    flush()
                pending.append(block)
        flush()
    for i, chunk in enumerate(result, 1):
        chunk['ordinal'] = i
    return result
