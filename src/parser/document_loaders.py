"""
Document Loaders
Supports PDF, HTML, TXT, MD.
Returns plain text.
"""
import re
from pathlib import Path
from html.parser import HTMLParser

def normalize_text(text: str) -> str:
    text = text.replace("\t", " ")
    text = re.sub(r"[\u00a0\u2000-\u200b\ufeff]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_text(path: Path) -> str:
    content = path.read_text(encoding="utf-8", errors="replace")
    return normalize_text(content)

class _HTMLTextExtractor(HTMLParser):
    SKIP_TAGS = {"script", "style", "noscript", "head"}
    BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4",
                  "h5", "h6", "blockquote", "section", "article"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth > 0:
            return
        self.parts.append(data)

    def get_text(self) -> str:
        return "".join(self.parts)

def load_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    parser = _HTMLTextExtractor()
    parser.feed(raw)
    return normalize_text(parser.get_text())

def load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf required. pip install pypdf")
    
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        parts.append(normalize_text(page_text))
    return "\n\n".join(parts)

def load_document(path: str | Path) -> str:
    path = Path(path)
    ext = path.suffix.lower()
    
    if ext == ".pdf":
        return load_pdf(path)
    if ext in (".html", ".htm"):
        return load_html(path)
    if ext in (".txt", ".md", ".markdown", ".rst"):
        return load_text(path)
        
    raise ValueError(f"Unsupported document format: {ext}")
