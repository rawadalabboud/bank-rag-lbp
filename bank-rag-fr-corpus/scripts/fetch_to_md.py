#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import yaml
import hashlib  # typo? fix to correct 'hashlib'
from pathlib import Path
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# Try to import trafilatura; script works without it
try:
    import trafilatura  # type: ignore
except Exception:
    trafilatura = None

# ---------- Paths ----------
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
SRC = BASE / "scripts" / "sources_fr.yaml"

# ---------- HTTP ----------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Referer": "https://www.labanquepostale.fr/",
}


# ---------- Helpers ----------
def sanitize_filename(url: str) -> str:
    """Create a safe, unique filename from URL."""
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    parsed = urlparse(url)
    path = (parsed.netloc + parsed.path).strip("/").replace("/", "_") or "index"
    path = re.sub(r"[^A-Za-z0-9._-]+", "_", path)  # keep it filesystem-safe
    return f"{path[:120]}_{h}.md"


def fetch_html(url: str) -> str:
    """Fetch raw HTML (prefer trafilatura, fallback to requests)."""
    if trafilatura is not None:
        try:
            html = trafilatura.fetch_url(url, no_ssl=True)
            if html:
                return html
        except Exception:
            pass
    r = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
    r.raise_for_status()
    return r.text


def main_content_only(html: str) -> str:
    """Strip boilerplate and keep the main/article area when possible."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove obvious noise
    for sel in [
        "header",
        "footer",
        "nav",
        "aside",
        ".cookie",
        "#cookie",
        ".cookies",
        ".breadcrumb",
        ".breadcrumbs",
        ".share",
        ".social",
        ".newsletter",
        ".cta",
        ".btn",
        ".related",
        ".promo",
        ".banner",
        "script",
        "noscript",
        "style",
    ]:
        for node in soup.select(sel):
            node.decompose()

    # Keep the most relevant content region if available
    candidates = soup.select("main, article, #main, .main, .content, #content, .article")
    if candidates:
        html = "".join(str(c) for c in candidates)
    else:
        html = str(soup)
    return html


def to_markdown_from_html(html: str) -> str:
    """HTML → Markdown (prefer trafilatura extract; fallback to markdownify)."""
    if trafilatura is not None:
        try:
            md_text = trafilatura.extract(html, include_links=True, output="markdown")
            if md_text and len(md_text.split()) >= 50:
                return md_text
        except Exception:
            pass
    return md(html)


def extract_title_from_html(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    # Fallback to URL path as title
    parsed = urlparse(url)
    base = Path(parsed.path).name or parsed.netloc
    base = re.sub(r"[-_]+", " ", base).strip() or parsed.netloc
    return base.title()


def title_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name or parsed.netloc
    name = re.sub(r"\.pdf$", "", name, flags=re.I)
    name = re.sub(r"[-_]+", " ", name).strip()
    return name.title() or "Document PDF"


def write_md(url: str, title: str, md_text: str) -> None:
    """Write Markdown with YAML front-matter."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fname = sanitize_filename(url)
    out_path = DATA_DIR / fname

    meta = {
        "source_url": url,
        "source_title": str(title),
        "date_acces": time.strftime("%Y-%m-%d"),
        "langue": "fr",
    }

    front = "---\n" + yaml.safe_dump(meta, allow_unicode=True) + "---\n\n" + f"# {title}\n\n"
    out_path.write_text(front + md_text, encoding="utf-8")
    print(f"[OK] {out_path}")


# ---------- Handlers ----------
def handle_html(url: str) -> None:
    try:
        raw = fetch_html(url)
    except Exception as e:
        print(f"[ERR] HTML fetch failed: {url} — {e}")
        return

    if not raw or len(raw) < 200:
        print(f"[SKIP] Aucun HTML : {url}")
        return

    cleaned = main_content_only(raw)
    title = extract_title_from_html(raw, url)
    md_text = to_markdown_from_html(cleaned)
    md_text = re.sub(r"\n{3,}", "\n\n", md_text).strip()
    if not md_text or len(md_text.split()) < 50:
        print(f"[SKIP] Contenu trop court : {url}")
        return
    write_md(url, title, md_text)


def handle_pdf(url: str) -> None:
    """Download PDF and extract text; try unstructured(fast) -> fall back to PyMuPDF -> pdfminer."""
    import tempfile
    import fitz  # PyMuPDF
    from pdfminer.high_level import extract_text as pdfminer_extract
    try:
        from unstructured.partition.pdf import partition_pdf
    except Exception:
        partition_pdf = None

    # Download
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERR] PDF download failed {url}: {e}")
        return

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(r.content); tmp.flush()

        text = ""
        # 1) Try unstructured fast (no poppler)
        if partition_pdf is not None:
            try:
                els = partition_pdf(filename=tmp.name, strategy="fast", infer_table_structure=False)
                text = "\n\n".join(el.text for el in els if getattr(el, "text", "").strip())
            except Exception as e:
                print(f"[WARN] unstructured fast failed: {e}")

        # 2) Fallback: PyMuPDF (often best on brochure-style PDFs)
        if len(text.split()) < 100:
            try:
                doc = fitz.open(tmp.name)
                pages = []
                for page in doc:
                    pages.append(page.get_text("text"))  # simple text; try "blocks" if needed
                text = "\n\n".join(pages)
            except Exception as e:
                print(f"[WARN] PyMuPDF extract failed: {e}")

        # 3) Last resort: pdfminer
        if len(text.split()) < 100:
            try:
                text = pdfminer_extract(tmp.name) or ""
            except Exception as e:
                print(f"[WARN] pdfminer extract failed: {e}")

    text = re.sub(r"\n{3,}", "\n\n", text or "").strip()
    if not text or len(text.split()) < 50:
        print(f"[SKIP] PDF texte trop court : {url}")
        return

    title = title_from_url(url) or "Tarifs La Banque Postale 2025"
    write_md(url, title, text)


# ---------- Main ----------
def main():
    if not SRC.exists():
        print(f"[ERR] Fichier des sources introuvable: {SRC}")
        return

    with open(SRC, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    urls = data.get("urls", [])
    if not urls:
        print("[ERR] Aucune URL dans sources_fr.yaml")
        return

    for url in urls:
        try:
            if url.lower().endswith(".pdf"):
                handle_pdf(url)
            else:
                handle_html(url)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[ERR] échec sur {url}: {e}")

if __name__ == "__main__":
    main()
