#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import List, Dict
import yaml

from langchain.docstore.document import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import FAISS

# Resolve paths relative to .../bank-rag-fr-corpus/scripts/ingest.py
BASE = Path(__file__).resolve().parents[1]            # → bank-rag-fr-corpus/
DATA_DIR = BASE / "data"
FAISS_PATH = BASE / "faiss_index"

# Embedding config (HF by default; can switch to OpenAI via env)
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "hf")      # "hf" | "openai"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # used if openai
HF_EMBED_MODEL = os.getenv(
    "HF_EMBED_MODEL",
    "BAAI/bge-m3"  # good default; requires trust_remote_code=True
)

def get_embeddings():
    """Return an embeddings object: HuggingFace (local) by default, or OpenAI if requested."""
    if EMBED_BACKEND.lower() == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=EMBED_MODEL)
    else:
        # Modern import (no deprecation warnings)
        from langchain_huggingface import HuggingFaceEmbeddings

        # trust_remote_code is needed for some newer HF models (e.g., nvidia/NV-Embed-v2)
        # normalize_embeddings=True → cosine-like behavior with FAISS
        return HuggingFaceEmbeddings(
            model_name=HF_EMBED_MODEL,
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )

def load_markdown_docs(data_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for p in data_dir.rglob("*.md"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        meta: Dict = {}
        body = text

        # Parse YAML front matter if present
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    meta = yaml.safe_load(parts[1]) or {}
                except Exception:
                    meta = {}
                body = parts[2]

        # keep only French (default to FR if no front matter)
        if str(meta.get("langue", "fr")).lower() != "fr":
            continue

        meta["path"] = str(p)
        meta["title"] = meta.get("source_title") or p.stem
        meta["source_url"] = meta.get("source_url") or ""

        docs.append(Document(page_content=body.strip(), metadata=meta))
    return docs

def split_docs(docs: List[Document]) -> List[Document]:
    headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    recur = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    out: List[Document] = []
    for d in docs:
        sections = header_splitter.split_text(d.page_content) or [
            Document(page_content=d.page_content, metadata=d.metadata)
        ]
        for s in sections:
            md = d.metadata.copy()
            md.update(s.metadata or {})
            sec = [md.get(k) for k in ("h1", "h2", "h3") if md.get(k)]
            if sec:
                md["section_title"] = " > ".join(sec)
            out.extend(
                recur.split_documents(
                    [Document(page_content=s.page_content, metadata=md)]
                )
            )
    return out

def main():
    print("Chargement des fichiers Markdown…")
    raw_docs = load_markdown_docs(DATA_DIR)
    print(f"- Fichiers FR chargés : {len(raw_docs)}")

    print("Découpage en chunks…")
    chunks = split_docs(raw_docs)
    print(f"- Chunks totaux : {len(chunks)}")

    backend_label = (
        f"OpenAI:{EMBED_MODEL}" if EMBED_BACKEND.lower() == "openai"
        else f"HF:{HF_EMBED_MODEL}"
    )
    print(f"Création des embeddings ({backend_label})…")
    embeddings = get_embeddings()

    print("Indexation FAISS…")
    # Prefer cosine distance if available in this LangChain version
    try:
        from langchain_community.vectorstores.faiss import DistanceStrategy
        index = FAISS.from_documents(
            chunks, embeddings, distance_strategy=DistanceStrategy.COSINE
        )
    except Exception:
        index = FAISS.from_documents(chunks, embeddings)

    FAISS_PATH.mkdir(parents=True, exist_ok=True)
    index.save_local(str(FAISS_PATH))
    print(f"- FAISS → {FAISS_PATH}")

    # petite stat
    by_src: Dict[str, int] = {}
    for c in chunks:
        k = c.metadata.get("source_url") or c.metadata.get("path")
        by_src[k] = by_src.get(k, 0) + 1

    print("\nTop sources par #chunks :")
    for k, v in sorted(by_src.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"- {v:4d}  {k}")

if __name__ == "__main__":
    main()
