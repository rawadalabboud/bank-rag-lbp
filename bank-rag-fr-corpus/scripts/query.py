#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
General-purpose French RAG chatbot over your Markdown corpus.

Env knobs (sensible defaults):
  EMBED_BACKEND=hf|openai              (default: hf)
  HF_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  EMBED_MODEL=text-embedding-3-small   (if EMBED_BACKEND=openai)

  LLM_BACKEND=hf|openai|ollama|none    (default: hf)
  HF_REPO_ID=Qwen/Qwen2.5-7B-Instruct  (any HF Instruct endpoint)
  HF_MAX_NEW_TOKENS=160  HF_TEMPERATURE=0.1  HF_TIMEOUT=60

  TOP_K=4                               (# retrieved docs)
  COMP_CHARS=900                        (context compression budget)
  VECTORSTORE=faiss|chroma              (default: faiss)
"""

import os, sys, re, time, textwrap
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS, Chroma
from langchain.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage

# ---------- Paths / config ----------
BASE = Path(__file__).resolve().parents[1]          # repo root
DATA_DIR = BASE / "data"
FAISS_PATH = str(BASE / "faiss_index")
CHROMA_PATH = str(BASE / "chroma")

EMBED_BACKEND = os.getenv("EMBED_BACKEND", "hf")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

LLM_BACKEND = os.getenv("LLM_BACKEND", "hf")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
HF_REPO_ID = os.getenv("HF_REPO_ID", "Qwen/Qwen2.5-7B-Instruct")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

TOP_K = int(os.getenv("TOP_K", "4"))
COMP_CHARS = int(os.getenv("COMP_CHARS", "900"))
VECTORSTORE = os.getenv("VECTORSTORE", "faiss").lower()

# ---------- Utils ----------
def safe_invoke(llm, messages, attempts: int = 4):
    """Retry a few times on transient rate limits/timeouts."""
    for i in range(attempts):
        try:
            return llm.invoke(messages)
        except Exception as e:
            msg = str(e)
            if any(t in msg.lower() for t in ("ratelimit", "too many requests", "timeout", "readtimeout")):
                time.sleep(2 * (i + 1))
                continue
            raise
    raise RuntimeError("LLM invoke failed after retries")

# ---------- Embeddings ----------
def get_embeddings():
    if EMBED_BACKEND.lower() == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=EMBED_MODEL)
    # HF default (prefer modern import, fall back if missing)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})

# ---------- Vector store ----------
def load_vs():
    emb = get_embeddings()
    if VECTORSTORE == "faiss":
        return FAISS.load_local(FAISS_PATH, emb, allow_dangerous_deserialization=True)
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=emb)

def retrieve(vs, query: str, k: int = TOP_K, oversample: int = 24) -> List[Document]:
    """Retrieve more, keep top-k, filter fr if metadata present."""
    if isinstance(vs, FAISS):
        docs = vs.similarity_search(query, k=oversample)
    else:
        docs = vs.as_retriever(search_kwargs={"k": oversample}).get_relevant_documents(query)
    # Prefer French docs if flagged
    fr = [d for d in docs if str(d.metadata.get("langue", "fr")).lower() == "fr"]
    return (fr or docs)[:k]

# ---------- Lightweight context compression ----------
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")

def compress_context(question: str, docs: List[Document], char_budget: int = COMP_CHARS) -> Tuple[str, List[Document]]:
    """Rank sentences by similarity to the query; pack up to char budget."""
    emb = get_embeddings()
    qv = emb.embed_query(question)

    cands: List[Tuple[float, str, Document]] = []
    for d in docs:
        for sent in _SENT_SPLIT.split(d.page_content):
            s = sent.strip()
            if 40 <= len(s) <= 350:
                sv = emb.embed_query(s)  # few hundred sentences total → fine
                score = sum(a * b for a, b in zip(qv, sv))  # vectors are normalized → dot≈cosine
                cands.append((score, s, d))

    cands.sort(key=lambda x: x[0], reverse=True)
    picked, used_docs, seen_keys = [], [], set()
    total = 0
    for _, s, d in cands:
        if total + len(s) + 1 > char_budget:
            continue
        picked.append((s, d))
        total += len(s) + 1
        key = d.metadata.get("source_url") or d.metadata.get("path", "")
        if key and key not in seen_keys:
            used_docs.append(d)
            seen_keys.add(key)
        if total >= char_budget:
            break

    if not picked:  # fallback: first 2 chunks raw
        text = "\n\n".join(d.page_content[:300] for d in docs[:2])
        return text, docs[:2]
    context = "\n".join(s for s, _ in picked)
    return context, used_docs

# ---------- Extractive fallback (no LLM) ----------
def extractive_answer(question: str, docs: List[Document]) -> Tuple[str, List[Document]]:
    emb = get_embeddings()
    qv = emb.embed_query(question)
    cands: List[Tuple[float, str, Document]] = []
    for d in docs:
        for sent in _SENT_SPLIT.split(d.page_content[:1200]):
            s = sent.strip()
            if 40 <= len(s) <= 350:
                sv = emb.embed_query(s)
                score = sum(a * b for a, b in zip(qv, sv))
                cands.append((score, s, d))
    if not cands:
        return "Information manquante dans les sources fournies.", []
    cands.sort(key=lambda x: x[0], reverse=True)
    top = cands[:4]
    answer = " ".join(s for _, s, _ in top)
    used, seen = [], set()
    for _, _, d in top:
        key = d.metadata.get("source_url") or d.metadata.get("path", "")
        if key and key not in seen:
            used.append(d); seen.add(key)
    return answer, used

# ---------- LLM ----------
def make_llm():
    backend = LLM_BACKEND.lower()
    if backend == "none":
        return None
    if backend == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except Exception:
            from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, temperature=0.2)
    if backend == "hf":
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        llm = HuggingFaceEndpoint(
            repo_id=HF_REPO_ID,
            task="text-generation",
            temperature=float(os.getenv("HF_TEMPERATURE", "0.1")),
            max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "160")),
            timeout=int(os.getenv("HF_TIMEOUT", "60")),
            do_sample=False,
            top_p=0.9,
            repetition_penalty=1.05,
            return_full_text=False,
        )
        return ChatHuggingFace(llm=llm)
    # OpenAI
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=CHAT_MODEL, temperature=0.2, max_retries=6, timeout=60)

# ---------- Prompting ----------
def build_messages(question: str, context: str) -> List:
    is_steps = question.strip().lower().startswith(("comment ", "comment?"))
    style = (
        "- Si la question commence par « comment », réponds par étapes numérotées (phrases courtes).\n"
        if is_steps else
        "- Réponds en 3–5 puces maximum, phrases courtes et concrètes.\n"
    )
    system = (
        "Tu es un assistant bancaire francophone spécialisé RAG.\n"
        "Règles:\n"
        "1) Tu utilises UNIQUEMENT le CONTEXTE fourni ci-dessous.\n"
        "2) Si l'information manque, écris exactement : « Information manquante dans les sources fournies. »\n"
        "3) Pas d'invention de montants, délais, liens ou procédures.\n"
        "4) Style : clair, opérationnel, sans jargon."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Contexte (extraits):\n{context}\n\n"
        "Consignes de réponse:\n"
        f"{style}"
        "- N'inclus pas d'URL dans le texte (les sources seront listées à part)."
    )
    return [SystemMessage(content=system), HumanMessage(content=user)]

def format_citations(docs: List[Document], max_items: int = 5) -> str:
    seen, out = set(), []
    for d in docs:
        title = d.metadata.get("title") or d.metadata.get("source_title") or "Source"
        url = d.metadata.get("source_url") or d.metadata.get("path", "")
        sec = d.metadata.get("section_title")
        key = (title, url, sec)
        if key in seen: 
            continue
        seen.add(key)
        line = f"- {title}"
        if sec: line += f" — {sec}"
        if url: line += f" — {url}"
        out.append(line)
        if len(out) >= max_items:
            break
    return "\n".join(out)

# ---------- Main ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/query.py \"ma question en français\"")
        sys.exit(1)

    question = sys.argv[1]
    vs = load_vs()
    docs = retrieve(vs, question, k=TOP_K, oversample=max(24, TOP_K * 6))

    # Build compact context + citations set
    context, cite_docs = compress_context(question, docs, char_budget=COMP_CHARS)
    llm = make_llm()

    if llm is None:
        answer, used = extractive_answer(question, docs)
        citations = format_citations(used or cite_docs or docs)
    else:
        try:
            resp = safe_invoke(llm, build_messages(question, context))
            answer = resp.content.strip()
            # if the model still dodges, backstop with extractive
            if not answer or "Information manquante" in answer and len(answer) < 80:
                ex, used = extractive_answer(question, docs)
                if ex:
                    answer = ex
                    cite_docs = used
            citations = format_citations(cite_docs or docs)
        except Exception:
            # Robust fallback
            answer, used = extractive_answer(question, docs)
            citations = format_citations(used or cite_docs or docs)

    print("\n=== Réponse ===\n")
    print(textwrap.fill(answer, width=100))
    print("\n=== Références ===\n")
    print(citations)

if __name__ == "__main__":
    main()
