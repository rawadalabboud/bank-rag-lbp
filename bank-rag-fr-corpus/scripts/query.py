#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
French RAG chatbot over your Markdown corpus (FAISS + BGE-M3 + OSS LLM).

Env knobs (sane defaults):
  # Embeddings (must match your ingest!)
  HF_EMBED_MODEL=BAAI/bge-m3
  EMBED_DEVICE=mps|cpu          (default: auto -> 'mps' if available else 'cpu')

  # LLM backends
  LLM_BACKEND=hf|ollama|none    (default: hf)
  HF_REPO_ID=meta-llama/Meta-Llama-3.1-70B-Instruct  (good OSS default)
  HF_MAX_NEW_TOKENS=180
  HF_TEMPERATURE=0.1
  HF_TIMEOUT=60

  OLLAMA_MODEL=mistral
  OLLAMA_TIMEOUT=60

  # Retrieval
  TOP_K=4
  OVERSAMPLE=24
  COMP_CHARS=900                 (context compression budget)

  # Optional re-ranking (cross-encoder)
  RERANK=true|false              (default: false)
  RERANK_MODEL=BAAI/bge-reranker-v2-m3
  RERANK_TOP_K=4                 (final K after re-ranking)
"""

import os, sys, re, time, textwrap
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage

# ---------- Paths / config ----------
BASE = Path(__file__).resolve().parents[1]          # repo root
FAISS_PATH = str(BASE / "faiss_index")

# Embeddings
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "BAAI/bge-m3")
EMBED_DEVICE = os.getenv("EMBED_DEVICE")  # if None -> auto

# LLM
LLM_BACKEND = os.getenv("LLM_BACKEND", "hf").lower()
HF_REPO_ID = os.getenv("HF_REPO_ID", "meta-llama/Meta-Llama-3.1-70B-Instruct")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "180"))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.1"))
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT", "60"))

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "4"))
OVERSAMPLE = int(os.getenv("OVERSAMPLE", "24"))
COMP_CHARS = int(os.getenv("COMP_CHARS", "900"))

# Re-ranking
RERANK = os.getenv("RERANK", "false").lower() in {"1", "true", "yes", "y"}
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", str(TOP_K)))

# ---------- Utils ----------
def safe_invoke(llm, messages, attempts: int = 4):
    """Retry a few times on transient rate limits/timeouts."""
    for i in range(attempts):
        try:
            return llm.invoke(messages)
        except Exception as e:
            msg = str(e).lower()
            if any(t in msg for t in ("ratelimit", "too many requests", "timeout", "readtimeout")):
                time.sleep(2 * (i + 1))
                continue
            raise
    raise RuntimeError("LLM invoke failed after retries")

def _auto_device() -> str:
    # Prefer Apple Silicon GPU if torch.mps is available
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
            return "mps"
    except Exception:
        pass
    return "cpu"

# ---------- Embeddings ----------
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    device = EMBED_DEVICE or _auto_device()
    return HuggingFaceEmbeddings(
        model_name=HF_EMBED_MODEL,
        # bge-m3 does not require trust_remote_code; leaving it False keeps load clean
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

# ---------- Vector store ----------
def load_vs():
    emb = get_embeddings()
    # allow_dangerous_deserialization=True is standard for FAISS in LangChain; only load your own index
    return FAISS.load_local(FAISS_PATH, emb, allow_dangerous_deserialization=True)

def retrieve(vs, query: str, k: int = TOP_K, oversample: int = OVERSAMPLE) -> List[Document]:
    """Retrieve more, prefer French, then trim to k."""
    docs = vs.similarity_search(query, k=oversample)
    fr = [d for d in docs if str(d.metadata.get("langue", "fr")).lower() == "fr"]
    return (fr or docs)[:k]

# ---------- Optional cross-encoder re-ranking ----------
def rerank_docs(query: str, docs: List[Document], top_k: int = RERANK_TOP_K) -> List[Document]:
    """Re-rank with a cross-encoder (BGE reranker)."""
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(RERANK_MODEL, device=_auto_device())
        pairs = [(query, d.page_content[:1200]) for d in docs]
        scores = model.predict(pairs, convert_to_numpy=True)
        ranked = sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)
        return [d for _, d in ranked[:top_k]]
    except Exception as e:
        print(f"[WARN] Re-ranking disabled (load error: {e}).")
        return docs[:top_k]

# ---------- Lightweight context compression ----------
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")

@lru_cache(maxsize=8192)
def _embed_for_cache(text: str) -> Tuple[float, ...]:
    """Tiny cache to avoid re-embedding the same sentences across queries."""
    emb = get_embeddings()
    v = emb.embed_query(text)
    # convert list[float] to tuple for cacheability
    return tuple(v)

def compress_context(question: str, docs: List[Document], char_budget: int = COMP_CHARS) -> Tuple[str, List[Document]]:
    """Rank sentences by cosine similarity to the query; pack up to char budget."""
    qv = _embed_for_cache("__Q__" + question)

    cands: List[Tuple[float, str, Document]] = []
    for d in docs:
        for sent in _SENT_SPLIT.split(d.page_content):
            s = sent.strip()
            if 40 <= len(s) <= 350:
                sv = _embed_for_cache(s)
                # vectors are normalized → dot ≈ cosine
                score = sum(a * b for a, b in zip(qv, sv))
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
    qv = _embed_for_cache("__Q__" + question)
    cands: List[Tuple[float, str, Document]] = []
    for d in docs:
        for sent in _SENT_SPLIT.split(d.page_content[:1200]):
            s = sent.strip()
            if 40 <= len(s) <= 350:
                sv = _embed_for_cache(s)
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
    if LLM_BACKEND == "none":
        return None
    if LLM_BACKEND == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except Exception:
            from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, temperature=0.2, num_ctx=8192, timeout=OLLAMA_TIMEOUT)

    # Hugging Face Inference Endpoint (OSS models, hosted)
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    llm = HuggingFaceEndpoint(
        repo_id=HF_REPO_ID,
        task="text-generation",
        temperature=HF_TEMPERATURE,
        max_new_tokens=HF_MAX_NEW_TOKENS,
        timeout=HF_TIMEOUT,
        do_sample=False,
        top_p=0.9,
        repetition_penalty=1.05,
        return_full_text=False,
    )
    return ChatHuggingFace(llm=llm)

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

    # retrieve
    docs = vs.similarity_search(question, k=max(OVERSAMPLE, TOP_K * 6))
    fr = [d for d in docs if str(d.metadata.get("langue", "fr")).lower() == "fr"]
    docs = (fr or docs)

    # optional re-ranking
    if RERANK:
        docs = rerank_docs(question, docs, top_k=RERANK_TOP_K)
    else:
        docs = docs[:TOP_K]

    # build compact context + citations
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
            if not answer or ("Information manquante" in answer and len(answer) < 80):
                ex, used = extractive_answer(question, docs)
                if ex:
                    answer = ex
                    cite_docs = used
            citations = format_citations(cite_docs or docs)
        except Exception as e:
            print(f"[WARN] LLM error, falling back extractive: {e}")
            answer, used = extractive_answer(question, docs)
            citations = format_citations(used or cite_docs or docs)

    print("\n=== Réponse ===\n")
    print(textwrap.fill(answer, width=100))
    print("\n=== Références ===\n")
    print(citations)

if __name__ == "__main__":
    main()
