#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end RAG evaluation (retrieval + compression + generation) over eval_bank_fr.yaml.

YAML schema (top-level key 'questions' or a plain list):
questions:
  - id: string
    question: string
    variants: [string, ...]              # optional
    expected_sources: [string, ...]      # URL fragments or full URLs/paths
    expected_regex: [regex, ...]         # optional, ALL must match the answer
    must_contain_any: [string, ...]      # optional, at least ONE must be present in the answer
    unanswerable: true|false             # optional. If true, a refusal is expected.
"""

import os, sys, re, json, csv, yaml, textwrap
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

# ---------- Paths ----------
BASE = Path(__file__).resolve().parents[1]
INDEX_DIR = BASE / "faiss_index"
EVAL_FILE = BASE / "scripts" / "eval_bank_fr.yaml"

# ---------- Embeddings / Device (must match your FAISS index) ----------
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "BAAI/bge-m3")
DEVICE = os.getenv("EMBED_DEVICE", "mps")  # "cpu" | "cuda" | "mps"

# ---------- Retrieval / Compression ----------
TOP_K = int(os.getenv("TOP_K", "5"))
OVERSAMPLE = int(os.getenv("OVERSAMPLE", "50"))
COMP_CHARS = int(os.getenv("COMP_CHARS", "1100"))

# ---------- Selective authority nudging (only for amount/tariff queries) ----------
AMOUNT_BOOST = os.getenv("AMOUNT_BOOST", "on").lower() in ("1", "true", "on", "yes")
BOOST_WEIGHT = float(os.getenv("BOOST_WEIGHT", "0.05"))
_AMOUNT_Q = re.compile(r"\b(frais|tarif|tarifs|cotisation|co[uû]t|prix|montant|commission|agios)\b", re.I)

# ---------- LLM Backend ----------
LLM_BACKEND = os.getenv("LLM_BACKEND", "hf").lower()  # "hf" | "openai" | "ollama" | "none"
HF_REPO_ID = os.getenv("HF_REPO_ID", "meta-llama/Meta-Llama-3.1-70B-Instruct")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "180"))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.1"))
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT", "60"))

# ---------- Grounding regex ----------
_EURO_RE  = re.compile(r"\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?\s*€")
_PHONE_RE = re.compile(r"(?:\+?\d[\s.-]?){6,}")

# ---------- Refusal matching (flexible) ----------
# Accept many natural refusal phrasings in French.
REFUSAL_PATTERNS = [
    r"\binformation manquante\s+dans\s+les\s+sources\s+fournies\b",
    r"\bje\s+n[’']ai\s+pas\s+trouv[ée]?\s+l['’]information\s+dans\s+les\s+sources\s+fournies\b",
    r"\baucune?\s+information\s+disponible\s+dans\s+les\s+sources\b",
    r"\binformation\s+indisponible\s+dans\s+les\s+sources\b",
    r"\bje\s+ne\s+dispose\s+pas\s+des?\s+informations?\s+nécessaires?\b",
    r"\binformation\s+non\s+disponible\s+dans\s+mes?\s+sources?\b",
    r"\bhors\s+p[ée]rim[èe]tre\s+des?\s+sources\b",
    r"\bje\s+ne\s+peux\s+pas\s+r[ée]pondre\s+avec\s+les?\s+sources?\s+fournies\b",
]
# Optional: add your own runtime regex (comma-separated) via env var
EXTRA_REFUSALS = os.getenv("REFUSAL_PATTERNS_EXTRA", "")
if EXTRA_REFUSALS.strip():
    for _pat in EXTRA_REFUSALS.split(","):
        _pat = _pat.strip()
        if _pat:
            REFUSAL_PATTERNS.append(_pat)

# Debug: log first N refusal mismatches
DEBUG_REFUSALS = os.getenv("DEBUG_REFUSALS", "off").lower() in ("1","true","on","yes")
DEFAULT_MAX_REFUSAL_LOGS = int(os.getenv("MAX_REFUSAL_LOGS", "5"))

FACT_EM_MODE = os.getenv("FACT_EM_MODE", "off").lower()  # off | report | strict

# ---------- Helpers ----------
def load_vs_and_emb():
    emb = HuggingFaceEmbeddings(
        model_name=HF_EMBED_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)
    return vs, emb

def _doc_id(d) -> str:
    return (d.metadata.get("source_url") or d.metadata.get("path") or "").lower()

def retrieve(vs, q: str, k: int = TOP_K, oversample: int = OVERSAMPLE):
    """
    Retrieve with FAISS similarity scores; optionally blend a tiny 'authority' boost
    ONLY for amount/tarif-style queries (frais, tarifs, cotisation, prix...).
    """
    pairs = vs.similarity_search_with_score(q, k=max(oversample, k))  # (Document, distance)
    sims = [(d, -float(dist)) for d, dist in pairs]  # distance -> similarity (higher better)

    if AMOUNT_BOOST and _AMOUNT_Q.search(q):
        def boost(doc):
            url = (doc.metadata.get("source_url") or doc.metadata.get("path") or "").lower()
            s = 0
            if url.endswith(".pdf"): s += 3
            if any(x in url for x in ("tarif", "tarifs", "conditions", "conventions")): s += 2
            if str(doc.metadata.get("langue", "fr")).lower() == "fr": s += 1
            return s
        sims = [(d, sim + BOOST_WEIGHT * boost(d)) for d, sim in sims]

    sims.sort(key=lambda x: x[1], reverse=True)
    docs_all = [d for d, _ in sims]
    return docs_all[:k], docs_all

def compress_context(question: str, docs, emb, char_budget: int = COMP_CHARS) -> Tuple[str, List]:
    """Rank sentences by similarity to the query; pack up to char budget."""
    SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")
    qv = emb.embed_query(question)
    cands = []
    for d in docs:
        for s in SENT_SPLIT.split(d.page_content):
            s = s.strip()
            if 40 <= len(s) <= 350:
                sv = emb.embed_query(s)
                score = sum(a * b for a, b in zip(qv, sv))
                cands.append((score, s, d))
    cands.sort(key=lambda x: x[0], reverse=True)
    picked, used, seen, total = [], [], set(), 0
    for _, s, d in cands:
        if total + len(s) + 1 > char_budget:
            continue
        picked.append((s, d)); total += len(s) + 1
        key = d.metadata.get("source_url") or d.metadata.get("path", "")
        if key and key not in seen:
            used.append(d); seen.add(key)
        if total >= char_budget:
            break
    if not picked:
        text = "\n\n".join(d.page_content[:300] for d in docs[:2])
        return text, docs[:2]
    return "\n".join(s for s, _ in picked), used

def make_llm():
    if LLM_BACKEND == "none":
        return None
    if LLM_BACKEND == "hf":
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
    if LLM_BACKEND == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except Exception:
            from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "mistral"), temperature=0.2)
    # OpenAI
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.getenv("CHAT_MODEL", "gpt-4o-mini"), temperature=0.2, max_retries=6, timeout=60)

def build_messages(q: str, ctx: str):
    system = (
        "Tu es un assistant bancaire francophone strictement fondé sur les sources (RAG).\n"
        "Règles:\n"
        "1) Tu utilises UNIQUEMENT le CONTEXTE fourni ; sinon, écris exactement : « Information manquante dans les sources fournies. »\n"
        "2) N'invente jamais de montants (€), numéros de téléphone, délais ou procédures.\n"
        "3) Préfère les PDF tarifaires/conditions en cas de conflit.\n"
        "4) Style : clair, concis, 3–5 puces."
    )
    user = f"Question:\n{q}\n\nContexte:\n{ctx}\n\nConsigne: Réponds en 3–5 puces maximum."
    return [SystemMessage(content=system), HumanMessage(content=user)]

def violates_grounding(answer: str, context: str) -> bool:
    """Flag if € amounts or phone numbers in the answer are absent from context."""
    for m in _EURO_RE.findall(answer):
        if m not in context:
            return True
    for m in _PHONE_RE.findall(answer):
        if m.strip() and m.strip() not in context:
            return True
    return False

def citation_precision(used_docs: List, retrieved_all: List) -> Optional[float]:
    """Share of used_docs that are contained in retrieved_all (ID match)."""
    ret_ids = {(d.metadata.get("source_url") or d.metadata.get("path") or "") for d in retrieved_all}
    if not used_docs:
        return 1.0
    ok = 0
    for d in used_docs:
        sid = d.metadata.get("source_url") or d.metadata.get("path") or ""
        ok += 1 if sid in ret_ids else 0
    return ok / len(used_docs) if used_docs else None

def recall_at_k(docs, expected: List[str], k: int) -> Optional[float]:
    if not expected: return None
    exp = [e.lower() for e in expected]
    for d in docs[:k]:
        did = _doc_id(d)
        if any(x in did for x in exp):
            return 1.0
    return 0.0

def mrr_at_k(docs, expected: List[str], k: int) -> Optional[float]:
    if not expected: return None
    exp = [e.lower() for e in expected]
    for i, d in enumerate(docs[:k], 1):
        did = _doc_id(d)
        if any(x in did for x in exp):
            return 1.0 / i
    return 0.0

def fact_em(answer: str, patterns: List[str]) -> Optional[int]:
    """All regex in patterns must match the answer."""
    if not patterns: return None
    for p in patterns:
        if not re.search(p, answer, flags=re.I):
            return 0
    return 1

def must_any(answer: str, tokens: List[str]) -> Optional[int]:
    """At least one token must be present (case-insensitive)."""
    if not tokens: return None
    for t in tokens:
        if re.search(re.escape(t), answer, flags=re.I):
            return 1
    return 0

def is_refusal(answer: str) -> bool:
    """Flexible check for refusal phrasing."""
    a = (answer or "").strip()
    if not a:
        return False
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, a, flags=re.I):
            return True
    return False

def _avg(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None

# ---------- Main ----------
def main():
    if not EVAL_FILE.exists():
        print(f"Eval set missing: {EVAL_FILE}"); sys.exit(1)

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    items = data.get("questions", data) or []
    if not isinstance(items, list) or not items:
        print("No questions found in eval set."); sys.exit(1)

    vs, emb = load_vs_and_emb()
    llm = make_llm()
    max_refusal_logs = DEFAULT_MAX_REFUSAL_LOGS

    rows = []
    for ex in items:
        base_q = ex["question"]
        queries = [base_q] + ex.get("variants", [])
        expected_sources = ex.get("expected_sources", [])
        expected_regex   = ex.get("expected_regex", [])
        must_any_tokens  = ex.get("must_contain_any", [])
        unanswerable     = ex.get("unanswerable", False)

        r5_list, r10_list, mrr10_list = [], [], []
        fact_list, grounded_list, citprec_list, refusal_list, must_any_list = [], [], [], [], []

        for q in queries:
            # 1) Retrieval
            topk, retrieved_all = retrieve(vs, q, k=TOP_K, oversample=OVERSAMPLE)

            # 2) Context compression
            ctx, cite_docs = compress_context(q, topk, emb, char_budget=COMP_CHARS)

            # Retrieval metrics per variant (computed on retrieved_all)
            r5_list.append(recall_at_k(retrieved_all, expected_sources, 5))
            r10_list.append(recall_at_k(retrieved_all, expected_sources, 10))
            mrr10_list.append(mrr_at_k(retrieved_all, expected_sources, 10))

            # 3) Generation
            if unanswerable:
                # Deterministic refusal for out-of-scope questions
                answer = "Information manquante dans les sources fournies." 
            else:
                if llm is None:
                    answer = "Information manquante dans les sources fournies." if not ctx \
                            else textwrap.shorten(ctx, width=600, placeholder=" …")
                else:
                    try:
                        resp = llm.invoke(build_messages(q, ctx))
                        answer = (resp.content or "").strip()
                    except Exception:
                        answer = ""
            # 4) Scoring
            if unanswerable:
                ok = 1 if is_refusal(answer) else 0
                refusal_list.append(ok)
                grounded_list.append(1)  # N/A → considered OK
                fact_list.append(None)
                must_any_list.append(None)
                citprec_list.append(citation_precision(cite_docs, retrieved_all))

                # DEBUG: show a handful of mismatches
                if DEBUG_REFUSALS and ok == 0 and max_refusal_logs > 0:
                    print("\n[DEBUG: refusal-miss]")
                    print("Q :", q)
                    print("Ans:", (answer or "")[:400])
                    print("----")
                    max_refusal_logs -= 1
            else:
                grounded_ok = 0 if violates_grounding(answer, ctx) else 1
                f_em = None
                if FACT_EM_MODE in ("report", "strict") and expected_regex:
                    f_em = fact_em(answer, expected_regex)
                m_any = must_any(answer, must_any_tokens)

                if (not answer) or (grounded_ok == 0):
                    answer = "Information manquante dans les sources fournies."
                    grounded_ok = 1

                grounded_list.append(grounded_ok)
                fact_list.append(f_em)
                must_any_list.append(m_any)
                citprec_list.append(citation_precision(cite_docs, retrieved_all))
                refusal_list.append(None)

        rows.append({
            "id": ex.get("id", ""),
            "n_variants": len(queries),
            "recall@5":  _avg(r5_list),
            "recall@10": _avg(r10_list),
            "mrr@10":    _avg(mrr10_list),
            "fact_em":   _avg([x for x in fact_list if x is not None]),
            "must_any":  _avg([x for x in must_any_list if x is not None]),
            "grounded_ok": _avg(grounded_list),
            "citation_precision": _avg([x for x in citprec_list if x is not None]),
            "refusal_accuracy":   _avg([x for x in refusal_list if x is not None]),
        })

    # ---------- Aggregate summary ----------
    def avg_field(field):
        vals = [r[field] for r in rows if r[field] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "n": len(rows),
        "retrieval": {
            "recall@5":  avg_field("recall@5"),
            "recall@10": avg_field("recall@10"),
            "mrr@10":    avg_field("mrr@10"),
        },
        "generation": {
            "fact_em":            avg_field("fact_em"),
            "must_any":           avg_field("must_any"),
            "grounded_ok":        avg_field("grounded_ok"),
            "citation_precision": avg_field("citation_precision"),
            "refusal_accuracy":   avg_field("refusal_accuracy"),
        }
    }

    # ---------- Outputs ----------
    out_csv = BASE / "scripts" / "eval_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out_json = BASE / "scripts" / "eval_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Per-item CSV: {out_csv}")
    print(f"Summary JSON: {out_json}")

if __name__ == "__main__":
    main()
