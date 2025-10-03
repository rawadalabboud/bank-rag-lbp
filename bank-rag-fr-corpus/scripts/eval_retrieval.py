#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, csv, yaml
from pathlib import Path
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BASE = Path(__file__).resolve().parents[1]
INDEX_DIR = BASE / "faiss_index"
EVAL_FILE = BASE / "scripts" / "eval_bank_fr.yaml"

HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "BAAI/bge-m3")
DEVICE = os.getenv("EMBED_DEVICE", "mps")

def load_vs():
    emb = HuggingFaceEmbeddings(
        model_name=HF_EMBED_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)
    return vs

def _doc_id(d) -> str:
    return (d.metadata.get("source_url") or d.metadata.get("path") or "").lower()

def _recall_at_k(docs, expected: List[str], k: int) -> Optional[float]:
    if not expected: return None
    exp = [e.lower() for e in expected]
    for d in docs[:k]:
        did = _doc_id(d)
        if any(x in did for x in exp):
            return 1.0
    return 0.0

def _mrr_at_k(docs, expected: List[str], k: int) -> Optional[float]:
    if not expected: return None
    exp = [e.lower() for e in expected]
    for i, d in enumerate(docs[:k], 1):
        did = _doc_id(d)
        if any(x in did for x in exp):
            return 1.0 / i
    return 0.0

def _avg(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs)/len(xs), 4) if xs else None

def main():
    if not EVAL_FILE.exists():
        print(f"Eval set missing: {EVAL_FILE}"); sys.exit(1)

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    items = data.get("questions", data) or []


    vs = load_vs()
    rows = []

    for ex in items:
        q_base = ex["question"]
        queries = [q_base] + ex.get("variants", [])
        expected_sources = ex.get("expected_sources", [])

        r5_list, r10_list, mrr10_list = [], [], []
        for q in queries:
            docs = vs.similarity_search(q, k=10)
            r5_list.append(_recall_at_k(docs, expected_sources, 5))
            r10_list.append(_recall_at_k(docs, expected_sources, 10))
            mrr10_list.append(_mrr_at_k(docs, expected_sources, 10))

        rows.append({
            "id": ex.get("id",""),
            "n_variants": len(queries),
            "recall@5":  _avg(r5_list),
            "recall@10": _avg(r10_list),
            "mrr@10":    _avg(mrr10_list),
        })

    # aggregate
    def avg_field(field):
        vals = [r[field] for r in rows if r[field] is not None]
        return round(sum(vals)/len(vals), 4) if vals else None

    summary = {
        "n": len(rows),
        "recall@5":  avg_field("recall@5"),
        "recall@10": avg_field("recall@10"),
        "mrr@10":    avg_field("mrr@10"),
    }

    out_csv = BASE / "scripts" / "eval_retrieval_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Per-item CSV: {out_csv}")

if __name__ == "__main__":
    main()
