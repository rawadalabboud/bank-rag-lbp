# Assistant LBP (RAG) — Chatbot bancaire FR 🇫🇷💳

_Un assistant bancaire basé sur **Retrieval-Augmented Generation (RAG)** pour répondre en français aux questions clients à partir des pages officielles de **La Banque Postale**, **Service-Public.fr**, et **Banque de France**._

> ✅ Réponses **sourcées** (titre + URL officiel)  
> ✅ Mode **extraction factuelle** (sans LLM) ou **génératif** (Hugging Face / OpenAI / Ollama)  
> ✅ Évaluations automatiques (retrieval & génération) sur un **jeu YAML de 63 cas métier**  
> ✅ Petite app web (FastAPI + HTML/CSS) avec logo **La Banque Postale**

---

## Sommaire

- [Aperçu](#aperçu)
- [Architecture](#architecture)
- [Arborescence](#arborescence)
- [Installation](#installation)
- [Ingestion des données](#ingestion-des-données)
- [Recherche & Réponse](#recherche--réponse)
- [Application Web](#application-web)
- [Évaluation](#évaluation)
- [Configuration](#configuration)
- [Ajouter des sources](#ajouter-des-sources)
- [Dépannage](#dépannage)
- [Licence & mentions](#licence--mentions)

---

## Aperçu

Le pipeline repose sur trois étapes principales :

1. **Récupération** des passages pertinents via embeddings + FAISS.  
2. **Compression** du contexte (scoring par phrase) pour rester efficace et éviter le bruit.  
3. **Génération contrôlée** :  
   - Mode **LLM** (HF, OpenAI, Ollama)  
   - Mode **fallback extractif** (sans LLM, renvoie directement les phrases pertinentes)  

Chaque réponse cite toujours les **sources officielles**.

---

## Architecture

```
[Sources YAML/Markdown] -> ingest.py -> [FAISS index]
                                      |
query.py: Question -> Retrieve -> Boost (optionnel) -> Context Compress
         -> (LLM ou Extractif) -> Réponse + Sources
```

- **Corpus** : pages HTML/PDF converties en Markdown avec front-matter  
- **Embeddings** : Hugging Face (`BAAI/bge-m3`) par défaut  
- **Vector store** : FAISS  
- **LLM** : Hugging Face Inference (Meta Llama 3.1 / Qwen2.5) ou OpenAI GPT  

---

## Arborescence

```
bank-rag-fr-corpus/
├─ app/                     # FastAPI UI
│  └─ main.py
├─ data/                    # Markdown (corpus)
├─ faiss_index/             # Index FAISS (généré)
├─ scripts/
│  ├─ ingest.py             # Split + embeddings + FAISS
│  ├─ query.py              # Pipeline RAG
│  ├─ eval_bank_fr.yaml     # Jeu de 63 cas métier (tarifs, virements, sécurité…)
│  ├─ eval_retrieval.py     # Éval. retrieval (recall@k, mrr@k)
│  ├─ eval_rag.py           # Éval. RAG (fact_em, grounded_ok, refusals…)
│  └─ fetch_to_md.py        # (optionnel) conversion HTML/PDF -> MD
├─ .env.prod                # Config (sans secrets)
├─ requirements.txt
└─ README.md
```

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Configurer l’environnement :

```bash
cp .env.prod .env   # version de prod, sans secrets
```

---

## Ingestion des données

1. **Préparer le corpus** (`data/*.md`) :  
   - soit généré via `fetch_to_md.py`  
   - soit copié depuis le dépôt (LBP, Banque de France, Service-Public)  

2. **Créer l’index FAISS** :  

```bash
HF_EMBED_MODEL="BAAI/bge-m3" VECTORSTORE=faiss python3 scripts/ingest.py
```

---

## Recherche & Réponse

### Mode extraction (sans LLM)

```bash
LLM_BACKEND=none python3 scripts/query.py "Quels sont les frais d’un virement SEPA ?"
```

→ renvoie les phrases les plus pertinentes + URL source.

### Mode LLM (Hugging Face)

```bash
LLM_BACKEND=hf HF_REPO_ID="meta-llama/Meta-Llama-3.1-70B-Instruct" TOP_K=5 COMP_CHARS=1000 python3 scripts/query.py "Comment activer Certicode Plus ?"
```

### Mode OpenAI

```bash
LLM_BACKEND=openai CHAT_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-... python3 scripts/query.py "Quels sont les plafonds d’une Visa Premier ?"
```

---

## Application Web

```bash
uvicorn app.main:app --reload --port 8000
```

- Interface simple type **chat**  
- Réponses sourcées, citations cliquables  
- Logo LBP dans `app/static/`

---

## Évaluation

### 1. Évaluer le **retrieval seul**

```bash
HF_EMBED_MODEL="BAAI/bge-m3" python3 scripts/eval_retrieval.py
```

→ calcule `recall@5`, `recall@10`, `mrr@10`

### 2. Évaluer le **RAG complet**

```bash
LLM_BACKEND=hf HF_REPO_ID="meta-llama/Meta-Llama-3.1-70B-Instruct" TOP_K=5 COMP_CHARS=1100 python3 scripts/eval_rag.py
```

→ calcule :
- **Retrieval** : recall/mrr  
- **Génération** : grounded_ok, citation_precision, refusals  
- **fact_em** : activable si regex fournis dans YAML (par défaut désactivé en prod)

Résultats sauvegardés dans :
- `eval_results.csv` (par item)
- `eval_summary.json` (moyennes)

---

## Configuration

Variable | Rôle
---|---
`HF_EMBED_MODEL` | modèle d’embeddings (par défaut `BAAI/bge-m3`)
`VECTORSTORE` | FAISS uniquement
`LLM_BACKEND` | none, hf, openai, ollama
`HF_REPO_ID` | modèle HF (ex: llama-3.1-70B-instruct)
`CHAT_MODEL` | modèle OpenAI (ex: gpt-4o-mini)
`TOP_K` | nb docs à récupérer
`COMP_CHARS` | budget contexte compressé
`AMOUNT_BOOST` | `on/off` boost des PDF tarifs
`FACT_EM_MODE` | `on/off` validation regex factuelles

---

## Ajouter des sources

1. Ajouter l’URL dans `sources_fr.yaml`  
2. `python3 scripts/fetch_to_md.py`  
3. `python3 scripts/ingest.py`  

---

## Dépannage

- **LibreSSL warning (macOS)** → sans impact  
- **Réponses trop longues** → baisser `COMP_CHARS` ou `HF_MAX_NEW_TOKENS`  
- **Pas de LLM dispo** → `LLM_BACKEND=none` pour extraction factuelle  
- **fact_em trop bas** → ajuster ou désactiver (`FACT_EM_MODE=off`)  

---

## Licence & mentions

- Code : **MIT**  
- Corpus : **La Banque Postale**, **Service-Public**, **Banque de France**  
- Logo **LBP** : utilisé uniquement pour la démonstration (marque déposée)
