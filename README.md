# Assistant LBP (RAG) — Chatbot bancaire FR

_Retrieval-Augmented Generation (RAG) pour répondre en français aux questions clients à partir des pages officielles **La Banque Postale** (FAQ, Tarifs PDF, etc.)._

> ✅ Réponses sourcées (titre + URL)  
> ✅ Mode sans LLM (extraction factuelle) **ou** avec LLM (Hugging Face / OpenAI / Ollama)  
> ✅ Évaluation rapide sur des cas métiers (tarifs, virements, RIB/IBAN…)  
> ✅ Petite app web (FastAPI + HTML/CSS) avec joli UI et logo LBP

---

## Sommaire

- [Aperçu](#aperçu)
- [Architecture](#architecture)
- [Arborescence](#arborescence)
- [Installation](#installation)
- [Données & Ingestion](#données--ingestion)
- [Recherche & Réponse](#recherche--réponse)
- [Application Web](#application-web)
- [Évaluation](#évaluation)
- [Configuration (env)](#configuration-env)
- [Ajouter des sources](#ajouter-des-sources)
- [Dépannage](#dépannage)
- [Licence & mentions](#licence--mentions)

---

## Aperçu

Le projet construit un corpus **Markdown** à partir des pages officielles (LBP, Service-Public, Banque de France…).  
On découpe en *chunks*, on crée des **embeddings** puis on indexe en **FAISS** (par défaut) pour la similarité.  
Lors d’une question :

1. **Récupération** des passages pertinents.  
2. **Compression** du contexte (sentences scoring) pour limiter les tokens.  
3. **Génération** avec un LLM (ou **fallback extractif** sans LLM).  
4. **Citations** systématiques (titre + URL).

---

## Architecture

```
[Sources YAML] -> fetch_to_md.py -> [Markdown] -> ingest.py
                           |                         |
                           v                         v
                         Nettoyage              Split (1000/150)
                                                Embeddings (HF/OpenAI)
                                                Vector Store (FAISS/Chroma)

query.py:
Question -> Retrieve -> Rerank -> Context Compress -> (LLM or Extractif) -> Réponse + Sources
```

> Par défaut :  
> - **Embeddings** : `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingue, gratuit)  
> - **Vector store** : **FAISS**  
> - **LLM** : Hugging Face Inference (`Qwen/Qwen2.5-7B-Instruct` conseillé) ou **mode sans LLM**

---

## Arborescence

```
bank-rag-fr-corpus/
├─ app/
│  ├─ static/              # logo, styles
│  └─ main.py              # FastAPI + page chat
├─ data/                   # Markdown générés (corpus)
├─ eval/
│  └─ cases.yaml           # Jeux de questions pour tests rapides
├─ faiss_index/            # Index FAISS (généré)
├─ scripts/
│  ├─ fetch_to_md.py       # Récupération -> Markdown
│  ├─ ingest.py            # Split + embeddings + FAISS/Chroma
│  ├─ query.py             # Pipeline RAG (LLM ou extractif)
│  └─ eval.py              # Petit harness d’évaluation
├─ sources_fr.yaml         # URLs officielles
├─ .env.example            # Variables d'environnement (modèle, clés, etc.)
├─ requirements.txt
└─ README.md
```

---

## Installation

```bash
# 1) Créer l'environnement
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
pip install -r requirements.txt

# 2) Configurer l'environnement (copier et adapter)
cp .env.example .env
# puis exportez vos clés si besoin (OpenAI/HF)
```

---

## Données & Ingestion

### 1) Récupérer les pages officielles → Markdown

```bash
python3 scripts/fetch_to_md.py
```

- Lit `sources_fr.yaml`
- Ajoute un front-matter (langue, URL, titre, date), nettoie l’HTML, sauvegarde `data/*.md`.

### 2) Créer l’index (embeddings + FAISS)

```bash
# Option gratuite (HF embeddings) + FAISS
EMBED_BACKEND=hf VECTORSTORE=faiss python3 scripts/ingest.py
```

- Split : `chunk_size=1000`, `chunk_overlap=150`
- Embeddings : `HF_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Index : `faiss_index/`

> Vous pouvez basculer vers OpenAI :
>
> ```bash
> export OPENAI_API_KEY=sk-...
> EMBED_BACKEND=openai EMBED_MODEL=text-embedding-3-small python3 scripts/ingest.py
> ```

---

## Recherche & Réponse

### Sans LLM (fallback extractif, rapide et robuste)

```bash
EMBED_BACKEND=hf LLM_BACKEND=none python3 scripts/query.py "Frais d'un virement SEPA au guichet ?"
```

> Le script sélectionne les phrases les plus proches sémantiquement et renvoie une réponse **strictement factuelle** + **citations**.

### Avec LLM (Hugging Face)

```bash
export HUGGING_FACE_HUB_TOKEN=hf_...
EMBED_BACKEND=hf LLM_BACKEND=hf HF_REPO_ID=Qwen/Qwen2.5-7B-Instruct HF_MAX_NEW_TOKENS=160 HF_TEMPERATURE=0 HF_TIMEOUT=60 TOP_K=3 COMP_CHARS=900 python3 scripts/query.py "Explique la différence entre un virement SEPA et un virement instantané (avec sources)."
```

- Le LLM est **chainé** et **contraint** : « Réponds uniquement à partir du contexte. Sinon, dis que l’info manque ».
- Le contexte est **compressé** (sentence-level scoring) pour rester dans le budget de tokens.
- Les **citations** sont affichées séparément (titre + URL).

### Avec OpenAI (option)

```bash
export OPENAI_API_KEY=sk-...
EMBED_BACKEND=openai LLM_BACKEND=openai CHAT_MODEL=gpt-4o-mini python3 scripts/query.py "Comment activer Certicode Plus ?"
```

### Paramètres utiles

- `TOP_K` : nb de documents récupérés (ex : `3`)
- `COMP_CHARS` : budget de caractères pour la compression de contexte (ex : `900`)
- `HF_MAX_NEW_TOKENS`, `HF_TEMPERATURE`, `HF_TIMEOUT` : contrôle des générations

---

## Application Web

Lancement de l’API + interface légère :

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Page d’accueil : champ de question, boutons modèle, citations cliquables, indicateur « l’assistant tape… ».
- Logo **La Banque Postale** affiché (à placer dans `app/static/logo-lbp.png`).

---

## Évaluation

Cas minimalistes pour tester la robustesse métier (tarifs, virements, retraits, etc.) :

```bash
python3 scripts/eval.py
```

- Lit `eval/cases.yaml`
- Exécute `query.py` et vérifie des **assertions textuelles** (ex : présence d’un montant).
- Compte les réussites/échecs.

---

## Configuration (env)

Variable | Rôle | Valeur par défaut
---|---|---
`EMBED_BACKEND` | `hf` \| `openai` | `hf`
`HF_EMBED_MODEL` | Modèle d’embedding HF | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
`EMBED_MODEL` | Modèle OpenAI | `text-embedding-3-small`
`VECTORSTORE` | `faiss` \| `chroma` | `faiss`
`LLM_BACKEND` | `hf` \| `openai` \| `ollama` \| `none` | `hf`
`HF_REPO_ID` | Modèle HF Inference | `Qwen/Qwen2.5-7B-Instruct`
`CHAT_MODEL` | Modèle OpenAI | `gpt-4o-mini`
`OLLAMA_MODEL` | Modèle local | `mistral`
`TOP_K` | Docs récupérés | `4`
`COMP_CHARS` | Budget de contexte (car.) | `1200`
`HF_MAX_NEW_TOKENS` | Longueur génération | `160`
`HF_TEMPERATURE` | Température | `0.1`
`HF_TIMEOUT` | Timeout (s) | `60`

---

## Ajouter des sources

1. Éditer `sources_fr.yaml` (URLs officielles).
2. `python3 scripts/fetch_to_md.py`
3. `python3 scripts/ingest.py` (recrée l’index)

> **NB** : pour les PDFs lourds (Tarifs), le découpage Markdown améliore la précision du retrieval.

---

## Dépannage

- **429 / rate limit** : réduire `TOP_K`, `HF_MAX_NEW_TOKENS`, `HF_TEMPERATURE`, ou passer en **mode extractif** (`LLM_BACKEND=none`) pour tester.  
- **Lent sur HF** : privilégier des modèles compacts (`Qwen2.5-7B-Instruct`, `gemma-2-2b-it`), baisser `COMP_CHARS`.  
- **LibreSSL warning (macOS entreprise)** : sans incidence fonctionnelle.  
- **Ollama indisponible** : utiliser HF Inference (pas besoin d’installer quoi que ce soit en local).

---

## Licence & mentions

- Code sous licence **MIT** (adapter si besoin).  
- Les contenus Markdown proviennent de sites officiels (La Banque Postale, Service-Public, Banque de France). Respecter leurs **conditions d’utilisation**.  
- Le **logo La Banque Postale** est une marque déposée et n’est utilisé ici qu’à des fins de démonstration.

---

### Exemples rapides

```bash
# Frais virement au guichet (extractif)
EMBED_BACKEND=hf LLM_BACKEND=none python3 scripts/query.py "Frais d'un virement SEPA au guichet ?"

# Virement instantané (LLM Hugging Face)
EMBED_BACKEND=hf LLM_BACKEND=hf HF_REPO_ID=Qwen/Qwen2.5-7B-Instruct python3 scripts/query.py "Le virement instantané est-il disponible 24/7 ?"
```

> Les réponses s’accompagnent toujours des **références** (titre + URL) pour vérification immédiate.
