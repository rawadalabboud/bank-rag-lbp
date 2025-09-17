# app/main.py
import os, re, time, textwrap
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from langchain_community.vectorstores import FAISS, Chroma
from langchain.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage

# -------- Paths / config --------
BASE = Path(__file__).resolve().parents[1] / "bank-rag-fr-corpus" if (Path(__file__).resolve().parents[1] / "bank-rag-fr-corpus").exists() else Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
FAISS_PATH = str(BASE / "faiss_index")
CHROMA_PATH = str(BASE / "chroma")

EMBED_BACKEND = os.getenv("EMBED_BACKEND", "hf")      # "hf" | "openai"
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

LLM_BACKEND = os.getenv("LLM_BACKEND", "hf")          # "hf" | "ollama" | "openai" | "none"
HF_REPO_ID = os.getenv("HF_REPO_ID", "Qwen/Qwen2.5-7B-Instruct")
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.1"))
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "160"))
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT", "60"))

TOP_K = int(os.getenv("TOP_K", "4"))
COMP_CHARS = int(os.getenv("COMP_CHARS", "900"))
MAX_SNIPPET = int(os.getenv("MAX_SNIPPET", "900"))
VECTORSTORE = os.getenv("VECTORSTORE", "faiss")       # "faiss" | "chroma"

PRICE_RE = re.compile(r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*€")
FREE_RE = re.compile(r"\bgratuit[e]?\b|0\s*€", re.IGNORECASE)

# -------- App --------
app = FastAPI(title="LBP RAG Chatbot", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Serve /static/*
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -------- Embeddings / LLM factories --------
def get_embeddings():
    if EMBED_BACKEND.lower() == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=EMBED_MODEL)
    else:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=HF_EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )

def make_llm():
    backend = LLM_BACKEND.lower()
    if backend == "none":
        return None
    if backend == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except Exception:
            from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "mistral"), temperature=0.2)
    if backend == "hf":
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        llm = HuggingFaceEndpoint(
            repo_id=HF_REPO_ID,
            task="text-generation",
            temperature=HF_TEMPERATURE,
            max_new_tokens=HF_MAX_NEW_TOKENS,
            timeout=HF_TIMEOUT,
            do_sample=False,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False,
        )
        return ChatHuggingFace(llm=llm)
    # default OpenAI
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.getenv("CHAT_MODEL", "gpt-4o-mini"), temperature=0.2, max_retries=6, timeout=60)

def safe_invoke(llm, messages, attempts=4):
    for i in range(attempts):
        try:
            return llm.invoke(messages)
        except Exception as e:
            msg = str(e)
            if any(t in msg for t in ("RateLimit", "Too Many Requests", "ReadTimeout", "timeout")):
                time.sleep(2 * (i + 1))
                continue
            raise
    raise RuntimeError("LLM invoke failed after retries")

# -------- Vector store --------
_VS = None
_EMB = None
_LLM = None

def load_vs():
    global _VS, _EMB
    if _VS is not None: return _VS
    _EMB = get_embeddings()
    if VECTORSTORE.lower() == "faiss":
        _VS = FAISS.load_local(FAISS_PATH, _EMB, allow_dangerous_deserialization=True)
    else:
        _VS = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=_EMB)
    return _VS

def retrieve(vs, query: str, k: int = TOP_K, oversample: int = None) -> List[Document]:
    if oversample is None: oversample = max(30, k*6)
    if isinstance(vs, FAISS):
        docs = vs.similarity_search(query, k=oversample)
        docs = [d for d in docs if str(d.metadata.get("langue", "fr")).lower() == "fr"]
        return docs[:k]
    return vs.as_retriever(search_kwargs={"k": k, "filter": {"langue": "fr"}}).get_relevant_documents(query)

# -------- Context compression + citations --------
def compress_context(question: str, docs: List[Document], char_budget: int = COMP_CHARS) -> Tuple[str, List[Document]]:
    emb = _EMB or get_embeddings()
    qv = emb.embed_query(question)
    cands = []
    for d in docs:
        for sent in re.split(r"(?<=[\.\!\?])\s+", d.page_content):
            s = sent.strip()
            if 40 <= len(s) <= 300:
                sv = emb.embed_query(s)
                score = sum(a*b for a, b in zip(qv, sv))
                cands.append((score, s, d))
    cands.sort(key=lambda x: x[0], reverse=True)
    picked, total, used_docs, seen = [], 0, [], set()
    for _, s, d in cands:
        if total + len(s) > char_budget:
            continue
        picked.append((s, d)); total += len(s) + 1
        key = d.metadata.get("source_url") or d.metadata.get("path")
        if key not in seen:
            used_docs.append(d); seen.add(key)
        if total >= char_budget: break
    compact = "\n".join(s for s, _ in picked) or "\n\n".join(d.page_content[:300] for d in docs[:2])
    return compact, used_docs

def format_citations(docs: List[Document]) -> List[Dict[str, str]]:
    seen = set()
    cites = []
    for d in docs:
        title = d.metadata.get("title") or d.metadata.get("source_title") or "Source"
        url = d.metadata.get("source_url") or d.metadata.get("path", "")
        sec = d.metadata.get("section_title")
        key = (title, url, sec)
        if key in seen:
            continue
        seen.add(key)
        cites.append({"title": title, "url": url, "section": sec or ""})
    return cites

# -------- System prompt (balanced; RAG-first, graceful fallback) --------
def build_messages(question: str, context: str, history: List[Dict[str, str]] = None):
    history = history or []
    sys_prompt = (
        "Tu es un assistant bancaire francophone pour La Banque Postale.\n"
        "Réponds de façon claire et concise en 3–6 puces ou en étapes si la question commence par « Comment… ».\n"
        "PRIORITÉ: Utilise EXCLUSIVEMENT les informations du CONTEXTE ci-dessous. Si une information n’y figure pas, écris : "
        "« Information manquante dans les sources fournies. »\n"
        "Ne fabrique pas d’URL. Les citations seront fournies séparément."
    )
    chat = [SystemMessage(content=sys_prompt)]
    if history:
        # keep last 3 exchanges for style continuity (not for facts)
        for turn in history[-3:]:
            if turn.get("role") == "user":
                chat.append(HumanMessage(content=turn.get("content","")))
            elif turn.get("role") == "assistant":
                chat.append(SystemMessage(content=f"[Style hint only]\n{turn.get('content','')}"))
    user = (
        f"Question:\n{question}\n\n"
        f"Contexte (extraits):\n{context}\n\n"
        "Contraintes:\n- Réponse en français.\n- 3–6 puces max; pas de hors-sujet.\n"
    )
    chat.append(HumanMessage(content=user))
    return chat

# -------- Extractive fallback (no LLM) --------
def extractive_answer(question: str, docs: List[Document]) -> Tuple[str, List[Document]]:
    emb = _EMB or get_embeddings()
    q_vec = emb.embed_query(question)

    def _split_sentences(text: str):
        return re.split(r"(?<=[\.\?\!])\s+", text)

    def _dot(u, v):
        return sum(a*b for a, b in zip(u, v))

    candidates = []
    for d in docs:
        for sent in _split_sentences(d.page_content[:MAX_SNIPPET]):
            s = sent.strip()
            if len(s) < 40: continue
            s_vec = emb.embed_query(s)
            candidates.append((_dot(q_vec, s_vec), s, d))
    if not candidates:
        return "Information manquante dans les sources fournies.", []
    candidates.sort(key=lambda x: x[0], reverse=True)
    picked, seen = [], set()
    for _, s, d in candidates:
        if s in seen: continue
        picked.append((s, d)); seen.add(s)
        if len(picked) >= 4: break
    answer = " ".join(s for s, _ in picked)
    cited, seen_docs = [], set()
    for _, d in picked:
        key = d.metadata.get("source_url") or d.metadata.get("path")
        if key not in seen_docs:
            cited.append(d); seen_docs.add(key)
    return answer, cited

# -------- Service: answer one question --------
def answer_question(question: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    vs = load_vs()
    docs = retrieve(vs, question, k=TOP_K)
    context, used_docs = compress_context(question, docs, char_budget=COMP_CHARS)
    cites = format_citations(used_docs if used_docs else docs)

    llm = _LLM or make_llm()
    if llm is None:
        text, cited = extractive_answer(question, docs)
        return {"text": text, "sources": format_citations(cited if cited else docs)}

    try:
        resp = safe_invoke(llm, build_messages(question, context, history))
        text = resp.content.strip()
        # Guardrail: if model punts, fallback extractive
        if "Information manquante" in text and any(k in context.lower() for k in ("virement", "certicode", "3d secure", "iban", "tarif", "frais")):
            raise RuntimeError("Guardrail fallback")
        return {"text": text, "sources": cites}
    except Exception:
        text, cited = extractive_answer(question, docs)
        return {"text": text, "sources": format_citations(cited if cited else docs)}

# -------- Routes --------
@app.get("/health")
def health():
    # cheap check that index is present
    ok = Path(FAISS_PATH).exists() or Path(CHROMA_PATH).exists()
    return {"ok": ok, "vectorstore": VECTORSTORE, "index_path": FAISS_PATH if VECTORSTORE=="faiss" else CHROMA_PATH}

INDEX_HTML = """<!doctype html>
<html lang="fr">
<head>
<link rel="preload" as="image" href="/static/lbp-logo.svg.png">
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LBP RAG Chatbot</title>
<style>
:root{
  --bg:#0b1220; --panel:#0f172a; --muted:#64748b;
  --accent:#0ea5e9; --accent2:#22d3ee; --chip:#0b2840;
  --user:#1f6feb; --assistant:#0ea5e9;
}
*{box-sizing:border-box}
body{
  margin:0; background:linear-gradient(180deg,#0b1220, #0b1220 40%, #0f172a);
  color:#e5e7eb; font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
}
.wrap{max-width:980px;margin:24px auto;padding:0 16px}
.header{
  display:flex;align-items:center;gap:14px;margin:18px 0 12px;
}
.badge{
  width:44px;height:44px;
  display:inline-flex;align-items:center;justify-content:center;
  background:#fff;                 /* ← force white background */
  border-radius:12px;
  padding:6px;                     /* little inner breathing room */
  border:1px solid rgba(0,0,0,.08);
  box-shadow:0 6px 20px rgba(2,6,23,.35);
}
.badge img{
  width:100%;height:100%;object-fit:contain;display:block;
}

.title{font-size:20px;font-weight:700;letter-spacing:.3px}
.sub{color:var(--muted);font-size:13px;margin-top:2px}
.panel{
  background:rgba(2,6,23,.5);border:1px solid rgba(148,163,184,.15);
  border-radius:16px;box-shadow:0 10px 30px rgba(2,6,23,.45);padding:14px;
}
#chat{
  height:62vh;overflow:auto;padding:8px;
  scroll-behavior:smooth;
}
.msg{display:flex;gap:10px;margin:10px 0}
.avatar{
  width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  background:#0b2840;flex:0 0 32px;font-size:13px;color:#cbd5e1
}
.avatar.user{background:#132c4b}
.bubble{
  max-width:78%; padding:10px 12px; border-radius:14px;
  box-shadow:0 8px 18px rgba(2,6,23,.35); line-height:1.45; font-size:15px;
}
.user .bubble{background:#0b2840;border:1px solid rgba(148,163,184,.15)}
.assistant .bubble{background:#071d2f;border:1px solid rgba(148,163,184,.12)}
.sources{margin-top:8px;display:flex;flex-wrap:wrap;gap:6px}
.chip{
  font-size:12px;padding:6px 8px;border-radius:999px;background:rgba(14,165,233,.12);
  border:1px solid rgba(14,165,233,.25)
}
.chip a{color:#bae6fd;text-decoration:none}
.footer{
  display:flex;gap:10px;margin-top:12px
}
input{
  flex:1;padding:12px 14px;border-radius:12px;border:1px solid rgba(148,163,184,.2);
  background:#0b1a2a;color:#e5e7eb;outline:none
}
button{
  padding:12px 16px;border:none;border-radius:12px;background:linear-gradient(90deg,var(--accent),var(--accent2));
  color:#041725;font-weight:700;cursor:pointer;box-shadow:0 8px 18px rgba(14,165,233,.35)
}
button:disabled{opacity:.6;cursor:not-allowed;box-shadow:none}
small.hint{color:var(--muted);display:block;margin-top:6px}
.typing{display:inline-flex;gap:3px;align-items:center}
.dot{
  width:6px;height:6px;border-radius:50%;background:#7dd3fc;opacity:.6;
  animation:blink 1.2s infinite ease-in-out;
}
.dot:nth-child(2){animation-delay:.15s}
.dot:nth-child(3){animation-delay:.3s}
@keyframes blink{
  0%,80%,100%{transform:translateY(0);opacity:.35}
  40%{transform:translateY(-3px);opacity:1}
}
a{color:#93c5fd}
</style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <span class="badge" title="La Banque Postale">
        <img src="/static/lbp-logo.svg.png" alt="La Banque Postale" />
      </span>
      <div>
        <div class="title">Assistant LBP (RAG)</div>
        <div class="sub">Réponses sourcées depuis les pages officielles & Tarifs</div>
      </div>
    </div>

    <div class="panel">
      <div id="chat"></div>
      <div class="footer">
        <input id="q" placeholder="Posez votre question (ex : Frais virement SEPA au guichet ?)" />
        <button id="send" onclick="ask()">Envoyer</button>
      </div>
      <small class="hint">Astuce : appuyez sur Entrée pour envoyer • Les sources sont cliquables.</small>
    </div>
  </div>

<script>
const chat = document.getElementById('chat');
const box = document.getElementById('q');
const btn = document.getElementById('send');

function addMessage(role, html, sources){
  const row = document.createElement('div');
  row.className = 'msg ' + role;
  const av = document.createElement('div');
  av.className = 'avatar ' + role;
  av.innerHTML = role === 'user' ? 'Vous' : 'LBP';
  const bb = document.createElement('div');
  bb.className = 'bubble';
  bb.innerHTML = html.replaceAll('\\n','<br>');

  if (role === 'assistant' && sources && sources.length){
    const s = document.createElement('div'); s.className = 'sources';
    for (const src of sources){
      const chip = document.createElement('span'); chip.className = 'chip';
      const link = document.createElement('a'); link.target = '_blank';
      link.textContent = src.title || 'Source';
      link.href = src.url || '#';
      chip.appendChild(link);
      if (src.section){ const i = document.createElement('span'); i.style.marginLeft='6px'; i.style.opacity='.7'; i.innerHTML = `<i>${src.section}</i>`; chip.appendChild(i); }
      s.appendChild(chip);
    }
    bb.appendChild(s);
  }

  row.appendChild(av); row.appendChild(bb);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}

let typingEl = null;
function showTyping(){
  typingEl = document.createElement('div');
  typingEl.className = 'msg assistant';
  typingEl.innerHTML = `
    <div class="avatar assistant">LBP</div>
    <div class="bubble"><span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span></div>`;
  chat.appendChild(typingEl);
  chat.scrollTop = chat.scrollHeight;
}
function hideTyping(){
  if (typingEl){ typingEl.remove(); typingEl = null; }
}

async function ask(){
  const q = box.value.trim();
  if(!q) return;
  addMessage('user', q);
  box.value = ''; box.focus();
  btn.disabled = true; box.disabled = true;
  showTyping();
  try{
    const r = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message:q})});
    const data = await r.json();
    hideTyping();
    addMessage('assistant', data.text || 'Pas de réponse.', data.sources || []);
  }catch(e){
    hideTyping();
    addMessage('assistant', "Oups, une erreur est survenue. Réessayez.");
  }finally{
    btn.disabled = false; box.disabled = false; box.focus();
  }
}

box.addEventListener('keydown', (e)=>{
  if(e.key === 'Enter' && !e.shiftKey){
    e.preventDefault();
    ask();
  }
});

// Optional: welcome message
addMessage('assistant', "Bonjour 👋 Comment puis-je vous aider aujourd’hui ?");
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)

@app.post("/chat")
async def chat(payload: Dict[str, Any]):
    msg = (payload or {}).get("message","").strip()
    history = (payload or {}).get("history", [])
    if not msg:
        return JSONResponse({"text": "Veuillez saisir une question.", "sources": []})
    out = answer_question(msg, history=history)
    return JSONResponse(out)
