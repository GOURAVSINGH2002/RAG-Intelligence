"""
RAG Intelligence — FastAPI Backend  (v6 Universal)
Serves the chatbot frontend and runs all 3 pipeline phases per query.

Install:
    pip install fastapi uvicorn groq duckduckgo-search

Run:
    python server.py
    Then open http://localhost:8000
"""

import time, re, json
import os
from pathlib import Path
from collections import Counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from groq import Groq
from ddgs import DDGS

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL          = "llama-3.3-70b-versatile"
DOCS_PER_QUERY = 5
TOP_DOCS       = 8
PRECISION_K    = 3
MAX_RETRIES    = 3
RELEVANCE_THRESHOLD = 0.30
MIN_DOC_WORDS       = 12
NER_BOOST           = 5

client = Groq(api_key=GROQ_API_KEY)
app    = FastAPI(title="RAG Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
STOPWORDS = {
    'the','a','an','is','are','was','were','in','on','at','to',
    'for','of','and','or','but','it','its','this','that','with',
    'as','by','from','have','has','had','be','been','not','no',
    'who','what','when','where','how','which','i','you','we','they',
    'their','our','your','his','her','also','just','more','than',
    'then','so','if','do','did','does','will','would','could','after',
    'before','during','over','under','about','into','through','between',
    'each','such','only','other','some','these','those','very','can',
    'get','got','may','might','must','shall','been','being','am',
    'said','says','according','per','via','like','new','one','two'
}

EVASION_PHRASES = [
    "not mentioned","does not mention","not provided","not available",
    "i don't know","cannot find","no information","not stated",
    "not found","information provided does not","cannot determine",
    "not explicitly","unable to","no specific","does not contain",
    "no results","not specified","as of my knowledge cutoff",
    "as of my last update","my training data","i cannot confirm",
    "i do not have","outside my","beyond my","i'm not sure",
    "i am not sure","no data","insufficient"
]

# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def content_words(text):
    return [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 2]

def get_ngrams(tokens, n):
    return {tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)}

def ask_ai(prompt, system="You are a helpful assistant.", temp=0.1):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
        temperature=temp
    )
    return r.choices[0].message.content.strip()

def extract_named_entities(text):
    tokens = re.findall(r'(?<!\.\s)\b[A-Z][a-zA-Z]{2,}\b', text)
    return {t.lower() for t in tokens if t.lower() not in STOPWORDS}

def check_completeness(answer):
    if any(p in answer.lower() for p in EVASION_PHRASES): return False
    if len(content_words(answer)) < 5: return False
    return True

# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════
def compute_precision_at_k(query, docs, k=PRECISION_K):
    qt = set(tokenize(query))
    rel = sum(1 for d in docs[:k]
              if len(qt & set(tokenize(d)))/max(len(qt),1) > RELEVANCE_THRESHOLD)
    return round(rel/min(k,len(docs)), 4) if docs else 0.0

def compute_recall_at_k(query, docs, k=PRECISION_K):
    qt = set(tokenize(query))
    def rel(d): return len(qt & set(tokenize(d)))/max(len(qt),1) > RELEVANCE_THRESHOLD
    total = sum(1 for d in docs if rel(d))
    if total == 0: return 0.0
    return round(sum(1 for d in docs[:k] if rel(d))/total, 4)

def compute_f1_at_k(p, r):
    return round(2*p*r/(p+r), 4) if p+r else 0.0

def compute_mrr(query, docs):
    qt = set(tokenize(query))
    for i, d in enumerate(docs, 1):
        if len(qt & set(tokenize(d)))/max(len(qt),1) > RELEVANCE_THRESHOLD:
            return round(1.0/i, 4)
    return 0.0

def compute_source_coverage(ans, docs):
    ac = set(content_words(ans))
    return round(sum(1 for d in docs if len(ac & set(content_words(d)))>=2)/max(len(docs),1), 4)

def compute_ctx_util(ans, docs):
    ac = set(content_words(ans))
    if not ac: return 0.0
    ctx = set(tokenize(" ".join(docs)))
    return round(len(ac & ctx)/len(ac), 4)

def compute_hallucination(ans, docs):
    full_ctx    = " ".join(docs).lower()
    ctx_uni     = set(tokenize(full_ctx))
    ctx_bi      = get_ngrams(tokenize(full_ctx), 2)
    ans_tok     = tokenize(ans)
    ans_cw      = content_words(ans)
    if not ans_cw: return 0.0
    unsup       = [w for w in ans_cw if w not in ctx_uni]
    ans_bi      = get_ngrams(ans_tok, 2)
    rescued     = {w for bg in ans_bi if bg in ctx_bi for w in bg}
    truly_unsup = [w for w in unsup if w not in rescued]
    ans_ner     = extract_named_entities(ans)
    ctx_ner     = extract_named_entities(" ".join(docs))
    ner_penalty = len(ans_ner - ctx_ner) * 2
    num = len(truly_unsup) + ner_penalty
    den = len(ans_cw) + ner_penalty
    return round(min(num/den, 1.0), 4)

def compute_entity_grounding(ans, docs):
    ne = extract_named_entities(ans)
    if not ne: return 1.0
    ctx_ne = extract_named_entities(" ".join(docs))
    return round(len(ne & ctx_ne)/len(ne), 4)

def compute_rouge_l(ans, ctx):
    pred = tokenize(ans)
    ref  = tokenize(ctx)[:500]
    m, n = len(ref), len(pred)
    if not m or not n: return 0.0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i][j] = dp[i-1][j-1]+1 if ref[i-1]==pred[j-1] else max(dp[i-1][j],dp[i][j-1])
    lcs = dp[m][n]
    p, r = lcs/n, lcs/m
    return round(2*p*r/(p+r), 4) if p+r else 0.0

def compute_tok_f1(a1, a2):
    c1, c2 = Counter(tokenize(a1)), Counter(tokenize(a2))
    common = sum((c1&c2).values())
    if not common: return 0.0, 0.0, 0.0
    t1, t2 = len(tokenize(a1)), len(tokenize(a2))
    p, r = common/t1, common/t2
    return round(2*p*r/(p+r),4), round(p,4), round(r,4)

def compute_lex_div(ans):
    t = tokenize(ans)
    return round(len(set(t))/len(t),4) if t else 0.0

def compute_len(ans):
    return {"total_tokens": len(tokenize(ans)), "content_tokens": len(content_words(ans))}

def compute_faithfulness(ans, ctx, answer_type="factual answer"):
    prompt = f"""Evaluate faithfulness. Answer type: {answer_type}
CONTEXT: {ctx[:3000]}
ANSWER: {ans}
CLAIMS:
1. [claim] → supported/unsupported
SUPPORTED: [N] / [TOTAL]
SCORE: [0.00 to 1.00]"""
    result = ask_ai(prompt, "You are a strict faithfulness evaluator.", temp=0.0)
    m = re.search(r'SCORE:\s*([0-9.]+)', result)
    return round(min(float(m.group(1)) if m else 0.5, 1.0), 4)

def compute_consistency(a1, a2, answer_type="factual answer"):
    prompt = f"""Compare two answers. Answer type: {answer_type}
CRITICAL: Different {answer_type}s = none agreement = 0.0
A: {a1}
B: {a2}
AGREEMENT: [full/partial/none]
SCORE: [1.0/0.6/0.0]
REASON: [one sentence]"""
    result = ask_ai(prompt, "You are a strict consistency judge.", temp=0.0)
    m = re.search(r'SCORE:\s*([0-9.]+)', result)
    return round(min(float(m.group(1)) if m else 0.5, 1.0), 4)

def compute_confidence(faith, consist, entity_g, ctx_util, hallu):
    return round(min(faith*30 + consist*25 + entity_g*20 + ctx_util*15 + (1-hallu)*10, 100), 1)

# ═══════════════════════════════════════════════════════════════
# RETRIEVAL
# ═══════════════════════════════════════════════════════════════
def analyse_query(query):
    prompt = f"""Analyse and produce a search strategy.
QUESTION: {query}
INTENT: [one word]
ANSWER_TYPE: [what kind of answer]
ANSWER_FORMAT: [how to phrase the answer]
PRIMARY_QUERY_1: [direct search]
PRIMARY_QUERY_2: [official source angle]
PRIMARY_QUERY_3: [recent news angle]
PRIMARY_QUERY_4: [alternate phrasing]
PRIMARY_QUERY_5: [key entities + latest]
VERIFY_QUERY_1: [independent check 1]
VERIFY_QUERY_2: [independent check 2]
VERIFY_QUERY_3: [independent check 3]"""
    raw = ask_ai(prompt, "Output ONLY the structured format. No extra text.", temp=0.0)
    result = {
        "intent_label": "general", "answer_type": "factual answer",
        "answer_format": "Direct factual statement",
        "primary_queries": [], "verify_queries": []
    }
    for line in raw.strip().split('\n'):
        line = line.strip()
        if line.startswith("INTENT:"):           result["intent_label"] = line.split(":",1)[1].strip().lower()
        elif line.startswith("ANSWER_TYPE:"):    result["answer_type"]  = line.split(":",1)[1].strip()
        elif line.startswith("ANSWER_FORMAT:"):  result["answer_format"]= line.split(":",1)[1].strip()
        elif re.match(r'PRIMARY_QUERY_\d+:', line):
            q = line.split(":",1)[1].strip()
            if q: result["primary_queries"].append(q)
        elif re.match(r'VERIFY_QUERY_\d+:', line):
            q = line.split(":",1)[1].strip()
            if q: result["verify_queries"].append(q)
    if not result["primary_queries"]:
        result["primary_queries"] = [query, f"{query} latest", f"{query} official", f"{query} current", f"{query} facts"]
    if not result["verify_queries"]:
        result["verify_queries"] = [f"{query} verified", f"{query} news", f"latest {query}"]
    return result

def retrieve_docs(queries, max_per_query=DOCS_PER_QUERY):
    seen, results = set(), []
    for q in queries:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(q, max_results=max_per_query):
                    if r['href'] not in seen:
                        seen.add(r['href'])
                        results.append({"url": r['href'], "content": r['body'], "title": r.get('title','')})
        except: pass
    results = [r for r in results if len(content_words(r['content'])) >= MIN_DOC_WORDS]
    q0_words = set(tokenize(queries[0]))
    q0_ner   = extract_named_entities(queries[0])
    for r in results:
        ct = set(tokenize(r['content']))
        tt = set(tokenize(r['title']))
        dn = extract_named_entities(r['content']+" "+r['title'])
        r['score'] = len(q0_words&ct) + len(q0_words&tt)*2 + len(q0_ner&dn)*NER_BOOST
    results.sort(key=lambda x: x['score'], reverse=True)
    top = results[:TOP_DOCS]
    reranked = ([top[0]] + top[2:] + [top[1]]) if len(top) >= 2 else top
    ctx_str  = ""
    raw_docs = []
    for i, r in enumerate(reranked):
        ctx_str += f"[Source {i+1}] {r['url']}\nTitle: {r['title']}\nContent: {r['content']}\n\n"
        raw_docs.append(r['content']+" "+r['title'])
    return ctx_str, raw_docs

def gen_answer(query, ctx, fmt="Direct factual statement", pass_num=1):
    sys = ("You are a precise factual assistant. Use ONLY the provided context. "
           "NEVER invent names, numbers, or facts. Cite as [Source N].")
    if pass_num == 1:
        prompt = f"Answer ONLY from sources below.\nExpected format: {fmt}\nSOURCES:\n{ctx}\nQUESTION: {query}\nANSWER:"
    elif pass_num == 2:
        prompt = f"Scan all sources. Extract the answer.\nFormat: {fmt}\nSOURCES:\n{ctx}\nQUESTION: {query}\nANSWER:"
    else:
        prompt = f"Last attempt from sources. If insufficient:\n'⚠️ Training knowledge (VERIFY ONLINE): [answer]'\nSOURCES:\n{ctx}\nQUESTION: {query}\nANSWER:"
    return ask_ai(prompt, sys, temp=0.0 if pass_num > 1 else 0.05)

# ═══════════════════════════════════════════════════════════════
# PHASE RUNNERS
# ═══════════════════════════════════════════════════════════════
def run_baseline(query):
    t0  = time.time()
    ans = ask_ai(query, "You are a helpful assistant. Answer from training knowledge. Admit uncertainty.", temp=0.1)
    lat = round(time.time()-t0, 2)
    return ans, {"length": compute_len(ans), "lexical_diversity": compute_lex_div(ans), "latency": lat}

def run_hallucinating(query):
    t0  = time.time()
    ans = ask_ai(query, "You are an overconfident bot. Never admit uncertainty. Always give a specific confident answer.", temp=0.9)
    lat = round(time.time()-t0, 2)
    return ans, {"length": compute_len(ans), "lexical_diversity": compute_lex_div(ans), "latency": lat}

def run_rag(query):
    t0       = time.time()
    analysis = analyse_query(query)
    ctx, raw = retrieve_docs(analysis['primary_queries'])
    ans, passes = None, 0
    for p in range(1, MAX_RETRIES+1):
        a = gen_answer(query, ctx, analysis['answer_format'], p)
        passes = p
        if check_completeness(a):
            ans = a; break
    if not ans: ans = a
    # Verification
    vctx, vdocs = retrieve_docs(analysis['verify_queries'], max_per_query=3)
    v_ans = gen_answer(query, vctx, analysis['answer_format'], 1)
    lat = round(time.time()-t0, 2)

    p_k    = compute_precision_at_k(query, raw)
    r_k    = compute_recall_at_k(query, raw)
    hallu  = compute_hallucination(ans, raw)
    eg     = compute_entity_grounding(ans, raw)
    cu     = compute_ctx_util(ans, raw)
    faith  = compute_faithfulness(ans, ctx, analysis['answer_type'])
    consist= compute_consistency(ans, v_ans, analysis['answer_type'])
    conf   = compute_confidence(faith, consist, eg, cu, hallu)
    tf1,tp,tr = compute_tok_f1(ans, v_ans)

    metrics = {
        "precision_at_k":      p_k,
        "recall_at_k":         r_k,
        "f1_at_k":             compute_f1_at_k(p_k, r_k),
        "mrr":                 compute_mrr(query, raw),
        "source_coverage":     compute_source_coverage(ans, raw),
        "context_utilization": cu,
        "hallucination_rate":  hallu,
        "entity_grounding":    eg,
        "rouge_l_vs_context":  compute_rouge_l(ans, ctx),
        "token_f1":            tf1,
        "token_precision":     tp,
        "token_recall":        tr,
        "lexical_diversity":   compute_lex_div(ans),
        "answer_length":       compute_len(ans),
        "faithfulness_score":  faith,
        "consistency_score":   consist,
        "confidence_score":    conf,
        "docs_primary":        len(raw),
        "docs_verify":         len(vdocs),
        "passes_used":         passes,
        "latency":             lat,
        "intent":              analysis['intent_label'],
        "answer_type":         analysis['answer_type'],
    }
    return ans, metrics

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def handle_query(req: QueryRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        base_ans,  base_met  = run_baseline(query)
        hallu_ans, hallu_met = run_hallucinating(query)
        rag_ans,   rag_met   = run_rag(query)
        return JSONResponse({
            "baseline_answer": base_ans,
            "baseline_metrics": base_met,
            "hallu_answer":    hallu_ans,
            "hallu_metrics":   hallu_met,
            "rag_answer":      rag_ans,
            "rag_metrics":     rag_met,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Frontend not found. Place chatbot_frontend.html next to server.py</h2>")

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}

# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    print("═"*55)
    print("  RAG Intelligence Server — v6 Universal")
    print(f"  Model  : {MODEL}")
    print("  Open   : http://localhost:8000")
    print("═"*55)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
