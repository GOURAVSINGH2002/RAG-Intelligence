# RAG Intelligence — Anti-Hallucination Chatbot (v6)

A full-stack chatbot that runs every query through **3 AI systems simultaneously**:
| Panel | Description |
|-------|-------------|
| 🤖 Baseline LLM | Pure parametric knowledge, no retrieval |
| 🎭 Hallucinating AI | Overconfident, ungrounded — shows dangers |
| 🔬 RAG v6 Universal | Live web retrieval + verified + fully metric-scored |

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install fastapi uvicorn groq duckduckgo-search
```

### 2. Place files together in one folder
```
your-project/
├── server.py
├── chatbot_frontend.html
└── rag_pipeline_v6.py   (optional — standalone CLI version)
```

### 3. Set your Groq API key
Edit the top of `server.py`:
```python
GROQ_API_KEY = "your-key-here"
```
Get a free key at: https://console.groq.com

### 4. Run the server
```bash
python server.py
```

### 5. Open in browser
```
http://localhost:8000
```

---

## 🔬 How It Works

Each query triggers 3 parallel AI runs:

```
User Query
    │
    ├──► Baseline LLM      → parametric answer, basic metrics
    ├──► Hallucinating AI  → overconfident answer, no grounding
    └──► RAG Pipeline v6
              │
              ├── LLM Query Analyser (detects intent, generates 5 search queries)
              ├── DuckDuckGo Retrieval (8 docs, NER-boosted reranking)
              ├── Answer Generation (up to 3 retry passes)
              ├── Cross-Verification (3 independent queries)
              └── Full Metric Suite:
                    Retrieval: Precision@K, Recall@K, F1@K, MRR
                    Generation: Hallucination Rate, Entity Grounding, ROUGE-L
                    Faithfulness: Claim-level LLM judge, Semantic Consistency
                    Composite: Confidence Score [0–100]
```

---

## 📊 Metrics Explained

| Metric | What it measures | Target |
|--------|-----------------|--------|
| Confidence Score | Composite [0–100] | ≥72 = HIGH |
| Faithfulness | % claims supported by sources | ≥0.8 |
| Entity Grounding | Named entities found in context | ≥0.85 |
| Hallucination Rate | % words not in context | ≤0.15 |
| Semantic Consistency | Primary vs verification agreement | ≥0.8 |
| Context Utilization | Answer grounded in retrieved text | ≥0.7 |

---

## 🌐 API Endpoint

**POST** `/query`
```json
{ "query": "Who is the current PM of India?" }
```
Returns:
```json
{
  "baseline_answer": "...",
  "baseline_metrics": { ... },
  "hallu_answer": "...",
  "hallu_metrics": { ... },
  "rag_answer": "...",
  "rag_metrics": { "confidence_score": 84.5, ... }
}
```

---

## 🧪 Demo Mode
Open `chatbot_frontend.html` **directly in a browser** (without the server) to see the UI with mock data — no Python needed for UI testing.

---

## 🎨 Frontend Features
- Animated particle background with grid
- Three answer cards with live-updating metric panels
- Circular confidence score ring (animated)
- Progress bar metrics for all key scores
- Source citation highlighting `[S1]`, `[S2]`
- Intent detection badge
- Suggestion chips for quick testing
- Typing indicator & skeleton loading states
- Responsive: stacks to 1 column on mobile