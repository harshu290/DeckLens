import streamlit as st
import pdfplumber
import numpy as np
import requests
import re
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from html import unescape

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ── RAG helpers ───────────────────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file) -> list[str]:
    chunks = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                chunks.append(text.strip())
    return chunks


def build_rag_index(chunks: list[str]):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix


def retrieve(query: str, chunks, vectorizer, matrix, top_k=6) -> str:
    q_vec  = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return "\n\n---\n\n".join(chunks[i] for i in top_idx)


# ── ML scoring ────────────────────────────────────────────────────────────────
SIGNAL_KEYWORDS = {
    "problem":        ["problem", "pain point", "challenge", "issue", "gap"],
    "solution":       ["solution", "product", "platform", "technology", "innovation"],
    "market":         ["market", "tam", "sam", "som", "billion", "million", "opportunity", "industry"],
    "traction":       ["revenue", "users", "growth", "customers", "mrr", "arr", "retention", "traction"],
    "team":           ["founder", "ceo", "cto", "team", "experience", "background", "advisor"],
    "business_model": ["revenue model", "monetise", "monetize", "subscription", "saas", "freemium", "pricing"],
    "competition":    ["competitor", "competition", "differentiated", "moat", "advantage"],
    "financials":     ["financial", "projection", "forecast", "burn", "runway", "profitability", "ebitda"],
}

SCORE_REASONS = {
    "problem": {
        "high":   ["Clear problem statement identified", "Pain point well articulated", "Target customer pain is specific"],
        "medium": ["Problem mentioned but lacks depth", "Pain point somewhat vague", "Could use stronger evidence of pain"],
        "low":    ["No clear problem defined", "Problem statement is missing or generic", "Customer pain not demonstrated"],
    },
    "solution": {
        "high":   ["Product/solution clearly described", "Innovation or unique approach visible", "Technology differentiator present"],
        "medium": ["Solution mentioned but technical depth missing", "Product description is surface-level", "Differentiation not fully explained"],
        "low":    ["Solution unclear or absent", "No product description found", "No innovation signals detected"],
    },
    "market": {
        "high":   ["TAM/SAM/SOM figures present", "Market sizing is credible and specific", "Industry opportunity well framed"],
        "medium": ["Market mentioned but not quantified", "Some market signals present", "Lacks bottom-up market validation"],
        "low":    ["No market data found", "Market size not mentioned", "Industry opportunity not framed"],
    },
    "traction": {
        "high":   ["Revenue or user metrics present", "Growth trajectory visible", "Retention or engagement data shown"],
        "medium": ["Early traction signals mentioned", "Some customer references present", "Growth implied but not quantified"],
        "low":    ["No traction data found", "No revenue or user metrics mentioned", "Pre-revenue with no pilot evidence"],
    },
    "team": {
        "high":   ["Founding team clearly introduced", "Relevant domain expertise shown", "Advisory board or notable backers mentioned"],
        "medium": ["Team mentioned but bios are thin", "Experience listed without proof points", "Missing co-founder or key hire"],
        "low":    ["No team information found", "Founder background absent", "No evidence of domain expertise"],
    },
    "business_model": {
        "high":   ["Revenue model clearly explained", "Pricing strategy defined", "Unit economics or SaaS metrics present"],
        "medium": ["Monetization mentioned but unclear", "Pricing vague or missing", "Business model implied not stated"],
        "low":    ["No business model found", "No pricing or monetization discussed", "Revenue path entirely unclear"],
    },
    "competition": {
        "high":   ["Competitive landscape addressed", "Clear moat or differentiation stated", "Switching costs or defensibility shown"],
        "medium": ["Competitors mentioned without deep analysis", "Moat claimed but not substantiated", "Differentiation is generic"],
        "low":    ["No competitive analysis found", "No moat identified", "Competitive positioning absent"],
    },
    "financials": {
        "high":   ["Financial projections provided", "Burn rate and runway disclosed", "Path to profitability outlined"],
        "medium": ["Some financial mentions present", "Projections exist but lack detail", "Runway or burn partially discussed"],
        "low":    ["No financial data found", "No projections or forecasts", "Financial health entirely undisclosed"],
    },
}

def get_score_reasons(dim: str, score: float) -> list[str]:
    tier = "high" if score >= 6.5 else "medium" if score >= 3.5 else "low"
    return SCORE_REASONS.get(dim, {}).get(tier, ["Insufficient signal data"])


def ml_score(full_text: str) -> dict:
    text_lower = full_text.lower()
    word_count  = max(len(text_lower.split()), 1)
    raw = {}
    for dim, kws in SIGNAL_KEYWORDS.items():
        hits = sum(text_lower.count(kw) for kw in kws)
        raw[dim] = min(hits / word_count * 1000, 5.0)
    max_raw = max(raw.values()) or 1
    scores  = {k: round((v / max_raw) * 10, 1) for k, v in raw.items()}
    weights = {
        "problem": 0.15, "solution": 0.15, "market": 0.15,
        "traction": 0.20, "team": 0.15, "business_model": 0.10,
        "competition": 0.05, "financials": 0.05,
    }
    scores["composite"] = round(sum(scores[k] * weights[k] for k in weights), 1)
    return scores


# ── Groq helper ───────────────────────────────────────────────────────────────
def call_groq(prompt: str, max_tokens: int = 3000, temperature: float = 0.7) -> str:
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not found in .env file."
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json=payload,
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError:
        code = resp.status_code
        if code == 401:
            return "Error: Invalid Groq API key."
        if code == 429:
            return "Error: Rate limit hit. Wait a moment and retry."
        return f"Error: Groq HTTP {code}: {resp.text[:300]}"
    except Exception as e:
        return f"Error: {e}"


# ── Prompts ───────────────────────────────────────────────────────────────────
MAIN_PROMPT = """You are a seasoned startup investor (think Y-Combinator partner).
Below is content extracted from a startup pitch deck plus ML signal scores.

=== PITCH DECK CONTENT ===
{context}

=== ML SIGNAL SCORES (0-10) ===
{score_summary}
Composite Viability Score: {composite}/10

=== YOUR TASK ===
Produce EXACTLY the following six sections (use these exact headings):

## Executive Summary
3-sentence overview of what the startup does, who it serves, and the core value proposition.

## Investment Thesis
Why a VC might be excited — market timing, differentiation, founder-market fit.

## Key Strengths
Top 3 strengths. For each: bold the title, then 2-3 sentences of justification.

## Risk Factors
Top 3 risks. For each: bold the title, explain why it matters, suggest one concrete mitigation.

## Opportunities
Top 3 opportunities. For each: bold the title, explain the upside, suggest how to capture it.

## Verdict
One honest paragraph: would you invest at this stage? Why or why not? Be direct.

Be specific and data-driven. Do not add extra sections or preamble.
"""

COMPETITOR_PROMPT = """You are a startup research analyst.
Based on the pitch deck content below, identify real-world competitors and differentiation signals.

=== PITCH DECK CONTENT ===
{context}

Respond in EXACTLY this format (no preamble, no markdown outside the structure):

COMPETITORS:
- [Competitor 1]
- [Competitor 2]
- [Competitor 3]
- [Competitor 4]
- [Competitor 5]

DIFFERENTIATION:
- [Differentiator 1]
- [Differentiator 2]
- [Differentiator 3]

Only list real companies that exist. If the deck's domain is unclear, make educated guesses based on context. Be concise.
"""

INVESTOR_QS_PROMPT = """You are a sharp VC partner doing due diligence on a startup.
Based on this pitch deck content, list the 7 most important questions you would ask the founders.

=== PITCH DECK CONTENT ===
{context}

Respond with ONLY a numbered list of questions, one per line, no preamble:
1. [Question]
2. [Question]
...
"""

DD_PROMPT = """You are a due diligence analyst. Score this startup's investment readiness.

=== PITCH DECK CONTENT ===
{context}

=== ML SCORES ===
{score_summary}

Respond ONLY in this exact JSON format (no markdown, no backticks):
{{
  "investment_readiness": <0-100>,
  "market": <0-100>,
  "team": <0-100>,
  "financials": <0-100>,
  "traction": <0-100>,
  "product": <0-100>,
  "rationale": "<one sentence explanation>"
}}
"""

MENTOR_PROMPT = """You are a partner at {firm}, one of the world's top VC firms.
Evaluate this pitch deck with your firm's investment philosophy in mind.

=== PITCH DECK CONTENT ===
{context}

=== COMPOSITE SCORE ===
{composite}/10

Respond in EXACTLY this format:

DECISION: [Invest / Pass / Request More Info]

REASON:
[2-3 sentences explaining your decision from {firm}'s perspective]

SUGGESTED FIX:
[1-2 specific, actionable improvements the founder should make]

COMPARABLE:
[One portfolio company or known startup this reminds you of, and why]
"""

# ── Parsers ───────────────────────────────────────────────────────────────────
def parse_main_sections(text: str) -> dict:
    headings = [
        "Executive Summary", "Investment Thesis",
        "Key Strengths", "Risk Factors", "Opportunities", "Verdict"
    ]
    sections = {}
    for i, h in enumerate(headings):
        pattern = rf"##\s*{h}\s*\n(.*?)(?=##\s*(?:{'|'.join(headings[i+1:])})|\Z)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        sections[h] = m.group(1).strip() if m else ""
    if not any(sections.values()):
        sections["_raw"] = text
    return sections


def parse_competitors(text: str) -> tuple[list[str], list[str]]:
    competitors, diffs = [], []
    comp_section = re.search(r"COMPETITORS:(.*?)(?:DIFFERENTIATION:|$)", text, re.DOTALL)
    diff_section  = re.search(r"DIFFERENTIATION:(.*?)$", text, re.DOTALL)
    if comp_section:
        competitors = [l.strip("- ").strip() for l in comp_section.group(1).strip().splitlines() if l.strip().startswith("-")]
    if diff_section:
        diffs = [l.strip("- ").strip() for l in diff_section.group(1).strip().splitlines() if l.strip().startswith("-")]
    return competitors, diffs


def parse_dd_scores(text: str) -> dict | None:
    import json
    clean = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except Exception:
        return None


# ── UI helpers ────────────────────────────────────────────────────────────────
def get_bar_color(score: float) -> str:
    if score >= 6.5: return "#1a5c3a"
    if score >= 4.0: return "#7a5c00"
    return "#8b1a1a"

def get_tier_label(score: float) -> str:
    if score >= 6.5: return "Strong"
    if score >= 4.0: return "Moderate"
    return "Weak"

def get_verdict_style(composite: float) -> tuple[str, str, str]:
    if composite >= 7.5:
        return "#1a5c3a", "rgba(26,92,58,0.06)", "High Conviction"
    if composite >= 5.5:
        return "#7a5c00", "rgba(122,92,0,0.06)", "Promising"
    if composite >= 3.0:
        return "#b84c00", "rgba(184,76,0,0.06)", "Needs Work"
    return "#8b1a1a", "rgba(139,26,26,0.06)", "Early Stage"


# ── App ───────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeckLens · Pitch Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400;1,600&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=DM+Mono:wght@300;400;500&family=Lato:wght@300;400;700&display=swap');

:root {
    --cream:          #faf8f4;
    --cream-dark:     #f2ede4;
    --parchment:      #e8e0d0;
    --ink:            #1a1714;
    --ink-soft:       #3d3730;
    --ink-muted:      #7a7068;
    --ink-faint:      #b5aa9a;
    --rule:           rgba(26,23,20,0.10);
    --rule-strong:    rgba(26,23,20,0.20);
    --gold:           #c9a84c;
    --gold-soft:      rgba(201,168,76,0.12);
    --green-deep:     #1a5c3a;
    --green-soft:     rgba(26,92,58,0.08);
    --red-deep:       #8b1a1a;
    --red-soft:       rgba(139,26,26,0.08);
    --amber-deep:     #7a5c00;
    --amber-soft:     rgba(122,92,0,0.08);
    --radius-sm:      4px;
    --radius:         8px;
    --radius-lg:      14px;
    --radius-xl:      20px;
}

.stApp {
    background: var(--cream) !important;
    font-family: 'Lato', sans-serif !important;
    color: var(--ink) !important;
}
.stApp > header { background: transparent !important; }
#MainMenu, footer, .stDeployButton { display: none !important; }
.block-container { padding: 0 2.5rem 5rem !important; max-width: 1320px !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--cream-dark); }
::-webkit-scrollbar-thumb { background: var(--parchment); border-radius: 2px; }

[data-testid="stSidebar"] {
    background: var(--ink) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { font-family: 'Lato', sans-serif !important; color: var(--cream) !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: var(--radius) !important;
    color: var(--cream) !important;
}

.sidebar-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    font-weight: 400;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(250,248,244,0.35) !important;
    margin: 24px 0 8px;
}
.wordmark {
    font-family: 'Playfair Display', serif;
    font-size: 20px;
    font-weight: 600;
    letter-spacing: 0.01em;
    color: var(--cream);
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 0 22px;
    border-bottom: 1px solid rgba(255,255,255,0.10);
    margin-bottom: 24px;
}
.wordmark-badge {
    width: 28px; height: 28px;
    background: var(--gold);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Playfair Display', serif;
    font-size: 14px; font-weight: 700; color: var(--ink);
}

[data-testid="stFileUploader"] {
    background: white !important;
    border: 1.5px dashed var(--parchment) !important;
    border-radius: var(--radius-lg) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--gold) !important;
}

.stButton > button {
    background: var(--ink) !important;
    color: var(--cream) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'Lato', sans-serif !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 9px 20px !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.80 !important; }

[data-testid="stExpander"] {
    background: white !important;
    border: 1px solid var(--rule) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary { font-size: 12px !important; color: var(--ink-muted) !important; font-weight: 400 !important; font-family: 'DM Mono', monospace !important; letter-spacing: 0.04em !important; }
[data-testid="stExpander"] summary:hover { color: var(--ink) !important; }

hr { border: none !important; border-top: 1px solid var(--rule) !important; margin: 44px 0 !important; }

/* ─── TOP NAV ─── */
.top-nav {
    background: var(--ink);
    padding: 0 2.5rem;
    display: flex; align-items: center; justify-content: space-between;
    margin: 0 -2.5rem 0;
    position: sticky; top: 0; z-index: 100;
    height: 56px;
}
.nav-brand {
    font-family: 'Playfair Display', serif;
    font-size: 18px; font-weight: 600;
    color: var(--cream);
    letter-spacing: 0.01em;
    display: flex; align-items: center; gap: 10px;
}
.nav-brand-gem { color: var(--gold); font-size: 14px; }
.nav-tagline {
    font-family: 'DM Mono', monospace;
    font-size: 10px; color: rgba(250,248,244,0.38);
    letter-spacing: 0.12em; text-transform: uppercase;
}

/* ─── HERO ─── */
.hero-wrap {
    background: var(--ink);
    border-radius: var(--radius-xl);
    padding: 48px 52px;
    margin: 28px 0 36px;
    position: relative; overflow: hidden;
}
.hero-rule {
    position: absolute; top: 0; left: 52px; right: 52px; height: 1px;
    background: var(--gold); opacity: 0.35;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 9px; font-weight: 400;
    letter-spacing: 0.20em; text-transform: uppercase;
    color: var(--gold); margin-bottom: 18px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 32px; font-weight: 600;
    line-height: 1.2; color: var(--cream);
    letter-spacing: -0.01em; margin-bottom: 10px;
}
.hero-subtitle {
    font-size: 14px; color: rgba(250,248,244,0.52);
    margin-bottom: 36px; line-height: 1.65; font-weight: 300;
}
.hero-score-label {
    font-family: 'DM Mono', monospace;
    font-size: 9px; letter-spacing: 0.16em;
    text-transform: uppercase; color: rgba(250,248,244,0.38); margin-bottom: 10px;
}
.hero-score-number {
    font-family: 'Playfair Display', serif;
    font-size: 64px; font-weight: 700;
    line-height: 1; letter-spacing: -0.04em;
}
.hero-score-denom { font-size: 22px; color: rgba(250,248,244,0.30); font-weight: 300; }

/* ─── SECTION HEADERS ─── */
.section-header {
    display: flex; align-items: flex-start; gap: 14px;
    margin: 48px 0 24px; padding-bottom: 16px;
    border-bottom: 1px solid var(--rule);
}
.section-num {
    font-family: 'DM Mono', monospace;
    font-size: 10px; color: var(--gold);
    letter-spacing: 0.10em; padding-top: 4px;
    min-width: 24px;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 20px; font-weight: 600;
    color: var(--ink); letter-spacing: -0.01em;
}
.section-sub {
    font-size: 13px; color: var(--ink-muted); margin-top: 3px;
    font-weight: 300;
}

/* ─── SIGNAL CARDS ─── */
.signal-card {
    background: white;
    border: 1px solid var(--rule);
    border-radius: var(--radius-lg);
    padding: 22px;
    transition: box-shadow 0.2s, border-color 0.2s;
    height: 100%;
}
.signal-card:hover { border-color: var(--rule-strong); box-shadow: 0 4px 20px rgba(26,23,20,0.06); }
.signal-dim {
    font-family: 'DM Mono', monospace;
    font-size: 9px; font-weight: 400; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--ink-faint); margin-bottom: 18px;
}
.signal-score-val {
    font-family: 'Playfair Display', serif;
    font-size: 36px; font-weight: 700;
    line-height: 1; letter-spacing: -0.03em; margin-bottom: 4px;
}
.signal-denom { font-size: 14px; color: var(--ink-faint); font-weight: 300; }
.signal-tier-badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 3px; font-size: 10px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.06em; margin-bottom: 16px;
}
.bar-outer {
    height: 4px; background: var(--cream-dark);
    border-radius: 2px; overflow: hidden; margin-bottom: 4px;
}
.bar-inner { height: 100%; border-radius: 2px; }

/* ─── ANALYSIS CARDS ─── */
.a-card {
    background: white; border: 1px solid var(--rule);
    border-radius: var(--radius-lg); padding: 28px; height: 100%;
}
.a-card-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 9px; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--ink-faint); margin-bottom: 14px;
}
.a-card-body {
    font-size: 14px; line-height: 1.78;
    color: var(--ink-soft); font-weight: 400;
}
.a-card-body strong { font-weight: 700; color: var(--ink); }

/* ─── VERDICT CARD ─── */
.verdict-outer {
    border-radius: var(--radius-lg); padding: 26px 30px;
    border: 1px solid; height: 100%;
    position: relative; overflow: hidden;
}
.verdict-stripe {
    position: absolute; top: 0; left: 0; bottom: 0; width: 3px;
}

/* ─── COMPETITOR TABLE ─── */
.comp-table {
    background: white; border: 1px solid var(--rule);
    border-radius: var(--radius-lg); overflow: hidden;
}
.comp-header-row {
    padding: 13px 20px;
    border-bottom: 1px solid var(--rule);
    display: flex; align-items: center; justify-content: space-between;
}
.comp-header-label {
    font-family: 'DM Mono', monospace;
    font-size: 9px; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--ink-faint);
}
.comp-row {
    display: flex; align-items: center; padding: 13px 20px;
    border-bottom: 1px solid var(--rule);
    transition: background 0.12s;
}
.comp-row:last-child { border-bottom: none; }
.comp-row:hover { background: var(--cream); }
.comp-idx {
    font-family: 'DM Mono', monospace;
    font-size: 10px; color: var(--ink-faint);
    min-width: 22px;
}
.comp-name { font-size: 14px; font-weight: 400; color: var(--ink); flex: 1; }
.comp-tag {
    font-size: 10px; font-family: 'DM Mono', monospace;
    letter-spacing: 0.04em;
    padding: 3px 9px; border-radius: 3px;
    background: var(--red-soft); color: var(--red-deep);
}
.diff-row {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 13px 20px; border-bottom: 1px solid var(--rule);
}
.diff-row:last-child { border-bottom: none; }
.diff-mark {
    font-family: 'DM Mono', monospace;
    font-size: 12px; color: var(--green-deep);
    min-width: 16px; margin-top: 1px;
}
.diff-text { font-size: 13px; color: var(--ink-soft); line-height: 1.55; }

/* ─── QUESTIONS ─── */
.q-item {
    display: flex; align-items: flex-start; gap: 16px; padding: 18px 22px;
    background: white; border: 1px solid var(--rule);
    border-radius: var(--radius); margin-bottom: 8px;
    transition: border-color 0.15s;
}
.q-item:hover { border-color: var(--gold); }
.q-num {
    font-family: 'Playfair Display', serif;
    font-size: 18px; font-weight: 600;
    color: var(--gold); min-width: 24px; line-height: 1.5;
}
.q-text { font-size: 14px; color: var(--ink-soft); line-height: 1.65; }

/* ─── READINESS ─── */
.readiness-card {
    background: white; border: 1px solid var(--rule);
    border-radius: var(--radius-lg); padding: 30px; text-align: center;
}
.readiness-num {
    font-family: 'Playfair Display', serif;
    font-size: 64px; font-weight: 700;
    line-height: 1; letter-spacing: -0.04em;
}
.readiness-pct { font-size: 22px; color: var(--ink-faint); font-weight: 300; }
.readiness-note {
    font-size: 13px; color: var(--ink-muted); line-height: 1.65;
    margin-top: 14px; padding-top: 14px;
    border-top: 1px solid var(--rule); text-align: left;
}
.dd-row { margin-bottom: 16px; }
.dd-row-header { display: flex; justify-content: space-between; margin-bottom: 6px; }
.dd-row-label { font-size: 13px; color: var(--ink-soft); font-weight: 400; }
.dd-row-pct {
    font-family: 'DM Mono', monospace;
    font-size: 12px; font-weight: 500;
}
.dd-bar-outer { height: 5px; background: var(--cream-dark); border-radius: 2px; overflow: hidden; }
.dd-bar-inner { height: 100%; border-radius: 2px; }

/* ─── MENTOR ─── */
.mentor-card {
    background: white; border: 1px solid var(--rule);
    border-radius: var(--radius-lg); overflow: hidden;
}
.mentor-header-bar {
    background: var(--ink); padding: 16px 26px;
    display: flex; align-items: center; justify-content: space-between;
}
.mentor-firm {
    font-family: 'Playfair Display', serif;
    font-size: 15px; font-weight: 600; color: var(--cream);
}
.mentor-stamp {
    font-family: 'DM Mono', monospace;
    font-size: 9px; letter-spacing: 0.14em; color: rgba(250,248,244,0.32);
    text-transform: uppercase;
}
.mentor-body { padding: 28px; }
.mentor-decision-word {
    font-family: 'Playfair Display', serif;
    font-size: 36px; font-weight: 700;
    letter-spacing: -0.02em; line-height: 1;
    margin-bottom: 4px;
}
.mentor-section-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 9px; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--ink-faint);
    margin-bottom: 8px; margin-top: 22px;
}
.mentor-section-text {
    font-size: 14px; color: var(--ink-soft); line-height: 1.72;
}
.mentor-comparable-box {
    background: var(--cream); border: 1px solid var(--parchment);
    border-radius: var(--radius); padding: 14px 18px;
    font-size: 13px; color: var(--ink-soft); line-height: 1.6;
}

.raw-text {
    font-family: 'DM Mono', monospace; font-size: 11px;
    color: var(--ink-muted); line-height: 1.75;
    white-space: pre-wrap; word-break: break-word;
}

/* ─── LANDING ─── */
.feature-pill {
    background: white; border: 1px solid var(--rule);
    border-radius: var(--radius-lg); padding: 20px 24px;
    min-width: 175px; text-align: center;
}
.feature-pill-title {
    font-family: 'Playfair Display', serif;
    font-size: 14px; font-weight: 600; color: var(--ink); margin-bottom: 5px;
}
.feature-pill-sub {
    font-size: 12px; color: var(--ink-muted); font-weight: 300;
}

@media (max-width: 768px) {
    .block-container { padding: 0 1.2rem 3rem !important; }
    .hero-wrap { padding: 28px 24px; }
}
</style>
""", unsafe_allow_html=True)

# ── Top nav ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-nav">
    <div class="nav-brand">
        <span class="nav-brand-gem">◈</span>
        DeckLens
    </div>
    <div class="nav-tagline">Pitch Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="wordmark">
        <div class="wordmark-badge">D</div>
        DeckLens
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-eyebrow">Mentor Firm</div>', unsafe_allow_html=True)
    mentor_firm = st.selectbox(
        "Select VC firm",
        ["Sequoia", "Y Combinator", "Andreessen Horowitz", "Accel", "Tiger Global"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-eyebrow">About</div>', unsafe_allow_html=True)
    st.markdown(
        '<span style="font-size:12px;color:rgba(250,248,244,0.38);line-height:1.7;display:block;">Upload a PDF pitch deck and receive institutional-grade investment analysis powered by Groq AI.</span>',
        unsafe_allow_html=True,
    )

    if not GROQ_API_KEY:
        st.markdown("""
        <div style="background:rgba(139,26,26,0.15); border:1px solid rgba(139,26,26,0.30); border-radius:8px; padding:12px 14px; margin-top:18px;">
            <span style="color:#e87c7c; font-size:12px; font-family:'DM Mono',monospace;">GROQ_API_KEY missing</span>
        </div>
        """, unsafe_allow_html=True)


# ── Landing ────────────────────────────────────────────────────────────────────
if "uploaded" not in st.session_state:
    st.markdown("""
    <div style="padding: 68px 0 24px; text-align: center;">
        <div style="font-family:'DM Mono',monospace; font-size:9px; font-weight:400; letter-spacing:0.20em; text-transform:uppercase; color:#c9a84c; margin-bottom:20px;">
            AI-Powered Pitch Intelligence
        </div>
        <div style="font-family:'Playfair Display',serif; font-size:44px; font-weight:600; letter-spacing:-0.02em; color:#1a1714; line-height:1.18; margin-bottom:18px;">
            Evaluate Any Pitch Deck<br><em>Like a VC Partner</em>
        </div>
        <div style="font-size:15px; color:#7a7068; max-width:500px; margin:0 auto; line-height:1.72; font-weight:300;">
            ML signal scoring, competitor analysis, due diligence readiness, and simulated VC feedback — distilled from a single PDF.
        </div>
    </div>
    """, unsafe_allow_html=True)

uploaded = st.file_uploader("Upload pitch deck (PDF)", type=["pdf"], label_visibility="collapsed")

if not uploaded:
    st.markdown("""
    <div style="text-align:center; padding: 6px 0 44px;">
        <span style="font-size:12px; color:#b5aa9a; font-family:'DM Mono',monospace; letter-spacing:0.08em;">PDF only · Max 200MB</span>
    </div>
    <div style="display:flex; gap:14px; justify-content:center; flex-wrap:wrap; padding-bottom:48px;">
        <div class="feature-pill"><div class="feature-pill-title">ML Scoring</div><div class="feature-pill-sub">8-dimension viability model</div></div>
        <div class="feature-pill"><div class="feature-pill-title">Competitive Intel</div><div class="feature-pill-sub">Real competitor detection</div></div>
        <div class="feature-pill"><div class="feature-pill-title">VC Mentor Mode</div><div class="feature-pill-sub">Sequoia, a16z, YC simulated</div></div>
        <div class="feature-pill"><div class="feature-pill-title">Due Diligence</div><div class="feature-pill-sub">Investment readiness score</div></div>
    </div>
    """, unsafe_allow_html=True)

if uploaded and not GROQ_API_KEY:
    st.markdown("""
    <div style="background:#fff0f0; border:1px solid rgba(139,26,26,0.20); border-radius:12px; padding:18px 22px; margin-top:18px;">
        <span style="color:#8b1a1a; font-size:14px;">GROQ_API_KEY not found. Please add it to your .env file and restart.</span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if uploaded and GROQ_API_KEY:

    # ── 1. Extract ──────────────────────────────────────────────────────────
    progress_placeholder = st.empty()

    def show_progress(step: int):
        steps = [
            ("Extracting deck content", 1),
            ("Building signal index",   2),
            ("Evaluating dimensions",   3),
            ("Running AI analysis",     4),
            ("Generating report",       5),
        ]
        html = '<div style="background:white; border:1px solid rgba(26,23,20,0.10); border-radius:12px; padding:22px 28px; margin:20px 0;">'
        html += '<div style="font-family:\'DM Mono\',monospace; font-size:9px; font-weight:400; color:#b5aa9a; letter-spacing:0.18em; text-transform:uppercase; margin-bottom:16px;">Analysis in progress</div>'
        for i, (label, idx) in enumerate(steps):
            if idx < step:
                col, icon, wt = "#1a5c3a", "✓", "400"
            elif idx == step:
                col, icon, wt = "#c9a84c", "→", "700"
            else:
                col, icon, wt = "#b5aa9a", "·", "300"
            html += f'<div style="display:flex;align-items:center;gap:12px;padding:7px 0;font-size:13px;color:{col};font-weight:{wt};">'
            html += f'<span style="font-family:\'DM Mono\',monospace;font-size:12px;min-width:14px;">{icon}</span><span>{label}</span></div>'
        html += '</div>'
        progress_placeholder.markdown(html, unsafe_allow_html=True)

    show_progress(1)
    with st.spinner(""):
        chunks = extract_text_from_pdf(uploaded)

    if not chunks:
        st.markdown("""
        <div style="background:#fff8f8; border:1px solid rgba(139,26,26,0.20); border-radius:12px; padding:18px 22px;">
            <span style="color:#8b1a1a; font-size:14px;">No text found. The PDF may be scanned or image-only.</span>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    full_text = "\n\n".join(chunks)

    show_progress(2)
    scores = ml_score(full_text)
    dims   = [k for k in scores if k != "composite"]
    composite = scores["composite"]

    show_progress(3)
    with st.spinner(""):
        vectorizer, matrix = build_rag_index(chunks)
        context = retrieve(
            "problem solution market traction team business model competition financials risks opportunities",
            chunks, vectorizer, matrix, top_k=7,
        )
    score_summary = "\n".join(
        f"  {k.replace('_',' ').title()}: {v}/10"
        for k, v in scores.items() if k != "composite"
    )

    show_progress(4)
    raw_main   = call_groq(MAIN_PROMPT.format(context=context, score_summary=score_summary, composite=composite))
    comp_raw   = call_groq(COMPETITOR_PROMPT.format(context=context), max_tokens=800, temperature=0.4)
    qs_raw     = call_groq(INVESTOR_QS_PROMPT.format(context=context), max_tokens=600, temperature=0.5)
    dd_raw     = call_groq(DD_PROMPT.format(context=context, score_summary=score_summary), max_tokens=400, temperature=0.3)
    mentor_raw = call_groq(MENTOR_PROMPT.format(firm=mentor_firm, context=context, composite=composite), max_tokens=600, temperature=0.6)

    show_progress(5)
    progress_placeholder.empty()

    sections             = parse_main_sections(raw_main)
    competitors, diffs   = parse_competitors(comp_raw)
    dd                   = parse_dd_scores(dd_raw)
    verdict_border, verdict_bg, verdict_tier = get_verdict_style(composite)

    if composite >= 7.5:   comp_col = "#1a5c3a"
    elif composite >= 5.5: comp_col = "#7a5c00"
    elif composite >= 3.0: comp_col = "#b84c00"
    else:                  comp_col = "#8b1a1a"

    verdict_text = sections.get("Verdict", "") if "_raw" not in sections else ""
    deck_title   = uploaded.name.replace('.pdf','').replace('-',' ').replace('_',' ').title()

    # ═══════════════════════════════════════════════════════════════════════
    # HERO
    # ═══════════════════════════════════════════════════════════════════════
    hero_l, hero_r = st.columns([3, 1])
    with hero_l:
        st.markdown(f"""
        <div class="hero-wrap">
            <div class="hero-rule"></div>
            <div class="hero-eyebrow">Analysis Complete · {len(chunks)} pages extracted · Groq AI</div>
            <div class="hero-title">{deck_title}</div>
            <div class="hero-subtitle">
                AI investment analysis across 8 signal dimensions, competitive landscape,
                and VC-style due diligence readiness scoring.
            </div>
            <div style="display:flex; align-items:flex-end; gap:36px; flex-wrap:wrap;">
                <div>
                    <div class="hero-score-label">Composite Viability Score</div>
                    <div style="display:flex; align-items:baseline; gap:6px;">
                        <span class="hero-score-number" style="color:{comp_col};">{composite}</span>
                        <span class="hero-score-denom">/10</span>
                    </div>
                </div>
                <div style="flex:1; min-width:220px;">
                    <div class="hero-score-label">Signal Strength</div>
                    <div style="height:6px; background:rgba(255,255,255,0.10); border-radius:3px; overflow:hidden; margin-bottom:12px;">
                        <div style="height:100%; width:{composite*10}%; background:{comp_col}; border-radius:3px;"></div>
                    </div>
                    <span style="display:inline-block; padding:5px 14px; border-radius:3px; font-family:'DM Mono',monospace; font-size:10px; letter-spacing:0.08em; background:rgba(255,255,255,0.08); color:{comp_col}; border:1px solid {comp_col}40;">{verdict_tier}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with hero_r:
        if verdict_text:
            st.markdown(f"""
            <div class="verdict-outer" style="border-color:{verdict_border}30; background:{verdict_bg}; margin-top:28px;">
                <div class="verdict-stripe" style="background:{verdict_border};"></div>
                <div style="font-family:'DM Mono',monospace; font-size:9px; letter-spacing:0.16em; text-transform:uppercase; color:{verdict_border}; margin-bottom:12px; opacity:0.7;">Verdict</div>
                <div style="font-size:13px; line-height:1.72; color:#3d3730; font-style:italic; font-family:'Libre Baskerville',serif;">{verdict_text[:340]}{'...' if len(verdict_text)>340 else ''}</div>
            </div>
            """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # SIGNAL SCORES — with radar chart above cards
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <div class="section-num">01</div>
        <div>
            <div class="section-title">Signal Scores</div>
            <div class="section-sub">ML-derived strength across 8 investment dimensions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Radar chart (Chart.js via HTML component)
    radar_labels  = [d.replace("_", " ").title() for d in dims]
    radar_data    = [scores[d] for d in dims]
    radar_labels_js  = str(radar_labels)
    radar_data_js    = str(radar_data)

    radar_html = f"""
    <div style="display:flex; justify-content:center; margin-bottom:28px;">
      <div style="position:relative; width:420px; height:360px;">
        <canvas id="radarChart" role="img" aria-label="Radar chart of 8 investment signal dimensions">{', '.join(f'{l}: {v}' for l,v in zip(radar_labels, radar_data))}</canvas>
      </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
    (function() {{
      var ctx = document.getElementById('radarChart').getContext('2d');
      new Chart(ctx, {{
        type: 'radar',
        data: {{
          labels: {radar_labels_js},
          datasets: [{{
            label: 'Signal Score',
            data: {radar_data_js},
            backgroundColor: 'rgba(201,168,76,0.10)',
            borderColor: '#c9a84c',
            borderWidth: 1.5,
            pointBackgroundColor: '#c9a84c',
            pointBorderColor: '#faf8f4',
            pointBorderWidth: 2,
            pointRadius: 4,
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            r: {{
              min: 0, max: 10,
              ticks: {{
                stepSize: 2,
                color: '#b5aa9a',
                font: {{ family: 'DM Mono, monospace', size: 9 }},
                backdropColor: 'transparent',
              }},
              grid: {{ color: 'rgba(26,23,20,0.08)' }},
              angleLines: {{ color: 'rgba(26,23,20,0.08)' }},
              pointLabels: {{
                color: '#3d3730',
                font: {{ family: 'Lato, sans-serif', size: 11, weight: '400' }},
              }}
            }}
          }}
        }}
      }});
    }})();
    </script>
    """
    st.components.v1.html(radar_html, height=390)

    cols = st.columns(4)
    for i, dim in enumerate(dims):
        s = scores[dim]
        bar_color = get_bar_color(s)
        tier_lbl  = get_tier_label(s)

        if s >= 6.5:
            tier_bg, tier_fg = "rgba(26,92,58,0.08)", "#1a5c3a"
        elif s >= 4.0:
            tier_bg, tier_fg = "rgba(122,92,0,0.08)", "#7a5c00"
        else:
            tier_bg, tier_fg = "rgba(139,26,26,0.08)", "#8b1a1a"

        reasons = get_score_reasons(dim, s)
        reason_html = "".join(
            f'<div style="font-family:\'DM Mono\',monospace;font-size:11px;color:#7a7068;padding:4px 0;border-bottom:1px solid rgba(26,23,20,0.05);">{r}</div>'
            for r in reasons
        )

        with cols[i % 4]:
            st.markdown(f"""
            <div class="signal-card">
                <div class="signal-dim">{dim.replace('_',' ')}</div>
                <div style="display:flex; align-items:baseline; gap:4px; margin-bottom:6px;">
                    <span class="signal-score-val" style="color:{bar_color};">{s}</span>
                    <span class="signal-denom">/10</span>
                </div>
                <span class="signal-tier-badge" style="background:{tier_bg}; color:{tier_fg};">{tier_lbl}</span>
                <div class="bar-outer">
                    <div class="bar-inner" style="width:{s*10}%; background:{bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Signal evidence"):
                st.markdown(reason_html, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # CORE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    if "_raw" in sections:
        st.markdown("""
        <div class="section-header">
            <div class="section-num">02</div>
            <div><div class="section-title">AI Analysis</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="a-card"><div class="a-card-body">{sections["_raw"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="section-header">
            <div class="section-num">02</div>
            <div>
                <div class="section-title">Investment Analysis</div>
                <div class="section-sub">Generated from deck content and ML signals</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="a-card">
                <div class="a-card-eyebrow">Executive Summary</div>
                <div class="a-card-body" style="font-family:'Libre Baskerville',serif; font-size:13.5px;">{sections.get("Executive Summary","")}</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="a-card">
                <div class="a-card-eyebrow">Investment Thesis</div>
                <div class="a-card-body" style="font-style:italic; font-family:'Libre Baskerville',serif; font-size:13.5px;">{sections.get("Investment Thesis","")}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="a-card" style="margin-bottom:14px;">
            <div class="a-card-eyebrow">Key Strengths</div>
            <div class="a-card-body">{sections.get("Key Strengths","")}</div>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="a-card">
                <div class="a-card-eyebrow" style="color:#8b1a1a;">Risk Factors</div>
                <div class="a-card-body">{sections.get("Risk Factors","")}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="a-card">
                <div class="a-card-eyebrow" style="color:#1a5c3a;">Opportunities</div>
                <div class="a-card-body">{sections.get("Opportunities","")}</div>
            </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # COMPETITORS
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <div class="section-num">03</div>
        <div>
            <div class="section-title">Competitive Landscape</div>
            <div class="section-sub">Detected competitors and differentiation signals</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        comp_rows = "".join(
            f'<div class="comp-row"><span class="comp-idx">0{i+1}</span><span class="comp-name">{c}</span><span class="comp-tag">Competitor</span></div>'
            for i, c in enumerate(competitors)
        ) if competitors else '<div style="padding:20px;color:#b5aa9a;font-size:13px;font-family:\'DM Mono\',monospace;">No competitors detected</div>'
        st.markdown(f"""
        <div class="comp-table">
            <div class="comp-header-row">
                <span class="comp-header-label">Known Competitors</span>
                <span class="comp-header-label">{len(competitors)} detected</span>
            </div>
            {comp_rows}
        </div>""", unsafe_allow_html=True)

    with col2:
        diff_rows = "".join(
            f'<div class="diff-row"><div class="diff-mark">+</div><div class="diff-text">{d}</div></div>'
            for d in diffs
        ) if diffs else '<div style="padding:20px;color:#b5aa9a;font-size:13px;font-family:\'DM Mono\',monospace;">No differentiators detected</div>'
        st.markdown(f"""
        <div class="comp-table">
            <div class="comp-header-row">
                <span class="comp-header-label">Differentiation Signals</span>
                <span class="comp-header-label" style="color:#1a5c3a;">{len(diffs)} found</span>
            </div>
            {diff_rows}
        </div>""", unsafe_allow_html=True)

    with st.expander("Raw competitive analysis"):
        st.markdown(f'<div class="raw-text">{comp_raw}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # INVESTOR QUESTIONS
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <div class="section-num">04</div>
        <div>
            <div class="section-title">Due Diligence Questions</div>
            <div class="section-sub">Questions a VC partner would ask in a first meeting</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    lines = [l.strip() for l in qs_raw.strip().splitlines() if l.strip() and l.strip()[0].isdigit()]
    if lines:
        for q in lines:
            match = re.match(r"^(\d+)\.\s*(.*)", q)
            num  = match.group(1) if match else "·"
            text = match.group(2) if match else q
            st.markdown(f"""
            <div class="q-item">
                <div class="q-num">{num}</div>
                <div class="q-text">{text}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="a-card"><div class="a-card-body">{qs_raw}</div></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # DUE DILIGENCE READINESS — with bar chart
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <div class="section-num">05</div>
        <div>
            <div class="section-title">Investment Readiness</div>
            <div class="section-sub">Due diligence scoring across key dimensions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if dd:
        overall = dd.get("investment_readiness", 0)
        if overall >= 70:   rd_col = "#1a5c3a"
        elif overall >= 40: rd_col = "#7a5c00"
        else:               rd_col = "#8b1a1a"

        col_meta, col_chart, col_bars = st.columns([1, 1.4, 1.6])

        with col_meta:
            st.markdown(f"""
            <div class="readiness-card">
                <div style="font-family:'DM Mono',monospace; font-size:9px; letter-spacing:0.16em; text-transform:uppercase; color:#b5aa9a; margin-bottom:12px;">Readiness Score</div>
                <div style="display:flex;align-items:baseline;gap:4px;">
                    <span class="readiness-num" style="color:{rd_col};">{overall}</span>
                    <span class="readiness-pct">%</span>
                </div>
                <div style="height:5px;background:#e8e0d0;border-radius:2px;margin:18px 0 0;overflow:hidden;">
                    <div style="height:100%;width:{overall}%;background:{rd_col};border-radius:2px;"></div>
                </div>
                <div class="readiness-note">{dd.get("rationale","")}</div>
            </div>""", unsafe_allow_html=True)

        with col_chart:
            dims_dd = ["market", "team", "financials", "traction", "product"]
            dd_vals  = [dd.get(d, 0) for d in dims_dd]
            dd_labels_js = str([d.title() for d in dims_dd])
            dd_vals_js   = str(dd_vals)
            colors_js = str([
                "#1a5c3a" if v >= 70 else "#7a5c00" if v >= 40 else "#8b1a1a"
                for v in dd_vals
            ])

            bar_html = f"""
            <div style="position:relative; width:100%; height:{len(dims_dd)*52+80}px;">
              <canvas id="ddBar" role="img" aria-label="Horizontal bar chart of due diligence readiness dimensions">{', '.join(f'{l}: {v}%' for l,v in zip(dims_dd, dd_vals))}</canvas>
            </div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
            <script>
            (function() {{
              if (window.__ddBarDone) return; window.__ddBarDone = true;
              var ctx = document.getElementById('ddBar').getContext('2d');
              new Chart(ctx, {{
                type: 'bar',
                data: {{
                  labels: {dd_labels_js},
                  datasets: [{{
                    label: 'Score',
                    data: {dd_vals_js},
                    backgroundColor: {colors_js},
                    borderRadius: 3,
                    borderSkipped: false,
                  }}]
                }},
                options: {{
                  indexAxis: 'y',
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {{ legend: {{ display: false }} }},
                  scales: {{
                    x: {{
                      min: 0, max: 100,
                      grid: {{ color: 'rgba(26,23,20,0.06)' }},
                      ticks: {{
                        color: '#b5aa9a',
                        font: {{ family: 'DM Mono, monospace', size: 9 }},
                        callback: (v) => v + '%'
                      }},
                      border: {{ color: 'rgba(26,23,20,0.10)' }}
                    }},
                    y: {{
                      grid: {{ display: false }},
                      ticks: {{
                        color: '#3d3730',
                        font: {{ family: 'Lato, sans-serif', size: 11 }}
                      }},
                      border: {{ color: 'rgba(26,23,20,0.10)' }}
                    }}
                  }}
                }}
              }});
            }})();
            </script>
            """
            st.components.v1.html(bar_html, height=len(dims_dd)*52+100)

        with col_bars:
            bar_html2 = '<div class="a-card" style="padding:22px;">'
            for d in dims_dd:
                pct = dd.get(d, 0)
                if pct >= 70:   bc = "#1a5c3a"
                elif pct >= 40: bc = "#7a5c00"
                else:           bc = "#8b1a1a"
                bar_html2 += f"""
                <div class="dd-row">
                    <div class="dd-row-header">
                        <span class="dd-row-label">{d.title()}</span>
                        <span class="dd-row-pct" style="color:{bc};">{pct}%</span>
                    </div>
                    <div class="dd-bar-outer">
                        <div class="dd-bar-inner" style="width:{pct}%; background:{bc};"></div>
                    </div>
                </div>"""
            bar_html2 += "</div>"
            st.markdown(bar_html2, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#fffbf0; border:1px solid rgba(122,92,0,0.20); border-radius:12px; padding:18px 22px;">
            <span style="color:#7a5c00; font-size:14px;">Could not parse readiness scores.</span>
        </div>""", unsafe_allow_html=True)
        with st.expander("Raw response"):
            st.code(dd_raw)

    # ═══════════════════════════════════════════════════════════════════════
    # MENTOR
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(f"""
    <div class="section-header">
        <div class="section-num">06</div>
        <div>
            <div class="section-title">VC Mentor · {mentor_firm}</div>
            <div class="section-sub">Simulated partner feedback from {mentor_firm}'s investment lens</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    def strip_html(text: str) -> str:
        if not text:
            return ""
        text = unescape(text)
        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("<div>", "")
        text = text.replace("</div>", "")
        return text.strip()
    
    with st.expander("DEBUG"):
        st.code(mentor_raw)

    decision_match = re.search(
    r"DECISION:\s*(.+?)(?=\n[A-Z ]+:|$)",
    mentor_raw,
    re.DOTALL | re.IGNORECASE
    )
    
    reason_match   = re.search(r"REASON:\s*\n(.*?)(?=SUGGESTED FIX:|$)", mentor_raw, re.DOTALL | re.IGNORECASE)
    fix_match      = re.search(r"SUGGESTED FIX:\s*\n(.*?)(?=COMPARABLE:|$)", mentor_raw, re.DOTALL | re.IGNORECASE)
    comp_match     = re.search(r"COMPARABLE:\s*\n(.*?)$", mentor_raw, re.DOTALL | re.IGNORECASE)

    decision   = strip_html(decision_match.group(1).strip()) if decision_match else "Unknown"
    reason     = strip_html(reason_match.group(1).strip())   if reason_match   else ""
    fix        = strip_html(fix_match.group(1).strip())      if fix_match      else ""
    comparable = strip_html(comp_match.group(1).strip())     if comp_match     else ""

    if "invest" in decision.lower():
        dec_col = "#1a5c3a"
    elif "pass" in decision.lower():
        dec_col = "#8b1a1a"
    else:
        dec_col = "#7a5c00"

    import html as html_lib
    reason_safe     = html_lib.escape(reason)
    fix_safe        = html_lib.escape(fix)
    comparable_safe = html_lib.escape(comparable)
    decision_safe   = html_lib.escape(decision)

    reason_html     = f'<div class="mentor-section-eyebrow">Decision Rationale</div><div class="mentor-section-text" style="font-family:\'Libre Baskerville\',serif; font-style:italic; font-size:13.5px;">{reason_safe}</div>' if reason else ""
    fix_html        = f'<div class="mentor-section-eyebrow">Suggested Improvements</div><div class="mentor-section-text">{fix_safe}</div>' if fix else ""
    comparable_html = f'<div class="mentor-section-eyebrow">Comparable Investment</div><div class="mentor-comparable-box">{comparable_safe}</div>' if comparable else ""

    mentor_html = f"""
    <div class="mentor-card">
        <div class="mentor-header-bar">
            <span class="mentor-firm">{mentor_firm} · Partner Memo</span>
            <span class="mentor-stamp">AI Simulation</span>
        </div>
        <div class="mentor-body">
            <div style="padding-bottom:22px; border-bottom:1px solid rgba(26,23,20,0.08);">
                <div style="font-family:'DM Mono',monospace; font-size:9px; letter-spacing:0.16em; text-transform:uppercase; color:#b5aa9a; margin-bottom:8px;">Diligence Decision</div>
                <div class="mentor-decision-word" style="color:{dec_col};">{decision_safe}</div>
            </div>
            {reason_html}
            {fix_html}
            {comparable_html}
        </div>
    </div>
    """
    st.markdown(mentor_html, unsafe_allow_html=True)

    if not any([reason, fix, comparable]):
        st.markdown(f'<div class="a-card"><div class="a-card-body">{mentor_raw}</div></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # RAW TEXT
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    with st.expander("View extracted pitch deck text"):
        st.markdown(
            f'<div class="raw-text">{full_text[:8000]}{"...[truncated]" if len(full_text) > 8000 else ""}</div>',
            unsafe_allow_html=True,
        )