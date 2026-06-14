"""
Patent Claim Analyzer v3
構文解析 + グラフ理論 | マインドマップ | 複数特許対比表
"""

import re, io
import numpy as np
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
from collections import defaultdict

try:
    import pdfplumber
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_OK = True
except ImportError:
    SBERT_OK = False

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patent Claim Analyzer",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS — dark sidebar + clean tech main
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Reset / base ─────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
[data-testid="stAppViewContainer"] { background: #EEF2FF; }
[data-testid="block-container"] { padding-top: 16px !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Dark sidebar ─────────────────────────────── */
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #07101F 0%, #0A1628 100%);
    padding: 0;
    border-right: 1px solid rgba(37,99,235,0.18);
}
section[data-testid="stSidebar"] .block-container { padding: 0 !important; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #94A3B8; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #F1F5F9 !important; }
section[data-testid="stSidebar"] textarea {
    background: #0C1526 !important;
    color: #E2E8F0 !important;
    border: 1px solid rgba(37,99,235,0.22) !important;
    font-size: 11px !important;
    font-family: 'Menlo','Consolas',monospace !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: linear-gradient(135deg,#0C1526,#0F1C33);
    border: 1px dashed rgba(37,99,235,0.32);
    border-radius: 10px; padding: 8px;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] small { color: #475569 !important; }
section[data-testid="stSidebar"] [data-baseweb="radio"] label { color: #94A3B8 !important; }
section[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(135deg,#2563EB,#3B82F6) !important;
    border: none !important; color: white !important;
    font-weight: 700 !important; border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
    letter-spacing: 0.2px !important;
}
section[data-testid="stSidebar"] button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    color: #64748B !important; border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px;
}

/* ── Sidebar logo ─────────────────────────────── */
.sb-logo {
    padding: 22px 20px 16px;
    border-bottom: 1px solid rgba(37,99,235,0.18);
    margin-bottom: 20px;
    position: relative;
}
.sb-logo::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg,#2563EB,#06B6D4,#7C3AED);
}
.sb-logo-icon { font-size: 22px; margin-bottom: 8px; }
.sb-logo-title {
    font-size: 14px; font-weight: 800;
    background: linear-gradient(135deg,#60A5FA,#38BDF8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -0.3px;
}
.sb-logo-sub {
    font-size: 10px; color: #334155 !important;
    letter-spacing: 1px; text-transform: uppercase; margin-top: 4px;
}
.sb-section { padding: 0 16px 16px; }
.sb-label {
    font-size: 10px; font-weight: 700; color: #334155 !important;
    text-transform: uppercase; letter-spacing: 0.9px; margin-bottom: 10px;
}

/* Patent list item */
.pat-item {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px; margin-bottom: 6px;
}
.pat-item-dot {
    width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0;
}
.pat-item-name { font-size: 11px; color: #CBD5E1 !important; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pat-item-count {
    font-size: 10px; color: #475569 !important;
    background: rgba(255,255,255,0.06); padding: 2px 7px; border-radius: 4px;
}

/* ── Hero header strip ─────────────────────────── */
.hero-strip {
    background: linear-gradient(135deg,#0C1A44 0%,#1246B0 55%,#0B2160 100%);
    border-radius: 14px; padding: 26px 32px;
    margin-bottom: 20px; position: relative; overflow: hidden;
    box-shadow: 0 8px 40px rgba(37,99,235,0.28);
}
.hero-strip::before {
    content: '';
    position: absolute; top: -70px; right: -50px;
    width: 220px; height: 220px;
    background: radial-gradient(circle,rgba(59,130,246,0.18),transparent 70%);
    border-radius: 50%;
}
.hero-strip::after {
    content: '';
    position: absolute; bottom: -50px; left: 25%;
    width: 200px; height: 200px;
    background: radial-gradient(circle,rgba(6,182,212,0.12),transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 24px; font-weight: 900; color: #FFFFFF;
    letter-spacing: -0.8px; margin-bottom: 5px;
    position: relative; z-index: 1;
}
.hero-title span {
    background: linear-gradient(90deg,#60A5FA,#38BDF8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub { font-size: 12px; color: rgba(255,255,255,0.48); position: relative; z-index: 1; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(59,130,246,0.22);
    border: 1px solid rgba(59,130,246,0.45);
    color: #93C5FD; font-size: 11px; font-weight: 700;
    padding: 5px 14px; border-radius: 20px;
    position: relative; z-index: 1; margin-top: 14px;
    letter-spacing: 0.2px;
}
.hero-ghost {
    position: absolute; right: 32px; top: 50%;
    transform: translateY(-50%);
    font-size: 72px; font-weight: 900; color: rgba(255,255,255,0.05);
    font-family: 'SF Mono','Menlo',monospace; line-height: 1; z-index: 0;
    letter-spacing: -4px;
}

/* ── Metric cards ──────────────────────────────── */
.metric-card {
    background: #FFFFFF;
    border: 1px solid rgba(226,232,240,0.9);
    border-radius: 12px; padding: 20px 22px;
    position: relative; overflow: hidden;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.mc-blue::before   { background: linear-gradient(90deg,#2563EB,#60A5FA); }
.mc-cyan::before   { background: linear-gradient(90deg,#0891B2,#22D3EE); }
.mc-green::before  { background: linear-gradient(90deg,#059669,#34D399); }
.mc-purple::before { background: linear-gradient(90deg,#7C3AED,#A78BFA); }
.mc-blue   { box-shadow: 0 4px 24px rgba(37,99,235,0.10); }
.mc-cyan   { box-shadow: 0 4px 24px rgba(8,145,178,0.10); }
.mc-green  { box-shadow: 0 4px 24px rgba(5,150,105,0.10); }
.mc-purple { box-shadow: 0 4px 24px rgba(124,58,237,0.10); }
.mc-label  { font-size: 10px; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.9px; margin-bottom: 8px; }
.mc-value  { font-size: 38px; font-weight: 800; line-height: 1; font-family: 'SF Mono','Menlo',monospace; margin-bottom: 6px; }
.mc-value.blue   { color: #2563EB; }
.mc-value.cyan   { color: #0891B2; }
.mc-value.green  { color: #059669; }
.mc-value.purple { color: #7C3AED; }
.mc-sub { font-size: 11px; color: #64748B; }

/* ── Tabs ──────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 2px solid #E2E8F0;
    gap: 0; background: #FFFFFF;
    border-radius: 12px 12px 0 0;
    padding: 0 8px;
    box-shadow: 0 2px 10px rgba(15,23,42,0.05);
}
[data-testid="stTabs"] button {
    font-size: 13px !important; font-weight: 500 !important;
    color: #64748B !important; padding: 13px 26px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
}
[data-testid="stTabs"] button:hover { color: #2563EB !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2563EB !important; border-bottom-color: #2563EB !important;
    font-weight: 700 !important;
    background: rgba(37,99,235,0.05) !important;
}

/* ── Claim chart table ─────────────────────────── */
.ct-wrap {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 12px; overflow-x: auto;
    box-shadow: 0 4px 24px rgba(15,23,42,0.07);
}
.ct { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.ct thead tr { background: linear-gradient(135deg,#1B3A8A 0%,#2563EB 100%); }
.ct th {
    color: rgba(255,255,255,0.92) !important; font-weight: 600;
    padding: 13px 16px; border-bottom: none;
    border-right: 1px solid rgba(255,255,255,0.12); text-align: center;
    white-space: nowrap; min-width: 90px; position: sticky; top: 0;
    background: transparent; letter-spacing: 0.3px;
}
.ct th.elem-col { text-align: left; min-width: 240px; max-width: 380px; }
.ct td {
    padding: 11px 16px; border-bottom: 1px solid #F1F5F9;
    border-right: 1px solid #F8FAFC;
    text-align: center; vertical-align: middle;
}
.ct td.elem-cell {
    text-align: left; color: #1E293B; font-weight: 500;
    white-space: normal; word-break: break-word;
    min-width: 260px; max-width: 480px; line-height: 1.65; font-size: 12px;
}
.ct tbody tr:nth-child(even) td { background: #FAFBFF; }
.ct tbody tr:hover td { background: #EFF6FF !important; }
.ct tr:last-child td { border-bottom: none; }
.c-direct  { color: #059669; font-weight: 700; font-size: 18px; }
.c-added   { color: #2563EB; font-weight: 700; font-size: 14px; }
.c-inherit { color: #CBD5E1; font-size: 15px; }
.c-none    { color: #E2E8F0; font-size: 15px; }

/* Badges — ライト背景用 */
.badge { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 9px; font-weight: 700; letter-spacing: 0.5px; margin-top: 4px; }
.b-ind   { background: rgba(37,99,235,0.10); color: #1D4ED8; border: 1px solid rgba(37,99,235,0.20); }
.b-dep   { background: #F1F5F9; color: #475569; border: 1px solid #E2E8F0; }
.b-multi { background: rgba(124,58,237,0.10); color: #5B21B6; border: 1px solid rgba(124,58,237,0.22); }
/* Badges — ダーク背景（テーブルヘッダー）用 */
.ct thead .b-ind   { background: rgba(255,255,255,0.18); color: #fff; border: 1px solid rgba(255,255,255,0.35); }
.ct thead .b-dep   { background: rgba(255,255,255,0.10); color: rgba(255,255,255,0.78); border: 1px solid rgba(255,255,255,0.22); }
.ct thead .b-multi { background: rgba(167,139,250,0.30); color: #EDE9FE; border: 1px solid rgba(167,139,250,0.55); }

/* Breadth score bar */
.bw-wrap { width:80%; height:3px; background:rgba(255,255,255,0.18); border-radius:2px; margin:6px auto 3px; }
.bw-bar  { height:3px; border-radius:2px; transition:width 0.4s; }
.bw-lbl  { font-size:9px; font-weight:700; display:block; margin-top:2px; }

/* Legend */
.legend {
    display: flex; gap: 24px; flex-wrap: wrap;
    font-size: 11px; color: #64748B;
    padding: 12px 18px; border-top: 1px solid #F1F5F9;
    background: #FAFBFF; border-radius: 0 0 12px 12px;
}

/* ── Comparison table ──────────────────────────── */
.cov-high { color: #059669; font-weight: 700; }
.cov-mid  { color: #2563EB; font-weight: 600; }
.cov-low  { color: #94A3B8; }

/* ── Panel card ────────────────────────────────── */
.panel {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 12px; padding: 20px 24px;
    box-shadow: 0 4px 24px rgba(15,23,42,0.07);
}
.panel-title { font-size: 14px; font-weight: 700; color: #0F172A; margin-bottom: 4px; }
.panel-sub   { font-size: 11.5px; color: #94A3B8; margin-bottom: 14px; line-height: 1.6; }

/* ── Empty state ───────────────────────────────── */
.empty { text-align: center; padding: 120px 0; user-select: none; }
.empty-icon { font-size: 56px; margin-bottom: 20px; }
.empty-title { font-size: 16px; font-weight: 700; color: #94A3B8; margin-bottom: 8px; }
.empty-sub { font-size: 12.5px; color: #CBD5E1; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────────────────────
TERMINATORS = [
    'と、', 'を備え、', 'を備える、', 'を有し、', 'を有する、',
    'からなり、', 'からなる、', 'によって、', 'により、',
    'であって、', 'において、', 'を含み、', 'を含む、',
    'を記憶し、', 'を実行し、', 'を実現し、',
    'を受け付け、', 'を変換し、', 'を生成し、',
    'を算出し、', 'を出力し、', 'を行い、',
]
CONNECTIVE_STARTS = (
    'を備える', 'からなる', 'において', 'であって',
    'を実現する', 'の情報処理', 'の装置', 'の方法',
    'の端末', 'のシステム', 'のプログラム',
)
DEP_RE = re.compile(
    r'請求項\s*(\d+)'
    r'(?:\s*(?:または|若しくは|もしくは|又は)[、]?\s*(?:請求項\s*)?(\d+))*'
    r'\s*[、に]?(?:記載の|おける|係る|に係る)'
)
CLAIMS_SECTION_RE = re.compile(
    r'(?:【書類名】\s*特許請求の範囲|特\s*許\s*請\s*求\s*の\s*範\s*囲)'
    r'(.*?)(?=【書類名】|【発明の詳細な説明】|明\s*細\s*書|\Z)',
    re.DOTALL,
)

# ─────────────────────────────────────────────────────────────
# PDF 抽出
# ─────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    if not PDF_OK:
        return ""
    parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)


def extract_claims_section(raw: str) -> str:
    m = CLAIMS_SECTION_RE.search(raw)
    if m:
        return m.group(1).strip()
    if re.search(r'【請求項\d+】', raw):
        idx = re.search(r'【請求項1】', raw)
        return raw[idx.start():].strip() if idx else raw.strip()
    return raw.strip()


# ─────────────────────────────────────────────────────────────
# パーサー
# ─────────────────────────────────────────────────────────────
def parse_claims(text: str) -> dict:
    claims = {}
    blocks = re.split(r'【請求項(\d+)】', text.strip())
    i = 1
    while i < len(blocks) - 1:
        num = int(blocks[i])
        body = blocks[i + 1].strip()
        i += 2
        parents = []
        m = DEP_RE.search(body)
        if m:
            parents = [int(x) for x in re.findall(r'\d+', m.group(0))]
        is_ind = len(parents) == 0
        claims[num] = {
            'num': num, 'body': body,
            'parents': parents,
            'is_independent': is_ind,
            'is_multi_dep': len(parents) > 1,
            'elements': _extract_elements(body, is_ind),
            'type': _claim_type(body),
        }
    return claims


def _extract_elements(body: str, is_independent: bool) -> list:
    if not is_independent:
        body = DEP_RE.sub('', body).strip()
    positions = []
    for term in TERMINATORS:
        for m in re.finditer(re.escape(term), body):
            positions.append((m.start(), m.end()))
    positions.sort()
    elements = []
    prev = 0
    for start, end in positions:
        chunk = body[prev:end].strip()
        if chunk and len(chunk) > 3 and not chunk.startswith(CONNECTIVE_STARTS):
            elements.append(chunk)
        prev = end
    remainder = body[prev:].strip()
    remainder = re.sub(r'[、。]?\s*[぀-ヿ一-鿿\w]+[。]\s*$', '', remainder).strip()
    if remainder and len(remainder) > 3 and not remainder.startswith(CONNECTIVE_STARTS):
        elements.append(remainder)
    return elements


def _claim_type(body: str) -> str:
    if re.search(r'方法|工程|ステップ', body): return '方法'
    if re.search(r'システム|サーバ', body):    return 'システム'
    if re.search(r'プログラム|コンピュータ可読', body): return 'プログラム'
    return '装置'


def resolve_inherited(claims: dict) -> dict:
    resolved = {}
    def _r(n):
        if n in resolved: return resolved[n]
        c = claims[n]
        if c['is_independent']:
            resolved[n] = list(c['elements']); return resolved[n]
        base = []
        for p in c['parents']:
            if p in claims: base = _r(p)[:]; break
        resolved[n] = base + c['elements']; return resolved[n]
    for n in sorted(claims.keys()): _r(n)
    return resolved


# ─────────────────────────────────────────────────────────────
# グラフ構築
# ─────────────────────────────────────────────────────────────
def build_elem_graph(claims: dict, resolved: dict):
    G = nx.Graph()
    label_to_id = {}
    for num, c in claims.items():
        for elem in c['elements']:
            lbl = elem[:30] + '…' if len(elem) > 30 else elem
            if lbl not in label_to_id:
                eid = f"E{len(label_to_id)}"
                label_to_id[lbl] = eid
                G.add_node(eid, label=lbl, claims=set())
            G.nodes[label_to_id[lbl]]['claims'].add(num)
    for num, elems in resolved.items():
        ids = []
        for e in elems:
            lbl = e[:30] + '…' if len(e) > 30 else e
            if lbl in label_to_id:
                G.nodes[label_to_id[lbl]]['claims'].add(num)
                ids.append(label_to_id[lbl])
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                if G.has_edge(ids[i], ids[j]):
                    G[ids[i]][ids[j]]['weight'] += 1
                else:
                    G.add_edge(ids[i], ids[j], weight=1)
    return G


# ─────────────────────────────────────────────────────────────
# クレーム広狭スコア (USPTO手法: 語数 + 限定語密度 + 要素数)
# ─────────────────────────────────────────────────────────────
LIMITER_PATTERNS = [
    r'特定の', r'所定の', r'予め定め', r'あらかじめ定め',
    r'[0-9０-９]+\s*(?:以上|以下|未満|超過)',
    r'[0-9０-９]+\s*(?:%|％|mm|cm|m\b|kg|g\b)',
    r'[0-9０-９]+\s*〜\s*[0-9０-９]+',
    r'から.{1,6}まで', r'のうち[のい]?ず?れか',
    r'少なくとも[一1]つ', r'ただし', r'但し',
    r'のみ(?=[をがはに、。])',
]

def compute_breadth_scores(claims: dict) -> dict:
    scores = {}
    for num, c in claims.items():
        body = c['body']
        chars = len(re.sub(r'\s', '', body))
        n_lim = sum(len(re.findall(p, body)) for p in LIMITER_PATTERNS)
        n_elem = len(c['elements'])
        char_s = max(0.0, 1 - chars / 600) * 40
        lim_s  = max(0.0, 1 - n_lim / 8)  * 35
        elem_s = max(0.0, 1 - n_elem / 8)  * 25
        raw    = char_s + lim_s + elem_s
        level  = '広' if raw >= 62 else ('中' if raw >= 35 else '狭')
        color  = '#059669' if level == '広' else ('#D97706' if level == '中' else '#DC2626')
        scores[num] = {
            'score': round(raw), 'level': level, 'color': color,
            'chars': chars, 'n_lim': n_lim, 'n_elem': n_elem,
        }
    return scores


# ─────────────────────────────────────────────────────────────
# inner-claim 要素間依存グラフ (FLAN-Graph ACL 2024 手法)
# ─────────────────────────────────────────────────────────────
def build_inner_dag(claim: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    elems = claim['elements']
    for i, e in enumerate(elems):
        short = re.sub(r'^前記|^その|^当該|^上記', '', e)
        short = re.split(r'[をがはにでのと、。]', short)[0][:18]
        G.add_node(i, text=e, label=f'E{i+1}', short=short)
    for j in range(1, len(elems)):
        refs = re.findall(
            r'前記([぀-ヿ一-鿿\w]{2,20}?)(?=[をがはにでのと、。])',
            elems[j],
        )
        matched = set()
        for ref in refs:
            for i in range(j - 1, -1, -1):
                if ref in elems[i] and i not in matched:
                    G.add_edge(i, j, ref=ref)
                    matched.add(i)
                    break
    return G


def plot_inner_dag(claim: dict, claim_num: int) -> go.Figure:
    G = build_inner_dag(claim)
    n = len(G.nodes())
    if n == 0:
        return go.Figure()

    xs = list(range(n))
    ys = [0] * n

    fig = go.Figure()

    # 依存エッジ — 上向き放物線アーチ
    for (src, dst) in G.edges():
        x0, x1 = xs[src], xs[dst]
        t = np.linspace(0, 1, 40)
        mid = (x0 + x1) / 2
        arc_h = max(0.3, abs(x1 - x0) * 0.4)
        bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * mid + t ** 2 * x1
        by = (1 - t) ** 2 * 0  + 2 * (1 - t) * t * arc_h + t ** 2 * 0
        fig.add_trace(go.Scatter(
            x=list(bx) + [None], y=list(by) + [None],
            mode='lines',
            line=dict(color='#2563EB', width=1.5, dash='dot'),
            hoverinfo='skip', showlegend=False,
        ))
        # 矢印代わりの三角マーカー
        fig.add_annotation(
            x=x1, y=0, ax=bx[-4], ay=by[-4],
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowwidth=1.5, arrowcolor='#2563EB',
        )

    # ノード
    node_x = [xs[i] for i in G.nodes()]
    node_y = [ys[i] for i in G.nodes()]
    node_short = [G.nodes[i]['short'] for i in G.nodes()]
    node_label = [G.nodes[i]['label'] for i in G.nodes()]
    in_deg = [G.in_degree(i) for i in G.nodes()]
    colors = ['#2563EB' if d == 0 else '#7C3AED' for d in in_deg]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=28, color=colors,
                    line=dict(width=2, color='white')),
        text=node_label,
        textposition='middle center',
        textfont=dict(size=10, color='white', family='monospace'),
        customdata=node_short,
        hovertemplate='<b>%{text}</b><br>%{customdata}<extra></extra>',
        showlegend=False,
    ))

    # 要素名ラベル（ノード下）
    for i in G.nodes():
        fig.add_annotation(
            x=xs[i], y=-0.22, text=G.nodes[i]['short'],
            showarrow=False, font=dict(size=9, color='#475569'),
            xref='x', yref='y',
        )

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.7, n - 0.3]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.45, 0.85]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        dragmode=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# SBERT (multilingual-e5-small) ローダー
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_sbert():
    if not SBERT_OK:
        return None
    try:
        return SentenceTransformer('intfloat/multilingual-e5-small')
    except Exception:
        return None


def sbert_sim(e1: str, e2: str, model) -> float:
    if model is None:
        return 0.0
    embs = model.encode(
        [f'passage: {e1}', f'passage: {e2}'],
        normalize_embeddings=True,
    )
    return float(np.dot(embs[0], embs[1]))


def compute_metrics(claims: dict, elem_G) -> dict:
    n_ind = sum(1 for c in claims.values() if c['is_independent'])
    # max depth via simple BFS
    children = defaultdict(list)
    for n, c in claims.items():
        for p in c['parents']:
            if p in claims: children[p].append(n)
    roots = [n for n, c in claims.items() if c['is_independent']]
    max_depth = 0
    queue = [(r, 0) for r in roots]
    while queue:
        node, d = queue.pop(0)
        max_depth = max(max_depth, d)
        for kid in children.get(node, []):
            queue.append((kid, d+1))
    all_e = set()
    for c in claims.values(): all_e.update(c['elements'])
    return {
        'n_claims': len(claims),
        'n_ind': n_ind,
        'n_dep': len(claims) - n_ind,
        'n_multi': sum(1 for c in claims.values() if c['is_multi_dep']),
        'max_depth': max_depth,
        'n_elements': len(all_e),
    }


# ─────────────────────────────────────────────────────────────
# マインドマップ レイアウト
# ─────────────────────────────────────────────────────────────
def compute_tree_pos(claims: dict) -> dict:
    """Reingold-Tilford 風 top-down ツリーレイアウト"""
    children = defaultdict(list)
    roots = sorted([n for n, c in claims.items() if c['is_independent']])
    for n, c in claims.items():
        if not c['is_independent'] and c['parents']:
            children[c['parents'][0]].append(n)
    for k in children: children[k].sort()

    def leaves(n):
        kids = children.get(n, [])
        return max(1, sum(leaves(k) for k in kids))

    pos = {}
    def place(n, x_left, y, width):
        pos[n] = (x_left + width / 2, y)
        kids = children.get(n, [])
        if not kids: return
        total = leaves(n)
        cur = x_left
        for k in kids:
            kw = width * leaves(k) / total
            place(k, cur, y - 2.2, kw)
            cur += kw

    total_leaves = sum(leaves(r) for r in roots)
    cur = 0.0
    for r in roots:
        rw = len(claims) * 1.6 * leaves(r) / max(total_leaves, 1)
        place(r, cur, 0, rw)
        cur += rw + 1.0

    return pos


def bezier_curve(x0, y0, x1, y1, n=28):
    """S字ベジェ曲線の点列を返す"""
    t = np.linspace(0, 1, n)
    cy = (y0 + y1) / 2
    bx = (1-t)**3*x0 + 3*(1-t)**2*t*x0 + 3*(1-t)*t**2*x1 + t**3*x1
    by = (1-t)**3*y0 + 3*(1-t)**2*t*cy + 3*(1-t)*t**2*cy + t**3*y1
    return list(bx) + [None], list(by) + [None]


def _elem_keyword(elem: str) -> str:
    e = re.sub(r'^前記|^その|^当該|^上記', '', elem.strip())
    e = re.split(r'[をがはにでのと、]', e)[0].strip()
    return e[:12] if e else ''


def plot_claim_dag(claims: dict) -> go.Figure:
    """クレーム依存DAG — 矢印付き有向グラフ + 要素類似破線"""
    pos = compute_tree_pos(claims)
    if not pos: return go.Figure()

    parent_set = {n: set(c['parents']) for n, c in claims.items()}
    claim_texts = {n: ' '.join(c['elements']) for n, c in claims.items()}
    claim_nums = sorted(claims.keys())

    # ── 要素類似エッジ（非従属ペア）─────────────
    sim_pairs = []
    for i in range(len(claim_nums)):
        for j in range(i + 1, len(claim_nums)):
            n1, n2 = claim_nums[i], claim_nums[j]
            if n2 in parent_set[n1] or n1 in parent_set[n2]: continue
            if n1 not in pos or n2 not in pos: continue
            bg1 = _bigrams(claim_texts[n1]); bg2 = _bigrams(claim_texts[n2])
            union = bg1 | bg2
            sim = len(bg1 & bg2) / len(union) if union else 0
            if sim >= 0.15:
                sim_pairs.append((n1, n2, sim))
    sim_pairs.sort(key=lambda x: -x[2]); sim_pairs = sim_pairs[:10]

    fig = go.Figure()

    # 類似エッジ（破線・アンバー）
    for n1, n2, sim in sim_pairs:
        x0, y0 = pos[n1]; x1, y1 = pos[n2]
        alpha = min(0.25 + sim * 1.2, 0.75)
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode='lines',
            line=dict(width=1.8, color=f'rgba(245,158,11,{alpha:.2f})', dash='dot'),
            hoverinfo='text', hovertext=f'C{n1} ↔ C{n2}　要素類似: {sim:.0%}',
            showlegend=False, name='',
        ))

    # 従属エッジ（矢印アノテーション — 有向）
    annotations = []
    for n, c in claims.items():
        if n not in pos: continue
        x1, y1 = pos[n]
        for p in c['parents']:
            if p not in pos: continue
            x0, y0 = pos[p]
            annotations.append(dict(
                x=x1, y=y1, ax=x0, ay=y0,
                xref='x', yref='y', axref='x', ayref='y',
                arrowhead=2, arrowsize=1.2, arrowwidth=2.0,
                arrowcolor='#94A3B8',
                showarrow=True, text='',
                standoff=20,
            ))

    # ノード下ラベル
    for n in sorted(claims.keys()):
        if n not in pos: continue
        c = claims[n]; x, y = pos[n]
        kind = '独立' if c['is_independent'] else ('多重従属' if c['is_multi_dep'] else '従属')
        kw = _elem_keyword(c['elements'][0]) if c['elements'] else ''
        annotations.append(dict(
            x=x, y=y - 0.75,
            text=f"<span style='font-size:9px;color:#64748B'>{kind}" +
                 (f"<br><span style='font-size:8px'>{kw}</span>" if kw else '') + "</span>",
            showarrow=False, font=dict(size=9, color='#64748B'),
        ))

    # レジェンドダミー
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
        line=dict(color='#94A3B8', width=2), name='従属関係 ▶', showlegend=True))
    if sim_pairs:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
            line=dict(color='#F59E0B', width=2, dash='dot'), name='要素類似', showlegend=True))

    # ノード
    nx_, ny_, ntxt, nclr, nsz, nhov = [], [], [], [], [], []
    for n in sorted(claims.keys()):
        if n not in pos: continue
        c = claims[n]; x, y = pos[n]
        nx_.append(x); ny_.append(y); ntxt.append(str(n))
        dep = '独立項' if c['is_independent'] else f"→ C{', '.join(str(p) for p in c['parents'])}"
        preview = c['body'][:120].replace('\n', ' ') + ('…' if len(c['body']) > 120 else '')
        nhov.append(f"<b>請求項 {n}</b>　{dep}<br>要素数: {len(c['elements'])} ／ {c['type']}<br>"
                    f"<span style='color:#94A3B8;font-size:11px'>{preview}</span><extra></extra>")
        if c['is_independent']:   nclr.append('#2563EB'); nsz.append(48)
        elif c['is_multi_dep']:   nclr.append('#7C3AED'); nsz.append(40)
        else:                     nclr.append('#475569'); nsz.append(40)

    fig.add_trace(go.Scatter(
        x=nx_, y=ny_, mode='markers+text',
        marker=dict(size=nsz, color=nclr, line=dict(width=3, color='white')),
        text=ntxt, textposition='middle center',
        textfont=dict(color='white', size=14, family='SF Mono, Menlo, Arial Black'),
        hovertemplate=nhov, name='', showlegend=False,
    ))

    all_x = [pos[n][0] for n in pos]; all_y = [pos[n][1] for n in pos]
    px_ = max((max(all_x) - min(all_x)) * 0.10, 0.6)
    py_ = max((max(all_y) - min(all_y)) * 0.20, 1.0)

    fig.update_layout(
        annotations=annotations, showlegend=True,
        legend=dict(x=0.01, y=0.01, xanchor='left', yanchor='bottom',
                    bgcolor='rgba(241,245,249,0.92)', bordercolor='#E2E8F0', borderwidth=1,
                    font=dict(size=11, color='#475569')),
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[min(all_x)-px_, max(all_x)+px_]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[min(all_y)-py_, max(all_y)+py_]),
        height=420,
        hoverlabel=dict(bgcolor='#0F172A', font_color='#F1F5F9', font_size=12, bordercolor='#1E293B'),
    )
    return fig


def plot_claim_matrix(claims: dict) -> go.Figure:
    """クレーム間の要素類似度ヒートマップ（bigram Jaccard）"""
    nums = sorted(claims.keys())
    if len(nums) < 2: return go.Figure()

    texts = {n: ' '.join(claims[n]['elements']) for n in nums}
    parent_set = {n: set(claims[n]['parents']) for n in nums}

    z, hover, annotations = [], [], []
    labels = [f'C{n}' for n in nums]

    for i, ni in enumerate(nums):
        row_z, row_h = [], []
        for j, nj in enumerate(nums):
            if ni == nj:
                row_z.append(1.0); row_h.append('同一')
            else:
                bg1 = _bigrams(texts[ni]); bg2 = _bigrams(texts[nj])
                union = bg1 | bg2
                sim = len(bg1 & bg2) / len(union) if union else 0.0
                row_z.append(sim)
                dep_mark = ' ▶' if nj in parent_set[ni] or ni in parent_set[nj] else ''
                row_h.append(f'C{ni} ↔ C{nj}{dep_mark}<br>類似度: {sim:.0%}')
            # セルの数値テキスト
            val = row_z[-1]
            txt = f'{val:.0%}' if val > 0.01 else ''
            txt_color = '#fff' if val > 0.5 else '#1E293B'
            annotations.append(dict(
                x=j, y=i, text=txt, showarrow=False,
                font=dict(size=10, color=txt_color),
                xref='x', yref='y',
            ))
        z.append(row_z); hover.append(row_h)

    fig = go.Figure(data=go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale=[[0,'#F8FAFF'],[0.15,'#DBEAFE'],[0.45,'#60A5FA'],[0.75,'#2563EB'],[1,'#1D4ED8']],
        showscale=True,
        colorbar=dict(thickness=10, len=0.75,
                      tickfont=dict(size=10, color='#64748B'),
                      title=dict(text='類似度', font=dict(size=10, color='#64748B'), side='right')),
        customdata=hover,
        hovertemplate='%{customdata}<extra></extra>',
        zmin=0, zmax=1,
    ))

    # 従属関係セルに枠線
    for ni in nums:
        for p in claims[ni]['parents']:
            if p in nums:
                i = nums.index(ni); j = nums.index(p)
                for xi, yi in [(i, j), (j, i)]:
                    fig.add_shape(type='rect',
                        x0=xi-0.5, x1=xi+0.5, y0=yi-0.5, y1=yi+0.5,
                        line=dict(color='#2563EB', width=2),
                        xref='x', yref='y',
                    )

    fig.update_layout(
        annotations=annotations,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        height=300,
        xaxis=dict(tickfont=dict(size=12, color='#1E293B'), side='top',
                   showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(size=12, color='#1E293B'), autorange='reversed',
                   showgrid=False, zeroline=False),
        hoverlabel=dict(bgcolor='#0F172A', font_color='#F1F5F9', font_size=12),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# 構成要件ネットワーク
# ─────────────────────────────────────────────────────────────
def plot_elem_network(elem_G: nx.Graph, claims: dict) -> go.Figure:
    if len(elem_G.nodes()) == 0: return go.Figure()
    pos = nx.spring_layout(elem_G, seed=42, k=3.0/max(len(elem_G.nodes())**0.5,1))
    cent = nx.degree_centrality(elem_G)
    max_c = max(cent.values()) if cent else 1
    ex, ey = [], []
    for u, v in elem_G.edges():
        x0,y0=pos[u]; x1,y1=pos[v]
        ex+=[x0,x1,None]; ey+=[y0,y1,None]
    nx_,ny_,ntxt,nclr,nsz,nhov=[],[],[],[],[],[]
    nc = max(len(claims),1)
    for node in elem_G.nodes():
        x,y=pos[node]; nx_.append(x); ny_.append(y)
        lbl=elem_G.nodes[node].get('label',node)
        cs=elem_G.nodes[node].get('claims',set())
        cv=cent.get(node,0); cov=len(cs)/nc
        ntxt.append(lbl[:13]+'…' if len(lbl)>13 else lbl)
        nsz.append(10+int(cv/max_c*24))
        nclr.append('#2563EB' if cov>=0.6 else ('#059669' if cov>=0.3 else '#94A3B8'))
        nhov.append(f"<b>{lbl}</b><br>出現: {' · '.join(f'C{c}' for c in sorted(cs))}<br>次数中心性: {cv:.2f}<extra></extra>")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=ex,y=ey,mode='lines',line=dict(width=1,color='#E2E8F0'),hoverinfo='none'))
    fig.add_trace(go.Scatter(
        x=nx_,y=ny_,mode='markers+text',
        marker=dict(size=nsz,color=nclr,opacity=0.9,line=dict(width=2,color='white')),
        text=ntxt,textposition='top center',
        textfont=dict(size=9,color='#374151'),
        hovertemplate=nhov,name='',
    ))
    fig.update_layout(
        showlegend=False,margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor='#FFFFFF',plot_bgcolor='#FFFFFF',
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        height=360,
        hoverlabel=dict(bgcolor='#0F172A',font_color='#F1F5F9',font_size=12),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# 対比表（複数特許）
# ─────────────────────────────────────────────────────────────
PATENT_COLORS = ['#2563EB','#059669','#D97706','#DC2626','#7C3AED','#0891B2','#65A30D','#DB2777']


def _bigrams(s: str) -> set:
    s = re.sub(r'前記|その|当該|上記|また|さらに|[\s　]', '', s)
    return {s[i:i+2] for i in range(len(s)-1)} if len(s)>1 else {s}


def build_comparison(patents: dict, sbert_model=None) -> pd.DataFrame:
    """
    patents: {name: {claims, resolved}}
    全請求項（独立項 + 従属項の追加要素）を横断比較
    sbert_model が None のとき: bigram Jaccard ≥ 0.25
    sbert_model 指定時: SBERT コサイン類似度 ≥ 0.75 で同一要素とみなす
    groups: [[full_elem, claim_label, {pname: elem}], ...]
    """
    pnames = list(patents.keys())
    groups = []  # [full_elem_for_matching, claim_label, {pname: elem}]

    def _similar(a: str, b: str) -> bool:
        if sbert_model is not None:
            return sbert_sim(a, b, sbert_model) >= 0.75
        bg_a, bg_b = _bigrams(a), _bigrams(b)
        union = bg_a | bg_b
        return bool(union) and len(bg_a & bg_b) / len(union) >= 0.25

    for pname, data in patents.items():
        for num, c in data['claims'].items():
            claim_lbl = f"C{num}({'独' if c['is_independent'] else '従'})"
            for elem in c['elements']:
                matched = False
                for g in groups:
                    if _similar(elem, g[0]):
                        if pname not in g[2]:
                            g[2][pname] = elem
                        matched = True; break
                if not matched:
                    groups.append([elem, claim_lbl, {pname: elem}])

    shorts = {p: (p[:22]+'…' if len(p)>22 else p) for p in pnames}
    rows = []
    for full_elem, claim_lbl, presence in groups:
        row = {'構成要件': full_elem}
        n_present = 0
        for p in pnames:
            row[shorts[p]] = '✓' if p in presence else '—'
            if p in presence: n_present += 1
        row['_n'] = n_present
        row['カバー'] = f"{n_present}/{len(pnames)}"
        rows.append(row)

    df = pd.DataFrame(rows).sort_values('_n', ascending=False).drop('_n', axis=1)
    return df, shorts, pnames


def render_comparison(patents: dict, sbert_model=None) -> str:
    if len(patents) < 2:
        return "<p style='color:#94A3B8;padding:20px'>2件以上の特許を読み込むと対比表が表示されます。</p>"

    df, shorts, pnames = build_comparison(patents, sbert_model=sbert_model)
    if df.empty:
        return "<p style='color:#94A3B8;padding:20px'>構成要件を抽出できませんでした。</p>"

    rows = ['<div class="ct-wrap"><table class="ct">']

    # ヘッダー
    rows.append('<thead><tr><th class="elem-col">構成要件</th>')
    for i, p in enumerate(pnames):
        c = PATENT_COLORS[i % len(PATENT_COLORS)]
        rows.append(f'<th style="color:{c};border-top:3px solid {c}">{shorts[p]}</th>')
    rows.append('<th>カバー</th></tr></thead><tbody>')

    for _, row in df.iterrows():
        rows.append(f'<tr><td class="elem-cell">{row["構成要件"]}</td>')
        for i, p in enumerate(pnames):
            val = row.get(shorts[p], '—')
            c = PATENT_COLORS[i % len(PATENT_COLORS)]
            if val == '✓':
                rows.append(f'<td><span style="color:{c};font-weight:700;font-size:16px">✓</span></td>')
            else:
                rows.append('<td><span class="c-none">—</span></td>')
        cov = row['カバー']
        n, total = map(int, cov.split('/'))
        cls = 'cov-high' if n==total else ('cov-mid' if n/total>0.5 else 'cov-low')
        rows.append(f'<td class="{cls}">{cov}</td></tr>')

    rows.append('</tbody></table>')
    rows.append('<div class="legend">')
    for i, p in enumerate(pnames):
        c = PATENT_COLORS[i % len(PATENT_COLORS)]
        rows.append(f'<span><b style="color:{c}">■</b> {shorts[p]}</span>')
    rows.append('</div></div>')
    return ''.join(rows)


# ─────────────────────────────────────────────────────────────
# クレームチャート
# ─────────────────────────────────────────────────────────────
def render_claim_chart(claims: dict, resolved: dict) -> str:
    if not claims: return ""
    nums = sorted(claims.keys())
    all_elems, seen = [], set()
    for n in nums:
        for e in claims[n]['elements']:
            if e not in seen: all_elems.append(e); seen.add(e)
    if not all_elems:
        return "<p style='color:#94A3B8;padding:20px'>構成要件を抽出できませんでした。</p>"

    bs = compute_breadth_scores(claims)
    rows = ['<div class="ct-wrap"><table class="ct"><thead><tr><th class="elem-col">構成要件</th>']
    for n in nums:
        c = claims[n]; bsi = bs[n]
        if c['is_independent']:   badge = '<span class="badge b-ind">独立項</span>'
        elif c['is_multi_dep']:   badge = '<span class="badge b-multi">多重従属</span>'
        else:                     badge = f'<span class="badge b-dep">→ C{c["parents"][0]}</span>'
        rows.append(
            f'<th>C{n}<br>{badge}'
            f'<div class="bw-wrap"><div class="bw-bar" style="width:{bsi["score"]}%;background:{bsi["color"]}"></div></div>'
            f'<span class="bw-lbl" style="color:{bsi["color"]}">{bsi["score"]}点 {bsi["level"]}</span>'
            f'</th>'
        )
    rows.append('</tr></thead><tbody>')

    for elem in all_elems:
        rows.append(f'<tr><td class="elem-cell">{elem}</td>')
        for n in nums:
            c = claims[n]; res = resolved.get(n, [])
            if elem in c['elements']:
                rows.append('<td><span class="c-direct">✓</span></td>' if c['is_independent']
                            else '<td><span class="c-added">＋</span></td>')
            elif elem in res:
                rows.append('<td><span class="c-inherit">↗</span></td>')
            else:
                rows.append('<td><span class="c-none">—</span></td>')
        rows.append('</tr>')

    rows.append('</tbody></table>')
    rows.append('<div class="legend">'
                '<span><b style="color:#059669">✓</b> 直接記載（独立項）</span>'
                '<span><b style="color:#2563EB">＋</b> 追加（従属項）</span>'
                '<span><b style="color:#CBD5E1">↗</b> 継承</span>'
                '<span><b style="color:#E2E8F0">—</b> 非該当</span>'
                '</div></div>')
    return ''.join(rows)


# ─────────────────────────────────────────────────────────────
# サンプルデータ
# ─────────────────────────────────────────────────────────────
SAMPLE = """\
【請求項1】
プロセッサと、メモリと、を備える情報処理装置であって、
前記メモリは、入力テキストを受け付ける受付手段と、
前記入力テキストをトークン列に変換するトークナイズ手段と、
前記トークン列をエンコーダモデルに入力してベクトルを生成するベクトル化手段と、
を実現するプログラムを記憶し、
前記プロセッサは前記プログラムを実行する、情報処理装置。

【請求項2】
前記ベクトル化手段は、Sentence-BERT型のエンコーダモデルを用いる、請求項1に記載の情報処理装置。

【請求項3】
前記ベクトル化手段は、複数のトークン列を並列にエンコードする、請求項2に記載の情報処理装置。

【請求項4】
前記プログラムは、さらに、複数の前記ベクトル間のコサイン類似度を算出する類似度算出手段を実現する、請求項1または2に記載の情報処理装置。

【請求項5】
前記類似度算出手段は、算出したコサイン類似度に基づいて類似ドキュメントをランキングする、請求項4に記載の情報処理装置。"""


# ─────────────────────────────────────────────────────────────
# セッション初期化
# ─────────────────────────────────────────────────────────────
if 'patents' not in st.session_state:
    st.session_state.patents = {}   # {name: {claims, resolved, n_claims}}
if 'active' not in st.session_state:
    st.session_state.active = None


def add_patent(name: str, claim_text: str):
    claims = parse_claims(claim_text)
    if not claims: return False
    resolved = resolve_inherited(claims)
    st.session_state.patents[name] = {
        'claims': claims, 'resolved': resolved,
        'n_claims': len(claims), 'text': claim_text,
    }
    st.session_state.active = name
    return True


# ─────────────────────────────────────────────────────────────
# 手法説明ポップオーバー
# ─────────────────────────────────────────────────────────────
_HELP = {
    'breadth': (
        "**広狭スコア算出方法**",
        """
USPTO Patent Claims Methodology に基づく 3指標の加重合計（0〜100点）。

| 指標 | 計算式 | 配点 |
|---|---|---|
| 語数スコア | max(0, 1 − 文字数 ÷ 600) × 40 | 40点満点 |
| 限定語密度 | max(0, 1 − 限定語数 ÷ 8) × 35 | 35点満点 |
| 要素数スコア | max(0, 1 − 要素数 ÷ 8) × 25 | 25点満点 |

**判定ライン:** 広 ≥ 62点 ／ 中 ≥ 35点 ／ 狭 < 35点

**限定語の例:** 「所定の」「特定の」「N以上」「N〜M」「ただし」「のみ」など

**根拠:** 語数が多いほど権利範囲が狭くなる傾向は *Patent claims and patent scope* (ScienceDirect, 2019) で実証されています。
""",
    ),
    'dag': (
        "**クレーム依存グラフ（inter-claim DAG）算出方法**",
        """
**依存関係の検出**

```
正規表現:
請求項\\s*(\\d+)
(?:\\s*(?:または|若しくは|又は)\\s*(?:請求項\\s*)?(\\d+))*
\\s*[、に]?(?:記載の|おける|係る)
```

「請求項N（またはM）に記載の」のパターンを全請求項から抽出し、NetworkX の有向グラフ（DAG）として構築します。

**読み方**
- 独立項 = DAGのルート（親を持たない）→ 権利の軸
- 従属深さ = サポートの厚み → 深いほど段階的な権利主張が可能
- 多重従属 = 複数の親クレームを参照 → 対象製品のバリエーションに対応

**レイアウト:** Reingold-Tilford 風ツリー（根を上に配置）
""",
    ),
    'inner': (
        "**inner-claim 要素間依存グラフ算出方法**",
        """
**参考文献:** FLAN-Graph (ACL 2024, arXiv:2404.14372)
単純なグラフ手法がLLMを上回ることを実証した手法の内部依存部分を実装。

**検出ロジック**

```python
# 前記X パターンを正規表現で抽出
re.findall(
    r'前記([ひらがな・漢字・英数字]{2,20})[をがはにでのと、。]',
    要素テキスト
)
```

要素Ejに「前記X」が現れ、X が先行要素Eiの本文に含まれる場合、Ei → Ej のエッジを追加します。

**読み方**
- エッジなし = 構成要件が独立している（シンプルな構成）
- A → B → C の連鎖 = 発明の構造的コア
- 多く参照される要素 = 発明のキーコンポーネント（侵害判断で重要）
""",
    ),
    'jaccard': (
        "**bigram Jaccard 類似度算出方法**",
        """
**手順**

1. テキストから指示語・前置詞（「前記」「その」「当該」「上記」）と空白を除去
2. 2文字N-gram（bigram）集合を生成
3. Jaccard 係数を計算

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

**例:**
"プロセッサを備え" → {プロ, ロセ, セッ, ッサ, サを, を備, 備え}

**閾値**
- 対比表での同一要素判定: ≥ 0.25
- DAGの類似エッジ表示: ≥ 0.15

**特徴:** 表記ゆれ・語尾変化を吸収しながら同一技術要素をスクリーニングする決定論的手法。LLMを使わず、同一入力→同一出力で再現可能。
""",
    ),
    'sbert': (
        "**SBERT コサイン類似度算出方法**",
        """
**モデル:** `intfloat/multilingual-e5-small`（多言語対応・384次元）

**手順**

1. 入力テキストに `"passage: "` プレフィックスを付与（E5モデルの仕様）
2. エンコーダでベクトル化（L2正規化済み）
3. ドット積 = コサイン類似度

```
sim(e1, e2) =
  encode("passage: " + e1) · encode("passage: " + e2)
```

**閾値:** ≥ 0.75 で同一要素とみなす

**bigram Jaccard との違い**
| | bigram Jaccard | SBERT |
|---|---|---|
| 方式 | 表層文字列 | 意味ベクトル |
| 速度 | 高速 | 要モデルロード |
| 強み | 決定論的・再現可能 | 言い換え・同義語に対応 |

両者を組み合わせるとカバレッジが向上します。
""",
    ),
    'network': (
        "**構成要件ネットワーク中心性指標の算出方法**",
        """
**グラフ構造**
同一請求項に共起する構成要件間にエッジを張ります（エッジ重み = 共起クレーム数）。

**次数中心性（Degree Centrality）**
```
DC(v) = そのノードのエッジ数 / (総ノード数 − 1)
```
→ 値が高い = 多くのクレームに登場する **発明コア要素**
→ 侵害分析・無効化調査でまず注目すべき要素

**媒介中心性（Betweenness Centrality）**
```
BC(v) = Σ (vを通る最短経路数 / 全ペアの最短経路数)
```
→ 値が高い = 技術領域を橋渡しする **ハブ要素**
→ この要素を持つ請求項が技術的に広い範囲をカバー

**活用:** 中心性の高い要素が競合特許と重複している場合 → 権利範囲の重複リスク
""",
    ),
}


def _info_popover(key: str):
    title, body = _HELP[key]
    with st.popover(f"ℹ 算出方法", use_container_width=False):
        st.markdown(f"### {title}")
        st.markdown(body)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sb-logo">
            <div class="sb-logo-icon">⚖</div>
            <div class="sb-logo-title">Patent Claim Analyzer</div>
            <div class="sb-logo-sub">Graph Theory · Non-generative AI</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-section">', unsafe_allow_html=True)
        st.markdown('<div class="sb-label">入力モード</div>', unsafe_allow_html=True)
        mode = st.radio("", ["PDF アップロード", "テキスト入力"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-section">', unsafe_allow_html=True)

        if mode == "PDF アップロード":
            if not PDF_OK:
                st.error("pdfplumber が未インストールです。")
            else:
                files = st.file_uploader(
                    "PDF（最大8件 · 各200MB）",
                    type=["pdf"], accept_multiple_files=True,
                    label_visibility="collapsed",
                )
                if files:
                    if len(files) > 8:
                        st.warning("先頭8件のみ処理します。")
                        files = files[:8]
                    if st.button("読み込む", type="primary", use_container_width=True):
                        with st.spinner("PDF解析中…"):
                            for f in files:
                                if f.name not in st.session_state.patents:
                                    raw = extract_text_from_pdf(f.read())
                                    ct = extract_claims_section(raw)
                                    ok = add_patent(f.name, ct)
                                    if not ok:
                                        st.warning(f"{f.name}: 請求項を検出できませんでした")
                        st.rerun()

        else:  # テキスト入力
            if st.button("サンプルを読み込む", use_container_width=True):
                st.session_state['_sample_text'] = SAMPLE

            text_in = st.text_area(
                "", height=320,
                value=st.session_state.get('_sample_text', ''),
                placeholder="【請求項1】\n...\n\n【請求項2】\n...",
                label_visibility="collapsed",
                key='_text_input',
            )
            name_in = st.text_input("", placeholder="特許名（任意）", label_visibility="collapsed")

            if st.button("解　析", type="primary", use_container_width=True):
                if text_in.strip():
                    label = name_in.strip() if name_in.strip() else f"テキスト入力 #{len(st.session_state.patents)+1}"
                    ok = add_patent(label, text_in)
                    if not ok:
                        st.error("請求項を検出できませんでした。【請求項N】形式で入力してください。")
                    else:
                        st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # 読み込み済み特許リスト
        if st.session_state.patents:
            st.markdown('<div class="sb-section">', unsafe_allow_html=True)
            st.markdown('<div class="sb-label">読み込み済み特許</div>', unsafe_allow_html=True)

            for i, (name, data) in enumerate(st.session_state.patents.items()):
                col_a, col_b = st.columns([5, 1])
                is_active = (name == st.session_state.active)
                dot_color = PATENT_COLORS[i % len(PATENT_COLORS)]
                short = name[:22]+'…' if len(name)>22 else name
                col_a.markdown(
                    f'<div class="pat-item">'
                    f'<div class="pat-item-dot" style="background:{dot_color}"></div>'
                    f'<div class="pat-item-name" style="{"color:#FFFFFF" if is_active else ""}">{short}</div>'
                    f'<div class="pat-item-count">{data["n_claims"]}項</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if col_b.button("×", key=f"del_{name}", help="削除"):
                    del st.session_state.patents[name]
                    if st.session_state.active == name:
                        st.session_state.active = (
                            list(st.session_state.patents.keys())[-1]
                            if st.session_state.patents else None
                        )
                    st.rerun()

            if len(st.session_state.patents) > 1:
                st.markdown('<div style="margin-top:8px"></div>', unsafe_allow_html=True)
                sel = st.selectbox(
                    "分析対象", list(st.session_state.patents.keys()),
                    index=list(st.session_state.patents.keys()).index(st.session_state.active)
                    if st.session_state.active in st.session_state.patents else 0,
                    label_visibility="collapsed",
                )
                st.session_state.active = sel

            if st.button("すべてクリア", use_container_width=True):
                st.session_state.patents = {}
                st.session_state.active = None
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    # ── Main content ────────────────────────────────────────
    n_patents = len(st.session_state.patents)

    if n_patents == 0:
        st.markdown("""
        <div class="empty">
            <div class="empty-icon">📋</div>
            <div class="empty-title">特許クレームを読み込んでください</div>
            <div class="empty-sub">
                左パネルから PDF をアップロード、またはテキストを貼り付けます<br>
                J-PlatPat の PDF・【請求項N】形式のテキストに対応
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Hero header ─────────────────────────────────────────
    st.markdown(
        f'<div class="hero-strip">'
        f'<div class="hero-title">Patent Claim <span>Analyzer</span></div>'
        f'<div class="hero-sub">構文解析 + グラフ理論 ─ 非生成AI · 決定論的 · 再現可能</div>'
        f'<div><span class="hero-badge">⚡ {n_patents} 件読み込み済み</span></div>'
        f'<div class="hero-ghost">C</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Active patent data ───────────────────────────────────
    active = st.session_state.active
    if active not in st.session_state.patents:
        active = list(st.session_state.patents.keys())[0]
        st.session_state.active = active

    data = st.session_state.patents[active]
    claims, resolved = data['claims'], data['resolved']
    elem_G = build_elem_graph(claims, resolved)
    m = compute_metrics(claims, elem_G)

    # ── Metrics ─────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card mc-blue">'
                    f'<div class="mc-label">請求項数</div>'
                    f'<div class="mc-value blue">{m["n_claims"]}</div>'
                    f'<div class="mc-sub">独立 {m["n_ind"]} ／ 従属 {m["n_dep"]}</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card mc-cyan">'
                    f'<div class="mc-label">最大従属深さ</div>'
                    f'<div class="mc-value cyan">{m["max_depth"]}</div>'
                    f'<div class="mc-sub">多重従属 {m["n_multi"]} 件</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card mc-green">'
                    f'<div class="mc-label">独立項数</div>'
                    f'<div class="mc-value green">{m["n_ind"]}</div>'
                    f'<div class="mc-sub">権利範囲の軸</div></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card mc-purple">'
                    f'<div class="mc-label">ユニーク構成要件</div>'
                    f'<div class="mc-value purple">{m["n_elements"]}</div>'
                    f'<div class="mc-sub">要素ノード {len(elem_G.nodes())} 個</div></div>',
                    unsafe_allow_html=True)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    # ── Tabs ────────────────────────────────────────────────
    tabs = ["　クレームチャート　", "　クレーム構造グラフ　", "　要素ネットワーク　"]
    if n_patents >= 2:
        tabs.append("　対比表　")

    tab_objects = st.tabs(tabs)
    tab_chart, tab_map, tab_net = tab_objects[0], tab_objects[1], tab_objects[2]
    tab_comp = tab_objects[3] if n_patents >= 2 else None

    with tab_chart:
        _tc0, _ic0 = st.columns([9, 1])
        with _tc0:
            st.markdown(
                '<div style="font-size:12px;color:#64748B;margin-bottom:6px">'
                '各請求項の構成要件と広狭スコアを一覧表示します。'
                '独立項 = 権利の外縁、従属項 = サポートの厚みを確認できます。'
                '</div>',
                unsafe_allow_html=True,
            )
        with _ic0:
            _info_popover('breadth')
        st.markdown(render_claim_chart(claims, resolved), unsafe_allow_html=True)

    with tab_map:
        dag_col, mat_col = st.columns([1.4, 1], gap="medium")
        with dag_col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            _td1, _id1 = st.columns([8, 2])
            with _td1:
                st.markdown('<div class="panel-title">クレーム依存グラフ（DAG）</div>', unsafe_allow_html=True)
            with _id1:
                _info_popover('dag')
            st.markdown(
                '<div class="panel-sub">'
                '<b style="color:#2563EB">●</b> 独立項 &nbsp;·&nbsp; '
                '<b style="color:#7C3AED">●</b> 多重従属 &nbsp;·&nbsp; '
                '<b style="color:#475569">●</b> 従属項'
                '&nbsp;｜&nbsp;矢印 = 従属方向 &nbsp;·&nbsp; '
                '<b style="color:#F59E0B">- - -</b> 要素類似（≥ 15%）'
                '</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(plot_claim_dag(claims), use_container_width=True,
                            config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
        with mat_col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            _td2, _id2 = st.columns([7, 3])
            with _td2:
                st.markdown('<div class="panel-title">要素類似度マトリクス</div>', unsafe_allow_html=True)
            with _id2:
                _info_popover('jaccard')
            st.markdown(
                '<div class="panel-sub">'
                'bigram Jaccard 類似度（0〜100%）&nbsp;·&nbsp; '
                '<b style="color:#2563EB">青枠</b> = 従属関係ペア'
                '</div>',
                unsafe_allow_html=True,
            )
            if len(claims) >= 2:
                st.plotly_chart(plot_claim_matrix(claims), use_container_width=True,
                                config={'displayModeBar': False})
            else:
                st.markdown('<div style="color:#94A3B8;font-size:12px;padding:20px 0">'
                            '2件以上の請求項があると類似度マトリクスが表示されます。</div>',
                            unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # inner-claim 要素間依存グラフ
        n_inner_cols = min(len(claims), 3)
        inner_claims = sorted(
            [(n, c) for n, c in claims.items() if len(c['elements']) >= 2],
            key=lambda x: x[0],
        )[:n_inner_cols * 2]
        if inner_claims:
            st.markdown('<div class="panel" style="margin-top:12px">', unsafe_allow_html=True)
            _td3, _id3 = st.columns([8, 2])
            with _td3:
                st.markdown('<div class="panel-title">inner-claim 要素間依存グラフ（FLAN-Graph手法）</div>',
                            unsafe_allow_html=True)
            with _id3:
                _info_popover('inner')
            st.markdown(
                '<div class="panel-sub">'
                '<b style="color:#2563EB">●</b> 先行要素 &nbsp;·&nbsp; '
                '<b style="color:#7C3AED">●</b> 参照先要素 &nbsp;·&nbsp; '
                '<b style="color:#2563EB">---→</b> 前記X 参照関係'
                '</div>',
                unsafe_allow_html=True,
            )
            cols_inner = st.columns(min(len(inner_claims), 3))
            for idx, (cnum, cclaim) in enumerate(inner_claims):
                with cols_inner[idx % 3]:
                    st.markdown(f'<div style="font-size:11px;font-weight:600;color:#475569;margin-bottom:4px">C{cnum}</div>',
                                unsafe_allow_html=True)
                    ig = build_inner_dag(cclaim)
                    if len(ig.edges()) > 0:
                        st.plotly_chart(plot_inner_dag(cclaim, cnum),
                                        use_container_width=True,
                                        config={'displayModeBar': False},
                                        key=f'inner_{cnum}')
                    else:
                        st.markdown('<div style="font-size:11px;color:#94A3B8;padding:8px 0">前記参照なし</div>',
                                    unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_net:
        left, right = st.columns([1.1, 1], gap="medium")
        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            _td4, _id4 = st.columns([7, 3])
            with _td4:
                st.markdown('<div class="panel-title">構成要件ネットワーク</div>', unsafe_allow_html=True)
            with _id4:
                _info_popover('network')
            st.markdown('<div class="panel-sub">ノードサイズ = 次数中心性 ／ 🔵 高カバー ・ 🟢 中 ・ ⬤ 低</div>', unsafe_allow_html=True)
            st.plotly_chart(
                plot_elem_network(elem_G, claims),
                use_container_width=True,
                config={'displayModeBar': False},
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">中心性ランキング</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-sub">次数中心性が高い要素 = 多くのクレームに関わる発明コア</div>', unsafe_allow_html=True)
            cent_d = nx.degree_centrality(elem_G)
            cent_b = nx.betweenness_centrality(elem_G)
            rows_c = []
            for node in elem_G.nodes():
                lbl = elem_G.nodes[node].get('label', node)
                cs = elem_G.nodes[node].get('claims', set())
                rows_c.append({
                    '構成要件': lbl,
                    '出現請求項': ' · '.join(f"C{c}" for c in sorted(cs)),
                    '次数': round(cent_d.get(node, 0), 3),
                    '媒介': round(cent_b.get(node, 0), 3),
                })
            df_c = pd.DataFrame(rows_c).sort_values('次数', ascending=False)
            st.dataframe(df_c, use_container_width=True, hide_index=True, height=300)
            st.markdown('</div>', unsafe_allow_html=True)

    if tab_comp is not None:
        with tab_comp:
            _tc5, _ic5 = st.columns([7, 3])
            with _ic5:
                _info_popover('jaccard' if not SBERT_OK else 'sbert')
            use_sbert = False
            if SBERT_OK:
                use_sbert = st.toggle(
                    'SBERT 意味類似度を使用（multilingual-e5-small）',
                    value=False,
                    help='bigram Jaccard に加えてエンコーダ埋め込みのコサイン類似度で補完します。初回はモデルをダウンロードします。',
                )
            else:
                st.markdown(
                    '<div style="font-size:11px;color:#94A3B8;margin-bottom:8px">'
                    'sentence-transformers 未インストール — bigram Jaccard のみ使用します。'
                    '</div>',
                    unsafe_allow_html=True,
                )

            sbert_model = None
            if use_sbert:
                with st.spinner('モデルを読み込んでいます…'):
                    sbert_model = load_sbert()
                if sbert_model is None:
                    st.warning('モデルの読み込みに失敗しました。bigram Jaccard で代替します。')

            st.markdown(
                '<div style="font-size:12px;color:#64748B;margin-bottom:12px">'
                '全請求項の構成要件を横断比較します。'
                + ('SBERT コサイン類似度 ≥ 0.75 で同一要素とみなします。'
                   if use_sbert and sbert_model is not None
                   else 'bigram Jaccard ≥ 0.25 で同一要素とみなします。')
                + '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(render_comparison(st.session_state.patents, sbert_model=sbert_model),
                        unsafe_allow_html=True)


if __name__ == '__main__':
    main()
