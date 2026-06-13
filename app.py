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
[data-testid="stAppViewContainer"] { background: #F1F5F9; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Dark sidebar ─────────────────────────────── */
section[data-testid="stSidebar"] > div:first-child {
    background: #0A0F1C;
    padding: 0;
}
section[data-testid="stSidebar"] .block-container {
    padding: 0 !important;
}
/* Sidebar all text default */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #94A3B8; }
/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #F1F5F9 !important; }
/* Streamlit text_area and uploader inside sidebar */
section[data-testid="stSidebar"] textarea {
    background: #131B2E !important;
    color: #E2E8F0 !important;
    border: 1px solid #1E293B !important;
    font-size: 11px !important;
    font-family: 'Menlo','Consolas',monospace !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #131B2E;
    border: 1px dashed #334155;
    border-radius: 8px;
    padding: 4px;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: #64748B !important;
}
section[data-testid="stSidebar"] [data-baseweb="radio"] label {
    color: #94A3B8 !important;
}
/* Sidebar primary button */
section[data-testid="stSidebar"] button[kind="primary"] {
    background: #2563EB !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #1E293B !important;
    color: #64748B !important;
    border-radius: 6px !important;
}
/* Expander inside sidebar */
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #131B2E;
    border: 1px solid #1E293B !important;
    border-radius: 6px;
}

/* ── Sidebar logo block ───────────────────────── */
.sb-logo {
    padding: 20px 20px 8px;
    border-bottom: 1px solid #1E293B;
    margin-bottom: 16px;
}
.sb-logo-title {
    font-size: 15px;
    font-weight: 700;
    color: #F1F5F9 !important;
    letter-spacing: -0.3px;
}
.sb-logo-sub {
    font-size: 10px;
    color: #475569 !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-top: 2px;
}
.sb-section { padding: 0 16px 12px; }
.sb-label {
    font-size: 10px;
    font-weight: 600;
    color: #475569 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 8px;
}

/* Patent list item */
.pat-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    background: #131B2E;
    border: 1px solid #1E293B;
    border-radius: 6px;
    margin-bottom: 5px;
    cursor: pointer;
}
.pat-item-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #2563EB; flex-shrink: 0;
}
.pat-item-name { font-size: 11px; color: #CBD5E1 !important; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pat-item-count { font-size: 10px; color: #475569 !important; }

/* ── App header ────────────────────────────────── */
.app-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 4px;
}
.app-title { font-size: 20px; font-weight: 700; color: #0F172A; letter-spacing: -0.5px; }
.app-badge {
    font-size: 10px; font-weight: 600;
    background: #DBEAFE; color: #1D4ED8;
    padding: 3px 8px; border-radius: 20px;
    letter-spacing: 0.3px;
}
.app-sub { font-size: 11px; color: #94A3B8; margin-bottom: 20px; }

/* ── Metric cards ──────────────────────────────── */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
.mc-label { font-size: 10px; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px; }
.mc-value { font-size: 28px; font-weight: 700; color: #0F172A; line-height: 1; font-family: 'SF Mono','Menlo',monospace; }
.mc-sub   { font-size: 11px; color: #64748B; margin-top: 4px; }

/* ── Tabs ──────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] { border-bottom: 2px solid #E2E8F0; gap: 0; }
[data-testid="stTabs"] button {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #64748B !important;
    padding: 10px 20px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2563EB !important;
    border-bottom-color: #2563EB !important;
    font-weight: 600 !important;
}

/* ── Claim chart table ─────────────────────────── */
.ct-wrap {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    overflow-x: auto;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
.ct { width: 100%; border-collapse: collapse; font-size: 12px; }
.ct th {
    background: #F8FAFC; color: #475569; font-weight: 600;
    padding: 10px 14px; border-bottom: 2px solid #E2E8F0;
    border-right: 1px solid #F1F5F9; text-align: center;
    white-space: nowrap; min-width: 88px; position: sticky; top: 0;
}
.ct th.elem-col { text-align: left; min-width: 240px; max-width: 320px; }
.ct td {
    padding: 8px 14px; border-bottom: 1px solid #F8FAFC;
    border-right: 1px solid #F8FAFC; text-align: center; vertical-align: middle;
}
.ct td.elem-cell { text-align: left; color: #1E293B; font-weight: 500; white-space: normal; word-break: break-word; min-width: 260px; max-width: 480px; line-height: 1.5; }
.ct tbody tr:hover td { background: #F8FAFC; }
.ct tr:last-child td { border-bottom: none; }
.c-direct  { color: #059669; font-weight: 700; font-size: 16px; }
.c-added   { color: #2563EB; font-weight: 700; font-size: 13px; }
.c-inherit { color: #CBD5E1; font-size: 14px; }
.c-none    { color: #E2E8F0; font-size: 14px; }

/* Badges */
.badge { display: inline-block; padding: 2px 7px; border-radius: 20px; font-size: 9px; font-weight: 700; letter-spacing: 0.3px; margin-top: 3px; }
.b-ind   { background: #DBEAFE; color: #1D4ED8; }
.b-dep   { background: #F1F5F9; color: #475569; }
.b-multi { background: #EDE9FE; color: #5B21B6; }

/* Legend */
.legend { display: flex; gap: 20px; font-size: 11px; color: #64748B; padding: 12px 16px; border-top: 1px solid #F1F5F9; }

/* ── Comparison table ──────────────────────────── */
.cov-high { color: #059669; font-weight: 700; }
.cov-mid  { color: #2563EB; font-weight: 600; }
.cov-low  { color: #94A3B8; }

/* ── Panel card ────────────────────────────────── */
.panel {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
.panel-title { font-size: 13px; font-weight: 600; color: #0F172A; margin-bottom: 2px; }
.panel-sub   { font-size: 11px; color: #94A3B8; margin-bottom: 10px; }

/* ── Empty state ───────────────────────────────── */
.empty {
    text-align: center; padding: 100px 0;
    color: #CBD5E1; user-select: none;
}
.empty-icon { font-size: 48px; margin-bottom: 16px; }
.empty-title { font-size: 15px; font-weight: 600; color: #94A3B8; margin-bottom: 6px; }
.empty-sub { font-size: 12px; color: #CBD5E1; }
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
    """要素テキストから主要キーワードを抽出（最大12文字）"""
    e = re.sub(r'^前記|^その|^当該|^上記', '', elem.strip())
    e = re.split(r'[をがはにでのと、]', e)[0].strip()
    return e[:12] if e else ''


def plot_mindmap(claims: dict) -> go.Figure:
    pos = compute_tree_pos(claims)
    if not pos: return go.Figure()

    # ── 親子集合（スキップ判定用）──────────────────
    parent_set = {n: set(c['parents']) for n, c in claims.items()}

    # ── 要素類似度エッジ（非従属ペア）─────────────
    claim_texts = {n: ' '.join(c['elements']) for n, c in claims.items()}
    claim_nums = sorted(claims.keys())
    sim_pairs = []
    for i in range(len(claim_nums)):
        for j in range(i + 1, len(claim_nums)):
            n1, n2 = claim_nums[i], claim_nums[j]
            if n2 in parent_set[n1] or n1 in parent_set[n2]: continue
            if n1 not in pos or n2 not in pos: continue
            bg1 = _bigrams(claim_texts[n1])
            bg2 = _bigrams(claim_texts[n2])
            union = bg1 | bg2
            sim = len(bg1 & bg2) / len(union) if union else 0
            if sim >= 0.15:
                sim_pairs.append((n1, n2, sim))
    sim_pairs.sort(key=lambda x: -x[2])
    sim_pairs = sim_pairs[:10]  # 上位10ペアのみ表示

    # ── 従属エッジ（ベジェ曲線）──────────────────
    edge_x, edge_y = [], []
    for n, c in claims.items():
        if n not in pos: continue
        x1, y1 = pos[n]
        for p in c['parents']:
            if p in pos:
                x0, y0 = pos[p]
                bx, by = bezier_curve(x0, y0, x1, y1)
                edge_x.extend(bx); edge_y.extend(by)

    # ── ノード ──────────────────────────────────────
    nx_, ny_, ntxt, nclr, nsz, nhov = [], [], [], [], [], []
    for n in sorted(claims.keys()):
        if n not in pos: continue
        c = claims[n]
        x, y = pos[n]
        nx_.append(x); ny_.append(y)
        ntxt.append(str(n))
        dep = "独立項" if c['is_independent'] else f"→ C{', '.join(str(p) for p in c['parents'])}"
        preview = c['body'][:150].replace('\n', ' ') + ('…' if len(c['body']) > 150 else '')
        nhov.append(
            f"<b>請求項 {n}</b>　{dep}<br>"
            f"要素数: {len(c['elements'])} ／ 種別: {c['type']}<br>"
            f"<span style='color:#94A3B8;font-size:11px'>{preview}</span>"
            f"<extra></extra>"
        )
        if c['is_independent']:
            nclr.append('#2563EB'); nsz.append(48)
        elif c['is_multi_dep']:
            nclr.append('#7C3AED'); nsz.append(38)
        else:
            nclr.append('#475569'); nsz.append(38)

    fig = go.Figure()

    # 要素類似エッジ（破線・アンバー）
    for n1, n2, sim in sim_pairs:
        x0, y0 = pos[n1]; x1, y1 = pos[n2]
        alpha = min(0.25 + sim * 1.2, 0.80)
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=1.8, color=f'rgba(245,158,11,{alpha:.2f})', dash='dot'),
            hoverinfo='text',
            hovertext=f'C{n1} ↔ C{n2} 要素類似度: {sim:.0%}',
            showlegend=False, name='',
        ))

    # 従属エッジ（ベジェ・グレー）
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=2.2, color='#94A3B8'),
        hoverinfo='none', showlegend=False,
    ))

    # レジェンド用ダミートレース
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='#94A3B8', width=2),
        name='従属関係', showlegend=True,
    ))
    if sim_pairs:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='#F59E0B', width=2, dash='dot'),
            name='要素類似', showlegend=True,
        ))

    # ノード
    fig.add_trace(go.Scatter(
        x=nx_, y=ny_, mode='markers+text',
        marker=dict(size=nsz, color=nclr, line=dict(width=3, color='white')),
        text=ntxt,
        textposition='middle center',
        textfont=dict(color='white', size=14, family='SF Mono, Menlo, Arial Black'),
        hovertemplate=nhov, name='', showlegend=False,
    ))

    # ノード下ラベル（種別 + 最初の要素キーワード）
    annotations = []
    for n in sorted(claims.keys()):
        if n not in pos: continue
        c = claims[n]
        x, y = pos[n]
        kind = '独立' if c['is_independent'] else ('多重' if c['is_multi_dep'] else '従属')
        kw = _elem_keyword(c['elements'][0]) if c['elements'] else ''
        kw_str = f' · {kw}' if kw else ''
        annotations.append(dict(
            x=x, y=y - 0.72,
            text=f"<span style='font-size:9px;color:#64748B'>{kind}{kw_str}</span>",
            showarrow=False,
            font=dict(size=9, color='#64748B'),
        ))

    all_x = [pos[n][0] for n in pos]; all_y = [pos[n][1] for n in pos]
    px = max((max(all_x) - min(all_x)) * 0.08, 0.5)
    py = max((max(all_y) - min(all_y)) * 0.15, 0.8)

    fig.update_layout(
        annotations=annotations,
        showlegend=True,
        legend=dict(
            x=0.01, y=0.01, xanchor='left', yanchor='bottom',
            bgcolor='rgba(241,245,249,0.9)',
            bordercolor='#E2E8F0', borderwidth=1,
            font=dict(size=11, color='#475569'),
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[min(all_x)-px, max(all_x)+px]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[min(all_y)-py, max(all_y)+py]),
        height=500,
        hoverlabel=dict(bgcolor='#0F172A', font_color='#F1F5F9',
                        font_size=12, bordercolor='#1E293B'),
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


def build_comparison(patents: dict) -> pd.DataFrame:
    """
    patents: {name: {claims, resolved}}
    全請求項（独立項 + 従属項の追加要素）を横断比較
    bigram Jaccard ≥ 0.25 で同一要素とみなす
    groups: [[full_elem, claim_label, {pname: elem}], ...]
    """
    pnames = list(patents.keys())
    groups = []  # [full_elem_for_matching, claim_label, {pname: elem}]

    for pname, data in patents.items():
        for num, c in data['claims'].items():
            claim_lbl = f"C{num}({'独' if c['is_independent'] else '従'})"
            for elem in c['elements']:
                bg = _bigrams(elem)
                matched = False
                for g in groups:
                    canon_bg = _bigrams(g[0])  # g[0] = full text for matching
                    union = bg | canon_bg
                    if union and len(bg & canon_bg)/len(union) >= 0.25:
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


def render_comparison(patents: dict) -> str:
    if len(patents) < 2:
        return "<p style='color:#94A3B8;padding:20px'>2件以上の特許を読み込むと対比表が表示されます。</p>"

    df, shorts, pnames = build_comparison(patents)
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

    rows = ['<div class="ct-wrap"><table class="ct"><thead><tr><th class="elem-col">構成要件</th>']
    for n in nums:
        c = claims[n]
        if c['is_independent']:   badge = '<span class="badge b-ind">独立項</span>'
        elif c['is_multi_dep']:   badge = '<span class="badge b-multi">多重従属</span>'
        else:                     badge = f'<span class="badge b-dep">→ C{c["parents"][0]}</span>'
        rows.append(f'<th>C{n}<br>{badge}</th>')
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
# Main
# ─────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sb-logo">
            <div class="sb-logo-title">⚖ Patent Claim Analyzer</div>
            <div class="sb-logo-sub">Graph Theory · No AI</div>
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

    # ── Header ──────────────────────────────────────────────
    col_h, col_badge = st.columns([6, 1])
    with col_h:
        st.markdown(
            f'<div class="app-header">'
            f'<span class="app-title">Patent Claim Analyzer</span>'
            f'<span class="app-badge">{n_patents} 件読み込み済み</span>'
            f'</div>'
            f'<div class="app-sub">構文解析 + グラフ理論 — 非生成AI</div>',
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
        st.markdown(f'<div class="metric-card"><div class="mc-label">請求項数</div>'
                    f'<div class="mc-value">{m["n_claims"]}</div>'
                    f'<div class="mc-sub">独立 {m["n_ind"]} ／ 従属 {m["n_dep"]}</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="mc-label">最大従属深さ</div>'
                    f'<div class="mc-value">{m["max_depth"]}</div>'
                    f'<div class="mc-sub">多重従属 {m["n_multi"]} 件</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="mc-label">独立項数</div>'
                    f'<div class="mc-value">{m["n_ind"]}</div>'
                    f'<div class="mc-sub">権利範囲の軸</div></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="mc-label">ユニーク構成要件</div>'
                    f'<div class="mc-value">{m["n_elements"]}</div>'
                    f'<div class="mc-sub">要素ノード {len(elem_G.nodes())} 個</div></div>',
                    unsafe_allow_html=True)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    # ── Tabs ────────────────────────────────────────────────
    tabs = ["　クレームチャート　", "　マインドマップ　", "　要素ネットワーク　"]
    if n_patents >= 2:
        tabs.append("　対比表　")

    tab_objects = st.tabs(tabs)
    tab_chart, tab_map, tab_net = tab_objects[0], tab_objects[1], tab_objects[2]
    tab_comp = tab_objects[3] if n_patents >= 2 else None

    with tab_chart:
        st.markdown(render_claim_chart(claims, resolved), unsafe_allow_html=True)

    with tab_map:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">クレーム関係構造 — マインドマップ</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-sub">'
            '🔵 独立項 &nbsp;·&nbsp; 🟣 多重従属 &nbsp;·&nbsp; ⬤ 従属項'
            '&nbsp;｜&nbsp;'
            '<b style="color:#94A3B8">───</b> 従属関係 &nbsp;·&nbsp; '
            '<b style="color:#F59E0B">- - -</b> 要素類似（bigram Jaccard ≥ 15%）'
            '&nbsp;·&nbsp; ホバーで詳細表示'
            '</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            plot_mindmap(claims),
            use_container_width=True,
            config={'displayModeBar': False},
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_net:
        left, right = st.columns([1.1, 1], gap="medium")
        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">構成要件ネットワーク</div>', unsafe_allow_html=True)
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
            st.markdown('<div style="font-size:12px;color:#64748B;margin-bottom:12px">'
                        '独立項の構成要件を横断比較します。bigram類似度 ≥ 0.33 で同一要素とみなします。'
                        '</div>', unsafe_allow_html=True)
            st.markdown(render_comparison(st.session_state.patents), unsafe_allow_html=True)


if __name__ == '__main__':
    main()
