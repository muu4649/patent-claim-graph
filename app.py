"""
特許クレーム構造分析 — 非生成AI版
構文解析 + グラフ理論のみで動作
"""

import re
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
from collections import defaultdict

# ───────────────────────────────────────────────────────────────
# Page Config
# ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="クレーム構造分析",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────────────────────
# CSS
# ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base ──────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: #F0F2F6;
}
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Typography ─────────────────────────── */
.app-title {
    font-size: 18px;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.4px;
    margin: 0;
}
.app-sub {
    font-size: 11px;
    color: #94A3B8;
    margin: 2px 0 20px;
    letter-spacing: 0.3px;
}

/* ── Metric cards ───────────────────────── */
.cards-row { display: flex; gap: 12px; margin-bottom: 16px; }
.card {
    flex: 1;
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 16px 18px;
}
.card-label {
    font-size: 10px;
    font-weight: 600;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 4px;
}
.card-value {
    font-size: 26px;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.1;
}
.card-sub {
    font-size: 11px;
    color: #6B7280;
    margin-top: 3px;
}

/* ── Claim chart table ──────────────────── */
.ct-wrap {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 20px;
    overflow-x: auto;
}
.ct {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}
.ct th {
    background: #F8FAFC;
    color: #475569;
    font-weight: 600;
    padding: 9px 14px;
    text-align: center;
    border-bottom: 2px solid #E2E8F0;
    border-right: 1px solid #E2E8F0;
    white-space: nowrap;
    min-width: 80px;
}
.ct th.elem-col {
    text-align: left;
    min-width: 200px;
    max-width: 280px;
}
.ct td {
    padding: 8px 14px;
    border-bottom: 1px solid #F1F5F9;
    border-right: 1px solid #F1F5F9;
    color: #334155;
    text-align: center;
    vertical-align: middle;
}
.ct td.elem-cell {
    text-align: left;
    color: #1E293B;
    font-weight: 500;
    max-width: 280px;
}
.ct tr:last-child td { border-bottom: none; }

/* Cell states */
.c-direct   { color: #059669; font-weight: 700; font-size: 15px; }
.c-added    { color: #2563EB; font-weight: 700; font-size: 13px; }
.c-inherit  { color: #94A3B8; font-size: 14px; }
.c-none     { color: #E2E8F0; font-size: 14px; }

/* Badges */
.badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.3px;
    margin-top: 4px;
}
.b-ind   { background: #DBEAFE; color: #1D4ED8; }
.b-dep   { background: #F1F5F9; color: #475569; }
.b-multi { background: #EDE9FE; color: #5B21B6; }

/* ── Legend ─────────────────────────────── */
.legend {
    display: flex;
    gap: 18px;
    font-size: 11px;
    color: #6B7280;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #F1F5F9;
}
.legend span b { margin-right: 3px; }

/* ── Graph panel ────────────────────────── */
.graph-wrap {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 16px 18px;
}
.panel-title {
    font-size: 12px;
    font-weight: 600;
    color: #374151;
    margin-bottom: 2px;
}
.panel-sub {
    font-size: 10px;
    color: #9CA3AF;
    margin-bottom: 10px;
}

/* ── Tabs ───────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    gap: 4px;
    border-bottom: 1px solid #E5E7EB;
}
[data-testid="stTabs"] button {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #6B7280 !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 8px 18px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2563EB !important;
    border-bottom: 2px solid #2563EB !important;
    font-weight: 600 !important;
}

/* ── Sidebar ────────────────────────────── */
.sidebar-label {
    font-size: 11px;
    font-weight: 600;
    color: #374151;
    letter-spacing: 0.3px;
    margin-bottom: 6px;
}
[data-testid="stTextArea"] textarea {
    font-size: 12px !important;
    font-family: 'Menlo', 'Consolas', monospace !important;
    line-height: 1.6 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# 定数・パターン
# ───────────────────────────────────────────────────────────────

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


# ───────────────────────────────────────────────────────────────
# パーサー
# ───────────────────────────────────────────────────────────────

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
            # 「請求項1または2」「請求項1または請求項2」の両形式に対応
            parents = [int(x) for x in re.findall(r'\d+', m.group(0))]

        is_ind = len(parents) == 0
        elements = _extract_elements(body, is_ind)

        claims[num] = {
            'num': num,
            'body': body,
            'parents': parents,
            'is_independent': is_ind,
            'is_multi_dep': len(parents) > 1,
            'elements': elements,
            'type': _claim_type(body),
        }
    return claims


def _extract_elements(body: str, is_independent: bool) -> list:
    if not is_independent:
        body = DEP_RE.sub('', body).strip()

    # ターミネータの出現位置を収集
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
    if re.search(r'方法|工程|ステップ', body):
        return '方法'
    if re.search(r'システム|サーバ', body):
        return 'システム'
    if re.search(r'プログラム|コンピュータ可読', body):
        return 'プログラム'
    return '装置'


def resolve_inherited(claims: dict) -> dict:
    resolved = {}

    def _resolve(num):
        if num in resolved:
            return resolved[num]
        c = claims[num]
        if c['is_independent']:
            resolved[num] = list(c['elements'])
            return resolved[num]
        base = []
        for p in c['parents']:
            if p in claims:
                base = _resolve(p)[:]
                break
        resolved[num] = base + c['elements']
        return resolved[num]

    for n in sorted(claims.keys()):
        _resolve(n)
    return resolved


# ───────────────────────────────────────────────────────────────
# グラフ構築
# ───────────────────────────────────────────────────────────────

def build_dep_graph(claims: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for n, c in claims.items():
        G.add_node(n, **{k: v for k, v in c.items() if k != 'elements'})
    for n, c in claims.items():
        for p in c['parents']:
            if p in claims:
                G.add_edge(n, p)  # 子 → 親
    return G


def build_elem_graph(claims: dict, resolved: dict):
    G = nx.Graph()
    label_to_id = {}

    for num, c in claims.items():
        for elem in c['elements']:
            lbl = elem[:30] + '…' if len(elem) > 30 else elem
            if lbl not in label_to_id:
                eid = f"E{len(label_to_id)}"
                label_to_id[lbl] = eid
                G.add_node(eid, label=lbl, full=elem, claims=set(), own_claim=num)
            else:
                eid = label_to_id[lbl]
            G.nodes[eid]['claims'].add(num)

    for num, elems in resolved.items():
        for elem in elems:
            lbl = elem[:30] + '…' if len(elem) > 30 else elem
            if lbl in label_to_id:
                G.nodes[label_to_id[lbl]]['claims'].add(num)

    for num, elems in resolved.items():
        ids = []
        for e in elems:
            lbl = e[:30] + '…' if len(e) > 30 else e
            if lbl in label_to_id:
                ids.append(label_to_id[lbl])
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if G.has_edge(ids[i], ids[j]):
                    G[ids[i]][ids[j]]['weight'] += 1
                else:
                    G.add_edge(ids[i], ids[j], weight=1)

    return G


# ───────────────────────────────────────────────────────────────
# グラフメトリクス
# ───────────────────────────────────────────────────────────────

def compute_metrics(G: nx.DiGraph, claims: dict) -> dict:
    n_ind = sum(1 for c in claims.values() if c['is_independent'])
    n_multi = sum(1 for c in claims.values() if c['is_multi_dep'])

    roots = [n for n, c in claims.items() if c['is_independent']]
    max_depth = 0
    for node in G.nodes():
        for root in roots:
            try:
                d = nx.shortest_path_length(G, node, root)
                max_depth = max(max_depth, d)
            except nx.NetworkXNoPath:
                pass

    in_deg = dict(G.in_degree())
    cited = max(in_deg, key=in_deg.get) if in_deg else None

    all_elems = set()
    for c in claims.values():
        all_elems.update(c['elements'])

    return {
        'n_claims': len(claims),
        'n_ind': n_ind,
        'n_dep': len(claims) - n_ind,
        'n_multi': n_multi,
        'max_depth': max_depth,
        'most_cited': cited,
        'most_cited_n': in_deg.get(cited, 0) if cited else 0,
        'n_elements': len(all_elems),
    }


# ───────────────────────────────────────────────────────────────
# Plotly: クレーム依存グラフ
# ───────────────────────────────────────────────────────────────

def _layered_pos(G: nx.DiGraph, claims: dict) -> dict:
    roots = [n for n, c in claims.items() if c['is_independent']]
    depth = {r: 0 for r in roots}
    queue = list(roots)
    while queue:
        cur = queue.pop(0)
        for pred in G.predecessors(cur):
            if pred not in depth:
                depth[pred] = depth[cur] + 1
                queue.append(pred)

    by_layer = defaultdict(list)
    for node, d in depth.items():
        by_layer[d].append(node)

    pos = {}
    for layer, nodes in by_layer.items():
        nodes = sorted(nodes)
        n = len(nodes)
        for i, node in enumerate(nodes):
            pos[node] = ((i + 1) / (n + 1), 1.0 - layer * 0.28)
    return pos


def plot_dep_graph(G: nx.DiGraph, claims: dict) -> go.Figure:
    pos = _layered_pos(G, claims)

    # エッジ
    ex, ey = [], []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ex += [x0, x1, None]
            ey += [y0, y1, None]

    # 矢印（アノテーション）
    annotations = []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            annotations.append(dict(
                x=x1, y=y1, ax=x0, ay=y0,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True,
                arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
                arrowcolor='#CBD5E1',
            ))

    nx_, ny_, ntxt, nclr, nsz, nhov = [], [], [], [], [], []
    for num in sorted(G.nodes()):
        if num not in pos:
            continue
        c = claims[num]
        x, y = pos[num]
        nx_.append(x); ny_.append(y)
        ntxt.append(str(num))

        dep = f"独立項" if c['is_independent'] else f"→ 請求項{'・'.join(str(p) for p in c['parents'])}"
        nhov.append(
            f"<b>請求項{num}</b><br>{dep}<br>"
            f"要素数: {len(c['elements'])}<br>"
            f"種別: {c['type']}<extra></extra>"
        )

        if c['is_independent']:
            nclr.append('#2563EB'); nsz.append(36)
        elif c['is_multi_dep']:
            nclr.append('#7C3AED'); nsz.append(28)
        else:
            nclr.append('#64748B'); nsz.append(28)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode='lines',
        line=dict(width=0),
        hoverinfo='none',
    ))
    fig.add_trace(go.Scatter(
        x=nx_, y=ny_,
        mode='markers+text',
        marker=dict(size=nsz, color=nclr,
                    line=dict(width=2.5, color='white'),
                    symbol='circle'),
        text=ntxt,
        textposition='middle center',
        textfont=dict(color='white', size=13, family='Arial Black, Arial'),
        hovertemplate=nhov,
        name='',
    ))
    fig.update_layout(
        annotations=annotations,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.3, 1.15]),
        height=300,
        hoverlabel=dict(bgcolor='#1E293B', font_color='white', font_size=12, bordercolor='#1E293B'),
    )
    return fig


# ───────────────────────────────────────────────────────────────
# Plotly: 構成要件ネットワーク
# ───────────────────────────────────────────────────────────────

def plot_elem_graph(elem_G: nx.Graph, claims: dict) -> go.Figure:
    if len(elem_G.nodes()) == 0:
        return go.Figure()

    pos = nx.spring_layout(elem_G, seed=42, k=3.0 / max(len(elem_G.nodes()) ** 0.5, 1))
    centrality = nx.degree_centrality(elem_G)
    max_c = max(centrality.values()) if centrality else 1

    ex, ey = [], []
    ew = []
    for u, v, d in elem_G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ex += [x0, x1, None]
        ey += [y0, y1, None]
        ew.append(d.get('weight', 1))

    nx_, ny_, ntxt, nclr, nsz, nhov = [], [], [], [], [], []
    n_claims = max(len(claims), 1)

    for node in elem_G.nodes():
        x, y = pos[node]
        nx_.append(x); ny_.append(y)

        lbl = elem_G.nodes[node].get('label', node)
        claim_set = elem_G.nodes[node].get('claims', set())
        c_val = centrality.get(node, 0)
        coverage = len(claim_set) / n_claims

        disp = lbl[:14] + '…' if len(lbl) > 14 else lbl
        ntxt.append(disp)

        size = 10 + int(c_val / max_c * 22)
        nsz.append(size)

        if coverage >= 0.6:
            nclr.append('#2563EB')
        elif coverage >= 0.3:
            nclr.append('#059669')
        else:
            nclr.append('#94A3B8')

        claims_str = ' · '.join(f"C{c}" for c in sorted(claim_set))
        nhov.append(
            f"<b>{lbl}</b><br>"
            f"出現: {claims_str}<br>"
            f"次数中心性: {c_val:.2f}<extra></extra>"
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode='lines',
        line=dict(width=1.0, color='#E2E8F0'),
        hoverinfo='none',
    ))
    fig.add_trace(go.Scatter(
        x=nx_, y=ny_,
        mode='markers+text',
        marker=dict(size=nsz, color=nclr, opacity=0.9,
                    line=dict(width=2, color='white')),
        text=ntxt,
        textposition='top center',
        textfont=dict(size=9, color='#374151'),
        hovertemplate=nhov,
        name='',
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=340,
        hoverlabel=dict(bgcolor='#1E293B', font_color='white', font_size=12, bordercolor='#1E293B'),
    )
    return fig


# ───────────────────────────────────────────────────────────────
# クレームチャート (HTML)
# ───────────────────────────────────────────────────────────────

def render_claim_chart(claims: dict, resolved: dict) -> str:
    if not claims:
        return ""

    nums = sorted(claims.keys())

    # 全構成要件リスト（独立項→従属項の順）
    all_elems = []
    seen = set()
    for n in nums:
        for e in claims[n]['elements']:
            if e not in seen:
                all_elems.append(e)
                seen.add(e)

    if not all_elems:
        return "<p style='color:#9CA3AF;font-size:13px'>構成要件を抽出できませんでした。</p>"

    rows = ['<div class="ct-wrap"><table class="ct">']

    # ヘッダー
    rows.append('<thead><tr><th class="elem-col">構成要件</th>')
    for n in nums:
        c = claims[n]
        if c['is_independent']:
            badge = f'<span class="badge b-ind">独立項</span>'
        elif c['is_multi_dep']:
            badge = f'<span class="badge b-multi">多重従属</span>'
        else:
            badge = f'<span class="badge b-dep">→ C{c["parents"][0]}</span>'
        rows.append(f'<th>C{n}<br>{badge}</th>')
    rows.append('</tr></thead>')

    # ボディ
    rows.append('<tbody>')
    for idx, elem in enumerate(all_elems):
        short = elem[:38] + '…' if len(elem) > 38 else elem
        rows.append(f'<tr><td class="elem-cell" title="{elem}">{short}</td>')

        for n in nums:
            c = claims[n]
            res = resolved.get(n, [])
            if elem in c['elements']:
                if c['is_independent']:
                    rows.append('<td><span class="c-direct">✓</span></td>')
                else:
                    rows.append('<td><span class="c-added">＋</span></td>')
            elif elem in res:
                rows.append('<td><span class="c-inherit">↗</span></td>')
            else:
                rows.append('<td><span class="c-none">—</span></td>')

        rows.append('</tr>')

    rows.append('</tbody></table>')
    rows.append('''
    <div class="legend">
        <span><b style="color:#059669">✓</b> 直接記載（独立項）</span>
        <span><b style="color:#2563EB">＋</b> 追加（従属項）</span>
        <span><b style="color:#94A3B8">↗</b> 親から継承</span>
        <span><b style="color:#E2E8F0">—</b> 非該当</span>
    </div>
    ''')
    rows.append('</div>')
    return ''.join(rows)


# ───────────────────────────────────────────────────────────────
# サンプル
# ───────────────────────────────────────────────────────────────

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


# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-label">請求項テキスト</div>', unsafe_allow_html=True)

        if st.button("サンプルを読み込む", use_container_width=True):
            st.session_state['input'] = SAMPLE

        text = st.text_area(
            "claims",
            value=st.session_state.get('input', ''),
            height=440,
            placeholder="【請求項1】\n...\n\n【請求項2】\n...",
            label_visibility="collapsed",
            key='input',
        )

        run = st.button("解 析", type="primary", use_container_width=True)

    # ── Header ───────────────────────────────────────────────
    st.markdown('<div class="app-title">⚖ 特許クレーム構造分析</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">非生成AI ・ 構文解析 + グラフ理論</div>', unsafe_allow_html=True)

    # ── 解析 ─────────────────────────────────────────────────
    if run and text.strip():
        claims = parse_claims(text)
        resolved = resolve_inherited(claims)
        st.session_state['claims'] = claims
        st.session_state['resolved'] = resolved

    claims = st.session_state.get('claims', {})
    resolved = st.session_state.get('resolved', {})

    if not claims:
        st.markdown("""
        <div style="
            text-align:center; padding:100px 0;
            color:#CBD5E1; user-select:none;
        ">
            <div style="font-size:44px; margin-bottom:14px;">📄</div>
            <div style="font-size:14px; font-weight:500;">
                左のパネルに請求項を貼り付けて「解析」を押してください
            </div>
            <div style="font-size:12px; margin-top:6px;">
                【請求項1】〜【請求項N】の形式に対応
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── グラフ構築 ────────────────────────────────────────────
    dep_G = build_dep_graph(claims)
    elem_G = build_elem_graph(claims, resolved)
    m = compute_metrics(dep_G, claims)

    # ── メトリクスカード ──────────────────────────────────────
    st.markdown('<div class="cards-row">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""<div class="card">
            <div class="card-label">請求項数</div>
            <div class="card-value">{m['n_claims']}</div>
            <div class="card-sub">独立 {m['n_ind']} ／ 従属 {m['n_dep']}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="card">
            <div class="card-label">最大従属深さ</div>
            <div class="card-value">{m['max_depth']}</div>
            <div class="card-sub">多重従属 {m['n_multi']} 件</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="card">
            <div class="card-label">最多被参照</div>
            <div class="card-value">C{m['most_cited']}</div>
            <div class="card-sub">{m['most_cited_n']} クレームから引用</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="card">
            <div class="card-label">構成要件数（ユニーク）</div>
            <div class="card-value">{m['n_elements']}</div>
            <div class="card-sub">要素ノード {len(elem_G.nodes())} 個</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── タブ ─────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["　クレームチャート　", "　グラフ構造　"])

    with tab1:
        chart_html = render_claim_chart(claims, resolved)
        st.markdown(chart_html, unsafe_allow_html=True)

    with tab2:
        left, right = st.columns(2, gap="medium")

        with left:
            st.markdown('<div class="graph-wrap">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">クレーム依存グラフ</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-sub">🔵 独立項 ・ 🟣 多重従属 ・ ⬤ 従属項 ／ 矢印 = 子→親</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                plot_dep_graph(dep_G, claims),
                use_container_width=True,
                config={'displayModeBar': False},
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="graph-wrap">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">構成要件ネットワーク</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-sub">ノードサイズ = 次数中心性 ／ 🔵 高被参照 ・ 🟢 中 ・ ⬤ 低</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                plot_elem_graph(elem_G, claims),
                use_container_width=True,
                config={'displayModeBar': False},
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # 中心性テーブル
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        centrality = nx.degree_centrality(elem_G)
        betweenness = nx.betweenness_centrality(elem_G)

        rows = []
        for node in elem_G.nodes():
            lbl = elem_G.nodes[node].get('label', node)
            claim_set = elem_G.nodes[node].get('claims', set())
            rows.append({
                '構成要件': lbl,
                '出現請求項': ' · '.join(f"C{c}" for c in sorted(claim_set)),
                '次数中心性': round(centrality.get(node, 0), 3),
                '媒介中心性': round(betweenness.get(node, 0), 3),
            })

        df = pd.DataFrame(rows).sort_values('次数中心性', ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True, height=200)


if __name__ == '__main__':
    main()
