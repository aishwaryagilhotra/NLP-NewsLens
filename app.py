import html
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Requires the tested inference function in ner_engine.py
from ner_engine import extract_entities

APP_TITLE = "NEWSLENS"
st.set_page_config(
    page_title="NewsLens — Entity Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual system ----------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --ink: #eaf1ff;
  --muted: #9caed0;
  --panel: #101a31;
  --panel2: #131f3a;
  --line: rgba(155,180,230,.16);
  --blue: #75a7ff;
  --pink: #ff83c7;
  --violet: #a99aff;
}
html, body, [class*="css"] {
  font-family: 'Manrope', sans-serif;
}
.stApp {
  background:
    radial-gradient(ellipse at 15% 0%, rgba(66,112,220,.17), transparent 38%),
    radial-gradient(ellipse at 90% 7%, rgba(255,96,185,.11), transparent 31%),
    #080e1d;
  color: var(--ink);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #111b34 0%, #0b1224 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
h1,h2,h3 { letter-spacing: -0.035em; color: #f3f6ff; }
h1 { font-weight: 800 !important; }
p, label, .stMarkdown { color: #dbe5fb; }
small, .muted { color: var(--muted) !important; }
div[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(23,37,67,.96), rgba(13,22,42,.96));
  border: 1px solid var(--line); border-radius: 16px; padding: 16px 18px;
}
div[data-testid="stMetricLabel"] { color: #aebfe0; }
div[data-testid="stMetricValue"] { color: #f4f7ff; font-weight: 800; }
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; border: 1px solid rgba(117,167,255,.35);
  background: linear-gradient(105deg, #477be0, #805bd9);
  color: white; font-weight: 750; min-height: 2.6rem;
  box-shadow: 0 7px 24px rgba(78,106,219,.14);
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--pink); color: white;
  box-shadow: 0 0 0 1px rgba(255,131,199,.22), 0 8px 30px rgba(255,94,185,.13);
}
.stTextArea textarea, .stTextInput input, [data-baseweb="select"] > div {
  background: #0c1529 !important; color: #edf3ff !important;
  border-color: rgba(145,170,220,.25) !important; border-radius: 10px !important;
}
div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
hr { border-color: var(--line); }
.nl-wordmark { font-size: 1.55rem; font-weight: 800; letter-spacing: .14em; color: #f4f7ff; }
.nl-wordmark span { color: #ff83c7; }
.nl-eyebrow { font: 500 .72rem 'DM Mono', monospace; letter-spacing: .16em; color: #8eabeb; text-transform: uppercase; }
.nl-hero {
  padding: 26px 28px; border: 1px solid rgba(135,163,225,.20);
  border-radius: 22px; background:
    radial-gradient(circle at 85% 0%, rgba(255,131,199,.15), transparent 35%),
    linear-gradient(130deg, rgba(25,42,77,.94), rgba(13,21,41,.97));
  margin-bottom: 22px;
}
.nl-hero h1 { margin: 8px 0 6px 0; font-size: clamp(2rem, 3vw, 3.15rem); }
.nl-hero p { color: #b9c8e7; max-width: 780px; margin-bottom: 0; }
.nl-card {
  background: linear-gradient(145deg, rgba(19,31,57,.96), rgba(12,20,39,.96));
  border: 1px solid var(--line); border-radius: 16px; padding: 18px 20px;
}
.nl-card h3 { font-size: 1rem; margin: 0 0 8px; letter-spacing: -.01em; }
.nl-card p { color: #aebfdf; font-size: .9rem; margin: 0; }
.nl-chip {
 display:inline-block; padding: 4px 9px; border-radius: 7px; margin: 2px 4px 2px 0;
 font: 500 .7rem 'DM Mono', monospace; letter-spacing: .03em;
 border: 1px solid rgba(117,167,255,.27); color: #c9dcff; background: rgba(77,126,227,.12);
}
.nl-divider { height:1px; background:var(--line); margin: 14px 0; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding: 7px 8px; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------- Session state ----------
if "library" not in st.session_state:
    st.session_state.library = []
if "active_article_id" not in st.session_state:
    st.session_state.active_article_id = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None

LABEL_GROUPS = {
    "Person": {"PERSON"},
    "Organization": {"ORG"},
    "Location": {"GPE", "LOC", "FAC"},
    "Date & time": {"DATE", "TIME"},
    "Numbers & values": {"CARDINAL", "ORDINAL", "MONEY", "PERCENT", "QUANTITY"},
    "Other": {"NORP", "LAW", "WORK_OF_ART", "PRODUCT", "LANGUAGE", "EVENT"},
}
LABEL_COLORS = {
    "PERSON": "#ff83c7",
    "ORG": "#75a7ff",
    "GPE": "#80e2c2",
    "LOC": "#80e2c2",
    "FAC": "#80e2c2",
    "DATE": "#ffd27d",
    "TIME": "#ffd27d",
    "MONEY": "#c2a7ff",
    "PERCENT": "#c2a7ff",
    "CARDINAL": "#c2a7ff",
    "ORDINAL": "#c2a7ff",
    "QUANTITY": "#c2a7ff",
    "NORP": "#ffa77b",
    "LAW": "#ffa77b",
    "WORK_OF_ART": "#ffa77b",
    "PRODUCT": "#ffa77b",
    "LANGUAGE": "#ffa77b",
    "EVENT": "#ffa77b",
}
MODEL_F1 = {
    "CARDINAL": .8573, "DATE": .8652, "EVENT": .6667, "FAC": .7117,
    "GPE": .9429, "LANGUAGE": .7027, "LAW": .5250, "LOC": .7493,
    "MONEY": .8668, "NORP": .9350, "ORDINAL": .8592, "ORG": .8675,
    "PERCENT": .8996, "PERSON": .9283, "PRODUCT": .6296,
    "QUANTITY": .7834, "TIME": .6364, "WORK_OF_ART": .6243,
}

def label_group(label):
    for group, labels in LABEL_GROUPS.items():
        if label in labels:
            return group
    return "Other"

def make_article(title, body, entities, source="Pasted text"):
    return {
        "id": f"N{datetime.now().strftime('%y%m%d%H%M%S%f')}",
        "title": title.strip() or "Untitled article",
        "body": body,
        "entities": entities,
        "source": source,
        "analyzed_at": datetime.now().strftime("%d %b %Y · %H:%M"),
    }

def article_frame(articles=None):
    rows = []
    for article in articles if articles is not None else st.session_state.library:
        for e in article["entities"]:
            rows.append({
                "Article ID": article["id"],
                "Article": article["title"],
                "Entity": e["text"],
                "Type": e["label"],
                "Category": label_group(e["label"]),
                "Confidence": e.get("confidence", None),
                "Start": e["start"],
                "End": e["end"],
            })
    return pd.DataFrame(rows)

def render_highlighted(article, entities, allowed_labels=None):
    body = article
    chosen = []
    for e in sorted(entities, key=lambda x: (x["start"], x["end"])):
        if allowed_labels is not None and e["label"] not in allowed_labels:
            continue
        s, end = int(e["start"]), int(e["end"])
        if s < 0 or end <= s or end > len(body):
            continue
        if body[s:end] != e["text"]:
            continue
        chosen.append(e)

    parts, cursor = [], 0
    for e in chosen:
        s, end = e["start"], e["end"]
        if s < cursor:
            continue
        parts.append(html.escape(body[cursor:s]))
        color = LABEL_COLORS.get(e["label"], "#aab8d2")
        escaped_text = html.escape(body[s:end])
        label = html.escape(e["label"])
        confidence = e.get("confidence")
        tip = f"{label}" + (f" · {confidence:.1%} confidence" if confidence is not None else "")
        parts.append(
            f'<span title="{tip}" style="background:{color}25; color:{color}; '
            f'border:1px solid {color}75; border-radius:5px; padding:2px 4px; '
            f'font-weight:700; white-space:normal;">{escaped_text}'
            f'<sup style="font:500 9px monospace; opacity:.8; margin-left:3px;">{label}</sup></span>'
        )
        cursor = end
    parts.append(html.escape(body[cursor:]))
    return (
        '<div style="white-space:pre-wrap; line-height:2.2; font-size:14px; '
        'padding:20px; background:#0b1427; color:#dce7ff; border:1px solid '
        'rgba(145,170,220,.16); border-radius:14px;">' + "".join(parts) + '</div>'
    )

def sentence_spans(text):
    # Practical sentence segmentation for a lightweight prototype.
    # Evidence is the sentence text; co-occurrence is not proof of a relationship.
    return [(m.start(), m.end(), m.group().strip())
            for m in re.finditer(r'[^.!?\n]+(?:[.!?]+|$)', text)
            if m.group().strip()]

def build_edges(articles, scope="sentence"):
    edges = Counter()
    evidence = defaultdict(list)
    for art in articles:
        ents = art["entities"]
        spans = sentence_spans(art["body"])
        for i, (s, end, sentence) in enumerate(spans):
            in_sentence = [e for e in ents if e["start"] < end and e["end"] > s]
            # Deduplicate repeated mentions of the same surface/type in one sentence.
            unique = {}
            for e in in_sentence:
                unique[(e["text"].casefold(), e["label"])] = e
            unique_entities = list(unique.values())
            for x in range(len(unique_entities)):
                for y in range(x + 1, len(unique_entities)):
                    a, b = unique_entities[x], unique_entities[y]
                    ka = f"{a['text']} [{a['label']}]"
                    kb = f"{b['text']} [{b['label']}]"
                    pair = tuple(sorted((ka, kb)))
                    edges[pair] += 1
                    if len(evidence[pair]) < 5:
                        evidence[pair].append({
                            "article": art["title"],
                            "sentence": sentence,
                        })
    return edges, evidence

def page_header(eyebrow, title, subtitle):
    st.markdown(
        f'<div class="nl-eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1 style="margin:5px 0 2px;">{html.escape(title)}</h1>'
        f'<p style="color:#9fb1d2;margin-top:0;margin-bottom:22px;">{html.escape(subtitle)}</p>',
        unsafe_allow_html=True
    )

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div class="nl-wordmark">NEWS<span>LENS</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="nl-eyebrow" style="margin-top:4px;">ENTITY INTELLIGENCE WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown("<div class='nl-divider'></div>", unsafe_allow_html=True)
    nav_options = ["Signal Desk", "Article Lab", "Entity Atlas", "Connection Map", "Model Card"]
    if "_go_page" in st.session_state:
        st.session_state["nav_page"] = st.session_state.pop("_go_page")
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Signal Desk"
    page = st.radio(
        "WORKSPACE",
        nav_options,
        key="nav_page",
        label_visibility="visible",
    )
    st.markdown("<div class='nl-divider'></div>", unsafe_allow_html=True)
    st.markdown('<div class="nl-eyebrow">SESSION LIBRARY</div>', unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:1.65rem;font-weight:800'>{len(st.session_state.library):02d}"
        f"<span style='font-size:.9rem;color:#ff83c7'> articles</span></div>",
        unsafe_allow_html=True
    )
    st.caption("Articles are kept in this browser session. Export the library to preserve it.")
    if st.session_state.library:
        export_library = [{
            "title": a["title"], "body": a["body"], "source": a["source"],
            "analyzed_at": a["analyzed_at"], "entities": a["entities"]
        } for a in st.session_state.library]
        st.download_button(
            "Export session library",
            json.dumps(export_library, ensure_ascii=False, indent=2),
            file_name="newslens_session_library.json",
            mime="application/json",
            use_container_width=True,
        )
    st.markdown("<div class='nl-divider'></div>", unsafe_allow_html=True)
    st.caption("BLUE = organizations  ·  PINK = people")
    st.caption("Prototype build · DistilBERT NER")

# ---------- Signal Desk ----------
if page == "Signal Desk":
    st.markdown(
        '<div class="nl-hero"><div class="nl-eyebrow">NEWS INTELLIGENCE / OVERVIEW</div>'
        '<h1>See the names behind the news.</h1>'
        '<p>Extract named entities, trace recurring mentions, and inspect co-occurrence patterns '
        'across the articles you analyze. One workspace, built around your NER model.</p></div>',
        unsafe_allow_html=True
    )
    df = article_frame()
    n_articles = len(st.session_state.library)
    n_mentions = len(df) if not df.empty else 0
    n_unique = df[["Entity", "Type"]].drop_duplicates().shape[0] if not df.empty else 0
    n_types = df["Type"].nunique() if not df.empty else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles analyzed", n_articles)
    c2.metric("Entity mentions", n_mentions)
    c3.metric("Distinct entity / type pairs", n_unique)
    c4.metric("Entity classes seen", n_types, "of 18 model classes")
    st.markdown("### Quick actions")
    a, b = st.columns([1.35, 1])
    with a:
        st.markdown('<div class="nl-card"><h3>01 / Article Lab</h3><p>Paste or upload a text article. Get highlighted names, categories, confidence scores and exportable records.</p></div>', unsafe_allow_html=True)
        if st.button("Open Article Lab ↗", key="quick_lab", use_container_width=True):
            st.session_state["_go_page"] = "Article Lab"
            st.rerun()
    with b:
        st.markdown('<div class="nl-card"><h3>02 / Entity Atlas</h3><p>Search mentions across your analyzed session library and inspect which articles contain them.</p></div>', unsafe_allow_html=True)
        if st.button("Open Entity Atlas ↗", key="quick_atlas", use_container_width=True):
            st.session_state["_go_page"] = "Entity Atlas"
            st.rerun()
    if st.session_state.library:
        st.markdown("### Recent articles")
        recent = st.session_state.library[-5:][::-1]
        for art in recent:
            labels = Counter(e["label"] for e in art["entities"])
            chip_html = " ".join(
                f'<span class="nl-chip">{html.escape(k)} {v}</span>'
                for k, v in labels.most_common(5)
            )
            st.markdown(
                f'<div class="nl-card" style="margin:8px 0;"><div style="display:flex;justify-content:space-between;gap:12px;">'
                f'<div><div style="font-weight:800;color:#edf3ff">{html.escape(art["title"])}</div>'
                f'<div style="font: .72rem DM Mono;color:#8fa5cf;margin-top:5px">{art["analyzed_at"]} · {len(art["entities"])} mentions</div></div>'
                f'<div>{chip_html}</div></div></div>',
                unsafe_allow_html=True
            )
    else:
        st.info("Your workspace is ready. Analyze your first article in Article Lab to populate the dashboard.")

# ---------- Article Lab ----------
elif page == "Article Lab":
    page_header("WORKSPACE / 01", "Article Lab", "Turn raw reporting into a structured map of people, organizations, places and events.")
    with st.form("article_form", clear_on_submit=False):
        title = st.text_input("Article title", placeholder="e.g. ISRO announces a new satellite mission")
        source_url = st.text_input("Source URL (optional)", placeholder="https://...")
        uploaded = st.file_uploader("Or upload a plain-text article", type=["txt", "md"])
        default_text = st.session_state.get("draft_article", "")
        body = st.text_area(
            "Article text",
            value=default_text,
            height=250,
            placeholder="Paste the full article text here…"
        )
        submitted = st.form_submit_button("Analyze article  →", use_container_width=True)
    if uploaded is not None and not body.strip():
        try:
            body = uploaded.getvalue().decode("utf-8", errors="replace")
        except Exception:
            body = ""
    if submitted:
        if not body.strip():
            st.error("Add article text or upload a .txt/.md file first.")
        else:
            with st.spinner("Running NewsLens entity extraction…"):
                try:
                    entities = extract_entities(body)
                    art_title = title.strip() or (uploaded.name if uploaded else "Untitled article")
                    art = make_article(art_title, body, entities, source_url.strip() or "Pasted text")
                    st.session_state.library.append(art)
                    st.session_state.active_article_id = art["id"]
                    st.session_state["draft_article"] = body
                    st.success(f"Analysis complete — {len(entities)} entity mentions found.")
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
    active = None
    if st.session_state.active_article_id:
        active = next((a for a in st.session_state.library if a["id"] == st.session_state.active_article_id), None)
    if active is None and st.session_state.library:
        active = st.session_state.library[-1]
    if active:
        ents = active["entities"]
        frame = pd.DataFrame(ents)
        st.markdown("<div class='nl-divider'></div>", unsafe_allow_html=True)
        st.markdown(f"### {active['title']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mentions", len(ents))
        m2.metric("Unique surface forms", len(set((e["text"].casefold(), e["label"]) for e in ents)))
        m3.metric("Classes found", len(set(e["label"] for e in ents)))
        m4.metric("Average confidence", f"{sum(e.get('confidence',0) for e in ents)/len(ents):.1%}" if ents else "—")
        group_options = list(LABEL_GROUPS.keys())
        chosen_groups = st.multiselect("Highlight categories", group_options, default=group_options)
        chosen_labels = set().union(*(LABEL_GROUPS[g] for g in chosen_groups)) if chosen_groups else set()
        st.markdown("#### Annotated article")
        st.markdown(render_highlighted(active["body"], ents, chosen_labels), unsafe_allow_html=True)
        st.markdown("#### Entity ledger")
        if ents:
            view = frame.copy()
            view["Category"] = view["label"].map(label_group)
            cat_filter = st.multiselect("Filter entity types", sorted(view["label"].unique()), default=sorted(view["label"].unique()), key="lab_filter")
            view = view[view["label"].isin(cat_filter)]
            st.dataframe(
                view.rename(columns={
                    "text": "Entity", "label": "Type", "confidence": "Confidence",
                    "start": "Start offset", "end": "End offset"
                })[["Entity", "Type", "Confidence", "Start offset", "End offset"]],
                use_container_width=True, hide_index=True
            )
            x, y, z = st.columns(3)
            x.download_button("Download CSV", view.to_csv(index=False).encode("utf-8"), "newslens_entities.csv", "text/csv", use_container_width=True)
            y.download_button("Download JSON", json.dumps(view.to_dict("records"), ensure_ascii=False, indent=2), "newslens_entities.json", "application/json", use_container_width=True)
            z.download_button("Download annotated article", active["body"].encode("utf-8"), "article.txt", "text/plain", use_container_width=True)
            counts = frame["label"].value_counts().rename_axis("Entity type").reset_index(name="Mentions")
            st.markdown("#### Mention distribution")
            st.bar_chart(counts.set_index("Entity type"), horizontal=True)
        else:
            st.info("No named entities were detected in this article.")
    elif not st.session_state.library:
        st.markdown('<div class="nl-card"><h3>Ready for input</h3><p>Run an article through the model to unlock entity highlighting, category counts, exports and the relationship workspace.</p></div>', unsafe_allow_html=True)

# ---------- Entity Atlas ----------
elif page == "Entity Atlas":
    page_header("WORKSPACE / 02", "Entity Atlas", "A searchable index of every entity mention in articles analyzed during this session.")
    df = article_frame()
    if df.empty:
        st.info("Analyze at least one article in Article Lab first. The atlas will populate automatically.")
    else:
        q = st.text_input("Search entity", placeholder="Search a name, organization, location…")
        col_a, col_b = st.columns(2)
        type_choices = ["All types"] + sorted(df["Type"].unique().tolist())
        chosen_type = col_a.selectbox("Entity type", type_choices)
        article_choices = ["All articles"] + [a["title"] for a in st.session_state.library]
        chosen_article = col_b.selectbox("Article", article_choices)
        view = df.copy()
        if q.strip():
            view = view[view["Entity"].str.contains(re.escape(q.strip()), case=False, na=False)]
        if chosen_type != "All types":
            view = view[view["Type"] == chosen_type]
        if chosen_article != "All articles":
            view = view[view["Article"] == chosen_article]
        c1, c2, c3 = st.columns(3)
        c1.metric("Matching mentions", len(view))
        c2.metric("Unique names", view["Entity"].nunique() if not view.empty else 0)
        c3.metric("Articles represented", view["Article"].nunique() if not view.empty else 0)
        if not view.empty:
            st.dataframe(
                view[["Entity", "Type", "Article", "Confidence", "Start", "End"]],
                use_container_width=True, hide_index=True
            )
            st.markdown("#### Most-mentioned surface forms")
            counts = view.groupby(["Entity", "Type"]).size().reset_index(name="Mentions").sort_values("Mentions", ascending=False).head(15)
            st.bar_chart(counts.set_index("Entity")["Mentions"], horizontal=True)
            st.download_button("Export filtered index as CSV", view.to_csv(index=False).encode("utf-8"), "newslens_entity_atlas.csv", "text/csv")
        else:
            st.warning("No matches for the current filters.")
        st.caption("Name matching here is based on surface forms and entity types. Alias resolution (for example, abbreviations) is not automatic.")

# ---------- Connection Map ----------
elif page == "Connection Map":
    page_header("WORKSPACE / 03", "Connection Map", "Explore which entities appear together in the same sentence, with the original sentence as evidence.")
    df = article_frame()
    if not st.session_state.library or df.empty:
        st.info("Analyze articles first. The map is built from entities detected by your model.")
    else:
        source_scope = st.radio("Graph scope", ["All analyzed articles", "Latest article"], horizontal=True)
        graph_articles = st.session_state.library if source_scope == "All analyzed articles" else [st.session_state.library[-1]]
        min_weight = st.slider("Minimum shared-sentence count", 1, 5, 1)
        edges, evidence = build_edges(graph_articles)
        edges = Counter({pair: count for pair, count in edges.items() if count >= min_weight})
        if not edges:
            st.warning("No entity pairs meet this threshold. Analyze a longer article or lower the threshold.")
        else:
            st.markdown(f"**{len(edges)} connections** · {len(set(n for pair in edges for n in pair))} nodes · Evidence-based co-occurrence")
            try:
                import plotly.graph_objects as go
                import networkx as nx

                graph = nx.Graph()
                for (a, b), weight in edges.items():
                    graph.add_edge(a, b, weight=weight)
                # Cap graph size for legibility.
                if graph.number_of_nodes() > 45:
                    keep_edges = edges.most_common(60)
                    graph = nx.Graph()
                    for (a, b), weight in keep_edges:
                        graph.add_edge(a, b, weight=weight)
                pos = nx.spring_layout(graph, seed=21, k=0.9)
                edge_x, edge_y = [], []
                for a, b in graph.edges():
                    edge_x += [pos[a][0], pos[b][0], None]
                    edge_y += [pos[a][1], pos[b][1], None]
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y, mode="lines",
                    line=dict(width=1.2, color="rgba(132,158,214,.30)"),
                    hoverinfo="none"
                )
                node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
                degree = dict(graph.degree(weight="weight"))
                for node in graph.nodes():
                    node_x.append(pos[node][0]); node_y.append(pos[node][1])
                    node_text.append(f"{node}<br>Weighted connections: {degree.get(node, 0)}")
                    if "[PERSON]" in node:
                        node_color.append("#ff83c7")
                    elif "[ORG]" in node:
                        node_color.append("#75a7ff")
                    else:
                        node_color.append("#80e2c2")
                    node_size.append(12 + min(18, degree.get(node, 0) * 2))
                node_trace = go.Scatter(
                    x=node_x, y=node_y, mode="markers+text",
                    text=[n if len(n) < 29 else n[:26] + "…" for n in graph.nodes()],
                    textposition="top center", hovertext=node_text, hoverinfo="text",
                    marker=dict(size=node_size, color=node_color, line=dict(width=1, color="#0b1427")),
                    textfont=dict(size=10, color="#dce7ff")
                )
                fig = go.Figure(data=[edge_trace, node_trace])
                fig.update_layout(
                    height=590, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.warning("Install plotly and networkx to render the graph: pip install plotly networkx")
            st.markdown("#### Strongest co-occurrences")
            edge_rows = []
            for (a, b), count in edges.most_common(30):
                edge_rows.append({"Entity A": a, "Entity B": b, "Shared sentences": count})
            st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)
            st.markdown("#### Evidence")
            pair_labels = [f"{a}  ↔  {b}" for (a, b), _ in edges.most_common(30)]
            selected_pair_label = st.selectbox("Inspect a connection", pair_labels)
            selected_pair = next(pair for pair, _ in edges.most_common(30) if f"{pair[0]}  ↔  {pair[1]}" == selected_pair_label)
            for ev in evidence[selected_pair]:
                st.markdown(
                    f'<div class="nl-card" style="margin:8px 0;"><div class="nl-eyebrow">{html.escape(ev["article"])}</div>'
                    f'<p style="margin-top:8px;color:#e5edff">“{html.escape(ev["sentence"])}”</p></div>',
                    unsafe_allow_html=True
                )
            st.caption("A connection means the entities were detected in the same sentence. It does not establish a real-world relationship or causation.")

# ---------- Model Card ----------
elif page == "Model Card":
    page_header("MODEL / 04", "Model Card", "Transparent performance reporting for the current OntoNotes-trained DistilBERT checkpoint.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test micro-F1", "88.04%")
    m2.metric("Test precision", "86.95%")
    m3.metric("Test recall", "89.17%")
    st.markdown("### Per-class test F1")
    f1_df = pd.DataFrame([{"Entity type": k, "F1": v, "F1 (%)": round(v*100, 2)} for k, v in MODEL_F1.items()]).sort_values("F1")
    st.bar_chart(f1_df.set_index("Entity type")["F1"], horizontal=True)
    st.dataframe(f1_df.sort_values("F1", ascending=False), use_container_width=True, hide_index=True)
    st.markdown("### Training history")
    history_path = Path(__file__).parent / "training_history.json"
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            st.json(history)
        except Exception:
            st.caption("Could not parse training_history.json.")
    else:
        st.caption("Place training_history.json beside app.py to display epoch history.")
    st.markdown("### Evaluation notes")
    st.markdown("""
    - Model: `distilbert/distilbert-base-cased`, fine-tuned for 37 BIO labels representing 18 entity types plus `O`.
    - Benchmark: OntoNotes 5.0 test split; the corpus includes multiple genres, not only contemporary news articles.
    - Evaluation used exact entity spans and the project's BIO repair policy.
    - Inputs were truncated to 256 tokens during the benchmark evaluation; the current inference function uses overlapping chunks for longer input.
    - Per-class scores are uneven, particularly for low-support classes such as LAW, LANGUAGE and PRODUCT.
    - The benchmark score is not a guarantee of accuracy on live news. Validate on contemporary articles before making comparative claims.
    """)
    st.warning("The model card reports the saved benchmark metrics. It does not claim the model is better than spaCy or other NER systems; that comparison has not been run.")
