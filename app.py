import re, io, builtins, threading, time
import streamlit.components.v1 as components
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
import streamlit as st, pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
from google import genai

st.set_page_config(page_title="DataPilot AI", page_icon="📊", layout="wide")
dark = st.sidebar.toggle("🌙 Dark mode", value=True, key="dark")
P = (dict(bg="#0a0e1a", bgs="rgba(10,14,26,.85)", panel="rgba(255,255,255,.045)", panel2="#0f1424", field="rgba(255,255,255,.06)", border="rgba(255,255,255,.10)", text="#e8ecf6", mute="#94a3b8", a1="#6366f1", a2="#22d3ee") if dark else
     dict(bg="#f3f5fb", bgs="rgba(243,245,251,.88)", panel="#ffffff", panel2="#ffffff", field="#f1f4fa", border="#e2e8f3", text="#0f172a", mute="#64748b", a1="#4f46e5", a2="#0891b2"))
COLORS = ["#6366f1", "#22d3ee", "#a78bfa", "#34d399", "#fbbf24", "#f472b6", "#fb923c"]
CSS = """
.stApp{background:radial-gradient(1200px 500px at 85% -10%,color-mix(in srgb,var(--a1) 18%,transparent),transparent),var(--bg)}
.stApp,.stApp p,.stApp span,.stApp label,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp small,.stApp summary{color:var(--text)}
[data-testid=stHeader]{background:transparent}
[data-testid=stSidebar]{background:var(--panel2);border-right:1px solid var(--border)}
.block-container{padding-top:3rem;max-width:1500px;padding-bottom:6rem}
html{font-size:15px}
.stApp input,.stApp textarea,[data-baseweb=select]>div,[data-baseweb=input]>div,[data-baseweb=textarea]>div{background:var(--field)!important;color:var(--text)!important;border-color:var(--border)!important;border-radius:10px}
[data-baseweb=popover] *{background-color:var(--panel2)!important;color:var(--text)!important}
.stApp button[kind=secondary]{background:var(--panel);border:1px solid var(--border);border-radius:12px;transition:.15s}
.stApp button[kind=secondary]:hover{border-color:var(--a1);transform:translateY(-1px)}
.stApp button[kind=primary]{background:linear-gradient(135deg,var(--a1),var(--a2));border:0;border-radius:12px}
.stApp button[kind=primary] p{color:#fff!important}
[data-testid=stExpander]{background:var(--panel);border:1px solid var(--border);border-radius:14px}
[data-testid=stExpander] details{border:0}
[data-testid=stFileUploaderDropzone]{background:var(--field);border:1px dashed var(--border);border-radius:12px}
[data-testid=stPlotlyChart]{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:.4rem}
[data-testid=stChatMessage]{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:.7rem 1rem;margin-bottom:.5rem}
[data-testid=stChatMessage] h1,[data-testid=stChatMessage] h2,[data-testid=stChatMessage] h3{font-size:1.05rem!important;margin:.3rem 0!important}
[data-testid=stBottom],[data-testid=stBottom]>div{background:var(--bg)!important}
[data-testid=stChatInput]{background:var(--panel);border:1px solid var(--border);border-radius:16px}
[data-testid=stChatInput] textarea{background:transparent!important}
.st-key-hdr{position:sticky;top:3.4rem;z-index:100;background:var(--bgs);backdrop-filter:blur(12px);padding:.3rem 0 .5rem;border-bottom:1px solid var(--border)}
.st-key-hdr button p{font-size:.78rem}.st-key-hdr button{padding:.15rem .4rem;min-height:2rem}
.st-key-hdr [data-testid=stRadio] label>div:first-child{display:none}
.st-key-hdr [data-testid=stRadio] label{padding:.35rem 1rem;border-radius:999px;border:1px solid var(--border);background:var(--panel);cursor:pointer}
.st-key-hdr [data-testid=stRadio] label:has(input:checked){background:linear-gradient(135deg,var(--a1),var(--a2));border-color:transparent}
.st-key-hdr [data-testid=stRadio] label:has(input:checked) p{color:#fff!important;font-weight:700}
.hero{font-size:1.75rem;font-weight:800;line-height:1.2;background:linear-gradient(90deg,var(--a1),var(--a2));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;width:fit-content}
.sub{color:var(--mute);font-size:.85rem;margin-bottom:.5rem}
.kpi{background:var(--panel);border:1px solid var(--border);border-top:3px solid var(--a1);border-radius:14px;padding:.8rem 1rem}
.kpi span{color:var(--mute);font-size:.78rem;font-weight:600}.kpi b{display:block;font-size:1.6rem;line-height:1.3}
.welcome{background:var(--panel);border:1px dashed var(--border);border-radius:18px;padding:1.6rem;text-align:center;margin:1rem 0}
.welcome p{color:var(--mute)}
.tbl-wrap{overflow:auto;max-height:320px;border:1px solid var(--border);border-radius:12px}
.tbl{width:100%;border-collapse:collapse;font-size:.8rem}
.tbl th{position:sticky;top:0;background:var(--panel2);color:var(--mute);text-align:left;padding:.5rem .7rem;white-space:nowrap}
.tbl td{padding:.4rem .7rem;border-top:1px solid var(--border);white-space:nowrap}
"""
st.markdown("<style>:root{" + "".join(f"--{k}:{v};" for k, v in P.items()) + "}" + CSS + "</style>", unsafe_allow_html=True)

def sty(f):
    f = go.Figure(f)
    f.update_layout(template="plotly_dark" if dark else "plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=COLORS, font=dict(color=P["text"]), margin=dict(l=10, r=10, t=50, b=10))
    return f

def html_table(d, n=30):
    return '<div class="tbl-wrap">' + d.head(n).to_html(index=False, border=0, classes="tbl", na_rep="–") + "</div>"

# ---------------- LLM (free Gemini) ----------------
@st.cache_resource
def gclient():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"], http_options={"timeout": 60000})  # 60s max, no endless waiting

def gemini(prompt):
    import time
    from google.genai import types
    main = st.secrets.get("GEMINI_MODEL", "gemini-3.8-flash")
    extra = [m.strip() for m in str(st.secrets.get("GEMINI_FALLBACKS", "")).split(",") if m.strip()]
    last = None
    for model in [main] + extra:               # try main model, then backups
        for attempt in range(3):               # retry on busy servers
            for use_cfg in (True, False):      # low thinking = faster; plain if unsupported
                try:
                    cfg = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="low")) if use_cfg else None
                    return gclient().models.generate_content(model=model, contents=prompt, config=cfg).text
                except Exception as e:
                    last = e
                    if use_cfg and "thinking" in str(e).lower(): continue
                    break
            msg = str(last)
            if any(k in msg for k in ("503", "429", "UNAVAILABLE", "overloaded", "RESOURCE_EXHAUSTED")):
                time.sleep(2 * (attempt + 1)); continue
            break                              # real error (404, bad key...) -> next model
    raise last

def groq(prompt):
    import requests
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}"},
        json={"model": st.secrets.get("GROQ_MODEL", "llama-3.3-70b-versatile"), "temperature": 0.2,
              "messages": [{"role": "user", "content": prompt}]}, timeout=60)
    if r.status_code != 200: raise RuntimeError(f"Groq {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]

def llm(prompt):
    """Tries providers in order; LLM_PRIMARY = "groq" or "gemini" (default gemini)."""
    order = [gemini, groq] if st.secrets.get("LLM_PRIMARY", "gemini") != "groq" else [groq, gemini]
    if not st.secrets.get("GROQ_API_KEY"): order = [gemini]
    last = None
    for fn in order:
        try: return fn(prompt)
        except Exception as e: last = e
    raise last

# ---------------- Memory (Google Sheet "Learning" tab) ----------------
def sheet_url(tab): return f"https://docs.google.com/spreadsheets/d/{st.secrets['SHEET_ID']}/gviz/tq?tqx=out:csv&sheet={tab}"

@st.cache_data(ttl=30)
def sheet_csv(tab): return pd.read_csv(sheet_url(tab))

def log(kind, sig, q="", code="", rating=0, note=""):
    try:
        import requests
        requests.post(st.secrets["SCRIPT_URL"], json=dict(token=st.secrets["SCRIPT_TOKEN"], kind=kind, sig=sig, question=q, code=code, rating=rating, note=note), timeout=20)
        sheet_csv.clear()
    except Exception: pass

def examples(sig):
    try: m = sheet_csv("Learning")
    except Exception: return ""
    if m.empty or "sig" not in m: return ""
    m = m[m["sig"] == sig]; out = []
    k = m[m["kind"] == "dataset_summary"].tail(1)
    if len(k): out.append("WHAT I LEARNED ABOUT THIS DATASET EARLIER: " + str(k["note"].iloc[0]))
    for _, r in m[(m["kind"] == "qa") & (m["rating"] == 1)].tail(3).iterrows():
        out.append(f"# Past GOOD example\nQ: {r['question']}\n```python\n{r['code']}\n```")
    bad = m[(m["kind"] == "qa") & (m["rating"] == -1)].tail(2)["question"].tolist()
    if bad: out.append("User disliked answers to: " + "; ".join(map(str, bad)) + ". Be more careful on similar questions.")
    return "\n\n".join(out)

def rate(m, v):
    log("qa", m["sig"], m["q"], m["code"], v)
    st.toast("Learned! Saved to the Learning tab." if v == 1 else "Noted, I will avoid this approach.")

# ---------------- Data loading ----------------
def gsheet_url(u):
    m = re.search(r"/d/([\w-]+)", u); g = re.search(r"gid=(\d+)", u)
    if not m: raise ValueError("Invalid Google Sheet link")
    return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv&gid={g.group(1) if g else 0}"

def clean(df):
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")
    for c in df.select_dtypes(["object", "string"]):
        if "date" in c.lower() or "time" in c.lower():
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def load():
    src = st.sidebar.radio("Data source", ["My Sheet (Data tab)", "Upload CSV / Excel", "Google Sheet link", "Paste data"])
    if src.startswith("My Sheet"):
        if st.sidebar.button("🔄 Reload data"): sheet_csv.clear()
        return clean(sheet_csv("Data").copy()), "sheet-data"
    if src == "Upload CSV / Excel":
        f = st.sidebar.file_uploader("File", type=["csv", "xlsx", "xls"])
        if not f: return None, ""
        if f.name.endswith(".csv"): return clean(pd.read_csv(f)), f.name
        x = pd.ExcelFile(f); sh = st.sidebar.selectbox("Sheet", x.sheet_names)
        return clean(x.parse(sh)), f"{f.name}:{sh}"
    if src == "Google Sheet link":
        u = st.sidebar.text_input("Sheet link (share: Anyone with link can view)")
        return (clean(pd.read_csv(gsheet_url(u))), "gsheet") if u else (None, "")
    t = st.sidebar.text_area("Paste from Google Sheets / Excel", height=200)
    return (clean(pd.read_csv(io.StringIO(t), sep=None, engine="python")), "pasted") if t.strip() else (None, "")

# ---------------- Profiling ----------------
def profile(df):
    out = [f"Rows={len(df)}, Cols={df.shape[1]}, Duplicate rows={int(df.duplicated().sum())}"]
    for c in df:
        out.append(f"- {c} [{df[c].dtype}] unique={df[c].nunique()} missing={int(df[c].isna().sum())} e.g. {df[c].dropna().unique()[:4].tolist()}")
    n = df.select_dtypes("number")
    if n.shape[1] > 1:
        c = n.corr().stack(); c = c[c.index.get_level_values(0) < c.index.get_level_values(1)]
        out.append("Top correlations: " + "; ".join(f"{a}~{b}={v:.2f}" for (a, b), v in c.reindex(c.abs().sort_values(ascending=False).index).head(5).items()))
    for col in n:
        q1, q3 = n[col].quantile([.25, .75]); i = q3 - q1
        o = int(((n[col] < q1 - 1.5 * i) | (n[col] > q3 + 1.5 * i)).sum())
        if o: out.append(f"Outliers in {col}: {o}")
    return "\n".join(out)

# ---------------- Safe code execution ----------------
SAFE = {k: getattr(builtins, k) for k in "len range min max sum sorted abs round enumerate zip list dict set tuple str int float bool isinstance any all map filter reversed print".split()}
BAD = ("import", "__", "open(", "exec", "eval", "os.", "sys.", "subprocess", "getattr", "globals", "input(", "st.")

def run(code, df):
    if any(b in code for b in BAD): raise ValueError("Unsafe code blocked")
    env = {"__builtins__": SAFE, "df": df.copy(), "pd": pd, "np": np, "px": px, "go": go}
    exec(code, env)
    return env.get("answer", ""), env.get("table"), env.get("fig")

LANGS = {"Auto (same as my question)": "", "हिन्दी": "Hindi (Devanagari script)", "Hinglish": "Hinglish (Hindi written in English letters)", "English": "English",
         "தமிழ்": "Tamil", "తెలుగు": "Telugu", "বাংলা": "Bengali", "मराठी": "Marathi", "ગુજરાતી": "Gujarati", "ਪੰਜਾਬੀ": "Punjabi", "ಕನ್ನಡ": "Kannada", "മലയാളം": "Malayalam"}
def lang_rule():
    l = LANGS.get(st.session_state.get("lang", "Auto (same as my question)"), "")
    if not l: return "Reply in the SAME language and script as the user's question (Hindi question -> Hindi, Hinglish -> Hinglish, English -> English)."
    return f"Reply ONLY in {l}."

def facts_for(sig):
    """Things the user taught the AI about this dataset (session + Learning tab)."""
    f = list(st.session_state.get("facts", {}).get(sig, []))
    try:
        m = sheet_csv("Learning")
        if not m.empty and "sig" in m:
            f += [str(x) for x in m[(m["sig"] == sig) & (m["kind"] == "definition")]["note"].tail(25)]
    except Exception: pass
    return list(dict.fromkeys(f))

def ask(q, df, prof, sig, hist=""):
    facts = facts_for(sig)
    facts_txt = ("USER-TAUGHT FACTS (always trust these):\n- " + "\n- ".join(facts)) if facts else ""
    prompt = "\n".join([
        "You are DataPilot, a friendly senior data analyst. A pandas DataFrame `df` is already loaded.",
        "SCHEMA:", prof, "SAMPLE:", df.head(5).to_string(), facts_txt,
        "RECENT CHAT:", hist or "(none)", "",
        "FIRST DECIDE what the user wants:",
        "1) Greeting, small talk, or a general-knowledge / non-data question -> reply EXACTLY `CHAT:` followed by a short, helpful answer. NO code.",
        "2) The user is EXPLAINING what a column, value or term means in this dataset (e.g. 'iska matlab ...', 'X means ...', 'remember that ...') -> reply `LEARN: <one-sentence fact in English using exact column names>` then a new line with a short acknowledgment.",
        "3) A question about the data -> reply with ONE ```python block only, following these rules:",
        "   - NO imports (pd, np, px, go are ready). Never modify df; use d = df.copy().",
        "   - Set `answer` = SHORT markdown (max 4 bullets or 5 lines) with real computed numbers (f-strings) and one clear insight. NO markdown headings (#).",
        "   - Optionally set `table` (DataFrame) and `fig` (plotly figure, titled, labelled, height<=380).",
        f"LANGUAGE: {lang_rule()} Applies to every reply, chart titles and labels. Keep variable and column names in English.",
        examples(sig), "", f"QUESTION: {q}"])
    err = None
    for _ in range(2):
        raw = llm(prompt).strip()
        up = raw.upper()
        if up.startswith("CHAT:"): return "", (raw[5:].strip(), None, None), {"kind": "chat"}
        if up.startswith("LEARN:"):
            fact, _, ack = raw[6:].strip().partition("\n")
            return "", (ack.strip() or "Got it, I will remember this! 🧠", None, None), {"kind": "learn", "fact": fact.strip()}
        code = re.sub(r"^```(?:python)?|```$", "", raw, flags=re.M).strip()
        try: return code, run(code, df), {"kind": "data"}
        except Exception as e:
            err = e; prompt += f"\n\nYour code failed with: {e}\nCODE:\n{code}\nFix it and reply with one python block."
    raise err

# ---------------- UI ----------------
@st.cache_data(show_spinner=False)
def auto_charts(df):
    num = [c for c in df.select_dtypes("number") if df[c].nunique() > 5 and not c.lower().endswith("id")][:3]
    dt = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cat = [c for c in df.select_dtypes(["object", "string", "category"]) if 2 <= df[c].nunique() <= 15][:3]
    figs = []
    if dt and num:
        t = df.groupby(df[dt[0]].dt.to_period("M").astype(str))[num[0]].sum().reset_index()
        figs.append(px.line(t, x=dt[0], y=num[0], markers=True, title=f"{num[0]} by month"))
    for c in cat:
        v = df[c].value_counts().head(10).reset_index(); v.columns = [c, "count"]
        figs.append(px.bar(v, x=c, y="count", title=f"{c}: count"))
    for c in num[:2]: figs.append(px.histogram(df, x=c, title=f"{c} distribution"))
    for f in figs: f.update_layout(height=290)
    return figs

def js(scroll=False):
    """Freezes the header (sticky on its wrapper too) and optionally smooth-scrolls to the newest message."""
    go_down = "[150,800,1800].forEach(t=>setTimeout(down,t));" if scroll else ""
    components.html("""<script>
const d=window.parent.document;
function stick(){let e=d.querySelector('.st-key-hdr');for(let i=0;i<4&&e&&!e.matches('[data-testid="stMainBlockContainer"]');i++){e.style.position='sticky';e.style.top='3.4rem';e.style.zIndex='100';e=e.parentElement}}
function down(){const a=d.getElementById('dp-end');if(a)a.scrollIntoView({behavior:'smooth',block:'end'})}
stick();[300,1200].forEach(t=>setTimeout(stick,t));""" + go_down + "</script>", height=0)

OV = "🔎 Overview"
hdr = st.container(key="hdr")
with hdr:   # frozen header
    st.markdown('<div class="hero">📊 DataPilot AI</div><div class="sub">Upload or paste data, then ask anything. It learns from your 👍 / 👎.</div>', unsafe_allow_html=True)
    view = st.radio("view", [OV, "💬 Ask AI"], horizontal=True, label_visibility="collapsed", key="view")
try: df, name = load()
except Exception as e: st.sidebar.error(str(e)); df = None
if df is None:
    st.info("👈 Pick a data source from the sidebar to begin."); st.stop()

st.sidebar.selectbox("🌐 Answer language", list(LANGS), key="lang")
sig = "|".join(sorted(df.columns)).lower()[:200]
prof = profile(df)
if st.session_state.get("ds") != sig + str(len(df)):
    st.session_state.update(ds=sig + str(len(df)), msgs=[], summary=None)

if view == OV:
    ks = [("📄 Rows", f"{len(df):,}"), ("🧱 Columns", df.shape[1]), ("🔢 Numeric cols", df.select_dtypes("number").shape[1]),
          ("🕳️ Missing cells", f"{int(df.isna().sum().sum()):,}"), ("♊ Duplicates", int(df.duplicated().sum()))]
    for col, (l, v) in zip(st.columns(5), ks): col.markdown(f'<div class="kpi"><span>{l}</span><b>{v}</b></div>', unsafe_allow_html=True)
    st.write("")
    figs = auto_charts(df)
    for k in range(0, len(figs), 3):
        for j, (col, f) in enumerate(zip(st.columns(3), figs[k:k + 3])): col.plotly_chart(sty(f), width="stretch", theme=None, key=f"ac{k}{j}")
    l, r = st.columns([3, 2])
    with l:
        with st.expander("📄 Data preview", expanded=True): st.markdown(html_table(df, 30), unsafe_allow_html=True)
    with r:
        with st.expander("🧠 AI summary", expanded=True):
            if st.button("Analyze this dataset", type="primary"):
                with st.spinner("Reading your data..."):
                    try:
                        st.session_state.summary = llm(f"You are a senior analyst. Explain this dataset in SHORT bullet points (max 8 bullets, no long paragraphs). {lang_rule()} Cover: what it is about, key patterns, data quality issues, and 4 smart questions to ask next.\n{prof}\nSAMPLE:\n{df.head(8).to_string()}")
                        log("dataset_summary", sig, note=st.session_state.summary[:1500])
                    except Exception as ex: st.error(ex)
            st.markdown(st.session_state.summary or "_Click the button to get a quick summary._")
    js()
else:
    nums = [c for c in df.select_dtypes("number") if not c.lower().endswith("id")]
    cats = [c for c in df.select_dtypes(["object", "string", "category"]) if 2 <= df[c].nunique() <= 15]
    chips = ["How many rows are there in total?", "Any data quality issues?"]
    if nums and cats: chips.insert(1, f"{nums[0]} by {cats[0]} bar chart")
    if nums: chips.append(f"Top 5 outliers in {nums[0]}")
    with hdr:
        for col, ch in zip(st.columns(len(chips)), chips):
            col.button(ch, key="chip" + ch, width="stretch", on_click=lambda ch=ch: st.session_state.update(pq=ch))
    if not st.session_state.msgs:
        st.markdown('<div class="welcome"><h3>👋 Ask anything about your data</h3><p>Try a suggestion above, type a question like <i>month-wise sales trend</i>, or just say hi.</p></div>', unsafe_allow_html=True)
    for i, m in enumerate(st.session_state.msgs):          # oldest -> newest
        with st.chat_message("user"): st.write(m["q"])
        with st.chat_message("assistant"):
            st.markdown(m["answer"] or "_Done._")
            if m["fig"] is not None: st.plotly_chart(sty(m["fig"]), width="stretch", theme=None, key=f"f{i}")
            if m["table"] is not None:
                t = m["table"] if isinstance(m["table"], pd.DataFrame) else pd.DataFrame(m["table"])
                st.markdown(html_table(t, 50), unsafe_allow_html=True)
            if m.get("kind") == "learn": st.caption(f"🧠 Saved to memory: {m['fact']}")
            if m["code"]:
                with st.expander("View code"): st.code(m["code"], language="python")
                x, y, _ = st.columns([1, 1, 8])
                x.button("👍", key=f"u{i}", on_click=rate, args=(m, 1)); y.button("👎", key=f"d{i}", on_click=rate, args=(m, -1))
    q = st.chat_input("Ask about your data… e.g. 'month-wise sales trend chart'") or st.session_state.pop("pq", None)
    if q:
        with st.chat_message("user"): st.write(q)
        with st.chat_message("assistant"): ph = st.empty()
    st.markdown('<div id="dp-end" style="scroll-margin-bottom:110px"></div>', unsafe_allow_html=True)
    js(scroll=bool(q) or st.session_state.pop("scroll", False))   # jump to the newest message
    if q:
        hist = "\n".join(f"User: {m['q']}\nAI: {str(m['answer'])[:200]}" for m in st.session_state.msgs[-3:])
        out = {}
        def work():
            try: out["r"] = ask(q, df, prof, sig, hist)
            except Exception as ex: out["e"] = ex
        t = threading.Thread(target=work); add_script_run_ctx(t, get_script_run_ctx()); t.start()
        steps = ["📖 Reading your data…", "🧠 Understanding your question…", "🧮 Crunching the numbers…", "📊 Drawing your chart…",
                 "🍳 Cooking up your result, almost ready…", "✨ Adding the final touches…", "⏳ Please wait, good things take a moment…"]
        t0, k = time.time(), 0
        while t.is_alive():
            ph.markdown(f"**{steps[min(k, len(steps) - 1)]}**  \n<span style='opacity:.6'>{int(time.time() - t0)}s elapsed</span>", unsafe_allow_html=True)
            time.sleep(1.4); k += 1
        if "r" in out:
            code, (ans, tbl, fig), meta = out["r"]
            if meta["kind"] == "learn" and meta.get("fact"):
                st.session_state.setdefault("facts", {}).setdefault(sig, []).append(meta["fact"])
                log("definition", sig, q=q, note=meta["fact"])
            st.session_state.msgs.append(dict(q=q, answer=ans, table=tbl, fig=fig, code=code, sig=sig, kind=meta["kind"], fact=meta.get("fact", "")))
        else:
            st.session_state.msgs.append(dict(q=q, answer=f"⚠️ Could not answer: {out.get('e')}. Try rephrasing.", table=None, fig=None, code="", sig=sig, kind="data"))
        st.session_state.scroll = True
        st.rerun()
