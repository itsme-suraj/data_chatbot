import re, io, builtins
import streamlit as st, pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
from google import genai

st.set_page_config(page_title="DataPilot AI", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.5rem;max-width:1200px}
h1{font-weight:800;letter-spacing:-.5px}
[data-testid=stMetric]{background:var(--secondary-background-color);padding:14px 16px;border-radius:12px;border-left:4px solid #0f766e}
</style>""", unsafe_allow_html=True)

# ---------------- LLM (free Gemini) ----------------
@st.cache_resource
def gclient():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"], http_options={"timeout": 60000})  # 60s max, no endless waiting

def llm(prompt):
    model = st.secrets.get("GEMINI_MODEL", "gemini-3.8-flash")
    try:  # low thinking = much faster answers
        from google.genai import types
        cfg = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="low"))
    except Exception:
        cfg = None
    try:
        return gclient().models.generate_content(model=model, contents=prompt, config=cfg).text
    except Exception as e:
        if cfg is not None and "thinking" in str(e).lower():  # model/SDK does not support it -> retry plain
            return gclient().models.generate_content(model=model, contents=prompt).text
        raise

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
    for c in df.select_dtypes("object"):
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

def ask(q, df, prof, sig):
    prompt = f"""You are a senior data analyst. A pandas DataFrame `df` is already loaded.
SCHEMA:
{prof}
SAMPLE:
{df.head(5).to_string()}

RULES: write python only. NO imports (pd, np, px, go are ready). Never modify df; use d = df.copy().
Set `answer` = markdown string with real computed numbers (use f-strings) and a clear business insight.
Optionally set `table` (DataFrame) and `fig` (plotly figure, template='plotly_white', titled, labelled).
Reply with ONE ```python block only.
{examples(sig)}

QUESTION: {q}"""
    err = None
    for _ in range(2):
        code = re.sub(r"^```(?:python)?|```$", "", llm(prompt).strip(), flags=re.M).strip()
        try: return code, run(code, df)
        except Exception as e:
            err = e; prompt += f"\n\nYour code failed with: {e}\nCODE:\n{code}\nFix it and reply with one python block."
    raise err

# ---------------- UI ----------------
st.title("📊 DataPilot AI")
st.caption("Upload or paste your data, then ask anything. It learns from your 👍 / 👎.")
try: df, name = load()
except Exception as e: st.sidebar.error(str(e)); df = None
if df is None:
    st.info("👈 Pick a data source from the sidebar to begin."); st.stop()

sig = "|".join(sorted(df.columns)).lower()[:200]
prof = profile(df)
if st.session_state.get("ds") != sig + str(len(df)):
    st.session_state.update(ds=sig + str(len(df)), msgs=[], summary=None)

t1, t2 = st.tabs(["🔎 Overview", "💬 Ask AI"])
with t1:
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(df):,}"); b.metric("Columns", df.shape[1])
    c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}"); d.metric("Duplicates", int(df.duplicated().sum()))
    st.dataframe(df.head(50), use_container_width=True)
    miss = df.isna().sum(); miss = miss[miss > 0]
    if len(miss): st.plotly_chart(px.bar(miss, title="Missing values by column", template="plotly_white"), use_container_width=True)
    if st.button("🧠 Analyze this dataset", type="primary"):
        with st.spinner("Reading your data..."):
            try: st.session_state.summary = llm(f"You are a senior analyst. Explain this dataset in simple Hinglish-friendly English: what it is about, key patterns, data quality issues, and 5 smart questions to ask next.\n{prof}\nSAMPLE:\n{df.head(8).to_string()}")
            except Exception as e: st.error(e)
        if st.session_state.summary: log("dataset_summary", sig, note=st.session_state.summary[:1500])
    if st.session_state.summary: st.markdown(st.session_state.summary)

with t2:
    for i, m in enumerate(st.session_state.msgs):
        with st.chat_message("user"): st.write(m["q"])
        with st.chat_message("assistant"):
            st.markdown(m["answer"] or "_Done._")
            if m["fig"] is not None: st.plotly_chart(m["fig"], use_container_width=True, key=f"f{i}")
            if m["table"] is not None: st.dataframe(m["table"], use_container_width=True)
            with st.expander("View code"): st.code(m["code"], language="python")
            x, y, _ = st.columns([1, 1, 8])
            x.button("👍", key=f"u{i}", on_click=rate, args=(m, 1)); y.button("👎", key=f"d{i}", on_click=rate, args=(m, -1))
    if q := st.chat_input("Ask about your data… e.g. 'month-wise sales trend chart'"):
        with st.spinner("Analyzing..."):
            try:
                code, (ans, tbl, fig) = ask(q, df, prof, sig)
                st.session_state.msgs.append(dict(q=q, answer=ans, table=tbl, fig=fig, code=code, sig=sig))
            except Exception as e:
                st.session_state.msgs.append(dict(q=q, answer=f"⚠️ Could not answer: {e}. Try rephrasing.", table=None, fig=None, code="", sig=sig))
        st.rerun()
