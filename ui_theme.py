"""Tema UI biru modern bergaya workspace percakapan untuk Streamlit."""
import streamlit as st

THEME_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
  color-scheme: light;
  --primary-color:#2563eb; --background-color:#f5f8ff; --secondary-background-color:#fff; --text-color:#172b4d;
  --blue-950:#102a56; --blue-800:#1746a2; --blue-700:#1d4ed8; --blue-600:#2563eb;
  --blue-100:#dbeafe; --blue-50:#eff6ff; --canvas:#f5f8ff; --ink:#172b4d;
  --muted:#526581; --line:#dce5f2; --radius:16px;
}
/* Typography: explicitly enforce readable dark text on light surfaces. */
html,body,.stApp,.stApp p,.stApp span,.stApp label,.stApp li,.stApp td,.stApp th,.stApp small,
.stApp [data-testid="stMarkdownContainer"],.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,.stApp [data-testid="stMarkdownContainer"] span,
.stApp [data-testid="stText"],.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stMetricLabel"],.stApp [data-testid="stMetricValue"],.stApp [data-testid="stMetricDelta"] {
  font-family:'Inter',system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  color:var(--ink);
}
.stApp{background:var(--canvas);color:var(--ink)}
.stApp [data-testid="stIconMaterial"],.stApp .material-icons,.stApp .material-symbols-rounded,.stApp .material-symbols-outlined{
  font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;font-feature-settings:'liga';letter-spacing:normal
}
.main .block-container{max-width:1320px;padding-top:1.8rem;padding-bottom:3.5rem}
header[data-testid="stHeader"]{background:rgba(245,248,255,.94)}
div[data-testid="stToolbar"]{visibility:hidden;height:0} div[data-testid="stStatusWidget"]{display:none}
h1,.stApp h1{color:var(--blue-950) !important;font-weight:800;letter-spacing:-.045em}
h2,h3,h4,h5,h6{color:var(--ink) !important;font-weight:700;letter-spacing:-.025em}
[data-testid="stCaptionContainer"],.stCaption,[data-testid="stCaptionContainer"] *{color:var(--muted) !important}

/* Sidebar */
section[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}
section[data-testid="stSidebar"]>div{padding-top:1.15rem}
section[data-testid="stSidebar"] img{display:block;margin:0 auto}
section[data-testid="stSidebar"] h3{color:var(--blue-950) !important;font-size:1.02rem}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *{color:var(--muted) !important}
.sidebar-app-brand{margin:1.25rem .2rem .25rem;padding:1rem .6rem;border-top:1px solid var(--line);color:var(--blue-800) !important;font-size:.86rem;font-weight:700;line-height:1.5;text-align:center}

/* One shared button style; primary buttons override it below. */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{
  min-height:40px;border:1px solid #cbd9ec;border-radius:12px;background:#fff;color:var(--blue-800) !important;
  font-weight:650;padding:.52rem 1rem;transition:background .16s ease,border-color .16s ease,box-shadow .16s ease,transform .16s ease
}
.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{background:var(--blue-50);border-color:#a9c8ff;color:var(--blue-800) !important;box-shadow:0 3px 10px rgba(37,99,235,.08)}
button[kind="primary"],button[kind="primaryFormSubmit"],button[data-testid="stBaseButton-primary"]{
  background:var(--blue-600) !important;border-color:var(--blue-600) !important;color:#fff !important;font-weight:700 !important;box-shadow:0 4px 12px rgba(37,99,235,.16)
}
button[kind="primary"] *,button[kind="primaryFormSubmit"] *,button[data-testid="stBaseButton-primary"] *{color:#fff !important}
button[kind="primary"]:hover,button[kind="primaryFormSubmit"]:hover,button[data-testid="stBaseButton-primary"]:hover{background:var(--blue-800) !important;border-color:var(--blue-800) !important;color:#fff !important}
button:focus-visible{outline:3px solid #bfdbfe !important;outline-offset:2px}
section[data-testid="stSidebar"] .stButton>button{width:100%;justify-content:flex-start;text-align:left;min-height:42px;border-color:transparent;border-radius:11px;background:transparent;color:#334966 !important;font-weight:600}
section[data-testid="stSidebar"] .stButton>button:hover{background:var(--blue-50);border-color:#d7e6ff;color:var(--blue-700) !important;transform:translateY(-1px)}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:var(--blue-600) !important;border-color:var(--blue-600) !important;color:#fff !important}
section[data-testid="stSidebar"] .stButton>button[kind="primary"] *{color:#fff !important}

/* Inputs, forms, select menus and placeholders */
[data-testid="stForm"]{background:#fff;border:1px solid var(--line);border-radius:18px;padding:1.4rem;box-shadow:0 8px 26px rgba(30,64,120,.045)}
div[data-baseweb="input"],div[data-baseweb="select"]>div,div[data-baseweb="textarea"]{border-radius:12px !important;background:#fff !important;border-color:#cbd9ec}
div[data-baseweb="input"] *,div[data-baseweb="select"] *,div[data-baseweb="textarea"] *{color:var(--ink) !important}
input,textarea,[data-baseweb="input"] input,[data-baseweb="textarea"] textarea{color:var(--ink) !important;-webkit-text-fill-color:var(--ink) !important;caret-color:var(--blue-700) !important}
input::placeholder,textarea::placeholder,[data-baseweb="input"] input::placeholder,[data-baseweb="textarea"] textarea::placeholder{color:#687b96 !important;-webkit-text-fill-color:#687b96 !important;opacity:1 !important}
div[data-baseweb="input"]:focus-within,div[data-baseweb="textarea"]:focus-within,div[data-baseweb="select"]>div:focus-within{border-color:var(--blue-600) !important;box-shadow:0 0 0 3px #dbeafe}
[data-testid="stFileUploader"] section{background:#fff;border:1.5px dashed #a8c5f5;border-radius:16px}
[data-testid="stFileUploader"] section:hover{border-color:var(--blue-600);background:var(--blue-50)}

/* Chat bubbles and composer */
[data-testid="stChatMessage"]{background:#fff;color:var(--ink) !important;border:1px solid #e0e8f4;border-radius:16px;padding:1rem 1.15rem;margin-bottom:.45rem}
[data-testid="stChatMessage"] *{color:var(--ink)}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){background:#edf4ff;border-color:#d5e5ff}
[data-testid="stChatInput"]{border:1px solid #cbd9ec !important;border-radius:20px !important;background:#fff !important;box-shadow:0 8px 26px rgba(31,67,130,.08)}
[data-testid="stChatInput"]:focus-within{border-color:var(--blue-600) !important;box-shadow:0 0 0 3px #dbeafe,0 8px 26px rgba(31,67,130,.08)}
[data-testid="stChatInput"] textarea{background:transparent !important;color:var(--ink) !important;-webkit-text-fill-color:var(--ink) !important}
[data-testid="stChatInput"] textarea::placeholder{color:#687b96 !important;-webkit-text-fill-color:#687b96 !important}

/* Tables and other widgets */
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid var(--line);border-radius:13px;overflow:hidden}
[data-testid="stImage"] img{border-radius:14px}
[data-testid="stAlert"]{border-radius:13px}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:#fff}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *{color:var(--ink) !important}
hr{border-color:var(--line)}
.login-brand{text-align:center;margin:2.2rem 0 1.3rem;padding:1.5rem 0 .5rem}
.login-brand h1{font-size:clamp(26px,3vw,38px);margin-bottom:.45rem;background:linear-gradient(100deg,#123b86 15%,#3b82f6 85%);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.login-brand p{color:var(--muted) !important;margin:0;font-size:.96rem}
footer{visibility:hidden} #MainMenu{visibility:hidden}
@media(max-width:700px){.main .block-container{padding-top:1rem;padding-left:1rem;padding-right:1rem}[data-testid="stForm"]{padding:1rem}}
</style>
"""


def apply_theme() -> None:
    """Terapkan tema biru modern pada seluruh halaman Streamlit."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)
