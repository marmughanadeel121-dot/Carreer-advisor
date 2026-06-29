"""
app.py — AI Career & University Advisor
================================================================================
A production-ready Streamlit web application powered by Google Gemini 2.5 Flash.

Architecture: Multi-step LLM orchestration with a state-machine design pattern.
The app cycles through three states (IDLE → QUESTIONING → READY) driven by a
"Missing Data Audit" loop that asks the LLM whether it has enough context
to produce definitive career and university recommendations.

INSTALLATION:
    pip install streamlit google-genai pypdf

USAGE:
    streamlit run app.py
================================================================================
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS
# All third-party imports wrapped in try/except so we can show friendly errors
# if a package is missing rather than crashing with a raw traceback.
# ══════════════════════════════════════════════════════════════════════════════
import streamlit as st    # Core web-framework that drives the entire UI
import io                 # In-memory byte buffer required by pypdf
import re                 # Regular-expression parsing of LLM output sections

# ── pypdf: PDF text extraction ────────────────────────────────────────────────
try:
    from pypdf import PdfReader
    _PYPDF_OK = True
except ImportError:
    _PYPDF_OK = False

# ── google-genai: modern unified SDK for Gemini models ───────────────────────
# (install: pip install google-genai)
try:
    from google import genai
    from google.genai import types as ga_types
    _GENAI_OK = True
except ImportError:
    _GENAI_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PAGE CONFIGURATION
# MUST be the very first Streamlit call in the script; any other st.* call
# before this will raise a StreamlitAPIException.
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Career & University Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "AI Career & University Advisor — "
            "Powered by Google Gemini 2.5 Flash. "
            "Built with Streamlit & pypdf."
        )
    },
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CUSTOM CSS INJECTION
# Streamlit's st.markdown(unsafe_allow_html=True) lets us inject a <style>
# block that overrides default Streamlit styles for a polished look.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* ── App title gradient text ─────────────────────────────────────────── */
    .app-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
        margin-bottom: 0;
    }
    .app-subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-top: 0.15rem;
        margin-bottom: 0.25rem;
    }

    /* ── Highlight cards (info / success) ───────────────────────────────── */
    .card-info {
        background: linear-gradient(135deg, #f0f4ff, #faf0ff);
        border-left: 5px solid #8b5cf6;
        padding: 0.85rem 1.4rem;
        border-radius: 0 0.75rem 0.75rem 0;
        margin: 0.6rem 0;
        color: #374151;
    }
    .card-success {
        background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
        border-left: 5px solid #10b981;
        padding: 0.85rem 1.4rem;
        border-radius: 0 0.75rem 0.75rem 0;
        margin: 0.6rem 0;
        color: #065f46;
    }

    /* ── File-upload preview boxes ───────────────────────────────────────── */
    .file-empty {
        border: 2px dashed #d1d5db;
        border-radius: 0.75rem;
        padding: 1.5rem;
        text-align: center;
        color: #6b7280;
        background: #f9fafb;
    }
    .file-ready {
        border: 2px solid #10b981;
        border-radius: 0.75rem;
        padding: 1.2rem;
        color: #065f46;
        background: #ecfdf5;
        line-height: 1.7;
    }

    /* ── Sidebar dark theme ──────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e7ff !important; }
    [data-testid="stSidebar"] hr { border-color: #4338ca !important; }
    [data-testid="stSidebar"] .stTextInput input {
        background: #3730a3 !important;
        border-color: #6366f1 !important;
        color: #e0e7ff !important;
    }

    /* ── Metric value colour ─────────────────────────────────────────────── */
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #6366f1 !important;
    }

    /* ── Progress bar gradient ───────────────────────────────────────────── */
    .stProgress > div > div { background: linear-gradient(90deg, #6366f1, #a855f7); }

    /* ── Tab strip ───────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }

    /* ── Footer ──────────────────────────────────────────────────────────── */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.77rem;
        padding: 1.5rem 0 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STATE MACHINE CONSTANTS
# Three named states drive every conditional block in the UI.
# ══════════════════════════════════════════════════════════════════════════════
_IDLE        = "IDLE"          # Initial state — waiting for profile submission
_QUESTIONING = "QUESTIONING"   # AI is asking clarifying questions
_READY       = "READY"         # Analysis is complete; show recommendations


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SESSION STATE INITIALISATION
#
# Streamlit re-executes the entire script on EVERY user interaction (button
# click, text input, file upload, etc.).  Session state is the only persistent
# memory across reruns.  We guard each key with "if key not in" so existing
# values are never overwritten when the script re-runs.
#
# IMPORTANT: list/dict defaults are always recreated as fresh objects to avoid
# Python shared-reference bugs where two variables point to the same list.
# ══════════════════════════════════════════════════════════════════════════════

# Master defaults table — used by both the initialiser and full_reset()
_DEFAULTS: dict = {
    "app_state":        _IDLE,    # current node in the state machine
    "cv_text":          "",        # raw text extracted from the uploaded PDF
    "hobbies_text":     "",        # user's typed hobbies
    "context_profile":  "",        # hidden accumulated context fed to the LLM
    "chat_history":     [],        # visible Q&A messages  [{role, content}, …]
    "recommendations":  "",        # final markdown analysis text
    "audit_pending":    False,     # True → audit engine must fire on this rerun
    "question_count":   0,         # number of clarifying Q&A rounds completed
    "followup_msgs":    [],        # post-results follow-up conversation
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        # Recreate mutable containers so session state gets its own objects
        if isinstance(_v, list):
            st.session_state[_k] = []
        elif isinstance(_v, dict):
            st.session_state[_k] = {}
        else:
            st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — HELPER / LOGIC FUNCTIONS
# Pure logic separated from UI code for readability and testability.
# ══════════════════════════════════════════════════════════════════════════════

def extract_pdf_text(uploaded_file) -> str:
    """
    Extract all readable text from a Streamlit UploadedFile (PDF).

    Process:
        1. Read the file bytes into an in-memory BytesIO buffer.
        2. Initialise pypdf's PdfReader from that buffer.
        3. Iterate over every page, calling page.extract_text().
        4. Return all page texts joined into one string.

    Args:
        uploaded_file: The Streamlit UploadedFile object from st.file_uploader.

    Returns:
        A single string with page-labelled text from every page.

    Raises:
        ValueError: PDF is encrypted, or contains no extractable text layer
                    (i.e., it is a scanned / image-only document).
        Exception:  Any pypdf or I/O error (corrupted file, unusual PDF variant).
    """
    # Read raw bytes and wrap in a seekable buffer (pypdf needs seekability)
    raw_bytes = uploaded_file.read()
    pdf_buffer = io.BytesIO(raw_bytes)

    reader = PdfReader(pdf_buffer)

    # Guard: encrypted / password-protected PDFs cannot be parsed
    if reader.is_encrypted:
        raise ValueError(
            "This PDF is password-protected or encrypted. "
            "Please remove the password before uploading."
        )

    # Accumulate text page-by-page with a page label for context
    page_texts: list[str] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            page_texts.append(f"[Page {page_num}]\n{text.strip()}")

    combined = "\n\n".join(page_texts)

    # Guard: if no text was extracted the PDF is likely image-based (scanned)
    if not combined.strip():
        raise ValueError(
            "No readable text was found in the PDF. "
            "It may be a scanned / image-based document. "
            "Please use a text-based PDF or export your CV from Word / Google Docs."
        )

    return combined


def call_llm(api_key: str, prompt: str, system: str = "") -> str:
    """
    Send a prompt to Google Gemini 2.5 Flash and return the response text.

    Uses the modern google-genai SDK pattern:
        client = genai.Client(api_key=...)
        response = client.models.generate_content(model=..., contents=..., config=...)

    Args:
        api_key: The user's Gemini API key.
        prompt:  The main user-facing content/instruction.
        system:  Optional system-level role instruction for the model.

    Returns:
        The model's plain-text response string.

    Raises:
        Any exception from the google-genai SDK (invalid key, quota, network, …).
    """
    client = genai.Client(api_key=api_key)

    if system:
        # Build a config object that includes the system instruction and tunables
        config = ga_types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.75,       # Slight creativity for nuanced career advice
            max_output_tokens=8192, # Generous limit for detailed recommendations
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )
    else:
        # Simple call without a system instruction (used for follow-up questions)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

    return response.text


def build_initial_context(cv_text: str, hobbies: str) -> str:
    """
    Assemble the initial hidden context profile from the user's CV and hobbies.

    This string is the seed for st.session_state.context_profile.
    It grows with each Q&A exchange as the user answers clarifying questions.

    Args:
        cv_text: Text extracted from the CV PDF.
        hobbies: User's typed hobbies and interests.

    Returns:
        A formatted multi-section string ready for LLM consumption.
    """
    return (
        "=== HOBBIES & PERSONAL INTERESTS (provided by the student) ===\n"
        f"{hobbies}\n\n"
        "=== CV / RESUME CONTENT (extracted from uploaded PDF) ===\n"
        f"{cv_text}\n"
    )


def run_missing_data_audit(api_key: str) -> tuple[str, str]:
    """
    The Missing-Data Audit — the core orchestration brain.

    Sends the accumulated context_profile to Gemini with a strict meta-prompt
    that forces the model to either:
        [NEEDS_INFO]  →  ask ONE more clarifying question, or
        [READY]       →  produce the full career + university analysis.

    Args:
        api_key: The user's Gemini API key.

    Returns:
        A 2-tuple:
            ("NEEDS_INFO", question_string)   — when more context is needed
            ("READY",      recommendations_md) — when analysis is complete

    Raises:
        Exception: Any API-level error propagated from call_llm().
    """

    # ── SYSTEM INSTRUCTION ────────────────────────────────────────────────────
    # Sets the LLM's persona and enforces the strict protocol.
    system_instruction = """\
You are Dr. Alex — a world-class AI Career Counsellor and University Admissions \
Expert with 25 years of global experience helping students find their ideal path.

ASSESSMENT PROTOCOL (follow exactly — no deviations):

STEP 1 — REVIEW the student profile provided.

STEP 2 — EVALUATE whether you have ENOUGH data for SPECIFIC, ACTIONABLE \
recommendations. The critical data you need a reasonable picture of:
  • Academic level and performance (grades, GPA, A-levels, SAT, IB, etc.)
  • University budget (approximate range or "open / scholarship only")
  • Geographic preferences (specific countries/regions, or open to anywhere)
  • Career direction sense (any career leanings, even vague ones)
  • Practical vs. academic preference (hands-on / vocational vs. research / theory)

STEP 3 — DECIDE:
  • If CRITICAL info is missing AND fewer than 4 questions have been asked →
      Reply: [NEEDS_INFO] followed by EXACTLY ONE warm, conversational question.
      Pick the MOST IMPORTANT missing piece. Never ask compound questions.
  • If you have SUFFICIENT info OR ≥ 4 questions have already been asked →
      Reply: [READY] followed by the complete analysis in the exact format shown.

ABSOLUTE FORMAT RULES:
1. Your response MUST begin with [NEEDS_INFO] or [READY]. Zero preamble.
2. After [NEEDS_INFO]: write exactly ONE question. Nothing else.
3. After [READY]: write the full analysis in the specified markdown structure.
4. Tone: warm, expert, encouraging — like a mentor who genuinely cares.\
"""

    # ── REQUIRED OUTPUT FORMAT (injected into the prompt when READY) ─────────
    # Defined as a string so it is easy to edit / extend without touching logic.
    output_format = """\
## 🎯 Top 3 Best-Fit Career Paths

### Career Path 1: [Job Title]
**Why this fits your profile:** [2–3 sentences connecting specific CV/hobby/answer details to this role]
**Day-to-day responsibilities:**
- [Responsibility 1]
- [Responsibility 2]
- [Responsibility 3]
**Key skills to develop:** [comma-separated list]
**Estimated salary range:** [realistic figures with region context]
**Growth trajectory:** [one sentence: Junior → Mid → Senior]
**Best entry industries:** [2–3 industries]

---

### Career Path 2: [Job Title]
**Why this fits your profile:** [2–3 specific sentences]
**Day-to-day responsibilities:**
- [Responsibility 1]
- [Responsibility 2]
- [Responsibility 3]
**Key skills to develop:** [comma-separated list]
**Estimated salary range:** [realistic range]
**Growth trajectory:** [one sentence]
**Best entry industries:** [2–3 industries]

---

### Career Path 3: [Job Title]
**Why this fits your profile:** [2–3 specific sentences]
**Day-to-day responsibilities:**
- [Responsibility 1]
- [Responsibility 2]
- [Responsibility 3]
**Key skills to develop:** [comma-separated list]
**Estimated salary range:** [realistic range]
**Growth trajectory:** [one sentence]
**Best entry industries:** [2–3 industries]

---

## 🎓 Top 3 Recommended University Majors & Programs

### Major 1: [Program / Major Name]
**Field:** [broad academic field]
**Why it suits your profile:** [2–3 specific sentences]
**Core subjects you will study:**
- [Subject 1]
- [Subject 2]
- [Subject 3]
**Top universities offering this program:**
- [University Name] — [Country]
- [University Name] — [Country]
- [University Name] — [Country]
- [University Name] — [Country]
**Typical duration:** [X years]
**Typical entry requirements:** [grades, test scores, prerequisites, language tests]
**Common career outcomes:** [where graduates typically end up]

---

### Major 2: [Program / Major Name]
**Field:** [broad academic field]
**Why it suits your profile:** [2–3 specific sentences]
**Core subjects you will study:**
- [Subject 1]
- [Subject 2]
**Top universities offering this program:**
- [University Name] — [Country]
- [University Name] — [Country]
- [University Name] — [Country]
**Typical duration:** [X years]
**Typical entry requirements:** [requirements]
**Common career outcomes:** [outcomes]

---

### Major 3: [Program / Major Name]
**Field:** [broad academic field]
**Why it suits your profile:** [2–3 specific sentences]
**Core subjects you will study:**
- [Subject 1]
- [Subject 2]
**Top universities offering this program:**
- [University Name] — [Country]
- [University Name] — [Country]
- [University Name] — [Country]
**Typical duration:** [X years]
**Typical entry requirements:** [requirements]
**Common career outcomes:** [outcomes]

---

## 📋 Your Personalised 6-Month Action Plan

1. **Month 1–2 — [Action Title]:** [Specific, measurable task with a clear outcome]
2. **Month 2–3 — [Action Title]:** [Specific task]
3. **Month 3–4 — [Action Title]:** [Specific task]
4. **Month 4–5 — [Action Title]:** [Specific task]
5. **Month 5–6 — [Milestone]:** [Achievement to celebrate and document]
6. **Ongoing habit:** [Daily or weekly practice that compounds over time]

---

## 💡 Your Unique Strength & Personal Insight

[Write 3–4 sentences identifying the student's most distinctive combination \
of traits drawn directly from their profile. Be specific, not generic. Say \
something they may not have articulated about themselves. End with a forward-looking, \
motivating statement about their particular potential.]\
"""

    # ── USER PROMPT ───────────────────────────────────────────────────────────
    # Contains the live context profile plus metadata that helps the LLM decide
    # when to force the READY state.
    user_prompt = (
        "Apply the Assessment Protocol to the student profile below.\n\n"
        f"{'=' * 64}\n"
        "ACCUMULATED STUDENT PROFILE:\n"
        f"{'=' * 64}\n"
        f"{st.session_state.context_profile}\n"
        f"{'=' * 64}\n\n"
        f"METADATA:\n"
        f"  Clarifying questions already asked: {st.session_state.question_count}\n"
        "  Rule: If this count is ≥ 4, you MUST reply [READY] regardless of any gaps.\n\n"
        f"IF YOUR DECISION IS [READY], use this exact markdown structure:\n\n"
        f"{output_format}"
    )

    # ── LLM CALL ─────────────────────────────────────────────────────────────
    raw_response = call_llm(
        api_key=api_key,
        prompt=user_prompt,
        system=system_instruction,
    ).strip()

    # ── RESPONSE PARSING ──────────────────────────────────────────────────────
    # Primary path: strict tag detection
    if raw_response.startswith("[NEEDS_INFO]"):
        question = raw_response.replace("[NEEDS_INFO]", "", 1).strip()
        return "NEEDS_INFO", question

    if raw_response.startswith("[READY]"):
        recommendations = raw_response.replace("[READY]", "", 1).strip()
        return "READY", recommendations

    # Fallback: LLM ignored the tag format — infer intent from response size
    # Short responses with a "?" are almost certainly clarifying questions.
    # Longer responses are almost certainly the analysis.
    if len(raw_response) < 500 and "?" in raw_response:
        return "NEEDS_INFO", raw_response
    return "READY", raw_response


def parse_sections(recommendations: str) -> dict[str, str]:
    """
    Split the full recommendations markdown into named display sections.

    Uses regex anchored on the exact emoji+heading patterns defined in the
    output_format template.  Returns empty strings for any section not found,
    so callers can apply fallback logic without crashing.

    Args:
        recommendations: The full markdown string returned by the LLM.

    Returns:
        dict with keys: 'careers', 'universities', 'action_plan', 'insight'.
    """
    patterns: dict[str, str] = {
        "careers":      r"(## 🎯 Top 3 Best-Fit Career Paths.*?)(?=\n## |\Z)",
        "universities": r"(## 🎓 Top 3 Recommended University Majors.*?)(?=\n## |\Z)",
        "action_plan":  r"(## 📋 Your Personalised 6-Month Action Plan.*?)(?=\n## |\Z)",
        "insight":      r"(## 💡 Your Unique Strength.*?)(?=\n## |\Z)",
    }

    sections: dict[str, str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, recommendations, re.DOTALL)
        sections[key] = match.group(1).strip() if match else ""

    # If NO section was extracted (LLM used different headings), dump everything
    # into the careers tab as a safe fallback so nothing is lost.
    if not any(sections.values()):
        sections["careers"] = recommendations

    return sections


def full_reset() -> None:
    """
    Reset ALL session state variables to their original default values.

    Always creates fresh mutable containers (lists / dicts) to avoid the Python
    shared-reference trap where two names point to the same list object.
    Called by the sidebar Reset button and the 'Analyse a New Profile' button.
    """
    for key, val in _DEFAULTS.items():
        if isinstance(val, list):
            st.session_state[key] = []
        elif isinstance(val, dict):
            st.session_state[key] = {}
        else:
            st.session_state[key] = val


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SIDEBAR LAYOUT
# Contains: API key input, How-to guide, session status, reset button, credits.
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🤖 AI Career Advisor")
    st.markdown("*Your intelligent planning companion*")
    st.markdown("---")

    # ── API Key input ─────────────────────────────────────────────────────────
    st.markdown("#### 🔑 API Configuration")
    api_key: str = st.text_input(
        label="Google Gemini API Key",
        type="password",                    # Masks the key visually
        placeholder="AIza…",
        help=(
            "Free API key from Google AI Studio:\n"
            "https://aistudio.google.com/app/apikey\n\n"
            "Your key is never stored — it lives only in this browser session."
        ),
        key="sidebar_api_key",
    )

    # Contextual feedback on key status
    if api_key and len(api_key.strip()) > 20:
        st.markdown(
            '<span style="color:#86efac;font-size:.85rem;">✅ Key provided — ready!</span>',
            unsafe_allow_html=True,
        )
    elif api_key:
        st.markdown(
            '<span style="color:#fca5a5;font-size:.85rem;">⚠️ Key looks short — please check.</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="color:#fbbf24;font-size:.85rem;">⚠️ API key required to begin.</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── How-to guide ──────────────────────────────────────────────────────────
    st.markdown("#### 📖 How to Use")
    st.markdown(
        """
1. **Enter your API key** above
2. **Describe your hobbies** in the text area
3. **Upload your CV** as a PDF
4. Click **🚀 Generate My Career Roadmap**
5. **Answer any clarifying questions**
6. **View your personalised results!**
"""
    )

    st.markdown("---")

    # ── Live session status ───────────────────────────────────────────────────
    st.markdown("#### 📊 Session Status")
    _sidebar_state = st.session_state.app_state
    _sidebar_qc = st.session_state.question_count

    if _sidebar_state == _IDLE:
        st.markdown(
            '<div style="color:#93c5fd;">⏳ Awaiting profile input</div>',
            unsafe_allow_html=True,
        )
    elif _sidebar_state == _QUESTIONING:
        st.markdown(
            f'<div style="color:#fcd34d;">💬 Gathering info ({_sidebar_qc} round(s) done)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="color:#86efac;">✅ Analysis complete!</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Reset control ─────────────────────────────────────────────────────────
    st.markdown("#### 🔄 Controls")
    if st.button(
        "🗑️ Reset Everything",
        use_container_width=True,
        key="sidebar_reset_btn",
        help="Clears all inputs, chat history, and results.",
    ):
        full_reset()
        st.rerun()   # Force immediate re-render from a clean blank state

    st.markdown("---")

    # ── Credits ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="color:#a5b4fc;font-size:.73rem;text-align:center;">'
        "Powered by Google Gemini 2.5 Flash<br>"
        "Built with Streamlit &amp; pypdf"
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN PAGE HEADER
# Rendered on every script run regardless of app state.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="app-title">🎓 AI Career & University Advisor</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="app-subtitle">'
    "Upload your CV, share your interests, and get a hyper-personalised career "
    "roadmap powered by Google Gemini AI."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — DEPENDENCY GATE
# If required packages are missing, show install instructions and stop.
# st.stop() raises a StopException that halts the rest of the script.
# ══════════════════════════════════════════════════════════════════════════════
_missing_pkgs: list[str] = []
if not _GENAI_OK:
    _missing_pkgs.append("google-genai     →   pip install google-genai")
if not _PYPDF_OK:
    _missing_pkgs.append("pypdf            →   pip install pypdf")

if _missing_pkgs:
    st.error("❌ **Missing required packages.** Install them and restart the app:")
    for _pkg in _missing_pkgs:
        st.code(_pkg, language="bash")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — API KEY GATE
# If no valid API key is present, show an info message and stop rendering.
# This prevents all downstream LLM calls from failing with cryptic errors.
# ══════════════════════════════════════════════════════════════════════════════
if not api_key or len(api_key.strip()) < 10:
    st.info(
        "👈 **Please enter your Google Gemini API key in the sidebar to begin.**\n\n"
        "You can get a **free** key in under 2 minutes at "
        "[Google AI Studio](https://aistudio.google.com/app/apikey).\n\n"
        "No credit card required — the free tier is generous enough for this app."
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — PROFILE INPUT FORM
# Only rendered when app_state == _IDLE (no active analysis in progress).
# Hides itself once the user submits so the chat / results area can take over.
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.app_state == _IDLE:

    st.markdown("### 📝 Build Your Profile")
    st.markdown(
        "*The more detail you provide, the more personalised your recommendations will be.*"
    )
    st.markdown("")

    # Two-column layout: hobbies on the left, CV upload on the right
    col_hobbies, col_cv = st.columns(2, gap="large")

    # ── Left column: hobbies & interests ──────────────────────────────────────
    with col_hobbies:
        st.markdown("#### 🎯 Hobbies & Personal Interests")
        st.markdown(
            "*What do you love doing? Include hobbies, passions, side projects, "
            "skills you enjoy, and activities that energise you — even if they seem "
            "unrelated to a career.*"
        )
        hobbies_input: str = st.text_area(
            label="Hobbies text area",
            label_visibility="collapsed",
            placeholder=(
                "Examples:\n"
                "• I love building electronics and tinkering with Arduino boards\n"
                "• I'm passionate about climate change and sustainability solutions\n"
                "• I enjoy strategy games, puzzles, and analytical challenges\n"
                "• I like teaching friends and explaining complex topics simply\n"
                "• I've written Python scripts to automate my daily tasks\n"
                "• I volunteer at a local youth coding club on weekends\n"
                "• I read widely about psychology and human behaviour\n\n"
                "Write as much as you like — be specific and honest!"
            ),
            height=295,
            key="hobbies_textarea",
        )
        # Character-count feedback to encourage detail
        _char_count = len(hobbies_input.strip()) if hobbies_input else 0
        if _char_count == 0:
            st.caption("📝 Start typing your hobbies and interests…")
        elif _char_count < 60:
            st.caption(f"✏️ {_char_count} chars — a little more detail will help a lot!")
        elif _char_count < 150:
            st.caption(f"✏️ {_char_count} chars — good start, keep going!")
        else:
            st.caption(f"✅ {_char_count} chars — great level of detail!")

    # ── Right column: CV / Resume uploader ────────────────────────────────────
    with col_cv:
        st.markdown("#### 📄 Upload Your CV / Resume")
        st.markdown(
            "*Upload a text-based PDF of your CV. We automatically extract your "
            "education, experience, and skills — nothing is ever stored or shared.*"
        )
        uploaded_file = st.file_uploader(
            label="CV PDF uploader",
            label_visibility="collapsed",
            type=["pdf"],
            help=(
                "PDF only. Must be text-based (not a scanned image). "
                "Export from Word or Google Docs for best results."
            ),
            key="cv_uploader",
        )

        # Dynamic upload feedback card
        if uploaded_file is not None:
            _bytes = len(uploaded_file.getvalue())
            _size_str = (
                f"{_bytes / 1024:.1f} KB"
                if _bytes < 1_048_576
                else f"{_bytes / 1_048_576:.2f} MB"
            )
            st.markdown(
                f'<div class="file-ready">'
                f"✅ <strong>File ready!</strong><br>"
                f"📎 <strong>{uploaded_file.name}</strong><br>"
                f"📦 Size: {_size_str}<br>"
                f"📋 Format: PDF Document"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="file-empty">'
                "📁 <strong>No file uploaded yet</strong><br><br>"
                "Drag &amp; drop your PDF here,<br>"
                "or click the <em>Browse files</em> button above."
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")  # Visual spacer

    # ── Centred call-to-action button ─────────────────────────────────────────
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        submit_clicked: bool = st.button(
            "🚀 Generate My Career Roadmap",
            type="primary",
            use_container_width=True,
            key="submit_btn",
        )

    # ── Handle submit click ────────────────────────────────────────────────────
    if submit_clicked:

        # Collect all validation failures before displaying any errors,
        # so the user sees everything at once instead of fixing one-by-one.
        _errors: list[str] = []
        if not hobbies_input or len(hobbies_input.strip()) < 30:
            _errors.append(
                "Your hobbies description is too short. "
                "Please write at least a few sentences (30+ characters)."
            )
        if uploaded_file is None:
            _errors.append(
                "No CV file detected. "
                "Please upload your CV or Resume as a PDF file."
            )

        # Display errors and abort if any exist
        for _err in _errors:
            st.error(f"❌ {_err}")

        # Only proceed when all inputs are valid
        if not _errors:

            # ── PDF text extraction ────────────────────────────────────────────
            # Wrapped in explicit try-except blocks as required by the spec.
            with st.spinner("📄 Extracting text from your CV… please wait."):
                try:
                    _cv_text = extract_pdf_text(uploaded_file)

                except ValueError as _ve:
                    # Known failure type: encrypted PDF or image-only PDF
                    st.error(
                        f"⚠️ **Could Not Read PDF Content**\n\n"
                        f"{_ve}\n\n"
                        f"*Tip: Try exporting your CV as a new PDF from Word or Google Docs.*"
                    )
                    st.stop()

                except Exception as _exc:
                    # Unknown failure: corrupted file, unusual PDF variant, etc.
                    st.error(
                        f"❌ **PDF Parsing Error**\n\n"
                        f"The file could not be read. It may be corrupted, "
                        f"in an unsupported PDF format, or missing a text layer.\n\n"
                        f"*Technical detail: {_exc}*\n\n"
                        f"*Please try re-saving or re-exporting your CV as a standard PDF.*"
                    )
                    st.stop()

            # ── Store data and build the initial context profile ───────────────
            st.session_state.cv_text = _cv_text
            st.session_state.hobbies_text = hobbies_input.strip()
            st.session_state.context_profile = build_initial_context(
                cv_text=_cv_text,
                hobbies=hobbies_input.strip(),
            )

            # ── Launch the audit loop ─────────────────────────────────────────
            # Setting audit_pending=True signals Section 12 to fire on the
            # next rerun.  Moving app_state out of _IDLE ensures the input form
            # (this section) does not re-render during the audit.
            st.session_state.audit_pending = True
            st.session_state.app_state = _QUESTIONING
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — AUDIT ENGINE  (the orchestration brain)
#
# Fires on every rerun where audit_pending is True.
# Makes a single LLM call via run_missing_data_audit() and transitions
# the state machine to either _QUESTIONING (needs more info) or _READY.
#
# Flow diagram:
#   [submit / answer] → set audit_pending=True → st.rerun()
#   → Section 12 fires → LLM call → update state → audit_pending=False → st.rerun()
#   → Section 13 (chat) OR Section 14 (results) renders
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.audit_pending:

    with st.spinner("🧠 AI is analysing your profile… this may take a few seconds."):
        try:
            _status, _content = run_missing_data_audit(api_key=api_key.strip())

            if _status == "NEEDS_INFO":
                # ── Transition: stay in QUESTIONING, add AI question to chat ──
                st.session_state.app_state = _QUESTIONING
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": _content}
                )
                st.session_state.question_count += 1

            else:
                # ── Transition: move to READY, store recommendations ───────────
                st.session_state.app_state = _READY
                st.session_state.recommendations = _content

        except Exception as _api_err:
            # ── Friendly, specific error messages based on error type ──────────
            _err_str = str(_api_err)
            if "API_KEY_INVALID" in _err_str or "invalid" in _err_str.lower():
                st.error(
                    "🔑 **Invalid API Key**\n\n"
                    "Gemini rejected your key. Please double-check it in the sidebar."
                )
            elif "RESOURCE_EXHAUSTED" in _err_str or "quota" in _err_str.lower():
                st.error(
                    "⚠️ **API Quota Exceeded**\n\n"
                    "You have hit your Gemini usage limit. "
                    "Please wait a moment and try again."
                )
            elif "DEADLINE_EXCEEDED" in _err_str or "timeout" in _err_str.lower():
                st.error(
                    "⏱️ **Request Timed Out**\n\n"
                    "The AI took too long to respond. "
                    "Please reset and try again."
                )
            else:
                st.error(
                    f"❌ **AI Processing Error**\n\n"
                    f"Something went wrong contacting the Gemini API.\n\n"
                    f"*Detail: {_api_err}*\n\n"
                    f"Please reset the app (sidebar) and try again."
                )
            # Reset to _IDLE so the user can start fresh
            full_reset()
            st.stop()

    # Clear the pending flag AFTER the LLM call succeeds.
    # Without this the audit would fire again on the very next rerun, creating
    # an infinite loop.
    st.session_state.audit_pending = False

    # Force a rerun so the UI reflects the new state (_QUESTIONING or _READY)
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — DYNAMIC CHAT INTERFACE  (QUESTIONING state only)
#
# Displays the accumulated Q&A conversation as proper chat bubbles and
# renders a st.chat_input bar for the user to type their next answer.
#
# When the user submits an answer:
#   1. Append it to the visible chat_history.
#   2. Append the Q&A pair to the hidden context_profile.
#   3. Set audit_pending=True and st.rerun() to trigger Section 12 again.
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.app_state == _QUESTIONING:

    st.markdown("### 💬 Profile Refinement")
    st.markdown(
        '<div class="card-info">'
        "🤖 <strong>The AI needs a little more information</strong> to tailor your "
        "recommendations perfectly. Please answer the question below as honestly "
        "and specifically as you can — there are no wrong answers!"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Progress indicator ────────────────────────────────────────────────────
    # Gives the user a sense of completion without being overly precise.
    _qc = st.session_state.question_count
    _prog = min(float(_qc) / 4.0, 0.85)   # Cap at 85%; the audit determines 100%
    st.progress(
        _prog,
        text=f"Profile completeness: ~{int(_prog * 100)}%  ({_qc} question(s) answered)",
    )
    st.markdown("")   # Visual spacer

    # ── Render full chat history ───────────────────────────────────────────────
    # Every message (AI questions and user answers) is shown so the user has
    # full context of the conversation at all times.
    for _msg in st.session_state.chat_history:
        _avatar = "🤖" if _msg["role"] == "assistant" else "👤"
        with st.chat_message(_msg["role"], avatar=_avatar):
            st.markdown(_msg["content"])

    # ── Chat input bar ────────────────────────────────────────────────────────
    # st.chat_input renders a sticky text bar pinned to the page bottom.
    # It returns the submitted text (once) when the user presses Enter,
    # and None on every other rerun — which prevents accidental double-fires.
    _answer_text = st.chat_input(
        placeholder="Type your answer here and press Enter to continue…",
        key="qa_chat_input",
    )

    # ── Handle submitted answer ────────────────────────────────────────────────
    if _answer_text and _answer_text.strip():

        # 1 — Add the user's answer to the VISIBLE chat history for display
        st.session_state.chat_history.append(
            {"role": "user", "content": _answer_text.strip()}
        )

        # 2 — Retrieve the most recent AI question to pair with this answer
        #     (needed for a clean log entry in the hidden context profile)
        _last_ai_q = next(
            (m["content"] for m in reversed(st.session_state.chat_history)
             if m["role"] == "assistant"),
            "Clarifying question",
        )

        # 3 — Append this Q&A exchange to the HIDDEN CONTEXT PROFILE
        #     The LLM receives this enriched profile on the next audit call,
        #     giving it progressively more signal to work with.
        st.session_state.context_profile += (
            f"\n\n--- CLARIFICATION ROUND {st.session_state.question_count} ---\n"
            f"AI QUESTION:     {_last_ai_q}\n"
            f"STUDENT ANSWER:  {_answer_text.strip()}\n"
        )

        # 4 — Flag the audit engine to run on the next rerun
        st.session_state.audit_pending = True

        # 5 — Trigger the rerun.
        #     st.rerun() raises a StopException that halts the current script
        #     execution immediately; Section 14 (Results) will NOT run this pass.
        #     On the next pass Section 12 (Audit Engine) fires first.
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — RESULTS DISPLAY  (READY state only)
#
# Renders the full recommendations in a three-tab layout:
#   Tab 1: 🎯 Best-Fit Job Paths
#   Tab 2: 🎓 Universities & Majors
#   Tab 3: 📋 Action Plan & Insights
#
# Also includes:
#   • Summary metric tiles
#   • Collapsible Q&A history
#   • An unlimited follow-up chat powered by the same Gemini model
#   • A 'Analyse a New Profile' reset button
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.app_state == _READY and st.session_state.recommendations:

    st.markdown("### 🎉 Your Personalised Career Roadmap")
    st.markdown(
        '<div class="card-success">'
        "✅ <strong>Analysis complete!</strong> "
        "Your recommendations are ready. Explore each tab below for full details "
        "on your best-fit career paths, university majors, and action plan."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Summary metric row ─────────────────────────────────────────────────────
    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
    with _mc1:
        st.metric(label="🎯 Career Paths",    value="3",  delta="Personalised for you")
    with _mc2:
        st.metric(label="🎓 Uni Majors",      value="3",  delta="Tailored to your profile")
    with _mc3:
        st.metric(label="💬 Q&A Rounds",      value=str(st.session_state.question_count),
                  delta="Completed")
    with _mc4:
        st.metric(label="📊 Analysis Status", value="Ready", delta="✓ Done")

    st.markdown("")

    # ── Parse recommendations into named sections ──────────────────────────────
    _sections = parse_sections(st.session_state.recommendations)

    # ── Three result tabs ──────────────────────────────────────────────────────
    _tab1, _tab2, _tab3 = st.tabs([
        "🎯 Best-Fit Job Paths",
        "🎓 Universities & Majors",
        "📋 Action Plan & Insights",
    ])

    # ── Tab 1: Career Paths ────────────────────────────────────────────────────
    with _tab1:
        st.markdown("")
        # Show the careers section, or fall back to the full text if parsing failed
        st.markdown(_sections["careers"] or st.session_state.recommendations)
        st.markdown("---")
        st.markdown(
            "💡 *Research these roles further on "
            "[LinkedIn Jobs](https://linkedin.com/jobs), "
            "[Glassdoor](https://glassdoor.com), and "
            "[Indeed](https://indeed.com) for live openings and salary data.*"
        )

    # ── Tab 2: Universities & Majors ───────────────────────────────────────────
    with _tab2:
        st.markdown("")
        if _sections["universities"]:
            st.markdown(_sections["universities"])
        else:
            st.info(
                "University details are integrated into the Career Paths tab. "
                "Please check there for programme recommendations."
            )
        st.markdown("---")
        st.markdown(
            "💡 *Always verify entry requirements, tuition fees, and application "
            "deadlines directly on each university's official website.*"
        )

    # ── Tab 3: Action Plan & Insights ─────────────────────────────────────────
    with _tab3:
        st.markdown("")
        _showed_anything = False

        if _sections["action_plan"]:
            st.markdown(_sections["action_plan"])
            _showed_anything = True

        if _sections["insight"]:
            if _showed_anything:
                st.markdown("---")
            st.markdown(_sections["insight"])
            _showed_anything = True

        if not _showed_anything:
            st.info(
                "Your action plan and personal insights are woven into the other tabs. "
                "Check the Career Paths and University tabs for your complete roadmap."
            )

        st.markdown("---")
        st.markdown(
            "💡 *Supplement these AI-generated recommendations with guidance from a "
            "qualified career counsellor or university admissions advisor.*"
        )

    st.markdown("---")

    # ── Collapsible: full Q&A conversation history ────────────────────────────
    # Lets the user review the clarifying questions and their own answers that
    # shaped the recommendations without cluttering the main view.
    if st.session_state.chat_history:
        with st.expander(
            "📜 View the full conversation that shaped these recommendations",
            expanded=False,
        ):
            st.markdown(
                "*Here are the clarifying questions and your answers that helped "
                "personalise your roadmap:*"
            )
            st.markdown("")
            for _hm in st.session_state.chat_history:
                _av = "🤖" if _hm["role"] == "assistant" else "👤"
                with st.chat_message(_hm["role"], avatar=_av):
                    st.markdown(_hm["content"])

    st.markdown("---")

    # ── Follow-up chat section ─────────────────────────────────────────────────
    # After results are shown, the user can ask unlimited follow-up questions.
    # Each call to Gemini receives the full recommendations + context profile
    # as background, so answers are specific to this student's situation.
    st.markdown("### 💬 Ask Follow-Up Questions")
    st.markdown(
        "*Still curious? Ask anything about your career paths, university options, "
        "skill requirements, application tips, or salary expectations.*"
    )

    # ── Display existing follow-up messages ────────────────────────────────────
    for _fm in st.session_state.followup_msgs:
        _fav = "🤖" if _fm["role"] == "assistant" else "👤"
        with st.chat_message(_fm["role"], avatar=_fav):
            st.markdown(_fm["content"])

    # ── Follow-up chat input ───────────────────────────────────────────────────
    # A separate key ("followup_chat_input") avoids Streamlit key conflicts with
    # the earlier Q&A chat input ("qa_chat_input").
    _fu_text = st.chat_input(
        placeholder="Ask a follow-up question about your recommendations…",
        key="followup_chat_input",
    )

    # ── Handle submitted follow-up question ────────────────────────────────────
    if _fu_text and _fu_text.strip():

        # Add the user's question to the follow-up message list
        st.session_state.followup_msgs.append(
            {"role": "user", "content": _fu_text.strip()}
        )

        # Build a context-rich prompt so Gemini can answer specifically
        _fu_system = """\
You are Dr. Alex, the AI Career Counsellor who produced the recommendations below.
A student has a follow-up question. Answer it helpfully, specifically, and \
encouragingly, drawing directly on their profile and the recommendations.
Keep your response concise and focused: 2–5 paragraphs or a short bulleted list.\
"""
        _fu_prompt = (
            f"RECOMMENDATIONS YOU PROVIDED:\n{'=' * 52}\n"
            f"{st.session_state.recommendations}\n"
            f"{'=' * 52}\n\n"
            f"STUDENT PROFILE CONTEXT:\n{'=' * 52}\n"
            f"{st.session_state.context_profile}\n"
            f"{'=' * 52}\n\n"
            f"FOLLOW-UP QUESTION: {_fu_text.strip()}\n\n"
            "Please answer the student's follow-up question now:"
        )

        # Call the LLM for a follow-up response
        try:
            _fu_response = call_llm(
                api_key=api_key.strip(),
                prompt=_fu_prompt,
                system=_fu_system,
            )
        except Exception as _fu_exc:
            _fu_response = (
                f"❌ I encountered an error while processing your question. "
                f"Please try asking again. *(Error: {_fu_exc})*"
            )

        # Store the AI response in session state
        st.session_state.followup_msgs.append(
            {"role": "assistant", "content": _fu_response}
        )

        # Rerun so the new messages appear via the display loop above,
        # ensuring a clean, duplication-free render.
        st.rerun()

    st.markdown("---")

    # ── 'Start over' button ────────────────────────────────────────────────────
    _, _rb_col, _ = st.columns([1, 2, 1])
    with _rb_col:
        if st.button(
            "🔄 Analyse a New Profile",
            use_container_width=True,
            key="restart_from_results_btn",
            help="Clears all data and returns to the profile input screen.",
        ):
            full_reset()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — PAGE FOOTER
# Shown at the bottom of every page state.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="footer">'
    "🎓 <strong>AI Career & University Advisor</strong> · "
    "Powered by Google Gemini 2.5 Flash · Built with Streamlit &amp; pypdf<br>"
    "<em>AI-generated recommendations are for informational purposes only. "
    "Always consult qualified career counsellors and academic advisors "
    "before making major life decisions.</em>"
    "</div>",
    unsafe_allow_html=True,
)
