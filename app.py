import streamlit as st
from google import genai
from google.genai import types
import pypdf
import io
import time
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# === Page config & styling ===
st.set_page_config(page_title="Elite AI Career & University Advisor", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main .block-container { padding-top: 1.5rem; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
    div[data-testid="stExpander"] { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# === Session state initialization ===
def init_state():
    defaults = {
        "chat_history": [],
        "cv_text": "",
        "hobbies": "",
        "app_state": "input",  # 'input', 'needs_info', 'ready'
        "full_context": "",
        "final_recommendations": "",
        "loading": False,
        "last_audit_request": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# === Secrets / Client init with guard ===
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ API Key Missing! Please add 'GEMINI_API_KEY' to your Streamlit Advanced Settings / Secrets.")
    st.stop()

def safe_init_client():
    try:
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client: {e}")
        st.stop()

client = safe_init_client()

# === LLM wrapper with retry ===
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8),
       retry=retry_if_exception_type(Exception))
def generate_with_retry(contents: str, system_instruction: str, temperature: float = 0.4):
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=temperature)
    )
    return resp

def run_orchestration_audit(user_profile_context):
    system_instruction = (
        "You are an Elite Career & University Guidance Counselor. Your goal is to review the user's "
        "CV profile and hobbies to provide a Top 3 Career Path match and Top 3 University/Major Roadmap.\n\n"
        "CRITICAL PROCESS RULE:\n"
        "Evaluate the details provided. If vital contextual constraints like preferred geographic study region, "
        "approximate budget limits, or current grades/qualifications are completely missing, you MUST "
        "ask a clarifying question. If you require more data, reply with exactly '[NEEDS_INFO]' followed by "
        "EXACTLY ONE clear, friendly conversational question. Do not output multiple questions.\n\n"
        "If you have sufficient data to make concrete accurate matching paths, reply with '[READY]' followed by "
        "a beautifully structured markdown analysis containing:\n"
        "1) 🌟 Executive Summary\n"
        "2) 📊 Top 3 Tailored Job Paths (with short descriptions and potential salary outlooks)\n"
        "3) 🗺️ Top 3 Recommended University Majors & Roadmap paths."
    )

    try:
        st.session_state.loading = True
        st.session_state.last_audit_request = {"timestamp": datetime.utcnow().isoformat(), "context_preview": user_profile_context[:200]}
        resp = generate_with_retry(user_profile_context, system_instruction, temperature=0.4)
        # defensive extraction
        text = getattr(resp, "text", None)
        if text is None:
            # some clients return choices or content differently
            try:
                text = resp.generated_text
            except Exception:
                text = None
        return text
    except Exception as e:
        # write minimal log to disk for later replay
        try:
            with open("output/llm_errors.log", "a", encoding="utf-8") as f:
                f.write(f"{datetime.utcnow().isoformat()} | ERROR | {e} | preview: {user_profile_context[:200]}\n")
        except Exception:
            pass
        st.error(f"API Generation Error: {e}")
        return None
    finally:
        st.session_state.loading = False

# === Robust PDF extraction ===
def extract_text_from_pdf(uploaded_file, max_bytes=5_000_000):
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    data = uploaded_file.read()
    if not data:
        st.error("Empty file uploaded.")
        return None
    if len(data) > max_bytes:
        st.warning("Uploaded PDF is large; only the first 5MB will be processed to avoid memory issues.")
        data = data[:max_bytes]
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(data))
        texts = []
        for i, page in enumerate(pdf_reader.pages):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                texts.append(t)
        return "\n".join(texts).strip()
    except pypdf.errors.PdfReadError:
        st.error("PDF parsing error: possible corrupt or scanned PDF (no embedded text). Consider OCR or a text copy.")
        return None
    except Exception as e:
        st.error(f"Unexpected PDF parsing error: {e}")
        return None

# === Reset helper ===
def reset_application():
    st.session_state.chat_history = []
    st.session_state.cv_text = ""
    st.session_state.hobbies = ""
    st.session_state.app_state = "input"
    st.session_state.full_context = ""
    st.session_state.final_recommendations = ""
    st.session_state.last_audit_request = None
    # Use experimental_rerun to be explicit
    st.experimental_rerun()

# === Sidebar ===
with st.sidebar:
    st.title("⚙️ Advisor Dashboard")
    st.success("🤖 AI Engine: ONLINE")
    st.caption("Connected securely via Streamlit Cloud Environment Keys")
    st.write("---")
    st.subheader("📋 Instructions")
    st.markdown("""
    1. Enter your **hobbies** & personal interests.
    2. Upload your **CV/Resume** as a PDF.
    3. Click **Generate Roadmap**.
    4. Answer any extra questions the AI asks!
    """)
    st.write("---")
    st.subheader("💡 Sandbox Toolkit")
    if st.button("✨ Load Test Demo Profile"):
        st.session_state.hobbies = "I love building mechanical setups, programming basic Python scripts, gaming, and organizing school group projects."
        st.session_state.cv_text = (
            "Name: Alex Smith\nEducation: International High School Student\nGrades: Straight A marks in Math, Physics, and Computer Science.\n"
            "Projects: Designed a custom web-app scraper tool; Leader of Robotics Club."
        )
        st.success("Demo data loaded! Click 'Generate My Custom Roadmap' below.")
    if st.button("🔄 Reset & Clear System"):
        reset_application()
    with st.expander("Advanced (QA)"):
        det = st.checkbox("Deterministic Mode (temp=0.0)", value=False, key="deterministic_mode")
        st.write("Toggle deterministic LLM responses for reproducible testing.")
    st.write("---")
    st.caption("Logs for failed LLM calls are saved to output/llm_errors.log")

# === Main UI ===
st.title("🎓 AI Career & University Roadmap Advisor")
st.write("Upload your background profile to uncover ideal jobs and university pathways engineered by advanced machine learning.")
st.write("---")

# PHASE 1: input (using form to prevent reruns)
if st.session_state.app_state == "input":
    with st.form(key="profile_form"):
        col1, col2 = st.columns([1, 1], gap="medium")
        with col1:
            st.subheader("🎨 Personal Background")
            hobbies_input = st.text_area(
                "What are your core hobbies, passions, or subjects you naturally enjoy?",
                value=st.session_state.hobbies,
                placeholder="Example: I love building remote control cars, writing short stories, video editing, and math...",
                height=160,
                key="hobbies_input"
            )
        with col2:
            st.subheader("📄 Professional/Academic Profile")
            uploaded_cv = st.file_uploader("Upload your CV / Academic Record (PDF format only)", type=["pdf"], key="uploaded_cv")
            if not uploaded_cv and st.session_state.cv_text:
                st.info("✅ Demo Academic Record loaded via sandbox panel.")
        submit_pressed = st.form_submit_button("🚀 Generate My Custom Roadmap")

    if submit_pressed:
        if not hobbies_input.strip():
            st.warning("⚠️ Please provide a few sentences describing your interests or hobbies first.")
        elif not uploaded_cv and not st.session_state.cv_text:
            st.warning("⚠️ Please upload your CV/Resume PDF to evaluate your educational background.")
        else:
            # extract PDF if provided
            if uploaded_cv:
                extracted_text = extract_text_from_pdf(uploaded_cv)
                if extracted_text:
                    st.session_state.cv_text = extracted_text
                else:
                    # stop processing if PDF extraction failed
                    st.warning("Could not extract text from uploaded PDF. You may continue with demo data or re-upload.")
                    st.stop()

            st.session_state.hobbies = hobbies_input.strip()

            # build initial context
            initial_context = (
                f"### USER DATA PROFILE ###\n"
                f"USER HOBBIES AND PASSIONS:\n{st.session_state.hobbies}\n\n"
                f"USER EXTRACTED CV PROFILE TEXT:\n{st.session_state.cv_text}\n\n"
                f"### CONVERSATION FLOW HISTORY ###\n"
            )
            st.session_state.full_context = initial_context

            # call audit
            with st.spinner("Extracting profile insights and compiling neural roadmaps..."):
                audit_result = run_orchestration_audit(st.session_state.full_context)
            if audit_result:
                if "[NEEDS_INFO]" in audit_result:
                    clean_question = audit_result.replace("[NEEDS_INFO]", "").strip()
                    st.session_state.chat_history.append({"role": "assistant", "content": clean_question})
                    st.session_state.app_state = "needs_info"
                elif "[READY]" in audit_result:
                    st.session_state.final_recommendations = audit_result.replace("[READY]", "").strip()
                    st.session_state.app_state = "ready"
                # let Streamlit re-render naturally

# PHASE 2: needs_info
elif st.session_state.app_state == "needs_info":
    st.subheader("🙋‍♂️ Career Advisor Follow-up Consultation")
    st.markdown("To provide an exceptionally accurate pathway, please clarify this final parameter for the advisor:")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_reply = st.chat_input("Provide your answer here...")
    if user_reply:
        if len(user_reply.strip()) < 8:
            st.warning("⚠️ Your response is a bit too short! Please provide a descriptive answer.")
        else:
            with st.chat_message("user"):
                st.write(user_reply)
            st.session_state.chat_history.append({"role": "user", "content": user_reply})

            # Safely find last assistant question
            last_assistant = None
            for msg in reversed(st.session_state.chat_history):
                if msg["role"] == "assistant":
                    last_assistant = msg["content"]
                    break

            if last_assistant:
                st.session_state.full_context += f"\nAdvisor Question: {last_assistant}\nUser Answer: {user_reply}\n"
            else:
                st.session_state.full_context += f"\nUser Answer: {user_reply}\n"

            with st.spinner("Processing context updates and calculating predictions..."):
                audit_result = run_orchestration_audit(st.session_state.full_context)

            if audit_result:
                if "[NEEDS_INFO]" in audit_result:
                    clean_question = audit_result.replace("[NEEDS_INFO]", "").strip()
                    st.session_state.chat_history.append({"role": "assistant", "content": clean_question})
                elif "[READY]" in audit_result:
                    st.session_state.final_recommendations = audit_result.replace("[READY]", "").strip()
                    st.session_state.app_state = "ready"

# PHASE 3: ready
elif st.session_state.app_state == "ready":
    if not st.session_state.final_recommendations:
        st.error("No recommendations found — please reset and try again.")
    else:
        st.balloons()
        st.success("🎯 Analysis Complete! Your Personalized Academic & Career Roadmap is Ready.")
        st.markdown(st.session_state.final_recommendations)

        st.download_button(
            "⬇️ Download Report (MD)",
            st.session_state.final_recommendations,
            file_name="roadmap.md",
            mime="text/markdown"
        )
        st.code(st.session_state.final_recommendations, language="markdown")
        if st.button("🆕 Analyze a New Profile Pathways Setup"):
            reset_application()
