import streamlit as st
from google import genai
from google.genai import types
import pypdf
import io
from datetime import datetime

# =====================================================================
# 1. PAGE CONFIGURATION & SETUP
# =====================================================================
st.set_page_config(
    page_title="Elite AI Career & University Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main .block-container { padding-top: 1.5rem; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
    div[data-testid="stExpander"] { border-radius: 8px; }
    .chat-wrap { max-width: 980px; margin: 0 auto; }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. INITIALIZE SESSION STATE VARIABLES
# =====================================================================
defaults = {
    "chat_history": [],
    "cv_text": "",
    "hobbies": "",
    "app_state": "input",   # input, needs_info, ready
    "full_context": "",
    "final_recommendations": "",
    "current_question": "",
    "last_user_answer": "",
    "error_log": [],
    "deterministic_mode": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =====================================================================
# 3. API CLIENT
# =====================================================================
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ API Key Missing! Please add 'GEMINI_API_KEY' to Streamlit Secrets.")
    st.stop()

try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"Failed to initialize Gemini Client: {str(e)}")
    st.stop()

# =====================================================================
# 4. HELPER FUNCTIONS
# =====================================================================
def log_error(message: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.error_log.append(f"[{ts}] {message}")
    try:
        with open("llm_error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
    except Exception:
        pass

def extract_text_from_pdf(uploaded_file):
    try:
        uploaded_file.seek(0)
        data = uploaded_file.read()
        if not data:
            st.error("Uploaded PDF is empty.")
            return None

        if len(data) > 5_000_000:
            st.warning("PDF is large. Processing only the first 5MB.")
            data = data[:5_000_000]

        pdf_reader = pypdf.PdfReader(io.BytesIO(data))
        extracted_text = []
        for page in pdf_reader.pages:
            try:
                text = page.extract_text()
                if text:
                    extracted_text.append(text)
            except Exception:
                continue

        result = "\n".join(extracted_text).strip()
        if not result:
            st.warning("No readable text found in this PDF.")
            return None
        return result

    except Exception as e:
        log_error(f"PDF parse error: {e}")
        st.error(f"⚠️ Failed to parse the PDF file: {str(e)}")
        return None

def run_orchestration_audit(user_profile_context):
    system_instruction = (
        "You are an Elite Career & University Guidance Counselor. Your goal is to review the user's "
        "CV profile and hobbies to provide a Top 3 Career Path match and Top 3 University/Major Roadmap.\n\n"
        "CRITICAL PROCESS RULE:\n"
        "Evaluate the details provided. If vital contextual constraints like preferred geographic study region, "
        "approximate budget limits, or current grades/qualifications are completely missing, you MUST ask a "
        "clarifying question. If you require more data, reply with exactly '[NEEDS_INFO]' followed by exactly "
        "one clear, friendly conversational question.\n\n"
        "If you have sufficient data to make concrete accurate matching paths, reply with '[READY]' followed by "
        "a beautifully structured markdown analysis containing:\n"
        "1) 🌟 Executive Summary\n"
        "2) 📊 Top 3 Tailored Job Paths\n"
        "3) 🗺️ Top 3 Recommended University Majors & Roadmap paths."
    )

    try:
        temp = 0.0 if st.session_state.deterministic_mode else 0.4
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_profile_context,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temp
            )
        )
        return getattr(response, "text", None)
    except Exception as e:
        log_error(f"Gemini generation error: {e}")
        st.error(f"API Generation Error: {str(e)}")
        return None

def reset_application():
    st.session_state.chat_history = []
    st.session_state.cv_text = ""
    st.session_state.hobbies = ""
    st.session_state.app_state = "input"
    st.session_state.full_context = ""
    st.session_state.final_recommendations = ""
    st.session_state.current_question = ""
    st.session_state.last_user_answer = ""
    st.rerun()

def handle_generate():
    if not st.session_state.hobbies_input.strip():
        st.warning("⚠️ Please provide a few sentences describing your interests or hobbies first.")
        return

    if not st.session_state.uploaded_cv and not st.session_state.cv_text:
        st.warning("⚠️ Please upload your CV/Resume PDF to evaluate your educational background.")
        return

    if st.session_state.uploaded_cv:
        extracted_text = extract_text_from_pdf(st.session_state.uploaded_cv)
        if extracted_text is None:
            return
        st.session_state.cv_text = extracted_text

    st.session_state.hobbies = st.session_state.hobbies_input.strip()

    initial_context = (
        f"### USER DATA PROFILE ###\n"
        f"USER HOBBIES AND PASSIONS:\n{st.session_state.hobbies}\n\n"
        f"USER EXTRACTED CV PROFILE TEXT:\n{st.session_state.cv_text}\n\n"
        f"### CONVERSATION FLOW HISTORY ###\n"
    )
    st.session_state.full_context = initial_context

    with st.spinner("Extracting profile insights and compiling roadmap..."):
        audit_result = run_orchestration_audit(st.session_state.full_context)

    if audit_result:
        if "[NEEDS_INFO]" in audit_result:
            clean_question = audit_result.replace("[NEEDS_INFO]", "").strip()
            st.session_state.current_question = clean_question
            st.session_state.chat_history.append({"role": "assistant", "content": clean_question})
            st.session_state.app_state = "needs_info"
        elif "[READY]" in audit_result:
            st.session_state.final_recommendations = audit_result.replace("[READY]", "").strip()
            st.session_state.app_state = "ready"

def handle_followup_answer():
    user_reply = st.session_state.followup_input.strip()

    if len(user_reply) < 3:
        st.warning("⚠️ Please write a little more detail.")
        return

    st.session_state.last_user_answer = user_reply
    st.session_state.chat_history.append({"role": "user", "content": user_reply})

    if st.session_state.current_question:
        st.session_state.full_context += (
            f"\nAdvisor Question: {st.session_state.current_question}\n"
            f"User Answer: {user_reply}\n"
        )
    else:
        st.session_state.full_context += f"\nUser Answer: {user_reply}\n"

    with st.spinner("Processing your answer and generating the next step..."):
        audit_result = run_orchestration_audit(st.session_state.full_context)

    if audit_result:
        if "[NEEDS_INFO]" in audit_result:
            clean_question = audit_result.replace("[NEEDS_INFO]", "").strip()
            st.session_state.current_question = clean_question
            st.session_state.chat_history.append({"role": "assistant", "content": clean_question})
            st.session_state.app_state = "needs_info"
            st.session_state.followup_input = ""
            st.rerun()

        elif "[READY]" in audit_result:
            st.session_state.final_recommendations = audit_result.replace("[READY]", "").strip()
            st.session_state.app_state = "ready"
            st.session_state.followup_input = ""
            st.rerun()

# =====================================================================
# 5. SIDEBAR
# =====================================================================
with st.sidebar:
    st.title("⚙️ Advisor Dashboard")
    st.success("🤖 AI Engine: ONLINE")
    st.caption("Connected securely via Streamlit Cloud Environment Keys")

    st.write("---")
    st.subheader("📋 Instructions")
    st.markdown("""
    1. Enter your **hobbies**.
    2. Upload your **CV/Resume** as a PDF.
    3. Click **Generate My Custom Roadmap**.
    4. Answer any follow-up question.
    """)

    st.write("---")
    st.subheader("💡 Sandbox Toolkit")

    if st.button("✨ Load Test Demo Profile"):
        st.session_state.hobbies = "I love building mechanical setups, programming basic Python scripts, gaming, and organizing school group projects."
        st.session_state.cv_text = "Name: Alex Smith\nEducation: International High School Student\nGrades: Straight A marks in Math, Physics, and Computer Science.\nProjects: Designed a custom web-app scraper tool; Leader of Robotics Club."
        st.success("Demo data loaded.")

    if st.button("🔄 Reset & Clear System"):
        reset_application()

    with st.expander("Advanced"):
        st.session_state.deterministic_mode = st.checkbox(
            "Deterministic Mode",
            value=st.session_state.deterministic_mode
        )

# =====================================================================
# 6. MAIN UI
# =====================================================================
st.title("🎓 AI Career & University Roadmap Advisor")
st.write("Upload your background profile to uncover ideal jobs and university pathways.")
st.write("---")

if st.session_state.app_state == "input":
    with st.form("profile_form"):
        col1, col2 = st.columns([1, 1], gap="medium")

        with col1:
            st.subheader("🎨 Personal Background")
            st.text_area(
                "What are your core hobbies, passions, or subjects you naturally enjoy?",
                value=st.session_state.hobbies,
                placeholder="Example: I love building remote control cars, writing short stories, video editing, and math...",
                height=160,
                key="hobbies_input"
            )

        with col2:
            st.subheader("📄 Professional/Academic Profile")
            st.file_uploader(
                "Upload your CV / Academic Record (PDF format only)",
                type=["pdf"],
                key="uploaded_cv"
            )

            if st.session_state.cv_text:
                st.info("✅ Demo Academic Record is available.")

        st.form_submit_button("🚀 Generate My Custom Roadmap", on_click=handle_generate)

elif st.session_state.app_state == "needs_info":
    st.subheader("🙋‍♂️ Career Advisor Follow-up Consultation")
    st.markdown("Please answer the question below to continue.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    st.form("followup_form")
    with st.form("followup_form"):
        st.text_input(
            "Your answer",
            key="followup_input",
            placeholder="Type your answer here..."
        )
        st.form_submit_button("Send Answer", on_click=handle_followup_answer)

elif st.session_state.app_state == "ready":
    st.balloons()
    st.success("🎯 Analysis Complete! Your Personalized Academic & Career Roadmap is Ready.")

    st.markdown(st.session_state.final_recommendations)

    st.download_button(
        "⬇️ Download Report (MD)",
        st.session_state.final_recommendations,
        file_name="roadmap.md",
        mime="text/markdown"
    )

    if st.button("🆕 Analyze a New Profile Pathways Setup"):
        reset_application()
