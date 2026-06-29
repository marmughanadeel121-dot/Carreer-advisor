import streamlit as st
from google import genai
from google.genai import types
import pypdf
import io

# =====================================================================
# 1. PAGE CONFIGURATION & SETUP
# =====================================================================
st.set_page_config(
    page_title="Elite AI Career & University Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling to improve UX and fix layout spacing
st.markdown("""
    <style>
    .main .block-container { padding-top: 1.5rem; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
    div[data-testid="stExpander"] { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. INITIALIZE SESSION STATE VARIABLES
# =====================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""
if "hobbies" not in st.session_state:
    st.session_state.hobbies = ""
if "app_state" not in st.session_state:
    st.session_state.app_state = "input"  # Modes: 'input', 'needs_info', 'ready'
if "full_context" not in st.session_state:
    st.session_state.full_context = ""
if "final_recommendations" not in st.session_state:
    st.session_state.final_recommendations = ""

# =====================================================================
# 3. SECURE BACKEND API INITIALIZATION (STREAMLIT SECRETS)
# =====================================================================
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ API Key Missing! Please add 'GEMINI_API_KEY' to your Streamlit Advanced Settings / Secrets.")
    st.stop()

try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"Failed to initialize Gemini Client: {str(e)}")
    st.stop()

# =====================================================================
# 4. HELPER FUNCTIONS
# =====================================================================
def extract_text_from_pdf(uploaded_file):
    """Safely extracts all visible text content from an uploaded resume PDF."""
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        extracted_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        return extracted_text.strip()
    except Exception as e:
        st.error(f"⚠️ Failed to parse the PDF file structure: {str(e)}")
        return None

def run_orchestration_audit(user_profile_context):
    """Sends current state to LLM to audit if extra questions are needed or if ready."""
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
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_profile_context,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4
            )
        )
        return response.text
    except Exception as e:
        st.error(f"API Generation Error: {str(e)}")
        return None

def reset_application():
    """Wipes state back to factory defaults to process a new profile."""
    st.session_state.chat_history = []
    st.session_state.cv_text = ""
    st.session_state.hobbies = ""
    st.session_state.app_state = "input"
    st.session_state.full_context = ""
    st.session_state.final_recommendations = ""
    st.rerun()

# =====================================================================
# 5. USER INTERFACE LAYOUT (SIDEBAR)
# =====================================================================
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
    # Interactive Demo Loader for premium testing presentation
    if st.button("✨ Load Test Demo Profile", type="secondary"):
        st.session_state.hobbies = "I love building mechanical setups, programming basic Python scripts, gaming, and organizing school group projects."
        st.session_state.cv_text = "Name: Alex Smith\nEducation: International High School Student\nGrades: Straight A marks in Math, Physics, and Computer Science.\nProjects: Designed a custom web-app scraper tool; Leader of Robotics Club."
        st.toast("Demo data loaded! Click 'Generate My Custom Roadmap' below.", icon="🚀")
    
    if st.button("🔄 Reset & Clear System", type="secondary"):
        reset_application()

# =====================================================================
# 6. MAIN PANEL VIEW LOGIC
# =====================================================================
st.title("🎓 AI Career & University Roadmap Advisor")
st.write("Upload your background profile to uncover ideal jobs and university pathways engineered by advanced machine learning.")
st.write("---")

# PHASE 1: COLLECT CORE INPUTS
if st.session_state.app_state == "input":
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.subheader("🎨 Personal Background")
        hobbies_input = st.text_area(
            "What are your core hobbies, passions, or subjects you naturally enjoy?",
            value=st.session_state.hobbies,
            placeholder="Example: I love building remote control cars, writing short stories, video editing, and math...",
            height=160
        )
        
    with col2:
        st.subheader("📄 Professional/Academic Profile")
        uploaded_cv = st.file_uploader("Upload your CV / Academic Record (PDF format only)", type=["pdf"])
        
        # Display alternative text status if the user loaded the visual sandboxed demo profile
        if not uploaded_cv and st.session_state.cv_text:
            st.info("✅ Demo Academic Record loaded via sandbox panel.")

    st.write("###")
    if st.button("🚀 Generate My Custom Roadmap", type="primary"):
        if not hobbies_input.strip():
            st.warning("⚠️ Please provide a few sentences describing your interests or hobbies first.")
        elif not uploaded_cv and not st.session_state.cv_text:
            st.warning("⚠️ Please upload your CV/Resume PDF to evaluate your educational background.")
        else:
            with st.spinner("Extracting profile insights and compiling neural roadmaps..."):
                # If a new physical file was added, extract it. Otherwise use the demo text
                if uploaded_cv:
                    extracted_text = extract_text_from_pdf(uploaded_cv)
                    if extracted_text:
                        st.session_state.cv_text = extracted_text
                
                st.session_state.hobbies = hobbies_input
                
                # Formulate structural audit history text block
                initial_context = (
                    f"### USER DATA PROFILE ###\n"
                    f"USER HOBBIES AND PASSIONS:\n{st.session_state.hobbies}\n\n"
                    f"USER EXTRACTED CV PROFILE TEXT:\n{st.session_state.cv_text}\n\n"
                    f"### CONVERSATION FLOW HISTORY ###\n"
                )
                st.session_state.full_context = initial_context
                
                # Evaluate input profile context data via Audit Block
                audit_result = run_orchestration_audit(st.session_state.full_context)
                
                if audit_result:
                    if "[NEEDS_INFO]" in audit_result:
                        clean_question = audit_result.replace("[NEEDS_INFO]", "").strip()
                        st.session_state.chat_history.append({"role": "assistant", "content": clean_question})
                        st.session_state.app_state = "needs_info"
                        st.rerun()
                    elif "[READY]" in audit_result:
                        st.session_state.final_recommendations = audit_result.replace("[READY]", "").strip()
                        st.session_state.app_state = "ready"
                        st.rerun()

# PHASE 2: ACTIVE DYNAMIC QUESTION-LOOP CONVERSATION 
elif st.session_state.app_state == "needs_info":
    st.subheader("🙋‍♂️ Career Advisor Follow-up Consultation")
    st.markdown("To provide an exceptionally accurate pathway, please clarify this final parameter for the advisor:")
    
    # Render interactive chat panel layout
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input parsing engine
    if user_reply := st.chat_input("Provide your answer here..."):
        
        # QUALITY CONTROL ACCURACY FILTER (Gibberish Guard)
        if len(user_reply.strip()) < 8:
            st.warning("⚠️ Your response is a bit too short! Please provide a descriptive answer so the AI can compute an accurate match.")
        else:
            with st.chat_message("user"):
                st.write(user_reply)
            
            st.session_state.chat_history.append({"role": "user", "content": user_reply})
            st.session_state.full_context += f"\nAdvisor Question: {st.session_state.chat_history[-2]['content']}\nUser Answer: {user_reply}\n"
            
            with st.spinner("Processing context updates and calculating predictions..."):
                audit_result = run_orchestration_audit(st.session_state.full_context)
                if audit_result:
                    if "[NEEDS_INFO]" in audit_result:
                        clean_question = audit_result.replace("[NEEDS_INFO]", "").strip()
                        st.session_state.chat_history.append({"role": "assistant", "content": clean_question})
                        st.rerun()
                    elif "[READY]" in audit_result:
                        st.session_state.final_recommendations = audit_result.replace("[READY]", "").strip()
                        st.session_state.app_state = "ready"
                        st.rerun()

# PHASE 3: COMPILE FINAL ADVISORY RESULTS
elif st.session_state.app_state == "ready":
    st.balloons()  # Premium celebration UX trigger effect!
    st.success("🎯 Analysis Complete! Your Personalized Academic & Career Roadmap is Ready.")
    
    # Display beautifully structured markdown report content 
    st.markdown(st.session_state.final_recommendations)
    
    st.write("---")
    if st.button("🆕 Analyze a New Profile Pathways Setup", type="secondary"):
        reset_application()
