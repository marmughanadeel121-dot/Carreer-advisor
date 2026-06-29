import streamlit as st
from google import genai
from google.genai import types
import pypdf
import io
import time  # Required for managing delay timing

# =====================================================================
# 1. PAGE CONFIGURATION & SETUP
# =====================================================================
st.set_page_config(
    page_title="AI Career & University Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling to improve UX
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
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
if "custom_api_key" not in st.session_state:
    st.session_state.custom_api_key = ""

# =====================================================================
# 3. DYNAMIC API CLIENT INITIALIZATION
# =====================================================================
def get_gemini_client():
    """Returns a GenAI client initialized with either the user's key or the global key."""
    # Check user custom key override first
    if st.session_state.custom_api_key.strip():
        try:
            return genai.Client(api_key=st.session_state.custom_api_key.strip())
        except Exception as e:
            st.sidebar.error(f"Invalid custom API key: {str(e)}")
            return None
            
    # Fallback to Streamlit Secrets global key
    if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"].strip():
        try:
            return genai.Client(api_key=st.secrets["GEMINI_API_KEY"].strip())
        except Exception as e:
            st.sidebar.error(f"Failed to initialize global Gemini Client: {str(e)}")
            return None
            
    return None

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
    """Sends current state to LLM with robust exponential backoff handling for 429 quota errors."""
    client = get_gemini_client()
    if not client:
        st.error("❌ API Key Missing! Please add a custom API key in the sidebar or verify global secrets.")
        return None

    system_instruction = (
        "You are an Elite Career & University Guidance Counselor. Your goal is to review the user's "
        "CV profile and hobbies to provide a Top 3 Career Path match and Top 3 University/Major Roadmap.\n\n"
        "CRITICAL PROCESS RULE:\n"
        "Evaluate the details provided. If vital contextual constraints like preferred geographic study region, "
        "approximate budget limits, or current high school grades/qualifications are completely missing, you MUST "
        "ask a clarifying question. If you require more data, reply with exactly '[NEEDS_INFO]' followed by "
        "EXACTLY ONE clear, friendly conversational question. Do not output multiple questions.\n\n"
        "If you have sufficient data to make concrete accurate matching paths, reply with '[READY]' followed by "
        "a beautifully structured markdown analysis containing: 1) Top 3 Job Paths (with descriptions), and 2) "
        "Top 3 Recommended University Majors/Roadmaps (with reasoning)."
    )

    # Standard Exponential Backoff wait delays: 1s, 2s, 4s, 8s, 16s
    delays = [1, 2, 4, 8, 16]
    
    for i, delay in enumerate(delays + [0]):  # 5 retry delays, then final exception on the last attempt
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
            error_msg = str(e)
            
            # Check if this error represents an API rate/quota limit (429 or RESOURCE_EXHAUSTED)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if i < len(delays):
                    # Show a professional temporary countdown spinner instead of crashing the page
                    with st.spinner(f"⏳ Rate limit triggered. Retrying automatically in {delay}s (Attempt {i+1}/5)..."):
                        time.sleep(delay)
                    continue  # Jump back to the start of the loop and try again!
            
            # If all retries fail or we hit a different, non-recoverable API error
            st.error("### ⚠️ System Quota Limit Reached")
            st.markdown(
                f"""
                The application encountered an API limit or configuration issue.
                
                **Technical Details:**
                `{error_msg}`
                
                **Suggestions & Fixes:**
                * 👉 **Supply a Custom Key (Recommended):** To bypass this global limit immediately, generate a FREE personal API key at [Google AI Studio](https://aistudio.google.com/) and paste it into the **API Key Override** box in the sidebar on the left!
                * If you just ran multiple requests quickly, wait 60 seconds and click **Type your response here...** again.
                """
            )
            return None
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
    st.title("⚙️ System Status")
    
    # Track the active client key state
    active_client = get_gemini_client()
    if active_client:
        if st.session_state.custom_api_key.strip():
            st.success("🤖 API Key: CUSTOM (Online)")
        else:
            st.success("🤖 API Key: GLOBAL (Online)")
    else:
        st.error("🤖 AI Brain Status: OFFLINE")

    # API Key Override Section
    st.write("---")
    st.subheader("🔑 API Key Override")
    custom_key_val = st.text_input(
        "Enter your custom Gemini API Key:",
        type="password",
        value=st.session_state.custom_api_key,
        help="If the shared daily limit is reached, obtain a FREE key at Google AI Studio and enter it here to instantly unlock the advisor."
    )
    if custom_key_val != st.session_state.custom_api_key:
        st.session_state.custom_api_key = custom_key_val
        st.rerun()
        
    if st.session_state.custom_api_key.strip():
        if st.button("🗑️ Clear Custom Key", type="secondary"):
            st.session_state.custom_api_key = ""
            st.rerun()

    st.write("---")
    st.subheader("📋 Instructions")
    st.markdown("""
    1. Enter your **hobbies** & personal interests.
    2. Upload your **CV/Resume** as a PDF.
    3. Click **Generate Roadmap**.
    4. Answer any extra questions the AI asks!
    """)
    
    st.write("---")
    if st.button("🔄 Reset & Clear Data", type="secondary"):
        reset_application()

# =====================================================================
# 6. MAIN PANEL VIEW LOGIC
# =====================================================================
st.title("🎓 AI Career & University Roadmap Advisor")
st.write("Upload your background profile to uncover ideal jobs and university pathways engineered by AI.")
st.write("---")

# Prevent operations if no client could be resolved
if not get_gemini_client():
    st.warning("⚠️ **API Setup Required**: Please configure a fallback `GEMINI_API_KEY` in your Streamlit secrets or input a personal override key in the sidebar to get started.")

# PHASE 1: COLLECT CORE INPUTS
if st.session_state.app_state == "input" and get_gemini_client():
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.subheader("🎨 Personal Background")
        hobbies_input = st.text_area(
            "What are your core hobbies, passions, or subjects you naturally enjoy?",
            placeholder="Example: I love building remote control cars, writing short stories, video editing, and math...",
            height=150
        )
        
    with col2:
        st.subheader("📄 Professional/Academic Profile")
        uploaded_cv = st.file_uploader("Upload your CV / Academic Record (PDF format only)", type=["pdf"])

    st.write("---")
    if st.button("🚀 Generate My Custom Roadmap", type="primary"):
        if not hobbies_input.strip():
            st.warning("⚠️ Please provide a few sentences describing your interests or hobbies first.")
        elif not uploaded_cv:
            st.warning("⚠️ Please upload your CV/Resume PDF to evaluate your educational background.")
        else:
            with st.spinner("Extracting profile data and calculating roadmaps..."):
                extracted_text = extract_text_from_pdf(uploaded_cv)
                if extracted_text:
                    st.session_state.cv_text = extracted_text
                    st.session_state.hobbies = hobbies_input
                    
                    # Create baseline comprehensive context
                    initial_context = (
                        f"### USER DATA PROFILE ###\n"
                        f"USER HOBBIES AND PASSIONS:\n{st.session_state.hobbies}\n\n"
                        f"USER EXTRACTED CV PROFILE TEXT:\n{st.session_state.cv_text}\n\n"
                        f"### CONVERSATION FLOW HISTORY ###\n"
                    )
                    st.session_state.full_context = initial_context
                    
                    # Evaluate profile complete sufficiency via AI Audit
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
elif st.session_state.app_state == "needs_info" and get_gemini_client():
    st.subheader("🙋‍♂️ Career Advisor Follow-up Questions")
    st.info("The AI is tailoring its recommendations, but wants a tiny bit more information to build the perfect roadmap.")
    
    # Render historical back-and-forth log layout
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat Input processing loop
    if user_reply := st.chat_input("Type your response here..."):
        with st.chat_message("user"):
            st.write(user_reply)
        
        st.session_state.chat_history.append({"role": "user", "content": user_reply})
        
        # Feed the answers directly back into the working backend context log
        st.session_state.full_context += f"\nAdvisor Question: {st.session_state.chat_history[-2]['content']}\nUser Answer: {user_reply}\n"
        
        with st.spinner("Analyzing answers..."):
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

# PHASE 3: COMPILE FINAL RESULTS DATA
elif st.session_state.app_state == "ready":
    st.balloons()
    st.success("🎯 Your Personalized Roadmap is Complete!")
    
    # Split layout options dynamically into clean visual tabs
    tab1, tab2 = st.columns([1, 1])
    
    st.write(st.session_state.final_recommendations)
    
    st.write("---")
    if st.button("🆕 Analyze New Profile Roadmap", type="secondary"):
        reset_application()
