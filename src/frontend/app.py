import uuid
import time
import streamlit as st
import httpx

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Loan Risk Bot + RAG",
    page_icon="🏦",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []


# ─────────────────────────────────────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def call_chat_api(message: str) -> dict:
    try:
        response = httpx.post(
            f"{API_BASE_URL}/chat",
            json={"session_id": st.session_state.session_id, "message": message},
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"response": f"❌ **Error:** {str(e)}", "tools_used": [], "citations": []}

def call_reset_api():
    try:
        httpx.post(f"{API_BASE_URL}/reset", params={"session_id": st.session_state.session_id}, timeout=10.0)
    except Exception:
        pass

def call_health_api():
    try:
        return httpx.get(f"{API_BASE_URL}/health", timeout=5.0).json()
    except Exception:
        return None

def upload_document(file_bytes, filename):
    try:
        files = {"file": (filename, file_bytes)}
        response = httpx.post(f"{API_BASE_URL}/ingest", files=files, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def poll_job_status(job_id: str):
    try:
        response = httpx.get(f"{API_BASE_URL}/ingest/status/{job_id}", timeout=5.0)
        return response.json()
    except Exception:
        return None

def call_evaluate_api():
    try:
        response = httpx.post(f"{API_BASE_URL}/evaluate/start", timeout=10.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def poll_eval_status(job_id: str):
    try:
        response = httpx.get(f"{API_BASE_URL}/evaluate/status/{job_id}", timeout=5.0)
        return response.json()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Controls")
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Session", use_container_width=True):
            call_reset_api()
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            call_reset_api()
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # --- WEEK 3: KNOWLEDGE BASE UPLOADER ---
    st.markdown("### 📚 Knowledge Base")
    st.caption("Upload policy manuals, guidelines, or FAQ documents (PDF, CSV, HTML, TXT).")
    
    uploaded_file = st.file_uploader("Add Document", label_visibility="collapsed")
    if uploaded_file is not None:
        if st.button("Upload & Index", type="primary", use_container_width=True):
            with st.spinner("Uploading..."):
                file_bytes = uploaded_file.getvalue()
                result = upload_document(file_bytes, uploaded_file.name)
                
            if "error" in result:
                st.error(f"Upload failed: {result['error']}")
            else:
                job_id = result["job_id"]
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                # Poll for background task completion
                while True:
                    status = poll_job_status(job_id)
                    if status:
                        progress_bar.progress(status["progress"])
                        status_text.caption(status["message"])
                        if status["state"] in ["completed", "failed"]:
                            if status["state"] == "completed":
                                st.success("Document indexed successfully!")
                            else:
                                st.error(f"Ingestion failed: {status.get('error')}")
                            break
                    time.sleep(1.5)

    st.divider()

    # --- WEEK 3: EVALUATION UI ---
    st.markdown("### 🧪 RAG Evaluation")
    st.caption("Run the LLM-as-a-judge evaluation suite against the golden dataset.")
    
    if st.button("Run Evaluation", use_container_width=True):
        eval_start = call_evaluate_api()
        if "error" in eval_start:
            st.error(f"Evaluation start failed: {eval_start['error']}")
        else:
            job_id = eval_start["job_id"]
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            while True:
                status = poll_eval_status(job_id)
                if status:
                    progress_bar.progress(status["progress"])
                    status_text.caption(status["message"])
                    if status["state"] in ["completed", "failed"]:
                        if status["state"] == "completed":
                            st.success("Evaluation completed successfully!")
                            eval_data = status["result"]
                            composite = (eval_data["summary"]["avg_context_precision"] + eval_data["summary"]["avg_context_recall"] + (eval_data["summary"]["avg_correctness"]/5.0)) / 3.0
                            st.success(f"Composite Score: {composite:.2f}")
                            with st.expander("View Full Metrics"):
                                st.json(eval_data)
                        else:
                            st.error(f"Evaluation failed: {status.get('error')}")
                        break
                time.sleep(1.5)
                    
    st.divider()

    st.markdown("### Backend Status")
    health = call_health_api()
    if health:
        st.success(f"✅ API Online")
        st.caption(f"**Model:** {health['model']}")
        st.caption(f"**Vector DB:** {health['vector_store'].upper()}")
    else:
        st.error("❌ Backend offline")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
st.title("Loan Underwriting Assistant")
st.caption("AI-powered risk analysis with multi-tool assessment and RAG grounding.")

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            # 1. Render Structured Output (Week 2)
            if message.get("structured_output"):
                so = message["structured_output"]
                with st.expander("📊 Structured Decision", expanded=False):
                    c1, c2 = st.columns(2)
                    c1.metric("Decision", so.get("decision", "N/A"))
                    c2.metric("Risk Level", so.get("risk_score", "N/A"))
                    for step in so.get("reasoning", []):
                        st.markdown(f"✅ {step}")
            
            # 2. Render Citations (Week 3)
            if message.get("citations"):
                with st.expander("📚 Source Citations", expanded=False):
                    for idx, cite in enumerate(message["citations"]):
                        score = f" (Score: {cite.get('relevance_score'):.2f})" if cite.get('relevance_score') else ""
                        st.markdown(f"**[{idx+1}] {cite.get('source_id', 'Document')}**{score}")
                        st.info(f"_{cite.get('snippet', '')}_")

            # 3. Render Tools Used
            if message.get("tools_used"):
                st.caption(f"🛠️ Tools used: {', '.join(message['tools_used'])}")


# Chat Input
if prompt := st.chat_input("Ex: What is the bank's policy on commercial real estate loans?"):
    # Render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Fetch and render assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            result = call_chat_api(prompt)

        response_text = result.get("response", "No response received.")
        structured = result.get("structured_output")
        citations = result.get("citations", [])
        tools_used = result.get("tools_used", [])

        st.markdown(response_text)

        # Render Structured Output
        if structured:
            with st.expander("📊 Structured Decision", expanded=True):
                c1, c2 = st.columns(2)
                c1.metric("Decision", structured.get("decision", "N/A"))
                c2.metric("Risk Level", structured.get("risk_score", "N/A"))
                for step in structured.get("reasoning", []):
                    st.markdown(f"✅ {step}")
                    
        # Render Citations
        if citations:
            with st.expander("📚 Source Citations", expanded=True):
                for idx, cite in enumerate(citations):
                    score = f" (Score: {cite.get('relevance_score'):.2f})" if cite.get('relevance_score') else ""
                    st.markdown(f"**[{idx+1}] {cite.get('source_id', 'Document')}**{score}")
                    st.info(f"_{cite.get('snippet', '')}_")

        if tools_used:
            st.caption(f"🛠️ Tools used: {', '.join(tools_used)}")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "structured_output": structured,
        "citations": citations,
        "tools_used": tools_used,
    })
