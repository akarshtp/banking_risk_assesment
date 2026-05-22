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
if "current_role" not in st.session_state:
    st.session_state.current_role = "junior_analyst"


# ─────────────────────────────────────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def call_chat_api(message: str) -> dict:
    try:
        response = httpx.post(
            f"{API_BASE_URL}/chat",
            json={
                "session_id": st.session_state.session_id, 
                "message": message,
                "role_name": st.session_state.current_role
            },
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

def get_pending_hitl():
    try:
        return httpx.get(f"{API_BASE_URL}/hitl/pending", timeout=5.0).json().get("pending_tasks", [])
    except Exception:
        return []

def resolve_hitl(task_id, decision, comments):
    try:
        httpx.post(f"{API_BASE_URL}/hitl/review/{task_id}", json={
            "decision": decision, "comments": comments, "reviewer": st.session_state.current_role
        }, timeout=5.0)
    except Exception:
        pass

def get_prompts():
    try:
        return httpx.get(f"{API_BASE_URL}/prompts", timeout=5.0).json().get("prompts", {})
    except Exception:
        return {}

def get_drift_analysis():
    try:
        return httpx.post(f"{API_BASE_URL}/eval/drift", timeout=5.0).json().get("drift_analysis", {})
    except Exception:
        return {}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Controls")
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")

    st.markdown("### 👤 User Role")
    roles = ["junior_analyst", "senior_underwriter", "credit_head", "auditor"]
    current_index = roles.index(st.session_state.current_role) if st.session_state.current_role in roles else 0
    st.session_state.current_role = st.selectbox(
        "Simulate Role",
        roles,
        index=current_index
    )

    try:
        auth_context = httpx.get(f"{API_BASE_URL}/auth/context", params={"role": st.session_state.current_role}, timeout=5.0).json()
        if "active_filters" in auth_context:
            filters = auth_context["active_filters"]
            if not filters:
                st.success("🔐 **Access Level: ALL** (No restrictions)")
            else:
                conf = filters.get('confidentiality', 'Unknown')
                if isinstance(conf, dict):
                    conf = conf.get('$in', conf)
                st.warning(f"🔐 **Access Level: RESTRICTED**\n\nAllowed: `{conf}`")
    except Exception:
        pass

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

    st.markdown("### Backend Status")
    health = call_health_api()
    if health:
        st.success(f"✅ API Online")
        st.caption(f"**Model:** {health['model']}")
        st.caption(f"**Vector DB:** {health['vector_store'].upper()}")
    else:
        st.error("❌ Backend offline")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN INTERFACE WITH TABS
# ─────────────────────────────────────────────────────────────────────────────
st.title("Loan Underwriting Assistant")
st.caption("AI-powered risk analysis with multi-tool assessment and RAG grounding.")

tab_chat, tab_hitl, tab_prompts, tab_eval, tab_mcp = st.tabs([
    "💬 Chat", "✅ HITL Approvals", "📝 Prompt Viewer", "📊 Eval Dashboard", "🔌 MCP Servers"
])

with tab_chat:
    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                if message.get("structured_output"):
                    so = message["structured_output"]
                    with st.expander("📊 Structured Decision", expanded=False):
                        c1, c2 = st.columns(2)
                        c1.metric("Decision", so.get("decision", "N/A"))
                        c2.metric("Risk Level", so.get("risk_score", "N/A"))
                        for step in so.get("reasoning", []):
                            st.markdown(f"✅ {step}")
                
                if message.get("citations"):
                    with st.expander("📚 Source Citations", expanded=False):
                        for idx, cite in enumerate(message["citations"]):
                            score = f" (Score: {cite.get('relevance_score'):.2f})" if cite.get('relevance_score') else ""
                            st.markdown(f"**[{idx+1}] {cite.get('source_id', 'Document')}**{score}")
                            st.info(f"_{cite.get('snippet', '')}_")

                if message.get("tools_used"):
                    st.caption(f"🛠️ Tools used: {', '.join(message['tools_used'])}")

    # Chat Input
    if prompt := st.chat_input("Ex: What is the bank's policy on commercial real estate loans?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                result = call_chat_api(prompt)

            response_text = result.get("response", "No response received.")
            structured = result.get("structured_output")
            citations = result.get("citations", [])
            tools_used = result.get("tools_used", [])

            st.markdown(response_text)

            if structured:
                with st.expander("📊 Structured Decision", expanded=True):
                    c1, c2 = st.columns(2)
                    c1.metric("Decision", structured.get("decision", "N/A"))
                    c2.metric("Risk Level", structured.get("risk_score", "N/A"))
                    for step in structured.get("reasoning", []):
                        st.markdown(f"✅ {step}")
                        
            if citations:
                with st.expander("📚 Source Citations", expanded=True):
                    for idx, cite in enumerate(citations):
                        score = f" (Score: {cite.get('relevance_score'):.2f})" if cite.get('relevance_score') else ""
                        st.markdown(f"**[{idx+1}] {cite.get('source_id', 'Document')}**{score}")
                        st.info(f"_{cite.get('snippet', '')}_")

            if tools_used:
                st.caption(f"🛠️ Tools used: {', '.join(tools_used)}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "structured_output": structured,
            "citations": citations,
            "tools_used": tools_used,
        })

with tab_hitl:
    st.header("Pending Human-in-the-Loop Reviews")
    tasks = get_pending_hitl()
    if not tasks:
        st.info("No pending tasks requiring manual review.")
    else:
        for t in tasks:
            with st.expander(f"Task {t.get('task_id', 'Unknown')} ({t.get('status', 'pending')})"):
                st.write(f"**Trigger Rule:** {t.get('rule', {}).get('id', 'Unknown')}")
                st.json(t.get('agent_context', '{}'))
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Approve", key=f"approve_{t.get('task_id')}"):
                        resolve_hitl(t.get('task_id'), "approve", "Approved via UI")
                        st.rerun()
                with col2:
                    if st.button("Reject", key=f"reject_{t.get('task_id')}"):
                        resolve_hitl(t.get('task_id'), "reject", "Rejected via UI")
                        st.rerun()

with tab_prompts:
    st.header("Versioned Prompt Registry")
    prompts = get_prompts()
    if not prompts:
        st.warning("No prompts loaded.")
    else:
        for name, meta in prompts.items():
            st.subheader(f"`{name}` (v{meta.get('version', '1.0.0')})")
            st.write(f"**Description:** {meta.get('description', 'N/A')}")
            st.divider()

with tab_eval:
    st.header("Evaluation & Drift Dashboard")
    st.markdown("Metrics from recent regression runs and data drift analysis.")
    
    st.markdown("### 🧪 RAG Evaluation Suite")
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
                            
                            st.markdown("#### Detailed Metrics")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Hit@3 (Retrieval)", f"{eval_data['summary']['avg_hit_at_3']:.2f}")
                            m2.metric("MRR (Retrieval)", f"{eval_data['summary']['avg_mrr']:.2f}")
                            m3.metric("Context Precision", f"{eval_data['summary']['avg_context_precision']:.2f}")
                            
                            m4, m5, m6 = st.columns(3)
                            m4.metric("Correctness (LLM Judge)", f"{eval_data['summary']['avg_correctness']:.1f} / 5")
                            m5.metric("Completeness (LLM Judge)", f"{eval_data['summary']['avg_completeness']:.1f} / 5")
                            m6.metric("Citation Quality", f"{eval_data['summary']['avg_citation_quality']:.1f} / 5")
                            
                            with st.expander("View Full Raw JSON"):
                                st.json(eval_data)
                        else:
                            st.error(f"Evaluation failed: {status.get('error')}")
                        break
                time.sleep(1.5)
                
    st.divider()

    st.markdown("### 📉 Data Drift Analysis")
    if st.button("Run Data Drift Analysis", use_container_width=True):
        with st.spinner("Analyzing data drift..."):
            drift_data = get_drift_analysis()
            if drift_data:
                st.write(f"**KL Divergence:** {drift_data.get('kl_divergence', 0):.4f}")
                st.write(f"**Drift Detected:** {drift_data.get('drift_detected', False)}")
                if drift_data.get('drift_detected'):
                    st.warning("Data drift detected! Retraining or human review recommended.")
                else:
                    st.success("No significant data drift detected.")
            else:
                st.error("Failed to fetch drift analysis.")

with tab_mcp:
    st.header("Connected MCP Servers")
    st.markdown("The following tools are dynamically loaded from external Model Context Protocol (MCP) servers.")
    
    if st.button("Refresh MCP Tools"):
        st.rerun()
        
    try:
        mcp_res = httpx.get(f"{API_BASE_URL}/mcp/tools", timeout=10.0).json()
        mcp_tools = mcp_res.get("tools", [])
        if not mcp_tools:
            st.info("No MCP tools currently connected. Check `config/mcp_servers.yaml`.")
        else:
            for mt in mcp_tools:
                with st.container(border=True):
                    st.subheader(f"🛠️ `{mt.get('name')}`")
                    st.write(mt.get('description'))
    except Exception as e:
        st.error(f"Failed to fetch MCP tools. Is the backend fully booted? Error: {e}")
