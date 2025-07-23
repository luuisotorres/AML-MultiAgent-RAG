"""
AML Compliance Chat Interface - Streamlit UI

Chat interface for the AML Multi-Agent RAG system.
"""

import streamlit as st
import requests
from datetime import datetime
import time

st.set_page_config(
    page_title="AML Compliance AI Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://localhost:8000"

if "messages" not in st.session_state:
    st.session_state.messages = []
if "backend_status" not in st.session_state:
    st.session_state.backend_status = "unknown"


def check_backend_status():
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/multi-agent/status",
            timeout=10,
            headers={"accept": "application/json"}
        )
        if response.status_code == 200:
            status_data = response.json()
            return "connected", status_data
        else:
            return "error", {"error": f"HTTP {response.status_code}"}
    except requests.exceptions.Timeout:
        return "disconnected", {"error": "Request timeout"}
    except requests.exceptions.ConnectionError:
        return "disconnected", {"error": "Connection refused"}
    except requests.exceptions.RequestException as e:
        return "disconnected", {"error": str(e)}
    except Exception as e:
        return "disconnected", {"error": f"Unexpected error: {str(e)}"}


def force_check_backend():
    return check_backend_status()


def query_single_agent(question):
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/query",
            json={"question": question},
            timeout=30
        )
        response.raise_for_status()
        return True, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def query_multi_agent(question, include_detailed=True):
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/multi-agent/query",
            json={
                "question": question,
                "include_detailed_analysis": include_detailed
            },
            timeout=60
        )
        response.raise_for_status()
        return True, response.json()
    except Exception as e:
        return False, {"error": str(e)}


st.title("🏛️ AML Compliance AI Agent")
st.markdown(
    "*AI Agent for Anti-Money Laundering compliance questions*")

with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🔄 Check Backend Status", type="secondary"):
        with st.spinner("Checking backend..."):
            status, data = force_check_backend()
            st.session_state.backend_status = status
            st.rerun()

    status, status_data = check_backend_status()
    if status == "connected":
        st.success("🟢 Backend Connected")
        if "agents" in status_data:
            st.write("**Agents Status:**")
            for agent, agent_status in status_data.get("agents", {}).items():
                emoji = "🟢" if agent_status == "healthy" else "🔴"
                st.write(f"{emoji} {agent.replace('_', ' ').title()}")
    elif status == "error":
        st.error("🔴 Backend Error")
        st.write(status_data.get("error", "Unknown error"))
    else:
        st.warning("🟡 Backend Disconnected")
        st.write("Make sure your backend is running on localhost:8000")

    st.divider()

    st.header("🤖 Query Mode")
    query_mode = st.radio(
        "Select processing mode:",
        ["Multi-Agent (Recommended)", "Single Agent (Fast)"],
        help="Multi-Agent provides detailed analysis and quality validation"
    )

    if query_mode == "Multi-Agent (Recommended)":
        include_detailed = st.checkbox(
            "Include detailed analysis",
            value=True,
            help="Show detailed breakdown of agent processing"
        )

    st.divider()

    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()

st.header("💬 Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.write(message["content"])

            if "quality_score" in message:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Quality",
                        f"{message['quality_score']:.1%}",
                        help=(
                            "Response quality (0-100%): accuracy, completeness"
                            ", citations"
                        )
                    )
                if "confidence_score" in message:
                    with col2:
                        st.metric(
                            "Confidence",
                            f"{message['confidence_score']:.1%}",
                            help=(
                                "Confidence level (0-100%): source relevance"
                                ", specificity"
                            )
                        )
                if "confidence_level" in message:
                    with col3:
                        st.metric(
                            "Level",
                            message["confidence_level"].title(),
                            help="High (≥80%), Medium (60-80%), Low (<60%)"
                        )

            if any(k in message for k in ["consistency_score",
                                          "is_consistent"]):
                col1, col2, col3 = st.columns(3)
                if "consistency_score" in message:
                    with col1:
                        st.metric(
                            "Consistency",
                            f"{message['consistency_score']:.1%}",
                            help="Alignment with sources, no contradictions"
                        )
                if "is_consistent" in message:
                    with col2:
                        consistent = message["is_consistent"]
                        status = "Consistent" if consistent else "Inconsistent"
                        st.metric(
                            "Status",
                            status,
                            help="Consistent (≥60%) or Inconsistent (<60%)"
                        )
                if "quality_assessment" in message:
                    with col3:
                        st.write("**Assessment**"
                                 )
                        st.write(f"{message['quality_assessment']}")

            if "recommendations" in message and message["recommendations"]:
                st.markdown("### 💡 **Recommendations**")
                for i, rec in enumerate(message["recommendations"], 1):
                    st.markdown(f"**{i}.** {rec}")
                st.markdown("---")

            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**Source {i}:**")

                        content = source.get('content')
                        if content and content != 'N/A':
                            st.write(f"- **Content:** {content}")

                        score = source.get('score')
                        if score and score != 'N/A':
                            st.write(f"- **Score:** {score}")

                        filename = source.get('filename')
                        if filename and filename != 'N/A':
                            st.write(f"- **Filename:** {filename}")

                        region = source.get('region')
                        if region and region != 'N/A':
                            st.write(f"- **Jurisdiction:** {region}")

                        language = source.get('language')
                        if language and language != 'N/A':
                            st.write(f"- **Language:** {language}")

                        metadata = source.get('metadata')
                        if metadata and metadata != 'N/A':
                            st.write(f"- **Metadata:** {metadata}")

                        if i < len(message["sources"]):
                            st.divider()

            if "quality_gates" in message and message["quality_gates"]:
                with st.expander("Quality Gates", expanded=False):
                    gates = message["quality_gates"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        consistency_status = "✅ Passed" if gates.get(
                            "consistency_passed") else "❌ Failed"
                        st.write(f"**Consistency:** {consistency_status}")
                    with col2:
                        confidence_status = "✅ Passed" if gates.get(
                            "confidence_passed") else "❌ Failed"
                        st.write(f"**Confidence:** {confidence_status}")
                    with col3:
                        overall_status = "✅ Passed" if gates.get(
                            "overall_passed") else "❌ Failed"
                        st.write(f"**Overall:** {overall_status}")

            response_keys = ["quality_score", "sources", "quality_gates",
                             "recommendations"]
            if any(key in message for key in response_keys):
                response_data = {}
                keys_to_include = [
                    "quality_score", "confidence_score", "confidence_level",
                    "consistency_score", "is_consistent", "quality_assessment",
                    "recommendations", "sources", "quality_gates"
                ]
                for key in keys_to_include:
                    if key in message and key != "content":
                        response_data[key] = message[key]

                if response_data:
                    with st.expander("Response Data", expanded=False):
                        st.json(response_data)

            if "processing_time" in message:
                st.caption(
                    f"⏱️ Processing time: {message['processing_time']:.2f}s")

        else:
            st.write(message["content"])

if prompt := st.chat_input("Ask about AML compliance requirements..."):
    current_status, _ = force_check_backend()
    if current_status != "connected":
        st.error(
            "⚠️ Backend is not connected. Please check your backend server."
        )
        st.info(
            "💡 Try clicking the 'Check Backend Status' button in the sidebar."
        )
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing your query..."):
                start_time = time.time()

                if query_mode == "Multi-Agent (Recommended)":
                    success, result = query_multi_agent(
                        prompt, include_detailed)
                else:
                    success, result = query_single_agent(prompt)

                processing_time = time.time() - start_time

                if success:
                    response_content = result.get(
                        "answer", "No answer provided")
                    st.write(response_content)

                    message_data = {
                        "role": "assistant",
                        "content": response_content,
                        "processing_time": processing_time,
                        "timestamp": datetime.now().isoformat()
                    }

                    keys_to_add = [
                        "quality_score", "confidence_score",
                        "confidence_level", "sources", "detailed_analysis",
                        "consistency_score", "is_consistent",
                        "quality_assessment", "recommendations",
                        "metadata", "quality_gates"
                    ]
                    for key in keys_to_add:
                        if key in result:
                            message_data[key] = result[key]

                    if "quality_score" in result:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            quality_score = result["quality_score"]
                            st.metric(
                                "Quality",
                                f"{quality_score:.1%}",
                                help=(
                                    "Response quality (0-100%): accuracy,"
                                    " completeness, citations"
                                )
                            )
                        if "confidence_score" in result:
                            with col2:
                                st.metric(
                                    "Confidence",
                                    f"{result['confidence_score']:.1%}",
                                    help=(
                                        "Confidence level (0-100%): source "
                                        "relevance, specificity"
                                    )
                                )
                        if "confidence_level" in result:
                            with col3:
                                confidence_level = result["confidence_level"]
                                st.metric(
                                    "Level",
                                    confidence_level.title(),
                                    help=(
                                        "High (≥80%), Medium (60-80%), "
                                        "Low (<60%)"
                                    )
                                )

                    consistency_keys = ["consistency_score", "is_consistent"]
                    if any(k in result for k in consistency_keys):
                        col1, col2, col3 = st.columns(3)
                        if "consistency_score" in result:
                            with col1:
                                consistency_score = result["consistency_score"]
                                st.metric(
                                    "Consistency",
                                    f"{consistency_score:.1%}",
                                    help=(
                                        "Cross-agent agreement (0-100%):"
                                        " response reliability"
                                    )
                                )
                        if "is_consistent" in result:
                            with col2:
                                is_consistent = result["is_consistent"]
                                status = "Consistent" if is_consistent else \
                                    "Inconsistent"
                                st.metric(
                                    "Status",
                                    status,
                                    help=(
                                        "Consistent (≥60%) or "
                                        "Inconsistent (<60%)"
                                    )
                                )
                        if "quality_assessment" in result:
                            with col3:
                                st.write(
                                    "**Assessment**"
                                    )
                                st.write(f"{result['quality_assessment']}")

                    recommendations_exist = result.get("recommendations")
                    if recommendations_exist:
                        st.markdown("### **Recommendations**")
                        for i, rec in enumerate(result["recommendations"], 1):
                            st.markdown(f"**{i}.** {rec}")
                        st.markdown("---")

                    if "sources" in result and result["sources"]:
                        with st.expander("📚 Sources", expanded=False):
                            for i, source in enumerate(result["sources"], 1):
                                st.write(f"**Source {i}:**")

                                filename = source.get('filename')
                                if filename and filename != 'N/A':
                                    st.write(f"- **Filename:** {filename}")

                                region = source.get('region')
                                if region and region != 'N/A':
                                    st.write(f"- **Jurisdiction:** {region}")

                                language = source.get('language')
                                if language and language != 'N/A':
                                    st.write(f"- **Language:** {language}")

                                score = source.get('score')
                                if score and score != 'N/A':
                                    st.write(f"- **Relevance Score:** {score}")

                                content = source.get('content')
                                if content and content != 'N/A':
                                    st.write(f"**Content:** {content}")

                                metadata = source.get('metadata')
                                if metadata and metadata != 'N/A':
                                    st.write(f"- **Metadata:** {metadata}")

                                if i < len(result["sources"]):
                                    st.divider()

                    if "quality_gates" in result and result["quality_gates"]:
                        with st.expander("Quality Gates", expanded=False):
                            gates = result["quality_gates"]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                consistency_status = "✅ Passed" if gates.get(
                                    "consistency_passed") else "❌ Failed"
                                st.write(
                                    f"**Consistency:** {consistency_status}")
                            with col2:
                                confidence_status = "✅ Passed" if gates.get(
                                    "confidence_passed") else "❌ Failed"
                                st.write(
                                    f"**Confidence:** {confidence_status}")
                            with col3:
                                overall_status = "✅ Passed" if gates.get(
                                    "overall_passed") else "❌ Failed"
                                st.write(f"**Overall:** {overall_status}")

                    with st.expander("Response Data", expanded=False):
                        st.json(result)

                    st.session_state.messages.append(message_data)

                else:
                    error = result.get('error', 'Unknown error')
                    error_msg = f"❌ Error: {error}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "processing_time": processing_time,
                        "timestamp": datetime.now().isoformat()
                    })

                st.caption(f"⏱️ Processing time: {processing_time:.2f}s")
