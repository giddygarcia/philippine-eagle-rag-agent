import logging
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import time

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


from tools import make_rag_tools
from ui import apply_styles, render_header, render_sidebar_header
from utils import (
    StreamlitStatusCallback,
    check_rate_limit,
    load_vectorstore,
    validate_input,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Philippine Eagle Conservation Research Agent",
    page_icon="🦅",
    layout="centered",
)


apply_styles()
render_header()
render_sidebar_header()


load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", None)

models = {
    "Gemma 4 31B (Free)": "google/gemma-4-31b-it:free",
    "Nemotron 3 Ultra (Free)": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "GPT-5 Mini": "openai/gpt-5-mini",
    "Gemini 3 Flash": "google/gemini-3-flash",
}

# CONFIG

LM_STUDIO_URL = "http://localhost:1234/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"


# EMBEDDINGS + CHROMA
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_vectorstore():
    embeddings = get_embeddings()
    return load_vectorstore("chroma_db", embeddings)


vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []
if "provider" not in st.session_state:
    st.session_state.provider = "OpenRouter"
if "model" not in st.session_state:
    st.session_state.model = "google/gemma-4-31b-it:free"
if "base_url" not in st.session_state:
    st.session_state.base_url = OPENROUTER_URL
if "api_key" not in st.session_state:
    st.session_state.api_key = OPENROUTER_KEY or "lm-studio"

# SIDEBAR SETTINGS
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    provider = st.selectbox(
        "Choose provider",
        ["OpenRouter", "LM Studio"],
        index=["OpenRouter", "LM Studio"].index(st.session_state.provider),
    )

    if provider == "OpenRouter":
        if not OPENROUTER_KEY:
            st.warning("OpenRouter key missing → switching to LM Studio")
            provider = "LM Studio"

    if provider == "LM Studio":
        base_url = LM_STUDIO_URL
        api_key = "lm-studio"
        model = "gemma"
        st.success("🟣 Using LM Studio (local)")

    elif provider == "OpenRouter":
        base_url = OPENROUTER_URL
        api_key = OPENROUTER_KEY

        selected_model_name = st.selectbox(
            "Model",
            list(models.keys()),
            index=(
                list(models.values()).index(st.session_state.model)
                if st.session_state.model in models.values()
                else 0
            ),
        )
        model = models[selected_model_name]
        st.success(f"🔵 Using OpenRouter - {selected_model_name}")

    # Persist to session state
    st.session_state.provider = provider
    st.session_state.model = model
    st.session_state.base_url = base_url
    st.session_state.api_key = api_key

    if st.session_state.messages:
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# LLM + AGENT
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.txt")


@st.cache_resource
def get_agent(model: str, base_url: str, api_key: str):
    """Create and cache the LangChain agent."""
    llm = ChatOpenAI(
        model=model, base_url=base_url, api_key=api_key, temperature=0, max_tokens=2048
    )

    tools = make_rag_tools(llm, retriever)

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )


agent = get_agent(
    st.session_state.model,
    st.session_state.base_url,
    st.session_state.api_key,
)
# CHAT UI

chat_box = st.container(height=550, border=True)

with chat_box:
    for msg in st.session_state.messages:
        with st.chat_message(
            msg["role"], avatar="🦅" if msg["role"] == "assistant" else "💬"
        ):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📚 Sources"):
                    st.markdown(msg["citations"])
            if msg.get("tool_activity"):
                with st.expander(f"🔧 Tool calls ({len(msg['tool_activity'])})"):
                    for tc in msg["tool_activity"]:
                        st.markdown(f"**`{tc['name']}`**")
                        if tc["args"]:
                            st.code(
                                "\n".join(f"{k}: {v}" for k, v in tc["args"].items()),
                                language="yaml",
                            )
                        if tc["result"]:
                            st.caption("Result preview:")
                            st.markdown(
                                tc["result"][:300] + "..."
                                if len(tc["result"]) > 300
                                else tc["result"]
                            )


# HANDLE INPUT
user_input = st.chat_input("Ask something about the Philippine Eagle...")

if user_input:
    is_valid, validation_msg = validate_input(user_input)
    if not is_valid:
        st.warning(validation_msg)
    else:
        allowed, rate_msg = check_rate_limit()
        if not allowed:
            st.warning(rate_msg)
        else:
            with chat_box:
                with st.chat_message("user", avatar="💬"):
                    st.markdown(user_input)

                with st.chat_message("assistant", avatar="🦅"):
                    citations = ""
                    tool_activity = []
                    response = ""

                    try:
                        with st.status("Thinking...", expanded=False) as status:
                            start_time = time.time()
                            logger.info(f"[REQ] {time.time()} | {user_input[:100]}")
                            callback = StreamlitStatusCallback(status)

                            messages = []
                            for msg in st.session_state.messages:
                                if msg["role"] == "user":
                                    messages.append(
                                        HumanMessage(content=msg["content"])
                                    )
                                else:
                                    messages.append(AIMessage(content=msg["content"]))
                            messages.append(HumanMessage(content=user_input))

                            result = agent.invoke(
                                {"messages": messages},
                                config={"callbacks": [callback]},
                            )
                            response = result["messages"][-1].content
                            status.update(
                                label="✅ Done", state="complete", expanded=False
                            )
                            for msg in result["messages"]:
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        tool_activity.append(
                                            {
                                                "name": tc["name"],
                                                "args": tc["args"],
                                                "result": None,
                                            }
                                        )
                                if isinstance(msg, ToolMessage):
                                    for tc in tool_activity:
                                        if (
                                            tc["result"] is None
                                            and tc["name"] == msg.name
                                        ):
                                            tc["result"] = msg.content.split(
                                                "<<<SOURCES>>>"
                                            )[0].strip()
                                            break
                                    if (
                                        isinstance(msg, ToolMessage)
                                        and "<<<SOURCES>>>" in msg.content
                                    ):
                                        citations = msg.content.split("<<<SOURCES>>>")[
                                            1
                                        ].strip()

                            logger.info(
                                f"Query handled in {time.time() - start_time:.2f}s"
                            )

                    except Exception as e:
                        logger.error(f"Agent call failed: {e}", exc_info=True)
                        error_str = str(e)

                        if "free-models-per-day" in error_str or "429" in error_str:
                            response = "⚠️ You've hit your OpenRouter free-tier limit for today. Try again tomorrow or switch to LM Studio."
                            st.error(
                                "🚫 OpenRouter quota exceeded (free-tier limit reached)",
                                icon="⛔",
                            )
                            with st.expander("🔍 Technical details"):
                                st.code(error_str)

                        elif "500" in error_str or "Internal Server Error" in error_str:
                            response = "The model is currently unavailable. Try switching models in the sidebar."
                            st.error("⚠️ Model error (500)", icon="🚨")

                        elif "401" in error_str or "unauthorized" in error_str.lower():
                            response = (
                                "Invalid API key. Please check your OpenRouter key."
                            )
                            st.error("🔑 Invalid API key", icon="🚨")

                        else:
                            response = "Unexpected error occurred. Please try again."
                            st.error(f"❌ Error: {error_str[:200]}", icon="🚨")

                        citations = ""
                        tool_activity = []

                    st.markdown(response)

                    if citations:
                        with st.expander("📚 Sources"):
                            st.markdown(citations)

                    if tool_activity:
                        with st.expander(f"🔧 Tool calls ({len(tool_activity)})"):
                            for tc in tool_activity:
                                st.markdown(f"**`{tc['name']}`**")
                                if tc["args"]:
                                    st.code(
                                        "\n".join(
                                            f"{k}: {v}" for k, v in tc["args"].items()
                                        ),
                                        language="yaml",
                                    )
                                if tc["result"]:
                                    st.caption("Result preview:")
                                    st.markdown(
                                        tc["result"][:300] + "..."
                                        if len(tc["result"]) > 300
                                        else tc["result"]
                                    )
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "citations": citations,
                    "tool_activity": tool_activity,
                }
            )
