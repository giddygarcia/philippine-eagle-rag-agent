import logging
import os
import re
import time

import streamlit as st
from langchain_chroma import Chroma


from langchain_core.callbacks import BaseCallbackHandler
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
import threading


class StreamlitStatusCallback(BaseCallbackHandler):
    def __init__(self, status):
        self.status = status
        self.ctx = get_script_run_ctx()

    def _add_ctx(self):
        add_script_run_ctx(threading.current_thread(), self.ctx)

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._add_ctx()
        self.status.write("🧠 Thinking...")

    def on_llm_end(self, response, **kwargs):
        self._add_ctx()
        self.status.write("✅ Response received")

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._add_ctx()
        name = serialized.get("name", "tool")
        self.status.write(f"🔧 Calling `{name}`...")

    def on_tool_end(self, output, **kwargs):
        self._add_ctx()
        self.status.write("✅ Tool finished")

    def on_retriever_start(self, serialized, query, **kwargs):
        self._add_ctx()
        self.status.write(f'🔍 Retrieving docs for: "{query[:60]}..."')

    def on_retriever_end(self, documents, **kwargs):
        self._add_ctx()
        self.status.write(f"📄 Retrieved {len(documents)} documents")

    def on_chain_start(self, serialized, inputs, **kwargs):
        pass

    def on_chain_end(self, outputs, **kwargs):
        pass


logger = logging.getLogger(__name__)


MAX_INPUT_LENGTH = 500
MIN_INPUT_LENGTH = 2

BLOCKED_PATTERNS = [
    # Prompt injection
    r"ignore (all|previous|above) instructions",
    r"disregard (all|previous|above)",
    r"reveal (the|your) system prompt",
    r"you are now (?!.*eagle)",
    # Sensitive location redaction
    r"(exact|gps|coordinates|location) of (nest|nesting site|roost)",
    r"where (exactly|specifically) (does|do) (the|philippine) eagle (nest|roost|live)",
]

RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60


def validate_input(text: str) -> tuple[bool, str]:
    """
    Validate user input for length and blocked patterns.

    Checks for empty input, minimum and maximum length, prompt injection
    attempts, and sensitive location requests specific to Philippine Eagle
    conservation (nesting sites, GPS coordinates, roost locations).

    Args:
        text: Raw input string from the user.

    Returns:
        A tuple of (is_valid, error_message). If valid, error_message is
        an empty string. If invalid, error_message describes the reason.
    """
    text = text.strip()

    if not text:
        return False, "Please enter a question."

    if len(text) < MIN_INPUT_LENGTH:
        return False, "Your question is too short — please add a bit more detail."

    if len(text) > MAX_INPUT_LENGTH:
        return False, f"Please keep questions under {MAX_INPUT_LENGTH} characters."

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning("Blocked prompt injection attempt: %s", text[:100])
            if "nest" in pattern or "roost" in pattern or "coordinates" in pattern:
                logger.warning("Blocked sensitive location request: %s", text[:100])
                return (
                    False,
                    "Specific nesting site locations are kept confidential to protect the Philippine Eagle from poaching.",
                )
            return False, "Sorry, I couldn't process that question. Please rephrase it."

    return True, ""


def check_rate_limit() -> tuple[bool, str]:
    """
    Enforce a sliding-window rate limit using Streamlit session state.

    Allows up to RATE_LIMIT_MAX_REQUESTS requests per
    RATE_LIMIT_WINDOW_SECONDS per browser session. Timestamps older than
    the window are discarded on each call.

    Returns:
        A tuple of (allowed, message). If allowed, message is an empty
        string. If rate limited, message includes the number of seconds
        the user should wait.
    """
    now = time.time()

    if "request_timestamps" not in st.session_state:
        st.session_state.request_timestamps = []

    st.session_state.request_timestamps = [
        t
        for t in st.session_state.request_timestamps
        if now - t < RATE_LIMIT_WINDOW_SECONDS
    ]

    if len(st.session_state.request_timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        wait = RATE_LIMIT_WINDOW_SECONDS - (
            now - st.session_state.request_timestamps[0]
        )
        logger.warning("Rate limit exceeded")
        return False, f"You're sending messages too quickly. Wait {int(wait)}s."

    st.session_state.request_timestamps.append(now)
    return True, ""


def load_vectorstore(persist_directory: str, embeddings) -> Chroma:
    """
    Load a persisted Chroma vectorstore from disk.

    Validates that the directory exists before attempting to load. Calls
    st.stop() on failure so the Streamlit app halts with a visible error
    rather than continuing in a broken state.

    Args:
        persist_directory: Path to the Chroma persistence directory.
        embeddings: LangChain-compatible embedding function used to
            initialise the vectorstore.

    Returns:
        A loaded Chroma vectorstore instance.
    """
    if not os.path.isdir(persist_directory):
        logger.error("Chroma directory not found: %s", persist_directory)
        st.error("Knowledge base not found. Run ingestion first.")
        st.stop()

    try:
        vs = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        logger.info("Vectorstore loaded successfully")
        return vs

    except Exception as e:
        logger.error("Failed to load vectorstore: %s", e, exc_info=True)
        st.error("Failed to load knowledge base.")
        st.stop()
