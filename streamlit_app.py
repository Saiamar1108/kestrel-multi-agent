from __future__ import annotations

import os
from typing import Any

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def _configure_environment() -> None:
    for key in (
        "GROQ_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGSMITH_TRACING",
        "LANGSMITH_PROJECT",
    ):
        try:
            os.environ[key] = str(st.secrets[key])
        except (KeyError, StreamlitSecretNotFoundError):
            continue


_configure_environment()

from app.graph.workflow import workflow


def _conversation_history(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if message["role"] in {"user", "assistant"}
    ]


def _show_result_details(details: dict[str, Any]) -> None:
    assessment = details.get("claim_assessments", [{}])[0]
    verdict = assessment.get("verdict", assessment.get("status", "unknown"))
    retrieved_chunks = details.get("retrieved_chunks", [])
    evidence = assessment.get("evidence", [])

    st.caption("Agent/workflow status: Planner -> Retriever -> Verifier -> Synthesizer completed.")
    st.write(f"**Verifier verdict:** `{verdict}`")

    citations = [
        f"[{chunk['chunk_id']} — {chunk['title']}]"
        for chunk in evidence
    ]
    st.write("**Citations:** " + (" ".join(citations) if citations else "None"))

    chunk_ids = [str(chunk["chunk_id"]) for chunk in retrieved_chunks]
    st.write("**Retrieved chunk IDs:** " + (", ".join(chunk_ids) if chunk_ids else "None"))


def main() -> None:
    st.set_page_config(page_title="Kestrel Multi-Agent Research Assistant")
    st.title("Kestrel Multi-Agent Research Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("details"):
                _show_result_details(message["details"])

    question = st.chat_input("Ask a question about Kestrel")
    if not question:
        return

    history = _conversation_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Researching the corpus..."):
            result = workflow.invoke(
                {
                    "question": question.strip(),
                    "conversation_history": history,
                }
            )
        answer = result.get("answer", "No answer was produced.")
        st.markdown(answer)
        _show_result_details(result)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "details": result}
    )


if __name__ == "__main__":
    main()