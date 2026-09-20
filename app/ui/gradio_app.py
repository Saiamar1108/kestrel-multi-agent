from __future__ import annotations

from typing import Any

import gradio as gr

from app.graph.workflow import workflow


def _to_conversation_history(history: list[Any] | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append({"role": role, "content": content})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_message, assistant_message = item
            if isinstance(user_message, str):
                messages.append({"role": "user", "content": user_message})
            if isinstance(assistant_message, str):
                messages.append({"role": "assistant", "content": assistant_message})
    return messages


def _format_citations(result: dict[str, Any]) -> str:
    assessments = result.get("claim_assessments", [])
    evidence = assessments[0].get("evidence", []) if assessments else []
    if not evidence:
        return "No citations available."
    return "\n".join(
        f"- [{chunk['chunk_id']} — {chunk['title']}]"
        for chunk in evidence
    )


def _format_retrieved_chunks(result: dict[str, Any]) -> str:
    chunks = result.get("retrieved_chunks", [])
    if not chunks:
        return "No chunks retrieved."
    return "\n".join(
        f"- `{chunk['chunk_id']}`: {chunk['title']} (score: {chunk['score']:.3f})"
        for chunk in chunks
    )


def respond(
    question: str,
    history: list[Any] | None,
) -> tuple[list[dict[str, str]], str, str, str, str, str]:
    if not question.strip():
        return (
            _to_conversation_history(history),
            "",
            "Please enter a question.",
            "No citations available.",
            "Status: waiting for a question.",
            "No chunks retrieved.",
        )

    conversation_history = _to_conversation_history(history)
    result = workflow.invoke(
        {
            "question": question.strip(),
            "conversation_history": conversation_history,
        }
    )
    answer = result.get("answer", "No answer was produced.")
    updated_history = conversation_history + [
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": answer},
    ]
    status = result.get("claim_assessments", [{}])[0].get(
        "status", "unknown"
    )
    status_text = (
        "Status: planner -> retriever -> verifier -> synthesizer completed. "
        f"Verification: `{status}`."
    )
    return (
        updated_history,
        "",
        answer,
        _format_citations(result),
        status_text,
        _format_retrieved_chunks(result),
    )


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Kestrel Assistant") as app:
        gr.Markdown("# Kestrel Assistant")
        chatbot = gr.Chatbot(label="Conversation")
        question = gr.Textbox(label="Question", placeholder="Ask about Kestrel")
        submit = gr.Button("Submit")
        answer = gr.Markdown(label="Answer")
        citations = gr.Markdown(label="Citations")
        status = gr.Markdown(label="Agent status")
        retrieved = gr.Markdown(label="Retrieved chunks")

        outputs = [chatbot, question, answer, citations, status, retrieved]
        submit.click(respond, inputs=[question, chatbot], outputs=outputs)
        question.submit(respond, inputs=[question, chatbot], outputs=outputs)

    return app


app = create_app()


if __name__ == "__main__":
    app.launch()
