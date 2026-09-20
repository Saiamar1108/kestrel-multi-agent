from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from app.agents.planner import plan
from app.agents.retriever import retrieve
from app.agents.synthesizer import synthesize
from app.agents.verifier import verify
from app.state import AgentState


def _traced_node(name: str, node):
    return traceable(node, name=name, run_type="chain")


def build_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", _traced_node("Planner", plan))
    workflow.add_node("retriever", _traced_node("Retriever", retrieve))
    workflow.add_node("verifier", _traced_node("Verifier", verify))
    workflow.add_node("synthesizer", _traced_node("Synthesizer", synthesize))
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "verifier")
    workflow.add_edge("verifier", "synthesizer")
    workflow.add_edge("synthesizer", END)
    return workflow.compile()


workflow = build_workflow()


if __name__ == "__main__":
    result = workflow.invoke({"question": "What is Kestrel?"})
    print(result["answer"])
