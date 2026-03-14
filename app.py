"""
Multi-Agent Customer Support System
Uses LangGraph + LangChain with a supervisor routing pattern.
"""

import json
import re
import uuid
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Final

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Multi-Agent State
# ---------------------------------------------------------------------------

class MultiAgentState(TypedDict):
    user_request: str        # original user message
    route: str               # "orders" | "billing" | "technical" | "subscription" | "general"
    agent_used: str          # which specialist handled it
    specialist_result: str   # raw output from specialist agent
    final_response: str      # final response returned to the user


# ---------------------------------------------------------------------------
# 2. Structured Handoff
# ---------------------------------------------------------------------------

@dataclass
class AgentHandoff:
    from_agent: str
    to_agent: str
    task: str
    context: dict
    priority: str   # "low" | "normal" | "high"
    timestamp: str

    def to_prompt_context(self) -> str:
        return (
            f"HANDOFF FROM {self.from_agent.upper()} TO {self.to_agent.upper()}:\n"
            f"Task: {self.task}\n"
            f"Priority: {self.priority}\n"
            f"Context: {self.context}\n"
            f"Received at: {self.timestamp}"
        )


# ---------------------------------------------------------------------------
# 3. Session Audit Log
# ---------------------------------------------------------------------------

@dataclass
class SessionAuditLog:
    session_id: str
    events: list[dict] = field(default_factory=list)
    total_cost_usd: float = 0.0

    def log(self, agent: str, action: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
        cost = (tokens_in * 0.000015 + tokens_out * 0.00006) / 1000
        self.total_cost_usd += cost
        self.events.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": agent,
                "action": action,
                "cost_usd": round(cost, 6),
            }
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "events": self.events,
        }


def persist_audit_log(audit: SessionAuditLog) -> None:
    path = Path("audit_log.jsonl")
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(audit.to_dict()) + "\n")


# ---------------------------------------------------------------------------
# 4. Injection Detection
# ---------------------------------------------------------------------------

INJECTION_PATTERNS: Final[list[str]] = [
    r"ignore (your |all |previous )?instructions",
    r"system prompt.*disabled",
    r"you are now a",
    r"repeat.*system prompt",
    r"jailbreak",
]


def detect_injection(user_input: str) -> bool:
    text = user_input.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def guard_request(user_input: str) -> str:
    if detect_injection(user_input):
        return "I can only assist with account and order support. (Request blocked.)"
    return user_input


# ---------------------------------------------------------------------------
# 5. Load Supervisor YAML Prompt
# ---------------------------------------------------------------------------

def load_supervisor_prompt(path: str = "prompts/supervisor_v1.yaml") -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["system"]


# ---------------------------------------------------------------------------
# 6. LLM Setup
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

VALID_ROUTES = {"orders", "billing", "technical", "subscription", "general"}

# Load supervisor prompt at module level so nodes can access it
supervisor_system_prompt = load_supervisor_prompt()

# ---------------------------------------------------------------------------
# 7. Graph Nodes
# ---------------------------------------------------------------------------

def supervisor_node(state: MultiAgentState) -> dict:
    messages = [
        SystemMessage(content=supervisor_system_prompt),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    route = response.content.strip().lower()
    if route not in VALID_ROUTES:
        route = "general"
    return {"route": route}


def route_to_specialist(state: MultiAgentState) -> str:
    route_map: dict[str, str] = {
        "orders": "orders_agent_node",
        "billing": "billing_agent_node",
        "technical": "technical_agent_node",
        "subscription": "subscription_agent_node",
        "general": "general_agent_node",
    }
    return route_map.get(state["route"], "general_agent_node")


def _make_handoff(from_agent: str, to_agent: str, state: MultiAgentState) -> AgentHandoff:
    return AgentHandoff(
        from_agent=from_agent,
        to_agent=to_agent,
        task=state["user_request"],
        context={"route": state["route"]},
        priority="normal",
        timestamp=datetime.utcnow().isoformat(),
    )


def orders_agent_node(state: MultiAgentState) -> dict:
    handoff = _make_handoff("supervisor", "orders", state)
    messages = [
        SystemMessage(content=(
            "You are the orders specialist. Help with returns, order status, "
            "tracking, and late deliveries. Be concise.\n\n"
            + handoff.to_prompt_context()
        )),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    return {
        "agent_used": "orders_agent",
        "specialist_result": response.content,
    }


def billing_agent_node(state: MultiAgentState) -> dict:
    handoff = _make_handoff("supervisor", "billing", state)
    messages = [
        SystemMessage(content=(
            "You are the billing specialist. Help with payments, refunds, "
            "double charges, and invoices. Be concise.\n\n"
            + handoff.to_prompt_context()
        )),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    return {
        "agent_used": "billing_agent",
        "specialist_result": response.content,
    }


def technical_agent_node(state: MultiAgentState) -> dict:
    handoff = _make_handoff("supervisor", "technical", state)
    messages = [
        SystemMessage(content=(
            "You are the technical support specialist. Help with app bugs, "
            "login issues, crashes, and errors. Be concise.\n\n"
            + handoff.to_prompt_context()
        )),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    return {
        "agent_used": "technical_agent",
        "specialist_result": response.content,
    }


def subscription_agent_node(state: MultiAgentState) -> dict:
    handoff = _make_handoff("supervisor", "subscription", state)
    messages = [
        SystemMessage(content=(
            "You are the subscription specialist. Help with plan upgrades/downgrades, "
            "cancellations, and pricing questions. Be concise.\n\n"
            + handoff.to_prompt_context()
        )),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    return {
        "agent_used": "subscription_agent",
        "specialist_result": response.content,
    }


def general_agent_node(state: MultiAgentState) -> dict:
    handoff = _make_handoff("supervisor", "general", state)
    messages = [
        SystemMessage(content=(
            "You are a general customer support agent. Help with business hours, "
            "locations, and any other general questions. Be concise.\n\n"
            + handoff.to_prompt_context()
        )),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    return {
        "agent_used": "general_agent",
        "specialist_result": response.content,
    }


def synthesize_response_node(state: MultiAgentState) -> dict:
    final = (
        f"[Handled by {state['agent_used']}]\n"
        f"{state['specialist_result']}"
    )
    return {"final_response": final}


# ---------------------------------------------------------------------------
# 8. Build Graph
# ---------------------------------------------------------------------------

def build_graph():
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("supervisor_node", supervisor_node)
    workflow.add_node("orders_agent_node", orders_agent_node)
    workflow.add_node("billing_agent_node", billing_agent_node)
    workflow.add_node("technical_agent_node", technical_agent_node)
    workflow.add_node("subscription_agent_node", subscription_agent_node)
    workflow.add_node("general_agent_node", general_agent_node)
    workflow.add_node("synthesize_response", synthesize_response_node)

    workflow.set_entry_point("supervisor_node")

    workflow.add_conditional_edges(
        "supervisor_node",
        route_to_specialist,
    )

    for specialist in [
        "orders_agent_node",
        "billing_agent_node",
        "technical_agent_node",
        "subscription_agent_node",
        "general_agent_node",
    ]:
        workflow.add_edge(specialist, "synthesize_response")

    workflow.add_edge("synthesize_response", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# 9. main()
# ---------------------------------------------------------------------------

def main() -> None:
    audit = SessionAuditLog(session_id=f"demo-{uuid.uuid4().hex[:8]}")
    graph = build_graph()

    requests = [
        "My order ORD-123 is late, can I return it?",
        "I want to upgrade from Basic to Pro. What will it cost?",
    ]

    for request in requests:
        safe_text = guard_request(request)

        # Log guard check
        audit.log(agent="guard", action=f"input_checked: '{request[:50]}'")

        state: MultiAgentState = {
            "user_request": safe_text,
            "route": "general",
            "agent_used": "",
            "specialist_result": "",
            "final_response": "",
        }

        result = graph.invoke(state)

        # Mock token counts for cost tracking
        audit.log(
            agent="supervisor",
            action=f"routed_to={result.get('route')}",
            tokens_in=120,
            tokens_out=10,
        )
        audit.log(
            agent=result.get("agent_used", "unknown"),
            action="specialist_response",
            tokens_in=200,
            tokens_out=150,
        )

        print(f"\nRequest : {request}")
        print(f"Route   : {result.get('route')}  |  Agent used: {result.get('agent_used')}")
        print(f"Final   : {result.get('final_response')}")
        print("-" * 60)

    print(f"\nTotal cost (USD): {round(audit.total_cost_usd, 6)}")
    persist_audit_log(audit)
    print("Audit log saved to audit_log.jsonl")


if __name__ == "__main__":
    main()
