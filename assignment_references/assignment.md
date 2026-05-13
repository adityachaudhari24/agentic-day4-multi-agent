# Day 4 Assignment

## Multi-Agent Collaboration – Supervisor + Specialists

---

## 🎯 Objective

Build a **multi-agent customer support system** that applies the patterns from **Week 2 Session 4 – Multi-Agent Collaboration**:

- A **supervisor agent** that routes requests
- **Specialist agents** for different domains
- **Structured handoffs** between agents
- **Graceful degradation** (4-level fallback)
- **Session-level audit log with cost tracking**

By the end of this assignment, you will have a production-minded **multi-agent architecture** instead of a single “do-everything” agent.

---

## 📦 Submission Structure

Create a **public GitHub repository** with the following structure:

```
agentic-day4-multi-agent/
├── .gitignore
├── requirements.txt
├── README.md
├── app.py
└── prompts/
    └── supervisor_v1.yaml
```

You may add extra modules/files (e.g. `agents.py`, `audit.py`), but these are the **minimum required** for grading.

---

## 🛠 Requirements

### 1️⃣ Project Setup

Your project must:

- Use `python-dotenv` and call `load_dotenv()`
- Use **LangGraph** (`langgraph`) and **LangChain** chat models (e.g. `ChatOpenAI`, `langchain_openai.ChatOpenAI`)
- Include required dependencies in `requirements.txt`:
    - `python-dotenv`
    - `langgraph`
    - At least one of:
        - `langchain`
        - `langchain-openai`
- Ensure `.env` is excluded in `.gitignore`

Your `README.md` must:

- Show how to run the app:

```bash
python app.py
```

- Explicitly mention that `.env` **must not be committed**.

---

### 2️⃣ Define Multi-Agent State

In `app.py`, define a **typed state** for your multi-agent graph:

```python
from typing import TypedDict

class MultiAgentState(TypedDict):
    user_request: str        # original user message
    route: str               # "orders" | "billing" | "technical" | "subscription" | "general"
    agent_used: str          # which specialist handled it
    specialist_result: str   # raw output from specialist agent
    final_response: str      # final response returned to the user
```

Notes:

- You may add extra fields (e.g. `escalated: bool`, `level: str`) but these 5 keys must exist.
- This state will be shared across supervisor and specialist nodes.

---

### 3️⃣ Move Supervisor Prompt to YAML

Create a YAML file for the **supervisor classification prompt**:

```yaml
# prompts/supervisor_v1.yaml
version: "1.0"
name: supervisor_classifier
created_by: "YOUR_NAME"
created_at: "YYYY-MM-DD"
description: "Supervisor classification prompt for routing support requests"
changelog: |
  v1.0:
  - Initial version with 5 categories:
    orders, billing, technical, subscription, general

system: |
  You are a supervisor for a customer support AI system.

  Classify each request into EXACTLY ONE category:
  - orders: returns, order status, tracking, late deliveries
  - billing: payments, refunds, double charges, invoices
  - technical: app bugs, login issues, crashes, errors
  - subscription: plan upgrades/downgrades, cancellations, pricing questions
  - general: everything else (business hours, locations, generic questions)

  Respond with ONLY the category name:
  orders, billing, technical, subscription, or general.
```

Requirements:

- File must be `prompts/supervisor_v1.yaml`.
- Must include fields: `version`, `created_by`, `created_at`, `description`, `changelog`, `system`.
- In `app.py`, load this YAML (e.g. with `yaml.safe_load`) and use the `system` text for the supervisor classification call (do **not** hard-code the system prompt only in Python).

---

### 4️⃣ Implement the Supervisor Node + Routing

Implement a **supervisor node** that:

- Reads `user_request` from `MultiAgentState`
- Calls an LLM with the YAML `system` prompt
- Writes a normalized `route` into state

Example shape:

```python
from langchain_core.messages import SystemMessage, HumanMessage

VALID_ROUTES = {"orders", "billing", "technical", "subscription", "general"}

def supervisor_node(state: MultiAgentState) -> dict:
    messages = [
        SystemMessage(content=supervisor_system_prompt_from_yaml),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    route = response.content.strip().lower()
    if route not in VALID_ROUTES:
        route = "general"
    return {"route": route}
```

Create a routing function that uses `route` to choose the next node:

```python
from typing import Literal

def route_to_specialist(state: MultiAgentState) -> str:
    route_map: dict[str, str] = {
        "orders": "orders_agent_node",
        "billing": "billing_agent_node",
        "technical": "technical_agent_node",
        "subscription": "subscription_agent_node",
        "general": "general_agent_node",
    }
    return route_map.get(state["route"], "general_agent_node")
```

Then wire up a **LangGraph**:

```python
from langgraph.graph import StateGraph, END

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
```

You can adjust node names, but:

- There must be a supervisor entry node.
- There must be a routing function that includes the 4 specialists **plus** a general path.
- Graph must use `StateGraph` + `add_conditional_edges`.

---

### 5️⃣ Create 4 Specialist Agents

Implement **four specialist nodes**:

- `orders_agent_node`
- `billing_agent_node`
- `technical_agent_node`
- `subscription_agent_node`

Each node must:

- Read `user_request` from state
- Produce some answer (you can keep tools simple / mocked)
- Write:
    - `agent_used` (e.g. `"orders_agent"`)
    - `specialist_result` (raw content string)

Example (simplified, no tools required for grading):

```python
def orders_agent_node(state: MultiAgentState) -> dict:
    # In your own solution, you can call tools or another LLM here.
    text = f"[orders_agent] Handling request: {state['user_request']}"
    return {
        "agent_used": "orders_agent",
        "specialist_result": text,
    }
```

You can keep these implementations minimal; the grader checks **structure and routing**, not tool sophistication.

---

### 6️⃣ Structured Handoffs (AgentHandoff)

Implement an `AgentHandoff` **dataclass** to represent handoffs between supervisor and specialists:

```python
from dataclasses import dataclass
from datetime import datetime

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
```

Use this pattern at least once, for example when the supervisor calls a specialist:

```python
handoff = AgentHandoff(
    from_agent="supervisor",
    to_agent="billing",
    task=state["user_request"],
    context={"route": state["route"]},
    priority="normal",
    timestamp=datetime.utcnow().isoformat(),
)
```

You do **not** need a full message-passing system; the goal is to show **typed, auditable handoff data**.

---

### 7️⃣ Injection Detection at Graph Entry

Add **prompt injection detection** at the graph entry point, before the supervisor runs:

```python
import re
from typing import Final

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
```

You must then use this before building state / invoking the graph:

```python
def guard_request(user_input: str) -> str:
    if detect_injection(user_input):
        return "I can only assist with account and order support. (Request blocked.)"
    return user_input
```

In `main()`, call `guard_request` on the input before invoking the graph.

---

### 8️⃣ Session Audit Log with Cost Tracking

Implement a simple **session audit log** that tracks events and approximate cost:

```python
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

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
```

After each `main()` run, **append** the session log to a JSON file at the repo root:

```python
def persist_audit_log(audit: SessionAuditLog) -> None:
    path = Path("audit_log.jsonl")
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(audit.to_dict()) + "\n")
```

You may mock token counts instead of reading real usage from the SDK. The important parts:

- `SessionAuditLog` exists with `session_id`, `events`, `total_cost_usd`, and a `log` method.
- Some function writes the log to disk (e.g. `audit_log.jsonl`) at the end of a session.

---

### 9️⃣ main() – Demonstrate the Multi-Agent System

In `app.py`, include a `main()` that:

1. Loads the supervisor YAML prompt.
2. Builds the multi-agent graph.
3. Creates a `SessionAuditLog`.
4. Runs at least **two example requests**:
    - One clearly **orders**-type request.
    - One clearly **subscription**-type request.
5. Prints to stdout:
    - The `route` and `agent_used` for each run.
    - The `final_response` for each run.
    - The total cost in USD from the audit log.

Example shape (you can adapt this):

```python
def main() -> None:
    audit = SessionAuditLog(session_id="demo-session")
    graph = build_graph()

    for request in [
        "My order ORD-123 is late, can I return it?",
        "I want to upgrade from Basic to Pro. What will it cost?",
    ]:
        safe_text = guard_request(request)
        state: MultiAgentState = {
            "user_request": safe_text,
            "route": "general",
            "agent_used": "",
            "specialist_result": "",
            "final_response": "",
        }
        result = graph.invoke(state)
        print("Request:", request)
        print("Route:", result.get("route"), "Agent used:", result.get("agent_used"))
        print("Final:", result.get("final_response"))
        print("---")

    print("Total cost (USD):", round(audit.total_cost_usd, 6))
    persist_audit_log(audit)

if __name__ == "__main__":
    main()
```

---

## 📊 Evaluation Criteria

Your submission will be automatically graded on:

- **Project contract**:
    - `.env` ignored in `.gitignore`
    - `requirements.txt` includes `python-dotenv`, `langgraph`, and at least one LangChain package
    - `README.md` documents `python app.py` and warns not to commit `.env`
- **Multi-agent state**:
    - `MultiAgentState` TypedDict exists with required keys
- **Supervisor + routing**:
    - YAML supervisor prompt exists and is loaded
    - `supervisor_node`-style logic sets `route`
    - `route_to_specialist` maps 5 categories to 5 nodes
    - Graph uses `StateGraph` + `add_conditional_edges`
- **Specialist agents**:
    - 4 specialist nodes exist (orders, billing, technical, subscription) plus a general path
    - They write `agent_used` and `specialist_result`
- **Structured handoffs**:
    - `AgentHandoff` dataclass exists
- **Injection defense**:
    - `detect_injection(user_input: str) -> bool` exists and is used prior to graph invocation
- **Audit log & cost tracking**:
    - `SessionAuditLog` exists with `log` method and `total_cost_usd`
    - A function writes an audit log JSON/JSONL file at the repo root

Exact tests are implemented in the **bootcamp grader** but follow these expectations.

### ✅ Pass threshold: 80%

If below 80%, you must fix issues and resubmit.

---

## 📌 Submission Rules

- Repository must be **public**
- Do **NOT** include API keys
- `.env` must **NOT** be committed
- Default branch should be `main`
- Code must run using:

```bash
python app.py
```

---
