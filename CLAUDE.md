# Multi-Agent Customer Support System

Day 4 assignment: LangGraph supervisor + specialist agents.

## Stack
- LangGraph (`StateGraph`, `add_conditional_edges`)
- LangChain (`ChatOpenAI`, `SystemMessage`, `HumanMessage`)
- OpenAI API key via `.env`

## Routing Model

```
START
  └─► supervisor_node          (classifies user_request → route)
        │
        ├─► orders_agent_node
        ├─► billing_agent_node
        ├─► technical_agent_node
        ├─► subscription_agent_node
        └─► general_agent_node
              │
              └─► synthesize_response ──► END
```

Routes: `orders` | `billing` | `technical` | `subscription` | `general`

Supervisor prompt loaded from `prompts/supervisor_v1.yaml` (not hardcoded).

## Key Components
- `MultiAgentState` — shared TypedDict across all nodes
- `AgentHandoff` — dataclass for typed supervisor→specialist handoffs
- `detect_injection()` — called before graph invocation
- `SessionAuditLog` — tracks events + cost, persisted to `audit_log.jsonl`


Remember 
Do not implement what is not asked for. Do not re-implement what is already implemented. Focus on the missing pieces.
You write simple and clean code with comments.