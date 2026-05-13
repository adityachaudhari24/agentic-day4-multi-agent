# Day 4 Learnings: Multi-Agent Customer Support System

## 1. LangGraph StateGraph + Conditional Routing

- `StateGraph(MultiAgentState)` is the graph container — all nodes and edges are registered on it
- `add_conditional_edges(node, fn)` routes dynamically based on values in state at runtime
- `START` and `END` are sentinel nodes; edges to/from them define entry and exit points
- `builder.compile()` freezes the graph into an executable runnable (validates edges, checks for orphan nodes)

```python
builder = StateGraph(MultiAgentState)
builder.add_edge(START, "supervisor")
builder.add_conditional_edges("handoff", route_to_specialist, {...})
graph = builder.compile()
```

## 2. Shared TypedDict State (`MultiAgentState`)

- All nodes read from and write to a single shared `TypedDict` — no function arguments passed between nodes
- Nodes return only the keys they modify (partial dicts); LangGraph merges them into the running state
- State is the communication bus: supervisor writes `route`, specialists read it and write `specialist_result`

```python
class MultiAgentState(TypedDict):
    user_request: str
    route: str
    handoff_context: AgentHandoff
    specialist_result: str
    final_response: str
    audit: SessionAuditLog
```

## 3. Supervisor + Specialist Pattern

- Supervisor classifies intent → writes `route` to state
- `handoff_node` creates an `AgentHandoff` record → writes `handoff_context` to state
- `route_to_specialist` (conditional edge function) reads `route` → returns next node name as a string
- Each specialist handles one domain, writes `agent_used` + `specialist_result`
- `synthesize_response` collapses specialist output into `final_response`

The supervisor never calls specialists directly — it only sets state. The graph engine handles dispatch.

## 4. Externalizing Prompts as Versioned YAML

- System prompts live in `prompts/supervisor_v1.yaml`, not hardcoded in Python
- `yaml.safe_load` (not `yaml.load`) prevents arbitrary code execution via malicious YAML
- YAML schema includes `version`, `name`, `description`, `changelog`, `system` fields
- `supervisor_node_factory(prompt_path)` loads YAML once at build time → returns a node function

```yaml
version: "1.0"
name: supervisor
system: |
  You are a customer support supervisor...
```

Versioning prompts as files enables iteration without touching Python code, and `git diff` shows prompt changes clearly.

## 5. `AgentHandoff` Dataclass for Typed Handoffs

- Captures `from_agent`, `to_agent`, `task`, `context`, `priority`, `timestamp`
- `to_prompt_context()` serialises the handoff into a formatted string injected into the specialist's system prompt
- Gives each specialist awareness of *why* it was called and what the supervisor determined

```python
@dataclass
class AgentHandoff:
    from_agent: str
    to_agent: str
    task: str
    context: str
    priority: str
    timestamp: str

    def to_prompt_context(self) -> str:
        return f"Task: {self.task}\nContext: {self.context}\nPriority: {self.priority}"
```

Using a dataclass (over a plain dict) provides IDE autocomplete and makes the contract between supervisor and specialists explicit.

## 6. Prompt Injection Defense (Defense-in-Depth)

Think of it like building security: Layer 1 is the **bouncer at the door** (never lets you in), Layer 2 is the **security camera inside** (catches anything that slipped past).

**Layer 1 — Pre-graph guard** (`guard_request` / `detect_injection` in `main()`)
- Runs *before* `graph.invoke()` — no LLM call happens at all
- Checks the raw user string via regex patterns
- Production case: a public API endpoint where you want to stop the request before it touches your LLM budget
- Limitation: only works when the entry point is `main()`. A webhook, REST endpoint, or batch job that calls `graph.invoke()` directly bypasses this entirely.

**Layer 2 — In-graph node** (`injection_check_node` + `_injection_route`)
- Runs as the first node *inside* the graph, after invocation has started
- Checks `state["user_request"]` — same regex, but inside graph execution
- Production case: multiple entry points (CLI, API, Slack bot, scheduled job) all funnel into the same graph. Layer 2 is the catch-all that works regardless of how the graph was called.
- Limitation: one LLM API call may already be charged before this node fires (depending on graph topology); but it still short-circuits before the expensive specialist nodes.

**Key difference:** Layer 1 protects the entry point; Layer 2 protects the graph itself.

```python
# Layer 1 — guards main() entry point only; bypassed if graph.invoke() is called directly
if guard_request(user_input):
    print("Blocked: potential injection detected")
    return

# Layer 2 — first node inside the graph; fires regardless of which entry point was used
def injection_check_node(state):
    if detect_injection(state["user_request"]):
        return {"final_response": "Request blocked."}
```

## 7. `SessionAuditLog` for Cost + Event Tracking

- Tracks every LLM call: agent name, action, token counts, cost estimate
- `log()` appends an event dict and accumulates `total_cost_usd`
- `to_dict()` / `persist_audit_log()` serialises to JSONL (append mode, one JSON line per session)
- Stored in `MultiAgentState["audit"]` so every node can log without importing a global

```python
audit.log(agent="supervisor", action="classify", input_tokens=120, output_tokens=15)
# Accumulates cost, stores event list
persist_audit_log(audit, path="audit_log.jsonl")
```

## 8. Node Factory Pattern

- `supervisor_node_factory(prompt_path)` is a closure: loads YAML once at build time, returns a node function
- The prompt string is captured in the closure scope — no global variables needed
- Pattern is reusable for any node that needs parameterized behavior

```python
def supervisor_node_factory(prompt_path: str):
    prompt = load_prompt(prompt_path)          # runs once at build time

    def supervisor_node(state: MultiAgentState):  # runs on every invocation
        ...use prompt...

    return supervisor_node

# At graph build time:
builder.add_node("supervisor", supervisor_node_factory("prompts/supervisor_v1.yaml"))
```

## 9. Project Structure Conventions

- `paths.py` — single source of truth for `project_root()`; avoids `os.getcwd()` fragility when running from different directories
- `.env` / `.env.example` — secrets never committed; example file shows required keys for onboarding
- `prompts/` directory — versioned YAML prompt files (`_v1`, `_v2` suffix convention)
- `audit_log.jsonl` at repo root — grader/CI-friendly; copy also written to `logs/` for organised storage

```
project/
├── paths.py               # project_root() helper
├── prompts/
│   └── supervisor_v1.yaml # versioned prompts
├── .env.example           # documents required secrets
└── audit_log.jsonl        # session audit trail
```

## 10. JSONL as Audit Format

- One JSON object per line → append-safe, streaming-friendly, easy to `grep` / `jq`
- `json.dumps(obj) + "\n"` written in append mode (`"a"`) — safe for concurrent writes from multiple sessions
- Unlike JSON arrays, JSONL doesn't require reading the whole file to append a new record

```python
with open(path, "a") as f:
    f.write(json.dumps(session_dict) + "\n")
```

JSONL is the right format when records accumulate over time and you want to inspect or stream them without loading everything into memory.
