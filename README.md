# agentic-day4-multi-agent

Multi-agent customer support system using LangGraph — supervisor + specialist agents.

## Setup

**1. Create and activate the virtual environment**

```bash
uv venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

**2. Install dependencies**

```bash
uv pip install -r requirements.txt
```

**3. Configure environment variables**

Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
```

> **Never commit `.env` to version control.** It is listed in `.gitignore`.

**4. Run**

```bash
python app.py
```
