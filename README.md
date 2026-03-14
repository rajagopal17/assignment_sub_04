# Multi-Agent Customer Support System

A multi-agent customer support system built with **LangGraph** and **LangChain**, featuring a supervisor agent that routes requests to specialist agents.

## Architecture

- **Supervisor agent** — classifies requests and routes to the correct specialist
- **Specialist agents** — Orders, Billing, Technical, Subscription, General
- **Structured handoffs** — typed `AgentHandoff` dataclass between agents
- **Injection detection** — guards against prompt injection at entry
- **Session audit log** — tracks events and approximate cost per session

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
# or
source venv/bin/activate       # macOS/Linux
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_key_here
```

> **Important:** Never commit your `.env` file. It is excluded via `.gitignore`.

## Run

```bash
python app.py
```

This will run two example requests through the multi-agent graph, print routes and responses, and save an audit log to `audit_log.jsonl`.

## Project Structure

```
├── app.py                    # Main application
├── prompts/
│   └── supervisor_v1.yaml    # Supervisor classification prompt
├── requirements.txt
├── README.md
├── .gitignore
└── audit_log.jsonl           # Generated at runtime (not committed)
```
