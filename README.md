# NamInsight: Autonomous Data Analytics Agent System (this project is not finished yet)

<div align="center">
  <img src="https://img.shields.io/badge/LangGraph-Agentic_AI-informational" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Docker-Sandboxed-blue" alt="Docker" />
  <img src="https://img.shields.io/badge/Qdrant-Vector_LTM-red" alt="Qdrant" />
  <img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/Security-Hardened-success" alt="Security" />
</div>

## Overview
NamInsight is a compound multi-agent orchestration system designed for secure, autonomous data analytics. Built on **LangGraph** and deployed via **Chainlit**, it delegates complex data analysis tasks across specialized autonomous agents. The system utilizes deterministic safeguards, ephemeral execution sandboxes, and long-term vector memory caching to reliably answer user queries, generate insights from a local PostgreSQL database, and render interactive Python visualizations.

## System Architecture & Multi-Agent Flow
The application's core logic operates synchronously across 5 bounded cognitive nodes:

1. **Architect (Planner):** Analyzes the user intent, searches long-term memory for past success patterns, and outputs a strict JSON tool-execution plan. Hardened strictly against jailbreak or system evasion prompt injections.
2. **Dispatcher (Executor):** Securely evaluates the plan and forwards execution to the requested programmatic tool (SQL, Python, or RAG), explicitly forbidden from unpredictably modifying the Architect's source payload.
3. **Critic (Evaluator):** Processes tool output to assess success or failure. Reports semantic validation back to the event-loop.
4. **Retry Manager (Self-Correction):** Triages failed code or SQL queries, initiating an autonomous self-debugging loop up to 3 times before abandoning the execution path.
5. **Governor (Circuit Breaker):** Monitors deterministic graph iteration limits to definitively halt infinite execution structures and safely exit to the client UI.

## Engine & Tooling Capabilities

### Secure SQL Engine
- Parses all incoming queries through `sqlglot` Abstract Syntax Trees (AST) to enforce strict Read-Only (`SELECT`) access control mechanisms.
- PostgreSQL connections are bound to a least-privilege `naminsight_reader` role, actively stripped of DML/DDL or system catalog (`pg_schema`) exposure capabilities.

### Docker-in-Docker Python Sandbox
- Provisions completely isolated, ephemeral containers dynamically during graph execution to generate data structures or visualization graphics (`matplotlib`, `seaborn`).
- **Zero-Trust Policy:** Drops all Linux system capabilities (`cap_drop=['ALL']`), severs internal container networking, forces non-root user execution, caps CPU cycles (0.5 cores max), and mounts read-only root filesystems to eradicate arbitrary code execution (ACE).

### Semantic Memory & Qdrant RAG
- **Long-Term Memory (LTM):** Indexes successful Architect planning pathways directly into a Qdrant cluster using embedded `BAAI/bge-small-en-v1.5`. Strong query matches bypass expansive LLM computation entirely for repetitive tasks, drastically dropping latency.
- **Decay Pruning:** Periodically cycles LTM entries using mathematical decay weights, proactively preventing cognitive drift on outdated analytical patterns.

## Security & Human-in-the-Loop (HITL)
NamInsight embraces transparent analytics generation that keeps end-users in strict control of operations.
- **HITL Verification:** The runtime automatically halts (`interrupt_before`) prior to the Dispatcher node. Users must explicitly authorize any data extraction or programmatic execution directly in the Chainlit UI.
- **Automated Self-Correction Approval:** While initial tool execution logic requires manual approval, secondary auto-correction retries organically bypass friction layers to ensure background remediation completes correctly and swiftly.
- **Rate-Limiting:** Front-end Chainlit environments strictly enforce anti-spam connection limits per interactive thread session constraint.

## Local Setup & Deployment

### Prerequisites
- Docker & Docker Compose
- [uv](https://docs.astral.sh/uv/) (Extremely fast Python package installer)

### Installation
1. Clone the repository to your host environment.
2. Configure active environment credentials inside a root `.env` file (OpenAI/vLLM endpoints).
3. Execute the standard Docker orchestration manifest:

```bash
docker-compose up -d --build
```
*(The workflow implicitly leverages `uv sync --frozen` natively during the target image build sequence, resulting in highly reproducible and lightning-fast package provisioning).*

### Usage
Instantiate the web application gateway via your browser:
```text
http://localhost:8000
```
Submit analytical questions, ask for cross-table correlations, or prompt the system to "Visualize the customer tier distribution," and interact with the NamInsight execution trace to monitor real-time logic.
