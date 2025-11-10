# AI Agent System For Data Analytics

An AI-powered data analysis system based on multiple agent architectures, capable of automatically solving Kaggle competition problems.

app_cag_multi_aux_en.py 与 cag_runner.py 目前并非CAG内容，请忽略

## 🎯 Overview

This project implements two AI agent architectures — **ReAct** and **RAG** — for end-to-end automated data analysis and Kaggle submission generation. By simply providing a Kaggle competition link and selecting an AI architecture, the system will automatically:

* Retrieve and analyze the dataset
* Generate and execute analytical/modeling code
* Export the `submission.csv`
* Record performance metrics and runtime logs

## 🏗️ System Architecture

### Architecture 1: ReAct Agent (Reasoning + Acting Loop)

* Iterative **think → act → observe** process
* Suitable for multi-step reasoning and exploratory data analysis tasks

### Architecture 2: RAG Agent (Retrieval-Augmented Generation)

* Retrieves similar competition data and code snippets from the **knowledge_base/**
* Combines retrieved templates to generate more robust and feature-rich analytical code

> Users can select the preferred agent type in the frontend; the system will automatically execute the corresponding workflow.

## 📁 Project Structure

```
ai-agent-analytics/
├── backend/
│   ├── agents/              # AI agent implementations (ReAct, RAG)
│   ├── kaggle/              # Kaggle API integration for data download and submission
│   ├── executor/            # Code execution engine (sandboxed with logging)
│   ├── evaluation/          # Evaluation metrics and performance tracking
|   ├── RAG_tool/            # Knowledge base for RAG
│   └── utils/               # Utility functions
├── frontend/
│   └── app.py               # Streamlit frontend entry point
├── .env.example             # Environment variable template
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1) Environment Setup

* **Python**: Recommended version 3.11 (newer versions may cause dependency conflicts)
* **OS**: macOS / Linux / Windows (PowerShell or CMD)

```bash
# Clone repository
git clone https://github.com/unsw-cse-comp99-3900/capstone-project-25t3-9900-w16a-bread.git
cd capstone-project-25t3-9900-w16a-bread

# Create virtual environment (recommended name: .venv)
python -m venv .venv

# Activate virtual environment
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (CMD)
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

> When using VS Code, `.venv` is automatically detected. Ensure `.venv/` is listed in `.gitignore`.

### 2) Configure Kaggle API

```bash
# Copy environment variable template
cp .env.example .env
```

Fill in your `.env` file:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
```

Get your API token from: [https://www.kaggle.com/settings/account](https://www.kaggle.com/settings/account)
You can also place your `kaggle.json` in:

* macOS/Linux: `~/.kaggle/kaggle.json`
* Windows: `%USERPROFILE%\.kaggle\kaggle.json`

### 3) Start LLM (Ollama)

```bash
# Install from https://ollama.ai/
ollama pull llama3
ollama serve
```

> To switch models, modify `OLLAMA_MODEL` in `.env`. Remote or cloud-based LLMs can also be integrated via compatible API endpoints.

### 4) Run Backend & Frontend

**You may need to restart your terminal before starting the app to ensure all dependencies are properly applied.**

**Frontend (Streamlit)**

```bash
# Run the Streamlit app
streamlit run frontend/app.py
```

Access the app at: `http://localhost:8501`

## 🧭 Usage

1. Open `http://localhost:8501`
2. Enter a Kaggle competition link (e.g. `https://www.kaggle.com/competitions/store-sales-time-series-forecasting`)
3. Select the AI architecture (**ReAct** or **RAG**)
4. Click **“Generate & Run”**
5. View real-time logs, generated code, and outputs
6. Download the resulting `submission.csv`

## 📊 Evaluation Metrics

* Execution time (overall and by stage)
* Prediction accuracy (Kaggle leaderboard / CV metrics)
* Explainability (feature importance, visualizations, text reports)
* Autonomy (human interaction count, tool usage, retry rate)
* Code complexity (lines, dependencies, cyclomatic complexity)
* Resource usage (memory, CPU/GPU time, LLM tokens)

## ❓ FAQ

**Q1: Dependency installation fails or version conflict?**
A: Use Python 3.11. If pip fails, try `pip install -r requirements.txt --use-pep517` or fix versions manually.

**Q2: Streamlit runs but backend returns 404?**
A: Ensure Uvicorn backend is running and the frontend `.env` or config points to `http://localhost:8000`.

**Q3: Kaggle API download fails?**
A: Verify `.env` or `~/.kaggle/kaggle.json` credentials and confirm you’ve accepted the competition rules on Kaggle.

**Q4: Ollama model too large or slow?**
A: Try lighter models like `llama3:instruct` or `mistral`, or simplify prompts to reduce context length.
