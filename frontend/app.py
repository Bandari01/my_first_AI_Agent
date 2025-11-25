"""
Kaggle AI Agent System - Streamlit Frontend (Rewritten)

Features:
- Choose Agent Type: ReAct, RAG, Multi-Agent
- Adjust Max Fix Attempts for code auto-fix
- Auto-fetch Kaggle competition data and analyze
- Real-time backend logs during execution
- Show generated code, total time, and download submission if available
"""

import sys
import os
import time
from dotenv import load_dotenv

load_dotenv()

import threading
import logging
from queue import Queue, Empty
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
import json
import subprocess
import tempfile
import shutil
import pandas as pd

import streamlit as st
import html

# Ensure repo root in sys.path for `backend.*` imports
REPO_ROOT = Path(__file__).resolve().parent.parent
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
	sys.path.insert(0, repo_root_str)

from backend.utils.logger import logger as loguru_logger  # loguru logger instance
from backend.agents.react_agent import ReactAgent
from backend.kaggle.data_fetcher import KaggleDataFetcher, CompetitionInfo

# Optional RAG imports
try:
	from backend.agents.rag_agent import RAGAgent
	from backend.RAG_tool.config import RAGConfig
	RAG_AVAILABLE = True
except Exception:
	RAG_AVAILABLE = False
	RAGAgent = None  # type: ignore
	RAGConfig = None  # type: ignore

# Optional Multi-Agent orchestrator import
try:
	from backend.agents.orchestrator import MultiAgentOrchestrator  # type: ignore
	MA_AVAILABLE = True
except Exception:
	MA_AVAILABLE = False
	MultiAgentOrchestrator = None  # type: ignore

# Optional DLP Agent import
try:
    from backend.agents.dlp_agent import run_cag
    DLP_AVAILABLE = True
except Exception:
    DLP_AVAILABLE = False
    run_cag = None


# ------------------------- Streamlit Page Config & Styles -------------------------
st.set_page_config(
	page_title="Kaggle AI Agent System",
	page_icon="🤖",
	layout="wide",
	initial_sidebar_state="expanded",
)

st.markdown(
	"""
<style>
.main-header { font-size: 2.2rem; font-weight: 700; text-align: center; color: #1f77b4; margin: 0.2rem 0 0.6rem 0; }
.sub-header { font-size: 1.1rem; text-align: center; color: #666; margin-bottom: 1.2rem; }
.status-box { padding: 0.6rem 0.8rem; border-radius: 0.5rem; margin: 0.4rem 0; }
.status-success { background: #e6ffed; border: 1px solid #b7eb8f; color: #135200; }
.status-error { background: #fff1f0; border: 1px solid #ffa39e; color: #a8071a; }
.status-info { background: #e6f7ff; border: 1px solid #91d5ff; color: #0050b3; }
.log-box { background: #0b1021; color: #e8f0ff; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; padding: 0.75rem; border-radius: 8px; height: 560px; overflow-y: auto; border: 1px solid #1f2a48; }
.code-box { background: #f8f9fc; color: #111; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; padding: 0.9rem; border-radius: 8px; height: 560px; overflow-y: auto; border: 1px solid #e3e6ef; white-space: pre; line-height: 1.35; }
.code-wrap pre { white-space: pre-wrap; word-wrap: break-word; }
</style>
""",
	unsafe_allow_html=True,
)


# ------------------------- Session State -------------------------
def _init_state():
	if "competition_info" not in st.session_state:
		st.session_state.competition_info = None
	if "agent_thread" not in st.session_state:
		st.session_state.agent_thread = None
	if "agent_running" not in st.session_state:
		st.session_state.agent_running = False
	if "agent_result" not in st.session_state:
		st.session_state.agent_result = None
	if "log_queue" not in st.session_state:
		st.session_state.log_queue = Queue()
	if "log_handler" not in st.session_state:
		st.session_state.log_handler = None
	if "loguru_sink_id" not in st.session_state:
		st.session_state.loguru_sink_id = None


_init_state()

# Global queues to avoid using Streamlit APIs inside background threads
LOG_QUEUE: "Queue[str]" = Queue()
RESULT_QUEUE: "Queue[Any]" = Queue()


# ------------------------- Logging Hookup -------------------------
class StreamToQueueHandler(logging.Handler):
	def emit(self, record: logging.LogRecord) -> None:
		try:
			msg = self.format(record)
		except Exception:
			msg = str(record.getMessage())
		try:
			LOG_QUEUE.put_nowait(msg)
		except Exception:
			pass


def _attach_log_handlers(level=logging.INFO):
	if st.session_state.log_handler is not None:
		return
	handler = StreamToQueueHandler()
	handler.setLevel(level)
	handler.setFormatter(logging.Formatter("%(message)s"))

	logger_names = [
		"backend",
		"backend.executor.code_executor",
		"backend.agents.react_agent",
		"backend.agents",
	]
	for name in logger_names:
		lg = logging.getLogger(name)
		lg.setLevel(level)
		lg.addHandler(handler)

	# loguru sink to capture loguru logs
	def make_sink():
		def _sink(message):
			try:
				LOG_QUEUE.put_nowait(message)
			except Exception:
				pass
		return _sink

	sink_id = loguru_logger.add(make_sink(), level="INFO")

	st.session_state.log_handler = handler
	st.session_state.loguru_sink_id = sink_id


def _detach_log_handlers():
	handler = st.session_state.log_handler
	if handler is not None:
		# Remove from known loggers
		for name in [
			"backend",
			"backend.executor.code_executor",
			"backend.agents.react_agent",
			"backend.agents",
		]:
			try:
				lg = logging.getLogger(name)
				lg.removeHandler(handler)
			except Exception:
				pass
	st.session_state.log_handler = None

	# Remove loguru sink
	if st.session_state.loguru_sink_id is not None:
		try:
			loguru_logger.remove(st.session_state.loguru_sink_id)
		except Exception:
			pass
	st.session_state.loguru_sink_id = None


# ------------------------- Helpers -------------------------
def parse_kaggle_competition(value: str) -> str:
	value = (value or "").strip()
	if not value:
		return value
	# Accept full URL like https://www.kaggle.com/competitions/titanic or id like "titanic"
	lowered = value.lower()
	if "/competitions/" in lowered:
		try:
			return lowered.split("/competitions/")[-1].split("?")[0].strip("/")
		except Exception:
			return value.strip("/")
	return value.strip("/")


def fetch_competition_info(competition: str) -> CompetitionInfo:
	fetcher = KaggleDataFetcher()
	return fetcher.fetch_complete_info(competition)


def validate_store_sales_data():
    """Verify the integrity of Store Sales data"""
    required_files = ["train.csv", "test.csv"]
    # Check in current directory or REPO_ROOT
    missing_files = []
    for f in required_files:
        if not os.path.exists(f) and not os.path.exists(os.path.join(REPO_ROOT, f)):
            missing_files.append(f)

    if missing_files:
        return False, f"Missing required files: {missing_files}"

    try:
        # Verify whether the key column exists
        train_path = "train.csv" if os.path.exists("train.csv") else os.path.join(REPO_ROOT, "train.csv")
        train_df = pd.read_csv(train_path, nrows=1)
        required_columns = ["date", "store_nbr", "family", "sales"]
        missing_columns = [col for col in required_columns if col not in train_df.columns]

        if missing_columns:
            return False, f"Missing columns in train.csv: {missing_columns}"

        return True, "All required data files and columns are present"

    except Exception as e:
        return False, f"Error validating data: {e}"


def _try_generate_submission(generated_code, competition_name):
    """Try to execute generated code to create submission file"""
    submission_output_path = None
    temp_dir = None
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        # Copy all data files to the temporary directory
        data_files = ["train.csv", "test.csv", "stores.csv", "oil.csv",
                      "holidays_events.csv", "transactions.csv", "sample_submission.csv"]
        
        for file in data_files:
            src_path = file
            if not os.path.exists(src_path):
                src_path = os.path.join(REPO_ROOT, file)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(temp_dir, file))

        # Create a temporary Python file
        temp_file = os.path.join(temp_dir, 'generated_solution.py')

        # Add debugging information to the generated code
        debug_code = """
import traceback
import sys

def debug_hook(type, value, tb):
    print("Exception occurred:", file=sys.stderr)
    traceback.print_exception(type, value, tb, file=sys.stderr)

sys.excepthook = debug_hook

""" + generated_code

        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(debug_code)

        # Execute the code
        result = subprocess.run([
            sys.executable, temp_file
        ], capture_output=True, text=True, timeout=300, cwd=temp_dir)

        # Check whether the submission file has been generated
        submission_path = os.path.join(temp_dir, 'submission.csv')
        if os.path.exists(submission_path):
            # Move to a persistent location
            output_dir = os.path.join(REPO_ROOT, "output")
            os.makedirs(output_dir, exist_ok=True)
            timestamp = int(time.time())
            submission_output_path = os.path.join(output_dir, f"submission_{timestamp}.csv")
            shutil.copy2(submission_path, submission_output_path)
            
    except Exception as e:
        pass
    finally:
        if temp_dir:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            
    return submission_output_path


# ------------------------- Agent Runners (normalized) -------------------------
def run_react_agent(ci: CompetitionInfo, cfg: dict) -> SimpleNamespace:
	agent = ReactAgent(
		max_tokens=cfg.get("max_tokens", 4000),
		timeout=cfg.get("max_execution_time", 600),
		max_fix_attempts=cfg.get("max_fix_attempts", 1),
		competition_name=ci.competition_name,
	)

	# Build minimal problem context
	problem_description = f"Kaggle Competition: {ci.competition_name} ({ci.competition_id})"
	data_info = {
		"train_files": getattr(ci, "train_files", []),
		"test_files": getattr(ci, "test_files", []),
		"columns": getattr(ci, "columns", {}),
		"all_files_info": (getattr(ci, "extra_info", {}).get("all_files", {}) if getattr(ci, "extra_info", None) else {}),
	}

	# Run async method in this thread
	import asyncio
	start = time.time()
	raw = asyncio.run(agent.run(problem_description=problem_description, data_info=data_info))

	ns = SimpleNamespace()
	ns.status = SimpleNamespace(value=("completed" if raw.success else "failed"))
	ns.generated_code = raw.generated_code or ""
	ns.retrieved_knowledge = []
	ns.reasoning_steps = []
	ns.execution_time = raw.execution_time
	ns.retrieval_count = 0
	ns.llm_calls = raw.llm_calls
	ns.error_message = raw.error
	ns.submission_file_path = raw.submission_path
	ns.thoughts = []
	ns.actions = []
	ns.observations = raw.observations or []
	ns.total_time = raw.total_time or (time.time() - start)
	ns.code_generation_time = raw.code_generation_time
	ns.code_lines = len((raw.generated_code or "").splitlines())
	return ns


def run_multi_agent(ci: CompetitionInfo, cfg: dict) -> SimpleNamespace:
	ns = SimpleNamespace()
	if not MA_AVAILABLE or MultiAgentOrchestrator is None:
		ns.status = SimpleNamespace(value="failed")
		ns.generated_code = ""
		ns.retrieved_knowledge = []
		ns.reasoning_steps = ["Multi-Agent orchestrator not available"]
		ns.execution_time = 0.0
		ns.retrieval_count = 0
		ns.llm_calls = 0
		ns.error_message = "Multi-Agent is not available"
		ns.submission_file_path = None
		ns.observations = []
		ns.total_time = 0.0
		ns.code_generation_time = 0.0
		ns.code_lines = 0
		return ns

	try:
		orchestrator = MultiAgentOrchestrator(
			max_fix_attempts=cfg.get("max_fix_attempts", 1),
            max_execution_time=cfg.get("max_execution_time", 600),
		)
		result_obj = orchestrator.run(
			competition_url=ci.competition_url,
			validate=True,
			force_download=False,
		)

		ns.status = SimpleNamespace(value=("completed" if getattr(result_obj, "success", False) else "failed"))
		ns.generated_code = getattr(result_obj, "generated_code", "") or ""
		ns.retrieved_knowledge = []
		ns.reasoning_steps = [f"Step {i+1}: {getattr(step, 'title', '')}" for i, step in enumerate(getattr(result_obj, "plan", SimpleNamespace(steps=[])).steps)]
		ns.execution_time = getattr(result_obj, "total_time_seconds", 0.0)
		ns.retrieval_count = 0
		ns.llm_calls = getattr(result_obj, "llm_calls", 0)
		errors = getattr(result_obj, "errors", None)
		ns.error_message = "; ".join(errors) if errors else None
		sub = getattr(result_obj, "submission_path", None)
		ns.submission_file_path = str(sub) if sub else None
		ns.thoughts = [getattr(getattr(result_obj, "plan", SimpleNamespace(goal="")), "goal", "")]
		ns.actions = [{"step": getattr(step, 'title', ''), "tool": getattr(getattr(step, 'tool_call', None), 'name', 'none')} for step in getattr(getattr(result_obj, "plan", SimpleNamespace(steps=[])), "steps", [])]
		ns.observations = [f"Step {getattr(a, 'step_id', '?')}: {'Success' if getattr(a, 'success', False) else 'Failed'}" for a in getattr(result_obj, "artifacts", [])]
		ns.total_time = getattr(result_obj, "total_time_seconds", 0.0)
		ns.code_generation_time = 0.0
		ns.code_lines = len(ns.generated_code.splitlines()) if ns.generated_code else 0
		return ns
	except Exception as e:
		ns.status = SimpleNamespace(value="failed")
		ns.generated_code = ""
		ns.retrieved_knowledge = []
		ns.reasoning_steps = [f"Error: {e}"]
		ns.execution_time = 0.0
		ns.retrieval_count = 0
		ns.llm_calls = 0
		ns.error_message = str(e)
		ns.submission_file_path = None
		ns.observations = []
		ns.total_time = 0.0
		ns.code_generation_time = 0.0
		ns.code_lines = 0
		return ns


def run_rag_agent(ci: CompetitionInfo, cfg: dict) -> SimpleNamespace:
	ns = SimpleNamespace()
	if not RAG_AVAILABLE or RAGAgent is None or RAGConfig is None:
		ns.status = SimpleNamespace(value="failed")
		ns.generated_code = "# RAG not available"
		ns.retrieved_knowledge = []
		ns.reasoning_steps = ["RAG framework not available"]
		ns.execution_time = 0.0
		ns.retrieval_count = 0
		ns.llm_calls = 0
		ns.error_message = "RAG not available"
		ns.submission_file_path = None
		ns.observations = []
		ns.total_time = 0.0
		ns.code_generation_time = 0.0
		ns.code_lines = 0
		return ns

	# Build minimal context
	problem_description = f"Kaggle Competition: {ci.competition_name} ({ci.competition_id})"
	data_info = {
		"train_files": getattr(ci, "train_files", []),
		"test_files": getattr(ci, "test_files", []),
		"columns": getattr(ci, "columns", {}),
		"all_files_info": (getattr(ci, "extra_info", {}).get("all_files", {}) if getattr(ci, "extra_info", None) else {}),
	}

	# Configure RAG
	rag_cfg = RAGConfig(
		knowledge_base_path=str(REPO_ROOT / "backend" / "RAG_tool" / "knowledge_base"),
		llm_model=cfg.get("rag_llm_model", "gpt-4o-mini"),
		max_tokens=cfg.get("rag_max_tokens", 4000),
		top_k_retrieval=cfg.get("rag_top_k", 5),
		similarity_threshold=cfg.get("rag_sim_threshold", 0.2),
		enable_validation=True,
		openai_api_key=cfg.get("openai_api_key"),
	)

	import asyncio
	agent = RAGAgent(rag_cfg)
	start = time.time()
	res = asyncio.run(agent.run(problem_description, data_info))

	# Try to generate submission if not already present
	submission_path = getattr(res, "submission_file_path", None)
	if not submission_path and getattr(res, "generated_code", "") and "store-sales" in ci.competition_name.lower():
		submission_path = _try_generate_submission(res.generated_code, ci.competition_name)

	ns.status = SimpleNamespace(value=getattr(res, "status", "failed"))
	ns.generated_code = getattr(res, "generated_code", "") or ""
	ns.retrieved_knowledge = getattr(res, "retrieved_knowledge", []) or []
	ns.reasoning_steps = getattr(res, "reasoning_steps", []) or []
	ns.execution_time = getattr(res, "execution_time", 0.0)
	ns.retrieval_count = getattr(res, "retrieval_count", 0)
	ns.llm_calls = getattr(res, "llm_calls", 0)
	ns.error_message = getattr(res, "error_message", None)
	ns.submission_file_path = submission_path
	ns.observations = []
	ns.total_time = getattr(res, "execution_time", None) or (time.time() - start)
	ns.code_generation_time = 0.0
	ns.code_lines = len(ns.generated_code.splitlines()) if ns.generated_code else 0
	return ns


def run_dlp_agent(ci: CompetitionInfo, cfg: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    if not DLP_AVAILABLE or run_cag is None:
        ns.status = SimpleNamespace(value="failed")
        ns.generated_code = "# DLP Agent not available"
        ns.retrieved_knowledge = []
        ns.reasoning_steps = ["DLP Agent not available"]
        ns.execution_time = 0.0
        ns.retrieval_count = 0
        ns.llm_calls = 0
        ns.error_message = "DLP Agent not available"
        ns.submission_file_path = None
        ns.observations = []
        ns.total_time = 0.0
        ns.code_generation_time = 0.0
        ns.code_lines = 0
        return ns

    # Prepare configuration for DLP
    # Assuming data is in REPO_ROOT/data/competitions/{competition_id} or current dir
    # We need to find where the data is.
    # Based on KaggleDataFetcher, data is downloaded to config.competitions_dir / competition_id
    # We can try to locate it.
    
    data_dir = REPO_ROOT / "data" / "competitions" / ci.competition_id
    if not data_dir.exists():
        # Fallback to current directory if data not found in standard location
        data_dir = Path(os.getcwd())

    # Identify files
    train_file = None
    test_file = None
    sample_file = None
    aux_files = []

    # Helper to find file case-insensitively
    def find_file(directory, pattern):
        if not directory.exists(): return None
        for f in os.listdir(directory):
            if pattern.lower() in f.lower() and f.lower().endswith('.csv'):
                return os.path.join(directory, f)
        return None

    train_file = find_file(data_dir, "train")
    test_file = find_file(data_dir, "test")
    sample_file = find_file(data_dir, "sample_submission") or find_file(data_dir, "submission")
    
    # Collect all other csvs as aux
    if data_dir.exists():
        for f in os.listdir(data_dir):
            if f.lower().endswith('.csv'):
                full_path = os.path.join(data_dir, f)
                if full_path not in [train_file, test_file, sample_file]:
                    aux_files.append(full_path)

    dlp_cfg = {
        "base_dir": str(data_dir),
        "train": train_file,
        "test": test_file,
        "sample_submission": sample_file,
        "aux": aux_files,
    }

    # Set environment variables for LLM if provided
    if cfg.get("openai_api_key"):
        os.environ["OPENAI_API_KEY"] = cfg["openai_api_key"]
        os.environ["USE_LLM"] = "1"
        os.environ["OPENAI_MODEL"] = cfg.get("dlp_llm_model", "gpt-4o-mini")
    else:
        os.environ["USE_LLM"] = "0"

    start = time.time()
    try:
        report = run_cag(dlp_cfg)
        
        ns.status = SimpleNamespace(value="completed")
        ns.generated_code = json.dumps(report.get("plan", {}), indent=2) # DLP doesn't generate code, it runs a plan. Show plan as code.
        ns.retrieved_knowledge = []
        ns.reasoning_steps = report.get("logs", [])
        ns.execution_time = time.time() - start
        ns.retrieval_count = 0
        ns.llm_calls = 0 # DLP doesn't report this easily, or we need to parse usage
        if "usage" in report:
             # Sum up tokens or calls if available
             pass
        
        ns.error_message = None
        
        # Handle submission path
        submission_path = report.get("outputs", {}).get("submission_path")
        if submission_path and os.path.exists(submission_path):
            # Move to a persistent location in output folder
            output_dir = REPO_ROOT / "output"
            output_dir.mkdir(exist_ok=True)
            timestamp = int(time.time())
            persistent_path = output_dir / f"submission_dlp_{timestamp}.csv"
            shutil.copy2(submission_path, persistent_path)
            ns.submission_file_path = str(persistent_path)
        else:
            ns.submission_file_path = None

        ns.observations = []
        ns.total_time = time.time() - start
        ns.code_generation_time = 0.0
        ns.code_lines = 0
        
        return ns

    except Exception as e:
        ns.status = SimpleNamespace(value="failed")
        ns.generated_code = ""
        ns.retrieved_knowledge = []
        ns.reasoning_steps = [f"Error: {e}"]
        ns.execution_time = time.time() - start
        ns.retrieval_count = 0
        ns.llm_calls = 0
        ns.error_message = str(e)
        ns.submission_file_path = None
        ns.observations = []
        ns.total_time = time.time() - start
        ns.code_generation_time = 0.0
        ns.code_lines = 0
        return ns


# ------------------------- UI Rendering -------------------------
st.markdown('<div class="main-header">🤖 Kaggle AI Agent System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Auto data fetch · Multi-agent analysis · Live logs · One-click submission</div>', unsafe_allow_html=True)


with st.sidebar:
	st.header("Configuration")
	agent_type = st.selectbox("Agent Type", ["ReAct", "RAG", "Multi-Agent", "DLP"], index=0)
			
	max_fix_attempts = st.slider("Max Fix Attempts (auto-fix)", min_value=0, max_value=5, value=1, step=1)
	max_execution_time = st.number_input("Execution Timeout (s)", min_value=60, max_value=3600, value=600, step=30)
	st.markdown("---")
	kaggle_input = st.text_input("Kaggle URL or ID", placeholder="e.g., https://www.kaggle.com/competitions/titanic or titanic")
	fetch_btn = st.button("Fetch & Analyze Competition")
	run_btn = st.button("Run Agent 🚀", type="primary")


# Data fetch
if fetch_btn:
	comp_id = parse_kaggle_competition(kaggle_input)
	if not comp_id:
		st.error("Please enter a valid Kaggle competition URL or ID")
	else:
		try:
			info = fetch_competition_info(comp_id)
			st.session_state.competition_info = info
		except Exception as e:
			st.error(f"Fetch failed: {e}")


# Competition summary panel
def _render_competition_overview(ci: CompetitionInfo):
	# Try to load persisted competition_info.json for rich details
	json_path = REPO_ROOT / "data" / "competitions" / ci.competition_id / "competition_info.json"
	ci_json = None
	if json_path.exists():
		try:
			with open(json_path, "r", encoding="utf-8") as f:
				ci_json = json.load(f)
		except Exception:
			ci_json = None

	# Header metrics
	c1, c2, c3 = st.columns(3)
	with c1:
		st.metric("Competition", ci.competition_id)
	with c2:
		st.metric("Problem Type", getattr(ci, "problem_type", "unknown"))
	with c3:
		st.metric("Metric", getattr(ci, "evaluation_metric", "unknown"))

	# Basic files and shapes
	c4, c5, c6 = st.columns(3)
	with c4:
		st.write("Train files")
		st.code("\n".join(getattr(ci, "train_files", []) or []), language="text")
	with c5:
		st.write("Test files")
		st.code("\n".join(getattr(ci, "test_files", []) or []), language="text")
	with c6:
		if getattr(ci, "sample_submission_file", None):
			st.write("Sample submission")
			st.code(str(ci.sample_submission_file), language="text")

	# Shapes and path
	with st.container():
		cc1, cc2, cc3 = st.columns(3)
		with cc1:
			train_shape = None
			if ci_json and isinstance(ci_json.get("train_shape"), list):
				train_shape = tuple(ci_json.get("train_shape"))
			st.metric("Train shape", value=str(train_shape) if train_shape else "-")
		with cc2:
			test_shape = None
			if ci_json and isinstance(ci_json.get("test_shape"), list):
				test_shape = tuple(ci_json.get("test_shape"))
			st.metric("Test shape", value=str(test_shape) if test_shape else "-")
		with cc3:
			st.metric("Data path", value=os.path.relpath(ci_json.get("data_path"), REPO_ROOT) if (ci_json and ci_json.get("data_path")) else "-")

	# Columns and types
	columns = None
	col_types = None
	if ci_json:
		columns = ci_json.get("columns")
		col_types = ci_json.get("column_types")
	else:
		columns = getattr(ci, "columns", None)
		col_types = getattr(ci, "column_types", None)

	if columns or col_types:
		st.markdown("**Columns & Types**")
		rows = []
		if isinstance(columns, list):
			for c in columns:
				dtype = col_types.get(c) if isinstance(col_types, dict) else None
				rows.append({"column": c, "dtype": dtype or "-"})
		elif isinstance(col_types, dict):
			for c, t in col_types.items():
				rows.append({"column": c, "dtype": t})
		if rows:
			try:
				import pandas as pd
				st.dataframe(pd.DataFrame(rows), width='stretch')
			except Exception:
				st.write(rows)

	# All files overview
	all_files = None
	if ci_json:
		all_files = (ci_json.get("extra_info") or {}).get("all_files")
	else:
		all_files = (getattr(ci, "extra_info", {}) or {}).get("all_files")

	if isinstance(all_files, dict) and all_files:
		st.markdown("**Files Summary**")
		table = []
		for fname, info in all_files.items():
			cols = info.get("columns") or []
			table.append({
				"file": fname,
				"n_cols": len(cols),
				"sample_rows": info.get("rows_sample", "-"),
				"columns": ", ".join(cols[:6]) + (" …" if len(cols) > 6 else ""),
			})
		try:
			import pandas as pd
			st.dataframe(pd.DataFrame(table), width='stretch')
		except Exception:
			st.write(table)

		# Per-file detail expanders
		for fname, info in all_files.items():
			with st.expander(f"Details: {fname}"):
				cols = info.get("columns") or []
				dtypes = info.get("dtypes") or {}
				sample_data = info.get("sample_data") or []
				st.write("Columns:", cols)
				if isinstance(dtypes, dict) and dtypes:
					try:
						import pandas as pd
						st.dataframe(pd.DataFrame([dtypes]).T.rename(columns={0: "dtype"}), width='stretch')
					except Exception:
						st.write(dtypes)
				if sample_data:
					st.write("Sample:")
					try:
						import pandas as pd
						st.dataframe(pd.DataFrame(sample_data), width='stretch')
					except Exception:
						st.write(sample_data)


if st.session_state.competition_info:
	ci = st.session_state.competition_info
	with st.expander("Competition & Data Overview", expanded=True):
		_render_competition_overview(ci)
		
		if "store-sales" in ci.competition_id.lower():
			st.markdown("---")
			if st.button("Validate Store Sales Data"):
				is_valid, message = validate_store_sales_data()
				if is_valid:
					st.success(f"✅ {message}")
				else:
					st.error(f"❌ {message}")
# Execution area (single column)
st.subheader("Live Logs")
log_placeholder = st.empty()

st.subheader("Generated Code & Results")
code_placeholder = st.empty()
result_placeholder = st.empty()


def _run_agent_thread(agent_type: str, cfg: dict, ci: CompetitionInfo):
	try:
		if agent_type == "ReAct":
			res = run_react_agent(ci, cfg)
		elif agent_type == "RAG":
			res = run_rag_agent(ci, cfg)
		elif agent_type == "DLP":
			res = run_dlp_agent(ci, cfg)
		else:
			res = run_multi_agent(ci, cfg)
		RESULT_QUEUE.put_nowait(res)
	except Exception as e:
		RESULT_QUEUE.put_nowait(SimpleNamespace(status=SimpleNamespace(value="failed"), error_message=str(e), generated_code=""))
	finally:
		pass


# Start run
if run_btn:
	if not st.session_state.competition_info:
		st.error("Please fetch competition data first")
	else:
		_attach_log_handlers()
		st.session_state.agent_running = True
		st.session_state.agent_result = None
		cfg = {
			"max_fix_attempts": max_fix_attempts,
			"max_execution_time": max_execution_time,
			"openai_api_key": os.environ.get("OPENAI_API_KEY"),
		}
		ci = st.session_state.competition_info
		t = threading.Thread(target=_run_agent_thread, args=(agent_type, cfg, ci), daemon=True)
		st.session_state.agent_thread = t
		t.start()


# While running: stream logs
if st.session_state.agent_running and st.session_state.agent_thread is not None:
	log_lines = []
	start_t = time.time()
	while st.session_state.agent_thread.is_alive():
		try:
			while True:
				msg = LOG_QUEUE.get_nowait()
				log_lines.append(msg)
		except Empty:
			pass
		log_placeholder.markdown(f"<div class='log-box'>{'<br/>'.join([html.escape(m) for m in log_lines[-1000:]])}</div>", unsafe_allow_html=True)
		time.sleep(0.25)

	# Drain remaining logs after thread exits
	try:
		while True:
			msg = LOG_QUEUE.get_nowait()
			log_lines.append(msg)
	except Empty:
		pass
	log_placeholder.markdown(f"<div class='log-box'>{'<br/>'.join([html.escape(m) for m in log_lines[-2000:]])}</div>", unsafe_allow_html=True)

	# Retrieve result from result queue
	try:
		st.session_state.agent_result = RESULT_QUEUE.get_nowait()
	except Empty:
		st.session_state.agent_result = SimpleNamespace(status=SimpleNamespace(value="failed"), error_message="No result returned", generated_code="")

	st.session_state.agent_running = False
	_detach_log_handlers()


# Render result if available
if st.session_state.agent_result is not None:
	res = st.session_state.agent_result
	status_val = getattr(getattr(res, "status", SimpleNamespace(value="unknown")), "value", "unknown")

	if status_val == "completed":
		st.success("Agent completed ✅")
	elif status_val == "failed":
		st.error(f"Agent failed ❌: {getattr(res, 'error_message', '')}")
	else:
		st.info("Agent status: " + status_val)

	# Generated code (extract from markdown fences if present, show in scrollable pre)
	def _extract_code_text(s: str) -> str:
		if not s:
			return ""
		try:
			if "```" in s:
				# Prefer ```python fenced content
				if "```python" in s.lower():
					start = s.lower().index("```python") + len("```python")
					end = s.find("```", start)
					if end != -1:
						return s[start:end].strip()
				# Generic first fenced block
				start = s.index("```") + 3
				end = s.find("```", start)
				if end != -1:
					return s[start:end].strip()
		except Exception:
			pass
		return s

	code_raw = getattr(res, "generated_code", "") or "# No generated code"
	display_code = _extract_code_text(code_raw)
	# Render in a copyable code box
	try:
		code_placeholder.code(display_code, language="python")
	except Exception:
		# Fallback to pre block if st.code fails for any reason
		code_placeholder.markdown("""<pre class='code-box'>""" + html.escape(display_code) + "</pre>", unsafe_allow_html=True)

	# Result metrics & download
	total_time = getattr(res, "total_time", None)
	exec_time = getattr(res, "execution_time", None)
	code_lines = getattr(res, "code_lines", 0)
	llm_calls = getattr(res, "llm_calls", 0)

	with result_placeholder.container():
		c1, c2, c3, c4 = st.columns(4)
		with c1:
			st.metric("Total Time (s)", f"{(total_time or 0):.2f}")
		with c2:
			st.metric("Execution Time (s)", f"{(exec_time or 0):.2f}")
		with c3:
			st.metric("Code Lines", code_lines)
		with c4:
			st.metric("LLM Calls", llm_calls)

		sub_path = getattr(res, "submission_file_path", None)
		if sub_path and os.path.exists(sub_path):
			with open(sub_path, "rb") as f:
				st.download_button(
					label="Download submission.csv",
					data=f.read(),
					file_name="submission.csv",
					mime="text/csv",
				)


st.markdown("---")
st.markdown(
	"""
<div style='text-align:center;color:#666'>
🤖 Kaggle AI Agent System | Supports Multi-Agent · ReAct · RAG · DLP
	</div>
""",
	unsafe_allow_html=True,
)

