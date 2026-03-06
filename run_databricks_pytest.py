"""
Submit credibility pytest suite to Databricks serverless compute.

Uses the REST API directly to submit a one-off run on serverless compute
(no cluster spec needed - workspace enforces serverless-only).
"""
import os
import sys
import time
import uuid
import base64
import json
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Load credentials
# ---------------------------------------------------------------------------
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

api_base = os.environ["DATABRICKS_HOST"].rstrip("/")
token = os.environ["DATABRICKS_TOKEN"]
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

RUN_ID = uuid.uuid4().hex[:8]
NOTEBOOK_PATH = "/Workspace/credibility/run_pytest"

# ---------------------------------------------------------------------------
# Read source files and embed as JSON
# ---------------------------------------------------------------------------
BASE = "/home/ralph/credibility"

def read_file(path):
    with open(path, "r") as f:
        return f.read()

files_map = {
    "_validation.py":          read_file(f"{BASE}/src/credibility/_validation.py"),
    "buhlmann_straub.py":      read_file(f"{BASE}/src/credibility/buhlmann_straub.py"),
    "hierarchical.py":         read_file(f"{BASE}/src/credibility/hierarchical.py"),
    "__init__.py":             read_file(f"{BASE}/src/credibility/__init__.py"),
    "conftest.py":             read_file(f"{BASE}/tests/conftest.py"),
    "test_buhlmann_straub.py": read_file(f"{BASE}/tests/test_buhlmann_straub.py"),
    "test_hierarchical.py":    read_file(f"{BASE}/tests/test_hierarchical.py"),
}

files_json = json.dumps(files_map)
pyproject_json = json.dumps("""[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "credibility"
version = "0.2.0"
requires-python = ">=3.9"
dependencies = ["numpy>=1.21", "polars>=0.20"]

[tool.hatch.build.targets.wheel]
packages = ["src/credibility"]
""")

# ---------------------------------------------------------------------------
# Build notebook source
# ---------------------------------------------------------------------------
# We use {json.dumps(...)!r} to embed the JSON blobs as Python string literals.
# The notebook itself does json.loads() to reconstruct the strings.
NOTEBOOK_SOURCE = f"""# Databricks notebook source
# MAGIC %pip install polars>=0.20 numpy>=1.21 pytest>=7.0 hatchling pandas --quiet

# COMMAND ----------

import json, os, sys, uuid, subprocess

pkg_id = uuid.uuid4().hex[:8]
pkg_dir = f"/tmp/credibility_{{pkg_id}}"
src_dir = f"{{pkg_dir}}/src/credibility"
tests_dir = f"{{pkg_dir}}/tests"
os.makedirs(src_dir, exist_ok=True)
os.makedirs(tests_dir, exist_ok=True)

FILES_JSON = {files_json!r}
PYPROJECT_JSON = {pyproject_json!r}

files_map = json.loads(FILES_JSON)
pyproject_src = json.loads(PYPROJECT_JSON)

src_files = {{"_validation.py", "buhlmann_straub.py", "hierarchical.py", "__init__.py"}}

for name, content in files_map.items():
    if name in src_files:
        path = f"{{src_dir}}/{{name}}"
    else:
        path = f"{{tests_dir}}/{{name}}"
    with open(path, "w") as f:
        f.write(content)

with open(f"{{pkg_dir}}/pyproject.toml", "w") as f:
    f.write(pyproject_src)

print(f"Written {{len(files_map) + 1}} files to {{pkg_dir}}")

# COMMAND ----------

r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", pkg_dir, "--quiet"],
    capture_output=True, text=True
)
if r.returncode != 0:
    print("Install error:", r.stderr[:500])
else:
    print("credibility installed from", pkg_dir)

# COMMAND ----------

r = subprocess.run(
    [sys.executable, "-m", "pytest", tests_dir, "-v", "--tb=short",
     "--no-header", "-p", "no:cacheprovider"],
    capture_output=True, text=True, cwd=pkg_dir
)

print(r.stdout)
if r.stderr:
    print("STDERR:", r.stderr[-500:])

if r.returncode == 0:
    print("\\n=== ALL TESTS PASSED ===")
    try:
        dbutils.notebook.exit("ALL TESTS PASSED")
    except NameError:
        pass
else:
    msg = f"TESTS FAILED (exit {{r.returncode}})"
    print(f"\\n=== {{msg}} ===")
    try:
        dbutils.notebook.exit(msg)
    except NameError:
        pass
"""

# ---------------------------------------------------------------------------
# Upload notebook
# ---------------------------------------------------------------------------
print(f"Uploading notebook to {NOTEBOOK_PATH} ...")
notebook_b64 = base64.b64encode(NOTEBOOK_SOURCE.encode("utf-8")).decode("ascii")

def api_call(method, endpoint, body=None):
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(
        f"{api_base}/{endpoint}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        raise RuntimeError(f"API {method} {endpoint} failed {e.code}: {err_body}")

result = api_call("POST", "api/2.0/workspace/import", {
    "path": NOTEBOOK_PATH,
    "format": "SOURCE",
    "language": "PYTHON",
    "content": notebook_b64,
    "overwrite": True,
})
print(f"Upload OK: {result}")

# ---------------------------------------------------------------------------
# Submit run on serverless compute
# For serverless, omit new_cluster entirely - Databricks uses serverless by default
# ---------------------------------------------------------------------------
print(f"Submitting serverless run {RUN_ID} ...")

submit_body = {
    "run_name": f"credibility-pytest-{RUN_ID}",
    "tasks": [
        {
            "task_key": "pytest",
            "notebook_task": {
                "notebook_path": NOTEBOOK_PATH,
                "source": "WORKSPACE",
            },
        }
    ],
}

result = api_call("POST", "api/2.1/jobs/runs/submit", submit_body)
run_id = result["run_id"]
print(f"Run submitted: run_id={run_id}")

# ---------------------------------------------------------------------------
# Poll
# ---------------------------------------------------------------------------
print("Polling (serverless starts faster than classic clusters) ...")
for i in range(90):
    time.sleep(15)
    run_state = api_call("GET", f"api/2.1/jobs/runs/get?run_id={run_id}")
    lc = run_state.get("state", {}).get("life_cycle_state", "UNKNOWN")
    rs = run_state.get("state", {}).get("result_state", "-")
    print(f"  [{i*15}s] {lc} / {rs}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break

print(f"\nFinal state: {lc} / {rs}")

# ---------------------------------------------------------------------------
# Fetch output
# ---------------------------------------------------------------------------
try:
    output = api_call("GET", f"api/2.1/jobs/runs/get-output?run_id={run_id}")
    notebook_result = output.get("notebook_output", {}).get("result", "")
    error = output.get("error", "")
    error_trace = output.get("error_trace", "")
    logs = output.get("logs", "")

    if notebook_result:
        print(f"\nExit value: {notebook_result}")
    if error:
        print(f"Error: {error}")
    if error_trace:
        print(f"Trace:\n{error_trace[:3000]}")
    if logs:
        print(f"\nLogs:\n{logs[-6000:]}")
except Exception as e:
    print(f"Could not fetch output: {e}")

if rs == "SUCCESS":
    print("\n=== PASS: All tests completed on Databricks. ===")
    sys.exit(0)
else:
    print(f"\n=== FAIL: Run ended with state {rs}. ===")
    sys.exit(1)
