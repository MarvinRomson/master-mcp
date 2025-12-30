# schema_editor.py
import os
import json
import hashlib
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse

import uvicorn
import subprocess
import socket
import time

file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(file_dir)   

SCHEMA_DIR = os.path.join(file_dir, "schemas")
os.makedirs(SCHEMA_DIR, exist_ok=True)

app = FastAPI()

def is_port_open(host: str, port: int) -> bool:
    """Check if a TCP port is open (service is listening)."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False
    
def _ensure_editor(editor_port):
    host = "127.0.0.1"
    if is_port_open(host, editor_port):
        #print(f"Schema editor already running at http://{host}:{editor_port}")
        return

    #print("Starting schema editor service...")
    proc = subprocess.Popen(
        ["uvicorn", "schema_editor:app", "--port", str(editor_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=file_dir
    )

    # wait until it’s listening
    for _ in range(10):
        if is_port_open(host, editor_port):
            #print(f"Schema editor started at http://{host}:{editor_port}")
            return
        time.sleep(0.5)

    #print("Warning: Schema editor did not start in time")

def schema_path(name="tools.json"):
    return os.path.join(SCHEMA_DIR, name)

@app.get("/", response_class=HTMLResponse)
async def index():
    """Render the schema editor page, loading the current tools schema."""
    try:
        with open(schema_path(), "r") as file:
            data = file.read()
        schema_data = json.loads(data)["modified"]
        # This is the JSON value we will load into the browser-side editor.
        # It is rendered into a <script type="application/json"> tag, then
        # parsed client-side to avoid HTML-escaping issues.
        schema = json.dumps(schema_data, ensure_ascii=False)
    except FileNotFoundError:
        schema = "[]"

    # Load HTML template from external file and inject the JSON schema.
    template_path = os.path.join(file_dir, "schema_editor.html")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            html_template = f.read()
    except FileNotFoundError:
        return "<html><body><h1>Schema Editor template not found</h1></body></html>"

    html = html_template.replace("SCHEMA_JSON_PLACEHOLDER", schema)
    return html

@app.post("/update")
async def update(schema: str = Form(...)):
    try:
        # Accept either a JSON array of tools or an object that already wraps
        # the tools in a "modified" field. We normalize to the internal
        # structure {"modified": [...], "modified_hash": "..."}.
        parsed = json.loads(schema)

        # If the user pasted the whole file structure, keep its "modified"
        # field; otherwise, treat the JSON value as the tools array.
        if isinstance(parsed, dict) and "modified" in parsed:
            tools = parsed["modified"]
        else:
            tools = parsed

        if not isinstance(tools, list):
            return {"error": "Top-level JSON must be an array of tools or an object with a 'modified' array"}

        hash_val = hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()
        with open(schema_path(), "w") as f:
            json.dump(
                {"modified": tools, "modified_hash": hash_val},
                f,
                indent=2
            )
        return RedirectResponse("/", status_code=303)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON"}
    
if __name__=="__main__":
    corn = False

    if corn:
        uvicorn.run(
            "schema_editor:app",
            host="0.0.0.0",
            port=8001,
            reload=True
        )
    else:
        import subprocess

        proc = subprocess.Popen(
            ["uvicorn", "schema_editor:app", "--reload", "--port", "8001"]
        )
    print(f"Started schema editor with PID {proc.pid}")
