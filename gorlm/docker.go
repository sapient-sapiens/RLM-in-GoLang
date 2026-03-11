package myrlm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const defaultImage = "python:3.11-slim"

// REPLResult holds the captured output from a single code execution
// inside the Docker container.
type REPLResult struct {
	Stdout        string
	Stderr        string
	Locals        map[string]string
	ExecutionTime time.Duration
	FinalAnswer   string
}

// DockerREPL manages a Docker container that runs Python code and can call
// back into the host Go server via llm_query()/rlm_query().
type DockerREPL struct {
	image       string
	containerID string
	tempDir     string
	rlmDepth    int
	closed      bool
}

// DockerConfig holds options for creating a DockerREPL.
type DockerConfig struct {
	Image string
	Model string
	Depth int
}

// NewDockerREPL creates and starts a Docker container with a mounted workspace.
// The container executes REPL code and uses HTTP callbacks to the Go server for
// llm_query()/rlm_query().
func NewDockerREPL(cfg DockerConfig) (*DockerREPL, error) {
	image := cfg.Image
	if image == "" {
		image = defaultImage
	}

	baseDir := os.Getenv("RLM_DOCKER_WORKSPACE_DIR")
	if baseDir == "" {
		baseDir = filepath.Join(os.TempDir(), "rlm_workspace")
	}
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return nil, fmt.Errorf("create workspace base dir: %w", err)
	}

	tempDir, err := os.MkdirTemp(baseDir, "docker_repl_")
	if err != nil {
		return nil, fmt.Errorf("create temp dir: %w", err)
	}

	d := &DockerREPL{
		image:    image,
		tempDir:  tempDir,
		rlmDepth: cfg.Depth,
	}

	success := false
	defer func() {
		if !success {
			d.Close()
		}
	}()

	if err := d.startContainer(); err != nil {
		return nil, fmt.Errorf("start container: %w", err)
	}

	if err := d.installDeps(); err != nil {
		return nil, fmt.Errorf("install deps: %w", err)
	}

	success = true
	return d, nil
}

func (d *DockerREPL) startContainer() error {
	serverPort := os.Getenv("RLM_SERVER_PORT")
	if serverPort == "" {
		serverPort = "9712"
	}

	out, err := exec.Command("docker", "run", "-d", "--rm",
		"--add-host=host.docker.internal:host-gateway",
		"-v", d.tempDir+":/workspace",
		"-e", "RLM_SERVER_PORT="+serverPort,
		d.image,
		"tail", "-f", "/dev/null",
	).Output()
	if err != nil {
		return fmt.Errorf("docker run: %w", err)
	}
	d.containerID = strings.TrimSpace(string(out))
	return nil
}

func (d *DockerREPL) installDeps() error {
	cmd := exec.Command("docker", "exec", d.containerID,
		"pip", "install", "-q", "dill")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("pip install failed: %w: %s", err, strings.TrimSpace(string(out)))
	}
	return nil
}

// LoadContext writes the context payload into the container's workspace
// and executes code to load it as the `context` variable.
func (d *DockerREPL) LoadContext(payload any) error {
	switch v := payload.(type) {
	case string:
		path := filepath.Join(d.tempDir, "context.txt")
		if err := os.WriteFile(path, []byte(v), 0644); err != nil {
			return fmt.Errorf("write context.txt: %w", err)
		}
		_, err := d.ExecuteCode("with open('/workspace/context.txt', 'r') as f:\n    context = f.read()")
		return err

	default:
		data, err := json.Marshal(v)
		if err != nil {
			return fmt.Errorf("marshal context: %w", err)
		}
		path := filepath.Join(d.tempDir, "context.json")
		if err := os.WriteFile(path, data, 0644); err != nil {
			return fmt.Errorf("write context.json: %w", err)
		}
		_, err = d.ExecuteCode("import json\nwith open('/workspace/context.json', 'r') as f:\n    context = json.load(f)")
		return err
	}
}

// WriteTempFile writes a file into the container's workspace directory.
// Returns the in-container path (always /workspace/<name>).
func (d *DockerREPL) WriteTempFile(name, content string) string {
	path := filepath.Join(d.tempDir, name)
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		log.Printf("[DockerREPL] warning: failed to write temp file %s: %v", name, err)
	}
	return "/workspace/" + name
}

// ExecuteCode runs a Python code string inside the Docker container.
// The code has access to: context, llm_query, llm_query_batched,
// FINAL_VAR, SHOW_VARS, and any variables from previous executions
// (persisted via dill).
func (d *DockerREPL) ExecuteCode(code string) (*REPLResult, error) {
	return d.ExecuteCodeCtx(context.Background(), code)
}

func (d *DockerREPL) ExecuteCodeCtx(ctx context.Context, code string) (*REPLResult, error) {
	script := buildExecScript(code, d.rlmDepth)

	cmd := exec.CommandContext(ctx, "docker", "exec", d.containerID, "python", "-c", script)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	start := time.Now()
	runErr := cmd.Run()
	elapsed := time.Since(start)

	if ctx.Err() != nil {
		return &REPLResult{
			Stderr:        "Execution cancelled (context deadline exceeded)",
			ExecutionTime: elapsed,
		}, fmt.Errorf("execution cancelled after %s: %w", elapsed, ctx.Err())
	}
	if runErr != nil {
		if stderr.Len() > 0 {
			stderr.WriteString("\n")
		}
		stderr.WriteString("docker exec error: ")
		stderr.WriteString(runErr.Error())
	}

	return parseExecOutput(stdout.String(), stderr.String(), elapsed)
}

// ExecuteCodeWithTimeout is like ExecuteCode but cancels after the given duration.
func (d *DockerREPL) ExecuteCodeWithTimeout(code string, timeout time.Duration) (*REPLResult, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	script := buildExecScript(code, d.rlmDepth)

	cmd := exec.CommandContext(ctx, "docker", "exec", d.containerID, "python", "-c", script)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	start := time.Now()
	err := cmd.Run()
	elapsed := time.Since(start)

	if ctx.Err() != nil {
		return &REPLResult{
			Stderr:        "Execution timed out",
			ExecutionTime: elapsed,
		}, fmt.Errorf("execution timed out after %s", timeout)
	}
	if err != nil {
		return parseExecOutput(stdout.String(), stderr.String(), elapsed)
	}
	return parseExecOutput(stdout.String(), stderr.String(), elapsed)
}

// Close stops the Docker container and removes the temp directory.
func (d *DockerREPL) Close() error {
	if d.closed {
		return nil
	}
	d.closed = true

	if d.containerID != "" {
		_ = exec.Command("docker", "stop", d.containerID).Run()
	}
	if d.tempDir != "" {
		os.RemoveAll(d.tempDir)
	}
	return nil
}

// buildExecScript generates a self-contained Python script that:
//   - loads persisted state from previous executions
//   - defines llm_query, llm_query_batched, rlm_query, rlm_query_batched
//     as HTTP POST calls to the Go server at RLM_SERVER_PORT
//   - defines FINAL_VAR and SHOW_VARS
//   - executes the user's code with stdout/stderr capture
//   - saves state and outputs JSON on the last line
func buildExecScript(code string, rlmDepth int) string {
	codeB64 := base64.StdEncoding.EncodeToString([]byte(code))

	return fmt.Sprintf(`
import sys, io, json, base64, traceback, os
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import concurrent.futures

sys.path.insert(0, "/workspace")

try:
    import dill
except ImportError:
    import pickle as dill

_SERVER_BASE = "http://host.docker.internal:" + os.environ["RLM_SERVER_PORT"]
_RLM_DEPTH = %d

def _post(path, body, timeout=1800):
    data = json.dumps(body).encode()
    req = Request(_SERVER_BASE + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()}"}
    except URLError as e:
        return {"error": f"Connection error: {e.reason}"}

def llm_query(prompt, model=None):
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [{"role": "user", "content": str(prompt)}]
    resp = _post("/llm_query", {"messages": messages, "model": model})
    return resp.get("text") or resp.get("error", "Unknown error")

def llm_query_batched(prompts, model=None):
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(prompts), 16)) as pool:
        futures = [pool.submit(llm_query, p, model) for p in prompts]
        return [f.result() for f in futures]

def _safe_context_payload():
    ctx = _locals.get("context")
    if ctx is None:
        return None
    try:
        json.dumps(ctx)
        return ctx
    except:
        return str(ctx)

_INHERIT = object()

def rlm_query(prompt, model=None, context=_INHERIT):
    ctx = _safe_context_payload() if context is _INHERIT else context
    resp = _post("/rlm_query", {
        "prompt": str(prompt),
        "model": model,
        "depth": _RLM_DEPTH,
        "context": ctx,
    })
    return resp.get("text") or resp.get("error", "Unknown error")

def rlm_query_batched(prompts, model=None, context=_INHERIT):
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(prompts), 4)) as pool:
        futures = [pool.submit(rlm_query, p, model, context) for p in prompts]
        return [f.result() for f in futures]

STATE = "/workspace/state.dill"

def load_state():
    if os.path.exists(STATE):
        try:
            with open(STATE, "rb") as f:
                return dill.load(f)
        except:
            pass
    return {}

def save_state(s):
    clean = {k: v for k, v in s.items() if not k.startswith("_")}
    for k in list(clean.keys()):
        try:
            dill.dumps(clean[k])
        except:
            del clean[k]
    with open(STATE, "wb") as f:
        dill.dump(clean, f)

_locals = load_state()
_final_answer = [None]
_combined_ref = [None]

def FINAL_VAR(name):
    name = name.strip().strip("\"'")
    ns = _combined_ref[0] if _combined_ref[0] is not None else _locals
    if name in ns:
        _final_answer[0] = str(ns[name])
        return _final_answer[0]
    available = [k for k in ns.keys() if not k.startswith("_")]
    if available:
        return f"Error: Variable '{name}' not found. Available: {available}"
    return f"Error: Variable '{name}' not found. No variables created yet."

def SHOW_VARS():
    available = {k: type(v).__name__ for k, v in _locals.items() if not k.startswith("_")}
    if not available:
        msg = "No variables created yet."
    else:
        parts = []
        for k, t in available.items():
            v = _locals[k]
            try:
                length = len(v)
                parts.append(f"  {k} ({t}, len={length})")
            except TypeError:
                parts.append(f"  {k} ({t})")
        msg = "Available variables:\n" + "\n".join(parts)
    print(msg)
    return msg

_globals = {
    "__builtins__": __builtins__,
    "__name__": "__main__",
    "llm_query": llm_query,
    "llm_query_batched": llm_query_batched,
    "rlm_query": rlm_query,
    "rlm_query_batched": rlm_query_batched,
    "FINAL_VAR": FINAL_VAR,
    "SHOW_VARS": SHOW_VARS,
}

code = base64.b64decode("%s").decode()
stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
old_stdout, old_stderr = sys.stdout, sys.stderr

try:
    sys.stdout, sys.stderr = stdout_buf, stderr_buf
    combined = {**_globals, **_locals}
    _combined_ref[0] = combined
    exec(code, combined, combined)
    for k, v in combined.items():
        if k not in _globals and not k.startswith("_"):
            _locals[k] = v
except:
    traceback.print_exc(file=stderr_buf)
finally:
    sys.stdout, sys.stderr = old_stdout, old_stderr

if "context_0" in _locals:
    _locals["context"] = _locals["context_0"]
if "history_0" in _locals:
    _locals["history"] = _locals["history_0"]

save_state(_locals)

output = {
    "stdout": stdout_buf.getvalue(),
    "stderr": stderr_buf.getvalue(),
    "locals": {k: repr(v)[:200] for k, v in _locals.items() if not k.startswith("_")},
    "final_answer": _final_answer[0],
}
print(json.dumps(output, ensure_ascii=False))
`, rlmDepth, codeB64)
}

// execOutput is the JSON structure printed by the exec script.
type execOutput struct {
	Stdout      string            `json:"stdout"`
	Stderr      string            `json:"stderr"`
	Locals      map[string]string `json:"locals"`
	FinalAnswer *string           `json:"final_answer"`
}

func parseExecOutput(stdout, stderr string, elapsed time.Duration) (*REPLResult, error) {
	lines := strings.Split(strings.TrimSpace(stdout), "\n")
	if len(lines) == 0 {
		return &REPLResult{
			Stdout:        stdout,
			Stderr:        stderr,
			ExecutionTime: elapsed,
		}, nil
	}

	lastLine := lines[len(lines)-1]
	var out execOutput
	if err := json.Unmarshal([]byte(lastLine), &out); err != nil {
		return &REPLResult{
			Stdout:        stdout,
			Stderr:        stderr + "\nJSON parse error: " + err.Error(),
			Locals:        map[string]string{},
			ExecutionTime: elapsed,
		}, nil
	}

	combinedStderr := strings.TrimSpace(strings.Join([]string{out.Stderr, stderr}, "\n"))
	result := &REPLResult{
		Stdout:        out.Stdout,
		Stderr:        combinedStderr,
		Locals:        out.Locals,
		ExecutionTime: elapsed,
	}
	if out.FinalAnswer != nil {
		result.FinalAnswer = *out.FinalAnswer
	}
	return result, nil
}

// FormatREPLResult formats a REPLResult into a string suitable for
// appending to the LLM message history (similar to format_execution_result
// in the reference Python implementation).
func FormatREPLResult(r *REPLResult) string {
	var parts []string
	if r.Stdout != "" {
		parts = append(parts, r.Stdout)
	}
	if r.Stderr != "" {
		parts = append(parts, "STDERR: "+r.Stderr)
	}
	if len(parts) == 0 {
		return "No output"
	}
	return strings.Join(parts, "\n\n")
}
