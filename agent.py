#!/usr/bin/env python3
"""Bounded Python -> Rust translation agent.

Set GEMINI_API_KEY and GEMINI_MODEL, then run: python agent.py
The agent uses Google Gemini directly and otherwise only the Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
RUST = HERE / "rust"
LIB = RUST / "src" / "lib.rs"
PYSRC = HERE / "reference" / "version.py"
LOGS = HERE / "logs"
MAX_CALLS = 40
CONTEXT_CHARS = 24_000
READ_CHARS = 9_000
MAX_TOOL_CALLS = 8


def clip(text: str, limit: int) -> str:
    """Keep the beginning and final diagnostics, marking omissions explicitly."""
    if len(text) <= limit:
        return text
    marker = "\n...[truncated; retrieve a narrower excerpt]...\n"
    room = limit - len(marker)
    return text[:room // 2] + marker + text[-(room - room // 2):]


def source_hash() -> str:
    return hashlib.sha256(LIB.read_bytes()).hexdigest()


def configured_model() -> str | None:
    return os.environ.get("GEMINI_MODEL")


def request_json(request, key: str) -> dict:
    """No hidden retries; sanitize provider errors before they reach logs."""
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        status, detail = exc.code, ""
        try:
            error = json.loads(exc.read(8192)).get("error", {})
            if isinstance(error, dict):
                detail = str(error.get("message", "")).replace(key, "[redacted]")[:500]
        except (ValueError, TypeError):
            pass
        finally:
            exc.close()
        raise RuntimeError(f"Model API HTTP {status}; {detail or 'check credentials, model, endpoint or quota.'}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Model request failed ({type(exc).__name__}); no automatic retry.") from None


def call_gemini(messages: list[dict], tools: list[dict]) -> dict:
    """Native Google generateContent function calling, using the same bounded packets.

    https://ai.google.dev/api/generate-content
    Each request starts afresh; no partial function-call/thought history is replayed.
    """
    key, model = os.environ.get("GEMINI_API_KEY"), os.environ.get("GEMINI_MODEL")
    if not key or not model:
        raise RuntimeError("Set GEMINI_API_KEY and GEMINI_MODEL before running.")
    declarations = []
    for tool in tools:
        declaration = {"name": tool["name"], "description": tool["description"]}
        if tool["parameters"].get("properties"):
            # Gemini's native Schema supports our fields except additionalProperties.
            declaration["parameters"] = {k: v for k, v in tool["parameters"].items()
                                         if k != "additionalProperties"}
        declarations.append(declaration)
    payload = {
        "systemInstruction": {"parts": [{"text": "\n\n".join(
            m["content"] for m in messages if m["role"] == "system")}]},
        "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                      "parts": [{"text": m["content"]}]}
                     for m in messages if m["role"] != "system"],
        "tools": [{"functionDeclarations": declarations}],
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        "generationConfig": {"maxOutputTokens": 12_000},
    }
    identifier = urllib.parse.quote(model.removeprefix("models/"), safe="")
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{identifier}:generateContent",
        data=json.dumps(payload).encode(), method="POST",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
    )
    data = request_json(request, key)
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidate; check prompt feedback or API configuration.")
    candidate = candidates[0]
    if candidate.get("finishReason") != "STOP":
        raise RuntimeError(f"Gemini response was not complete: {candidate.get('finishReason')}")
    text, calls = [], []
    for part in candidate.get("content", {}).get("parts", []):
        if part.get("text") and not part.get("thought"):
            text.append(part["text"])
        if "functionCall" in part:
            call = part["functionCall"]
            calls.append({"name": call.get("name"), "arguments": call.get("args", {})})
    return {"text": "\n".join(text) or None, "tool_calls": calls,
            "usage": data.get("usageMetadata", {}), "response_id": data.get("responseId"),
            "model": data.get("modelVersion"), "requested_model": model, "provider": "gemini"}


def call_model(messages: list[dict], tools: list[dict]) -> dict:
    """One Gemini HTTP request per budgeted call; no invisible retries."""
    return call_gemini(messages, tools)

def system_prompt() -> str:
    return """Translate the required public API of reference/version.py into
rust/src/lib.rs. The Python source and reference/test_*.py are the specification.
First inspect the Rust API, then retrieve Python parsing, comparison, formatting
and bump implementations. Port relevant reference cases into real Rust tests.
Implement parse, to_string, compare, bump_major, bump_minor and bump_patch.
Preserve public Version fields/types used by the harness. Only lib.rs may change.
Never edit main.rs, Cargo.toml, evaluate.py or reference/. Use std only: no unsafe,
external processes/Python, new dependencies, todo!, unimplemented! or panic!.
Avoid unnecessary cloning and unwraps.

Verify ASCII validation, empty identifiers, leading zeros in core/prerelease
versus build, build-independent precedence, numeric/non-numeric prereleases,
long numeric identifiers, prefix ordering and bump behavior against the source.
Do not confuse next_version with bump_patch. Python integers are unbounded but
the scaffold uses u64: document representational limits and handle overflow
deliberately; do not rely on release-mode wrapping.

Retrieve bounded excerpts, write the initial implementation, then prefer exact
replace_rust edits. Run cargo_build/test and evaluate for compiler diagnostics
and all practice families. Use probe for specific discrepancies. Improve the
semantics, not the evaluator or its fixtures. Tests must assert behavior.
Each turn has a fresh bounded context. save_notes persists concise findings,
failed approaches and next steps. Use read/search to retrieve code again.
The runner restores the best evaluated version on regressions: re-read after
rollback. Don't repeat unchanged actions or just narrate plans. Completion
requires a fresh full evaluation of the current file: successful build, nonempty
passing Rust tests, every differential case, precedence chain and clean quality.
You have at most 40 model calls.
"""


def result_of(event: dict) -> dict:
    return event.get("result", {}) if event.get("role") == "tool" else {}


def build_context(history: list[dict], step: int) -> list[dict]:
    """WRITE notes/logs, SELECT recent evidence, COMPRESS old steps without an LLM."""
    task = next((m["content"] for m in history if m["role"] == "user"), "Translate the module.")
    counts = Counter(m.get("name") for m in history if m["role"] == "tool")
    notes = next((result_of(m).get("notes") for m in reversed(history)
                  if m.get("name") == "save_notes" and result_of(m).get("ok")), "No notes yet.")
    evaluations = [result_of(m) for m in history if m.get("name") == "evaluate"]
    scores = [{"sha256": e.get("sha256"), "score": e.get("report", {}).get("differential_pct"),
               "build": e.get("report", {}).get("build"), "rollback": e.get("rollback", False)}
              for e in evaluations[-6:]]
    state = {"step": step, "sha256": source_hash(), "notes": notes,
             "tool_counts": dict(counts), "recent_evaluations": scores}
    prefix = (f"Task: {clip(task, 2000)}\n\nPersistent state:\n"
              + json.dumps(state, ensure_ascii=False) + "\n\nRecent observations (oldest first):\n")
    snippets = []
    for event in reversed(history[2:]):
        if event.get("role") == "tool":
            snippet = f"{event.get('name')}: {clip(event.get('content', ''), READ_CHARS)}"
        elif event.get("role") == "assistant":
            names = [str(c.get("name")) for c in event.get("tool_calls", [])]
            snippet = f"assistant: {clip(event.get('content', ''), 700)}; actions={names}"
        else:
            continue
        if len(snippets) >= 10:
            break
        snippets.append(snippet)
    system = history[0]["content"]
    remaining = CONTEXT_CHARS - len(system) - len(prefix)
    selected = []
    for snippet in snippets:
        if remaining < 300:
            break
        snippet = clip(snippet, min(remaining - 2, READ_CHARS))
        selected.append(snippet)
        remaining -= len(snippet) + 2
    return [{"role": "system", "content": system},
            {"role": "user", "content": prefix + "\n\n".join(reversed(selected))}]


def successful(report: dict) -> bool:
    tests, families = report.get("cargo_test") or {}, report.get("differential") or {}
    quality = report.get("quality") or {}
    return bool(report.get("build") and report.get("cargo_test_ok") and tests.get("passed", 0) > 0
                and tests.get("failed") == 0 and report.get("spec_precedence_chain")
                and not report.get("violations") and len(families) == 5
                and all(f.get("total", 0) > 0 and f.get("pass") == f["total"] for f in families.values())
                and all(quality.get(k, 1) == 0 for k in
                        ("unsafe_blocks", "todo_macros", "panic_macros", "extra_dependencies"))
                and quality.get("clone_calls", 21) <= 20 and quality.get("unwrap_calls", 11) <= 10)


def rank(report: dict) -> tuple:
    """Build/safety/test gates outrank correctness; ties prefer less copying."""
    tests, quality = report.get("cargo_test") or {}, report.get("quality") or {}
    clean = bool(report.get("build") and "violations" in report and not report["violations"])
    tested = bool(report.get("cargo_test_ok") and tests.get("passed", 0) > 0 and tests.get("failed") == 0)
    families = report.get("differential") or {}
    passed = sum(f.get("pass", 0) for f in families.values())
    total = sum(f.get("total", 0) for f in families.values())
    return (clean, bool(report.get("build")), tested, passed / total if total else 0,
            -quality.get("clone_calls", 0), -quality.get("unwrap_calls", 0))


def should_stop(history: list[dict], step: int, budget: int,
                last_score: float | None) -> tuple[bool, str]:
    evaluations = [result_of(m) for m in history if m.get("name") == "evaluate"]
    if evaluations:
        latest = evaluations[-1]
        if latest.get("sha256") == source_hash() and latest.get("ok") and successful(latest.get("report", {})):
            return True, "verified completion: build, tests, differential, precedence and quality passed"
    turns = [m for m in history if m.get("role") == "assistant"]
    if len(turns) >= 3 and all(not m.get("tool_calls") for m in turns[-3:]):
        return True, "stuck: three replies without tool actions"
    if len(turns) >= 3:
        # The same tests after a real edit are useful, not a repeated action.
        fingerprints = [json.dumps([m.get("source_hash"), m.get("tool_calls")], sort_keys=True)
                        for m in turns[-3:]]
        if len(set(fingerprints)) == 1:
            return True, "stuck: repeated identical actions on the same source three times"
    if len(evaluations) >= 5:
        recent = [rank(e.get("report", {})) for e in evaluations[-5:]]
        if max(recent[1:]) <= recent[0]:
            return True, f"stuck: four evaluations without improvement (latest score {last_score})"
    if step >= min(budget, MAX_CALLS):
        return True, f"budget exhausted ({min(budget, MAX_CALLS)} model calls)"
    return False, ""


def _run(cmd, cwd=None, timeout=180, input_text=None):
    try:
        p = subprocess.run(cmd, cwd=cwd, input=input_text, capture_output=True, text=True, timeout=timeout)
        return {"ok": p.returncode == 0, "returncode": p.returncode,
                "output": (p.stdout + p.stderr).strip() or "(no output)"}
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or b""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return {"ok": False, "error": f"TIMEOUT after {timeout}s", "output": output}
    except OSError as exc:
        return {"ok": False, "error": str(exc)}


def read_excerpt(path, args):
    lines = path.read_text().splitlines()
    start, count = args.get("start", 1), args.get("count", 100)
    if type(start) is not int or type(count) is not int or start < 1 or not 1 <= count <= 160:
        raise ValueError("start must be >=1 and count must be 1..160")
    text = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines) if start <= i + 1 < start + count)
    return {"ok": True, "total_lines": len(lines), "text": clip(text, READ_CHARS)}


def t_read_python(args):
    return read_excerpt(PYSRC, args)


def t_read_rust(args):
    return {**read_excerpt(LIB, args), "sha256": source_hash()}


def t_read_tests(args):
    if args["file"] not in ("test_parsing.py", "test_compare.py", "test_bump.py"):
        raise ValueError("Choose a supplied reference test file")
    return read_excerpt(HERE / "reference" / args["file"], args)


def t_search_source(args):
    paths = {"python": PYSRC, "rust": LIB, **{n: HERE / "reference" / n for n in
             ("test_parsing.py", "test_compare.py", "test_bump.py")}}
    path, needle = paths[args.get("file", "python")], args["query"]
    if not isinstance(needle, str) or not needle:
        raise ValueError("query must be nonempty literal text")
    lines = path.read_text().splitlines()
    hits = [i for i, line in enumerate(lines) if needle in line]
    selected = sorted({i for h in hits[:12] for i in range(max(0, h - 2), min(len(lines), h + 9))})
    return {"ok": True, "matches": len(hits),
            "text": clip("\n".join(f"{i + 1}: {lines[i]}" for i in selected), READ_CHARS)}


def t_write_rust(args):
    content = args["content"]
    if not isinstance(content, str) or not content.strip() or len(content) > 120_000:
        raise ValueError("content must be the whole Rust file (1..120000 characters)")
    with tempfile.NamedTemporaryFile(mode="w", dir=LIB.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        handle.write(content)
    temporary.replace(LIB)
    return {"ok": True, "bytes": len(content.encode()), "sha256": source_hash()}


def t_replace_rust(args):
    old, new, source = args["old"], args["new"], LIB.read_text()
    if not isinstance(old, str) or not isinstance(new, str) or not old or source.count(old) != 1:
        raise ValueError("old must match exactly once; read the current file before editing")
    if args.get("sha256") and args["sha256"] != source_hash():
        raise ValueError("Source changed since it was read; retrieve it again")
    return t_write_rust({"content": source.replace(old, new, 1)})


def t_cargo_build(_args):
    return _run(["cargo", "build", "--release"], cwd=RUST)


def t_cargo_test(_args):
    return _run(["cargo", "test", "--release"], cwd=RUST)


def t_evaluate(_args):
    """Full practice suite; structured JSON instead of a rounded console score."""
    with tempfile.TemporaryDirectory(prefix="py2rust-eval-") as directory:
        report_path = pathlib.Path(directory) / "report.json"
        result = _run([sys.executable, str(HERE / "evaluate.py"), "--n", "300",
                       "--json", str(report_path)], cwd=HERE, timeout=300)
        result["report"] = json.loads(report_path.read_text()) if report_path.exists() else {}
    if result["report"].get("build"):
        # evaluate.py only parses the first test summary and discards its exit code.
        # Check the process status too, including binary tests and doctests.
        tests = t_cargo_test({})
        result["report"]["cargo_test_ok"] = tests["ok"]
        if not tests["ok"]:
            result["test_failure"] = tests
    result["sha256"] = source_hash()
    return result


def t_probe(args):
    """One structured command against both oracle and Rust; no arbitrary Python."""
    import semver

    op, version = args["operation"], args["version"]
    other, kind = args.get("other", ""), args.get("kind", "patch")
    if any(not isinstance(s, str) or any(c.isspace() for c in s) for s in (version, other)):
        raise ValueError("The harness cannot represent whitespace in versions; use Rust unit tests")
    if op not in ("parse", "format", "compare", "bump") or kind not in ("major", "minor", "patch"):
        raise ValueError("Invalid operation or bump kind")
    if not version or (op == "compare" and not other):
        raise ValueError("Nonempty version (and other for compare) is required by the harness")
    try:
        parsed = semver.Version.parse(version)
        if op == "parse":
            expected = {"ok": True, **parsed.to_dict()}
        elif op == "compare":
            expected = {"ok": True, "cmp": parsed.compare(semver.Version.parse(other))}
        else:
            value = str(parsed) if op == "format" else str(getattr(parsed, f"bump_{kind}")())
            expected = {"ok": True, "version": value}
    except ValueError:
        expected = {"ok": False}
    build = t_cargo_build({})
    if not build["ok"]:
        return build
    command = f"{op} {version}"
    if op == "compare":
        command += f" {other}"
    elif op == "bump":
        command = f"bump {kind} {version}"
    result = _run([str(RUST / "target" / "release" / "harness")], input_text=command + "\n", timeout=15)
    if not result["ok"]:
        return result
    actual = json.loads(result["output"])
    matches = actual == expected if expected["ok"] else actual.get("ok") is False
    return {"ok": True, "matches": matches, "expected": expected, "actual": actual}


def t_save_notes(args):
    notes = args["notes"]
    if not isinstance(notes, str) or len(notes) > 2500:
        raise ValueError("Keep notes to at most 2500 characters")
    return {"ok": True, "notes": notes}


def tool(name, description, fn, properties=None, required=()):
    return {"name": name, "description": description, "fn": fn,
            "parameters": {"type": "object", "properties": properties or {},
                           "required": list(required), "additionalProperties": False}}


STRING = {"type": "string"}
RANGE = {"start": {"type": "integer", "minimum": 1}, "count": {"type": "integer", "minimum": 1, "maximum": 160}}
TEST_FILES = ["test_parsing.py", "test_compare.py", "test_bump.py"]
TOOLS = [
    tool("read_python", "Read numbered Python lines (default start=1, count=100).", t_read_python, RANGE),
    tool("read_rust", "Read numbered Rust lines and source hash.", t_read_rust, RANGE),
    tool("read_tests", "Read supplied Python tests to port to Rust.", t_read_tests,
         {**RANGE, "file": {"type": "string", "enum": TEST_FILES}}, ["file"]),
    tool("search_source", "Find literal text with surrounding source lines.", t_search_source,
         {"query": STRING, "file": {"type": "string", "enum": ["python", "rust", *TEST_FILES]}}, ["query"]),
    tool("write_rust", "Write WHOLE lib.rs including tests. Only this file can be edited.", t_write_rust,
         {"content": STRING}, ["content"]),
    tool("replace_rust", "Replace one exact unique snippet; optional sha256 rejects stale edits.", t_replace_rust,
         {"old": STRING, "new": STRING, "sha256": STRING}, ["old", "new"]),
    tool("cargo_build", "Compile the release harness; return status and diagnostics.", t_cargo_build),
    tool("cargo_test", "Run Rust tests; return status and diagnostics.", t_cargo_test),
    tool("evaluate", "Full practice evaluation with scores, failures and quality. May roll back regressions.", t_evaluate),
    tool("probe", "Build and compare one case against Python semver.", t_probe,
         {"operation": {"type": "string", "enum": ["parse", "format", "compare", "bump"]},
          "version": STRING, "other": STRING, "kind": {"type": "string", "enum": ["major", "minor", "patch"]}},
         ["operation", "version"]),
    tool("save_notes", "Persist findings, failed attempts and next steps (<=2500 characters).", t_save_notes,
         {"notes": STRING}, ["notes"]),
]
BY_NAME = {t["name"]: t for t in TOOLS}
SCHEMAS = [{k: t[k] for k in ("name", "description", "parameters")} for t in TOOLS]


def dispatch(call: dict) -> dict:
    """Tool mistakes become observations so the model can recover."""
    name, args = call.get("name"), call.get("arguments", {})
    if not isinstance(name, str) or name not in BY_NAME:
        return {"ok": False, "error": f"Unknown tool {name!r}"}
    if not isinstance(args, dict):
        return {"ok": False, "error": "Tool arguments must be a JSON object"}
    schema = BY_NAME[name]["parameters"]
    if set(schema["required"]) - args.keys() or args.keys() - schema["properties"].keys():
        return {"ok": False, "error": "Missing required or unexpected tool arguments"}
    try:
        return BY_NAME[name]["fn"](args)
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


class Run:
    """A run's durable evidence and best evaluated checkpoint."""

    def __init__(self, budget: int, task: str):
        if not 1 <= budget <= MAX_CALLS:
            raise ValueError("Model call budget must be 1..40")
        self.budget = budget
        LOGS.mkdir(exist_ok=True)
        self.directory = LOGS / f"run-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}"
        self.directory.mkdir()
        self.log = self.directory.with_suffix(".jsonl")
        self.history = [{"role": "system", "content": system_prompt()}, {"role": "user", "content": task}]
        self.best = None
        self.last_score = None
        self.step = 0
        (self.directory / "initial.rs").write_bytes(LIB.read_bytes())
        self.record(event="start", budget=budget, task=task, model=configured_model(),
                    provider="gemini",
                    source_hash=source_hash(), schemas=SCHEMAS)

    def record(self, **fields):
        with self.log.open("a") as handle:
            handle.write(json.dumps({"t": time.time(), **fields}, ensure_ascii=False) + "\n")

    def observe(self, call: dict) -> dict:
        result = dispatch(call)
        if call.get("name") == "evaluate":
            report = result.get("report", {})
            self.last_score = report.get("differential_pct")
            if self.best is not None and (not result.get("ok") or rank(report) < rank(self.best)):
                LIB.write_bytes((self.directory / "best.rs").read_bytes())
                result.update(rollback=True, restored_sha256=source_hash(),
                              restored_score=self.best.get("differential_pct"))
                self.record(event="rollback", step=self.step, candidate_report=report,
                            restored_sha256=source_hash())
            elif result.get("ok") and report.get("build"):
                self.best = report
                (self.directory / "best.rs").write_bytes(LIB.read_bytes())
                (self.directory / "best-report.json").write_text(json.dumps(report, indent=2))
        self.history.append({"role": "tool", "name": call.get("name"),
                             "content": json.dumps(result, ensure_ascii=False), "result": result})
        self.record(event="tool", step=self.step, name=call.get("name"), arguments=call.get("arguments"), result=result)
        if call.get("name") == "save_notes" and result.get("ok"):
            (self.directory / "notes.md").write_text(result["notes"])
        print(f"      -> {call.get('name')}: {'ok' if result.get('ok') else result.get('error', 'failed')}", flush=True)
        return result

    def execute(self, model=call_model) -> int:
        reason, failed = "", False
        try:
            self.observe({"name": "evaluate", "arguments": {}})
            while True:
                stop, reason = should_stop(self.history, self.step, self.budget, self.last_score)
                if stop:
                    break
                self.step += 1
                context = build_context(self.history, self.step)
                (self.directory / f"context-{self.step:02d}.json").write_text(json.dumps(context, indent=2))
                # Record before the request: even failed network calls consume budget.
                self.record(event="request", step=self.step, context_chars=sum(len(m["content"]) for m in context))
                revision = source_hash()
                reply = model(context, SCHEMAS)
                self.record(event="model", step=self.step, reply=reply)
                calls = reply.get("tool_calls") or []
                if not isinstance(calls, list) or any(not isinstance(c, dict) for c in calls):
                    raise ValueError("Model returned malformed tool_calls")
                self.history.append({"role": "assistant", "content": reply.get("text") or "",
                                     "tool_calls": calls, "source_hash": revision})
                if reply.get("text"):
                    print(f"[{self.step}] {reply['text'][:200]}", flush=True)
                for call in calls[:MAX_TOOL_CALLS]:
                    result = self.observe(call)
                    if result.get("rollback"):
                        # Subsequent edits in this batch may target the rejected source.
                        break
                if len(calls) > MAX_TOOL_CALLS:
                    self.record(event="batch_limit", step=self.step, skipped=len(calls) - MAX_TOOL_CALLS)
                (self.directory / "state.json").write_text(json.dumps({
                    "step": self.step, "budget": self.budget, "source_hash": source_hash(),
                    "last_score": self.last_score, "best_report": self.best}, indent=2))
        except (RuntimeError, OSError, ValueError, KeyboardInterrupt) as exc:
            reason = f"interrupted: {type(exc).__name__}: {exc}"
            failed = True
            self.record(event="error", step=self.step, error=reason)
        # Reuse fresh evidence; otherwise verify an edit made on the last call.
        final = next((result_of(m) for m in reversed(self.history) if m.get("name") == "evaluate"), {})
        if final.get("sha256") != source_hash() or final.get("rollback") or not final.get("ok"):
            final = self.observe({"name": "evaluate", "arguments": {}})
        if final.get("rollback"):
            final = self.observe({"name": "evaluate", "arguments": {}})
        if self.best is None and not final.get("report", {}).get("build"):
            original = (self.directory / "initial.rs").read_bytes()
            if LIB.read_bytes() != original:
                LIB.write_bytes(original)
                self.record(event="rollback", step=self.step, reason="no buildable checkpoint; restored initial source")
                final = self.observe({"name": "evaluate", "arguments": {}})
        (self.directory / "final-report.json").write_text(json.dumps(final, indent=2))
        complete = final.get("ok", False) and successful(final.get("report", {}))
        if complete and not failed:
            reason = "verified completion: build, tests, differential, precedence and quality passed"
        self.record(event="stop", reason=reason, steps=self.step, complete=complete, source_hash=source_hash())
        print(f"\n[stop] {reason}\ntrajectory: {self.log}\nfinal score: {self.last_score}")
        return 0 if complete and not failed else 1


def preflight() -> list[str]:
    problems = []
    for name in ("GEMINI_API_KEY", "GEMINI_MODEL"):
        if not os.environ.get(name):
            problems.append(f"Set {name} in the environment.")
    if shutil.which("cargo") is None:
        problems.append("Install Rust/Cargo (https://rustup.rs) and put cargo on PATH.")
    if importlib.util.find_spec("semver") is None:
        problems.append(f"Install the oracle: {sys.executable} -m pip install -r requirements.txt")
    for path in (PYSRC, LIB, RUST / "src" / "main.rs", HERE / "evaluate.py"):
        if not path.exists():
            problems.append(f"Missing {path.relative_to(HERE)}; fetch source or restore the scaffold.")
    return problems


def load_env_file(path: pathlib.Path) -> None:
    """Read explicitly selected credentials without evaluating shell code."""
    allowed = {"GEMINI_API_KEY", "GEMINI_MODEL"}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or name not in allowed:
            raise ValueError(f"Unsupported configuration on env-file line {number}")
        try:
            parts = shlex.split(value, comments=True)
        except ValueError:
            raise ValueError(f"Invalid quoting on env-file line {number}") from None
        if len(parts) > 1:
            raise ValueError(f"Quote values containing spaces on env-file line {number}")
        os.environ.setdefault(name, parts[0] if parts else "")


def fresh_run(budget: int, task: str) -> int:
    """Run the live model from a recreated scaffold without overwriting lib.rs."""
    parent = HERE / ".runs"
    parent.mkdir(exist_ok=True)
    workspace = pathlib.Path(tempfile.mkdtemp(prefix="live-", dir=parent))
    (workspace / "rust" / "src").mkdir(parents=True)
    shutil.copytree(HERE / "reference", workspace / "reference",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("agent.py", "evaluate.py"):
        shutil.copy2(HERE / name, workspace / name)
    for name in ("Cargo.toml", "Cargo.lock", "src/main.rs"):
        if (RUST / name).exists():
            shutil.copy2(RUST / name, workspace / "rust" / name)
    shutil.copy2(HERE / "fixtures" / "starter_lib.rs", workspace / "rust" / "src" / "lib.rs")
    print(f"Fresh workspace: {workspace}\nModel requested: {configured_model()}", flush=True)
    try:
        process = subprocess.run([sys.executable, str(workspace / "agent.py"),
                                  "--budget", str(budget), "--task", task], cwd=workspace)
    finally:
        # Archive complete evidence even when an API request fails.
        LOGS.mkdir(exist_ok=True)
        for path in (workspace / "logs").glob("run-*"):
            target = LOGS / path.name
            if path.is_dir():
                shutil.copytree(path, target)
                shutil.copy2(workspace / "agent.py", target / "agent-snapshot.py")
                (target / "provenance.json").write_text(json.dumps({
                    "mode": "live_api_fresh_scaffold", "workspace": str(workspace),
                    "requested_model": configured_model(),
                    "provider": "gemini",
                    "starter": "fixtures/starter_lib.rs; recreated original API and placeholders",
                }, indent=2))
            else:
                shutil.copy2(path, target)
        print(f"Run artifacts saved to {LOGS}; generated Rust remains in {workspace / 'rust/src/lib.rs'}", flush=True)
    return process.returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=40, help="maximum model calls (1..40)")
    parser.add_argument("--model", help="Gemini model identifier; overrides GEMINI_MODEL")
    parser.add_argument("--env-file", type=pathlib.Path, help="read API configuration from an ignored local file")
    parser.add_argument("--fresh", action="store_true", help="start from a stub in an isolated workspace and archive the live trajectory")
    parser.add_argument("--task", default="Translate reference/version.py into rust/src/lib.rs, including meaningful Rust tests.")
    parser.add_argument("--check", action="store_true", help="check prerequisites without model calls")
    args = parser.parse_args(argv)
    if not 1 <= args.budget <= MAX_CALLS:
        parser.error("--budget must be between 1 and 40")
    if args.env_file:
        try:
            load_env_file(args.env_file)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    if args.model:
        os.environ["GEMINI_MODEL"] = args.model
    problems = preflight()
    if problems:
        print("Setup required:\n" + "\n".join(f"  - {p}" for p in problems), file=sys.stderr)
        return 2
    if args.check:
        print("Ready: provider configuration, Cargo, oracle and source files found.")
        return 0
    if args.fresh:
        return fresh_run(args.budget, args.task)
    return Run(args.budget, args.task).execute()


if __name__ == "__main__":
    raise SystemExit(main())
