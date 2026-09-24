"""Offline agent regression tests. Model/compiler results are explicitly mocked."""
import contextlib
import copy
import io
import json
import os
import pathlib
import subprocess
import tempfile
import unittest
import urllib.error
from unittest.mock import Mock, patch

import agent


def report(passed=100, total=100):
    return {
        "build": True, "cargo_test_ok": True, "cargo_test": {"passed": 12, "failed": 0},
        "differential": {name: {"pass": passed, "total": total} for name in
                         ("parse valid", "parse invalid", "compare", "bump", "round-trip")},
        "differential_pct": round(100 * passed / total, 2),
        "spec_precedence_chain": True, "violations": [],
        "quality": {name: 0 for name in ("unsafe_blocks", "todo_macros", "panic_macros",
                                        "extra_dependencies", "clone_calls", "unwrap_calls")},
    }


def observation(name, result):
    return {"role": "tool", "name": name, "content": json.dumps(result), "result": result}


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = pathlib.Path(self.temp.name)
        (root / "rust" / "src").mkdir(parents=True)
        (root / "reference").mkdir()
        self.lib = root / "rust" / "src" / "lib.rs"
        self.lib.write_text("initial source\n")
        (self.lib.parent / "main.rs").write_text("protected harness\n")
        self.python = root / "reference" / "version.py"
        self.python.write_text("\n".join(f"line {i}" for i in range(1, 401)))
        self.patch = patch.multiple(agent, HERE=root, RUST=root / "rust", LIB=self.lib,
                                    PYSRC=self.python, LOGS=root / "logs")
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.history = [{"role": "system", "content": agent.system_prompt()},
                        {"role": "user", "content": "Translate the module."}]

    def evaluate(self, value):
        return observation("evaluate", {"ok": True, "sha256": agent.source_hash(), "report": value})

    def test_context_stays_bounded_and_preserves_notes_and_latest_error(self):
        history = self.history + [observation("save_notes", {"ok": True, "notes": "Keep build zeros."})]
        for _ in range(100):
            history += [{"role": "assistant", "content": "working", "tool_calls": [
                {"name": "write_rust", "arguments": {"content": "OLD_SOURCE" * 10_000}}]},
                observation("cargo_build", {"ok": False, "output": "compiler error\n" * 20_000})]
        history.append(observation("cargo_build", {"ok": False, "output": "LATEST_DIAGNOSTIC"}))
        context = agent.build_context(history, 39)
        content = "\n".join(m["content"] for m in context)
        self.assertLessEqual(sum(len(m["content"]) for m in context), agent.CONTEXT_CHARS)
        self.assertIn("Keep build zeros.", content)
        self.assertIn("LATEST_DIAGNOSTIC", content)
        self.assertNotIn("OLD_SOURCE", content)
        self.assertIn(agent.source_hash(), content)

    def test_source_reads_and_searches_are_bounded(self):
        self.assertIn("200: line 200", agent.t_read_python({"start": 200, "count": 2})["text"])
        self.assertNotIn("202:", agent.t_read_python({"start": 200, "count": 2})["text"])
        self.assertFalse(agent.dispatch({"name": "read_python", "arguments": {"count": 1000}})["ok"])
        search = agent.t_search_source({"query": "line 250"})
        self.assertEqual(search["matches"], 1)
        self.assertIn("250: line 250", search["text"])

    def test_tool_dispatch_known_unknown_and_malformed(self):
        self.assertTrue(agent.dispatch({"name": "read_rust", "arguments": {}})["ok"])
        for call in ({"name": "shell"}, {"name": []}, {"name": "write_rust", "arguments": []},
                     {"name": "write_rust", "arguments": {}},
                     {"name": "read_rust", "arguments": {"path": "../../secret"}}):
            with self.subTest(call=call):
                self.assertFalse(agent.dispatch(call)["ok"])

    def test_replace_checks_uniqueness_and_staleness(self):
        before = agent.source_hash()
        self.assertTrue(agent.dispatch({"name": "replace_rust", "arguments": {
            "old": "initial", "new": "updated", "sha256": before}})["ok"])
        self.assertFalse(agent.dispatch({"name": "replace_rust", "arguments": {
            "old": "updated", "new": "bad", "sha256": before}})["ok"])
        self.lib.write_text("repeat repeat")
        self.assertFalse(agent.dispatch({"name": "replace_rust", "arguments": {
            "old": "repeat", "new": "bad"}})["ok"])
        self.assertEqual((self.lib.parent / "main.rs").read_text(), "protected harness\n")

    def test_completion_requires_fresh_clean_nonempty_tests(self):
        history = self.history + [self.evaluate(report())]
        self.assertTrue(agent.should_stop(history, 2, 40, 100)[0])
        self.lib.write_text("changed after verification")
        self.assertFalse(agent.should_stop(history, 2, 40, 100)[0])
        for mutation in (lambda r: r.update(build=False),
                         lambda r: r.update(cargo_test_ok=False),
                         lambda r: r.update(cargo_test={"passed": 0, "failed": 0}),
                         lambda r: r.update(cargo_test={"passed": 2, "failed": 1}),
                         lambda r: r.update(spec_precedence_chain=False),
                         lambda r: r.update(violations=["unsafe_blocks"]),
                         lambda r: r["quality"].update(panic_macros=1),
                         lambda r: r["quality"].update(clone_calls=21)):
            value = report()
            mutation(value)
            self.assertFalse(agent.successful(value))

    def test_rounded_100_percent_does_not_hide_failure(self):
        value = report(99_999, 100_000)
        self.assertEqual(value["differential_pct"], 100.0)
        self.assertFalse(agent.successful(value))
        self.assertLess(agent.rank(value), agent.rank(report(100_000, 100_000)))

    def test_evaluate_checks_full_test_exit_status(self):
        def fake_process(cmd, **_kwargs):
            pathlib.Path(cmd[-1]).write_text(json.dumps(report()))
            return {"ok": True, "output": "evaluation complete", "returncode": 0}

        with patch("agent._run", side_effect=fake_process), \
                patch("agent.t_cargo_test", return_value={"ok": False, "returncode": 101, "output": "doctest failed"}):
            result = agent.t_evaluate({})
        self.assertFalse(agent.successful(result["report"]))
        self.assertEqual(result["test_failure"]["returncode"], 101)

    def test_failed_first_edit_restores_original_if_no_build_ever_succeeds(self):
        def failed_evaluation(_args):
            return {"ok": False, "sha256": agent.source_hash(), "report": {"build": False}}

        with patch.dict(agent.BY_NAME["evaluate"], fn=failed_evaluation), contextlib.redirect_stdout(io.StringIO()):
            run = agent.Run(1, "Mocked initially broken build test")
            self.assertEqual(run.execute(lambda *_: {"text": None, "tool_calls": [
                {"name": "write_rust", "arguments": {"content": "still broken"}}]}), 1)
        self.assertEqual(self.lib.read_text(), "initial source\n")

    def test_budget_no_actions_repeated_actions_and_plateau(self):
        self.assertIn("budget", agent.should_stop(self.history, 40, 500, None)[1])
        chatting = self.history + [{"role": "assistant", "content": "done", "tool_calls": []}] * 3
        self.assertIn("without tool", agent.should_stop(chatting, 3, 40, None)[1])
        turn = {"role": "assistant", "content": "", "source_hash": "same",
                "tool_calls": [{"name": "cargo_build", "arguments": {}}]}
        repeated = self.history + [copy.deepcopy(turn) for _ in range(3)]
        self.assertIn("identical", agent.should_stop(repeated, 3, 40, None)[1])
        repeated[-1]["source_hash"] = "new edit"
        self.assertFalse(agent.should_stop(repeated, 3, 40, None)[0])
        plateau = self.history + [self.evaluate(report(80)) for _ in range(5)]
        self.assertIn("without improvement", agent.should_stop(plateau, 8, 40, 80)[1])

    def fake_evaluation(self, _args):
        passed = {"initial source\n": 50, "good source": 100, "bad source": 30}.get(self.lib.read_text(), 60)
        return {"ok": True, "sha256": agent.source_hash(), "report": report(passed)}

    def test_full_loop_success_and_complete_trajectory(self):
        replies = iter([
            {"text": "Read first.", "tool_calls": [{"name": "read_rust", "arguments": {}}]},
            {"text": "Implement.", "tool_calls": [
                {"name": "write_rust", "arguments": {"content": "good source"}},
                {"name": "evaluate", "arguments": {}}]},
        ])
        with patch.dict(agent.BY_NAME["evaluate"], fn=self.fake_evaluation), contextlib.redirect_stdout(io.StringIO()):
            run = agent.Run(4, "Mocked offline integration test")
            status = run.execute(lambda *_: next(replies))
        self.assertEqual(status, 0)
        events = [json.loads(line) for line in run.log.read_text().splitlines()]
        self.assertEqual(events[0]["event"], "start")
        self.assertEqual(events[-1]["event"], "stop")
        self.assertTrue(events[-1]["complete"])
        self.assertEqual(events[-1]["steps"], 2)
        self.assertEqual(len([e for e in events if e["event"] == "request"]), 2)
        self.assertTrue((run.directory / "context-02.json").exists())
        self.assertEqual((run.directory / "best.rs").read_text(), "good source")

    def test_budget_end_checks_unverified_edit_and_restores_best(self):
        with patch.dict(agent.BY_NAME["evaluate"], fn=self.fake_evaluation), contextlib.redirect_stdout(io.StringIO()):
            run = agent.Run(1, "Mocked regression test")
            status = run.execute(lambda *_: {"text": None, "tool_calls": [
                {"name": "write_rust", "arguments": {"content": "bad source"}}]})
        self.assertEqual(status, 1)
        self.assertEqual(run.step, 1)
        self.assertEqual(self.lib.read_text(), "initial source\n")
        events = [json.loads(line) for line in run.log.read_text().splitlines()]
        self.assertTrue(any(e["event"] == "rollback" for e in events))

    def test_rollback_discards_remaining_stale_batch_edits(self):
        with patch.dict(agent.BY_NAME["evaluate"], fn=self.fake_evaluation), contextlib.redirect_stdout(io.StringIO()):
            run = agent.Run(1, "Mocked stale batch test")
            run.execute(lambda *_: {"text": None, "tool_calls": [
                {"name": "write_rust", "arguments": {"content": "bad source"}},
                {"name": "evaluate", "arguments": {}},
                {"name": "write_rust", "arguments": {"content": "should never be written"}}]})
        self.assertEqual(self.lib.read_text(), "initial source\n")
        events = [json.loads(line) for line in run.log.read_text().splitlines()]
        self.assertEqual(len([e for e in events if e["event"] == "tool" and e["name"] == "write_rust"]), 1)

    def test_api_failure_still_logs_one_call_and_final_evaluation(self):
        with patch.dict(agent.BY_NAME["evaluate"], fn=self.fake_evaluation), contextlib.redirect_stdout(io.StringIO()):
            run = agent.Run(40, "Mocked provider failure test")
            status = run.execute(Mock(side_effect=RuntimeError("Provider unavailable")))
        self.assertEqual(status, 1)
        self.assertEqual(run.step, 1)
        self.assertTrue((run.directory / "final-report.json").exists())
        events = [json.loads(line) for line in run.log.read_text().splitlines()]
        self.assertEqual(events[-1]["event"], "stop")
        self.assertIn("Provider unavailable", events[-1]["reason"])

    def test_process_errors_are_structured(self):
        with patch("agent.subprocess.run", side_effect=FileNotFoundError("cargo missing")):
            self.assertFalse(agent.t_cargo_build({})["ok"])
        with patch("agent.subprocess.run", side_effect=subprocess.TimeoutExpired("cargo", 1, output=b"diagnostic")):
            result = agent.t_cargo_test({})
        self.assertFalse(result["ok"])
        self.assertEqual(result["output"], "diagnostic")

    def test_cli_rejects_budget_above_cap(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            agent.main(["--budget", "41"])
        self.assertEqual(error.exception.code, 2)

    def test_env_file_is_data_and_preserves_exported_overrides(self):
        path = agent.HERE / '.env'
        path.write_text('OPENAI_API_KEY="dummy-key"\nOPENAI_MODEL=file-model\n'
                        'OPENAI_BASE_URL="https://example.test/v1" # comment\n')
        with patch.dict(os.environ, {"OPENAI_MODEL": "explicit-model"}, clear=True):
            agent.load_env_file(path)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "dummy-key")
            self.assertEqual(os.environ["OPENAI_MODEL"], "explicit-model")
            self.assertEqual(os.environ["OPENAI_BASE_URL"], "https://example.test/v1")
        path.write_text('OPENAI_MODEL="$(touch should-not-exist)"\n')
        with patch.dict(os.environ, {}, clear=True):
            agent.load_env_file(path)
            self.assertEqual(os.environ["OPENAI_MODEL"], "$(touch should-not-exist)")
        self.assertFalse((agent.HERE / 'should-not-exist').exists())

    def test_env_file_errors_do_not_echo_credentials(self):
        path = agent.HERE / '.env'
        for content in ('OPENAI_API_KEY="secret-value\n', 'UNKNOWN=secret-value\n'):
            path.write_text(content)
            with self.assertRaises(ValueError) as error:
                agent.load_env_file(path)
            self.assertNotIn('secret-value', str(error.exception))

    def test_fresh_run_preserves_solution_and_archives_child_trajectory(self):
        for name in ('agent.py', 'evaluate.py'):
            (agent.HERE / name).write_text('# mock test copy\n')
        (agent.RUST / 'Cargo.toml').write_text('[package]\n')
        (agent.HERE / 'fixtures').mkdir()
        (agent.HERE / 'fixtures/starter_lib.rs').write_text('starter stub\n')
        (agent.HERE / '.env').write_text('OPENAI_API_KEY=do-not-copy\n')

        def child(cmd, cwd):
            self.assertEqual((cwd / 'rust/src/lib.rs').read_text(), 'starter stub\n')
            self.assertFalse((cwd / '.env').exists())
            self.assertNotIn('--fresh', cmd)
            self.assertIn('40', cmd)
            logs = cwd / 'logs'
            logs.mkdir()
            (logs / 'run-test.jsonl').write_text('{"event":"mock"}\n')
            (logs / 'run-test').mkdir()
            (logs / 'run-test/best.rs').write_text('mock generated solution\n')
            (cwd / 'rust/src/lib.rs').write_text('mock generated solution\n')
            return subprocess.CompletedProcess(cmd, 0)

        with patch('agent.subprocess.run', side_effect=child), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(agent.fresh_run(40, 'mock fresh-run test'), 0)
        self.assertEqual(self.lib.read_text(), 'initial source\n')
        self.assertTrue((agent.LOGS / 'run-test.jsonl').exists())
        self.assertEqual((agent.LOGS / 'run-test/best.rs').read_text(), 'mock generated solution\n')
        self.assertTrue((agent.LOGS / 'run-test/agent-snapshot.py').exists())


class ModelTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"MODEL_PROVIDER": "openai"})
        environment.start()
        self.addCleanup(environment.stop)

    def test_native_gemini_request_and_verified_model_metadata(self):
        body = {"modelVersion": "gemini-2.5-flash", "responseId": "google-test-response",
                "usageMetadata": {"promptTokenCount": 123}, "candidates": [{
                    "finishReason": "STOP", "content": {"parts": [
                        {"text": "Inspecting source"},
                        {"functionCall": {"name": "read_python", "args": {"start": 100, "count": 20}}},
                    ]}}]}
        with patch.dict(os.environ, {"MODEL_PROVIDER": "gemini", "GEMINI_API_KEY": "google-test-key",
                                     "GEMINI_MODEL": "gemini-2.5-flash"}, clear=True), \
                patch("agent.urllib.request.urlopen", return_value=io.StringIO(json.dumps(body))) as request:
            reply = agent.call_model([{"role": "system", "content": "rules"},
                                      {"role": "user", "content": "task"}], agent.SCHEMAS)
        sent = request.call_args.args[0]
        self.assertEqual(sent.full_url, 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent')
        self.assertEqual(sent.get_header('X-goog-api-key'), 'google-test-key')
        self.assertNotIn('google-test-key', sent.data.decode())
        payload = json.loads(sent.data)
        self.assertEqual(payload['systemInstruction']['parts'][0]['text'], 'rules')
        declarations = payload['tools'][0]['functionDeclarations']
        self.assertNotIn('additionalProperties', declarations[0]['parameters'])
        no_args = next(d for d in declarations if d['name'] == 'evaluate')
        self.assertNotIn('parameters', no_args)
        self.assertEqual(reply['tool_calls'][0]['arguments']['start'], 100)
        self.assertEqual(reply['model'], 'gemini-2.5-flash')
        self.assertEqual(reply['response_id'], 'google-test-response')
        self.assertEqual(reply['provider'], 'gemini')
        self.assertEqual(request.call_count, 1)

    def test_gemini_incomplete_and_blocked_responses_are_not_executed(self):
        for body in ({'candidates': []}, {'candidates': [{'finishReason': 'MAX_TOKENS',
                     'content': {'parts': [{'functionCall': {'name': 'write_rust', 'args': {'content': 'partial'}}}]}}]}):
            with patch.dict(os.environ, {'MODEL_PROVIDER': 'gemini', 'GEMINI_API_KEY': 'test-key',
                                         'GEMINI_MODEL': 'test-model'}, clear=True), \
                    patch('agent.urllib.request.urlopen', return_value=io.StringIO(json.dumps(body))), \
                    self.assertRaises(RuntimeError):
                agent.call_model([], [])

    def test_json_provider_error_redacts_key(self):
        body = {'error': {'message': 'Invalid key: secret-test-key'}}
        error = urllib.error.HTTPError('https://example.test', 400, 'Bad request', {},
                                       io.BytesIO(json.dumps(body).encode()))
        with patch('agent.urllib.request.urlopen', side_effect=error), self.assertRaises(RuntimeError) as raised:
            agent.request_json(None, 'secret-test-key')
        self.assertNotIn('secret-test-key', str(raised.exception))
        self.assertIn('[redacted]', str(raised.exception))

    def test_responses_request_and_return_shape(self):
        body = {"status": "completed", "id": "resp_test", "model": "returned-model-snapshot",
                "usage": {"input_tokens": 123}, "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": "Inspecting source"}]},
            {"type": "function_call", "name": "read_python", "arguments": '{"start": 100, "count": 20}'}]}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"}, clear=True), \
                patch("agent.urllib.request.urlopen", return_value=io.StringIO(json.dumps(body))) as request:
            reply = agent.call_model([{"role": "system", "content": "rules"},
                                      {"role": "user", "content": "task"}], agent.SCHEMAS)
        sent = request.call_args.args[0]
        payload = json.loads(sent.data)
        self.assertEqual(sent.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(payload["instructions"], "rules")
        self.assertFalse(payload["store"])
        self.assertFalse(payload["parallel_tool_calls"])
        self.assertEqual(payload["tools"][0]["type"], "function")
        self.assertEqual(reply["tool_calls"][0]["arguments"]["start"], 100)
        self.assertEqual(reply["text"], "Inspecting source")
        self.assertEqual(reply["model"], "returned-model-snapshot")
        self.assertEqual(reply["requested_model"], "test-model")
        self.assertEqual(request.call_count, 1)

    def test_malformed_arguments_are_returned_as_recoverable_tool_error(self):
        body = {"output": [{"type": "function_call", "name": "read_rust", "arguments": "{bad json"}]}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "key", "OPENAI_MODEL": "model"}), \
                patch("agent.urllib.request.urlopen", return_value=io.StringIO(json.dumps(body))):
            reply = agent.call_model([], [])
        self.assertFalse(agent.dispatch(reply["tool_calls"][0])["ok"])

    def test_http_error_is_redacted_and_never_retried(self):
        error = urllib.error.HTTPError("https://example.test", 401, "secret-key", {}, io.BytesIO(b"secret-key"))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret-key", "OPENAI_MODEL": "model"}), \
                patch("agent.urllib.request.urlopen", side_effect=error) as request, \
                self.assertRaises(RuntimeError) as raised:
            agent.call_model([], [])
        self.assertIn("401", str(raised.exception))
        self.assertNotIn("secret-key", str(raised.exception))
        self.assertEqual(request.call_count, 1)

    def test_incomplete_output_cannot_be_written(self):
        body = {"status": "incomplete", "output": [{"type": "function_call", "name": "write_rust"}]}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "key", "OPENAI_MODEL": "model"}), \
                patch("agent.urllib.request.urlopen", return_value=io.StringIO(json.dumps(body))), \
                self.assertRaises(RuntimeError):
            agent.call_model([], [])


if __name__ == "__main__":
    unittest.main()
