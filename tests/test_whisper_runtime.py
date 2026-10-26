import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import whisper_runtime as runtime


class _Completed:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


class _FakeGuard:
    _next_pid = 3000

    def __init__(self, poll_value=None):
        self.pid = _FakeGuard._next_pid
        _FakeGuard._next_pid += 1
        self._poll = poll_value
        self.stdin = MagicMock()
        self.stdout = MagicMock()
        self.stdout.fileno.return_value = 0
        self.terminated = False
        self.waited = False

    def poll(self):
        return self._poll

    def wait(self, timeout=None):
        self.waited = True
        self._poll = 0
        return 0

    def terminate(self):
        self.terminated = True
        self._poll = 0


class WhisperRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.cli = root / "whisper-cli"
        self.server = root / "whisper-server"
        self.model_small = root / "ggml-small.bin"
        self.model_medium = root / "ggml-medium.bin"
        self.wav = root / "speech.wav"
        for path, contents in (
            (self.cli, b"cli-v1"), (self.server, b"server-v1"),
            (self.model_small, b"small-v1"), (self.model_medium, b"medium-v1"),
            (self.wav, b"RIFF" + b"audio"),
        ):
            path.write_bytes(contents)
            path.chmod(0o755)
        self.patches = [
            patch.object(runtime, "_KNOWN_BUILD", {
                "cli": hashlib.sha256(b"cli-v1").hexdigest(),
                "server": hashlib.sha256(b"server-v1").hexdigest(),
                "library": hashlib.sha256(b"library-v1").hexdigest(),
            }),
            patch.object(runtime.platform, "machine", return_value="arm64"),
            patch.object(runtime, "_stat_signature", side_effect=self._signature),
        ]
        for item in self.patches:
            item.start()
        self.addCleanup(self._stop_patches)

    def _stop_patches(self):
        for item in reversed(self.patches):
            item.stop()
        self.tempdir.cleanup()

    def _signature(self, path):
        path = Path(path)
        if not path.is_file():
            return {"path": str(path.absolute()), "missing": True}
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "path": str(path.absolute()), "resolved": str(path.resolve()),
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "inode": stat.st_ino,
            "device": stat.st_dev, "sha256": digest,
        }

    def _runtime(self, **kwargs):
        runtime_instance = runtime.WhisperRuntime(self.cli, self.server, **kwargs)
        runtime_instance._lsof = "/usr/sbin/lsof"
        def binary_identity():
            return {
                "cli": self._signature(self.cli),
                "server": self._signature(self.server),
                "libraries": {"whisper": {"sha256": hashlib.sha256(b"library-v1").hexdigest()}},
                "environment": {name: "" for name in (
                    "WHISPER_ARG_DEVICE", "GGML_BACKEND_PATH", "GGML_METAL_PATH_RESOURCES",
                    "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
                    "LD_LIBRARY_PATH", "LD_PRELOAD",
                )},
            }
        runtime_instance._binary_identity = binary_identity
        self.addCleanup(runtime_instance.close)
        return runtime_instance

    def _server(self, model, result=None):
        guard = _FakeGuard()
        instance = runtime._Server(guard, 12001, "/qwen-asr-test", self._signature(model), guard.pid + 1)
        instance.result = result or {"text": "服务识别"}
        return instance

    def test_request_has_no_expected_text_and_explicitly_resets_prompt_decoder(self):
        body, content_type = runtime._multipart(b"wav-bytes")
        self.assertTrue(content_type.startswith("multipart/form-data; boundary="))
        self.assertIn(b'name="prompt"\r\n\r\n\r\n', body)
        self.assertIn(b'name="carry_initial_prompt"\r\n\r\nfalse\r\n', body)
        self.assertIn(b'name="no_timestamps"\r\n\r\ntrue\r\n', body)
        self.assertIn(b'name="token_timestamps"\r\n\r\nfalse\r\n', body)
        self.assertIn(b'name="best_of"\r\n\r\n5\r\n', body)
        self.assertIn(b'name="beam_size"\r\n\r\n5\r\n', body)
        self.assertNotIn(b"expected", body.lower())
        self.assertNotIn(b"transcript", body.lower())

    def test_identity_is_json_serializable_and_changes_when_binary_changes(self):
        instance = self._runtime()
        first = instance.identity()
        json.dumps(first)
        self.assertEqual(first["compatibility"], "reviewed-homebrew-arm64-1.9.2")
        self.assertEqual(first["request_fields"]["prompt"], "")
        self.server.write_bytes(b"server-v2-with-a-new-size")
        second = instance.identity()
        self.assertNotEqual(first["binaries"], second["binaries"])
        self.assertEqual(instance.stats["server_stops"], 0)

    def test_cli_emergency_disable_does_not_start_server(self):
        instance = self._runtime()
        with patch.dict(os.environ, {"QWEN3_TTS_ASR_BACKEND": "cli"}), \
                patch.object(instance, "_cli", return_value=" CLI 输出\n") as cli:
            self.assertEqual(instance.transcribe(self.wav, self.model_small), " CLI 输出\n")
        cli.assert_called_once_with(self.wav, self.model_small)
        self.assertEqual(instance.stats["server_starts"], 0)
        self.assertEqual(instance.stats["fallback_reasons"], {"cli_requested": 1})

    def test_model_servers_are_distinct_and_never_more_than_two(self):
        instance = self._runtime()
        small = self._server(self.model_small)
        medium = self._server(self.model_medium)
        starts = [small, medium]
        with patch.object(instance, "_start_server", side_effect=starts) as start, \
                patch.object(instance, "_owns_listener", return_value=True), \
                patch.object(instance, "_http_json", side_effect=lambda server, *_args, **_kwargs: server.result), \
                patch.object(instance, "_stop_server"):
            self.assertEqual(instance.transcribe(self.wav, self.model_small), "服务识别")
            self.assertEqual(instance.transcribe(self.wav, self.model_medium), "服务识别")
            self.assertEqual(set(instance._servers), {
                str(self.model_small.absolute()), str(self.model_medium.absolute())
            })
            self.assertEqual(start.call_count, 2)
            third = self.tempdir.name + "/ggml-large.bin"
            Path(third).write_bytes(b"large")
            with patch.object(instance, "_cli", return_value="CLI") as cli:
                self.assertEqual(instance.transcribe(self.wav, third), "CLI")
            cli.assert_called_once_with(self.wav, third)
            self.assertEqual(instance.stats["fallback_reasons"], {"model_capacity": 1})

    def test_empty_result_disables_only_that_model_and_falls_back(self):
        instance = self._runtime()
        bad = self._server(self.model_small, {"text": "   "})
        with patch.object(instance, "_start_server", return_value=bad), \
                patch.object(instance, "_owns_listener", return_value=True), \
                patch.object(instance, "_http_json", return_value=bad.result), \
                patch.object(instance, "_stop_server") as stop, \
                patch.object(instance, "_cli", return_value="CLI") as cli:
            self.assertEqual(instance.transcribe(self.wav, self.model_small), "CLI")
            self.assertEqual(instance.transcribe(self.wav, self.model_small), "CLI")
        self.assertEqual(stop.call_count, 1)
        self.assertEqual(cli.call_count, 2)
        self.assertEqual(instance.stats["fallback_reasons"], {"server_failure": 1, "model_server_disabled": 1})
        self.model_small.write_bytes(b"small-v2")
        with patch.object(instance, "_start_server", return_value=bad), \
                patch.object(instance, "_owns_listener", return_value=True), \
                patch.object(instance, "_http_json", return_value={"text": "恢复"}):
            self.assertEqual(instance.transcribe(self.wav, self.model_small), "恢复")

    def test_timeout_and_crash_fall_back_and_remove_bad_instance(self):
        instance = self._runtime()
        bad = self._server(self.model_small)
        errors = [TimeoutError("deadline"), runtime._ServerError("crashed")]
        with patch.object(instance, "_start_server", return_value=bad), \
                patch.object(instance, "_owns_listener", return_value=True), \
                patch.object(instance, "_http_json", side_effect=errors), \
                patch.object(instance, "_stop_server") as stop, \
                patch.object(instance, "_cli", return_value="CLI") as cli:
            self.assertEqual(instance.transcribe(self.wav, self.model_small), "CLI")
            self.assertEqual(instance.transcribe(self.wav, self.model_small), "CLI")
        self.assertEqual(stop.call_count, 1)
        self.assertEqual(cli.call_count, 2)
        self.assertEqual(instance.stats["server_failures"], 1)

    def test_missing_or_unreadable_model_does_not_attempt_server_start(self):
        instance = self._runtime()
        for signature, reason in (
            ({"path": str(self.model_small), "missing": True}, "model_missing"),
            ({"path": str(self.model_small), "unreadable": True}, "model_signature_unavailable"),
        ):
            with self.subTest(reason=reason), \
                    patch.object(instance, "_signature", return_value=signature), \
                    patch.object(instance, "_start_server") as start, \
                    patch.object(instance, "_cli", return_value=None):
                self.assertIsNone(instance.transcribe(self.wav, self.model_small))
                self.assertEqual(instance.stats["fallback_reasons"][reason], 1)
                start.assert_not_called()

    def test_corrupt_results_and_dead_server_fall_back_without_publishing_text(self):
        for result in ({}, {"text": 42}, {"text": "错误\ufffd"}, {"text": "好", "error": "bad"}, "dead"):
            with self.subTest(result=result):
                instance = self._runtime()
                server = self._server(self.model_small)
                if result == "dead":
                    server.guard._poll = 1
                with patch.object(instance, "_start_server", return_value=server), \
                        patch.object(instance, "_http_json", return_value=result) as http, \
                        patch.object(instance, "_stop_server") as stop, \
                        patch.object(instance, "_cli", return_value="CLI") as cli:
                    self.assertEqual(instance.transcribe(self.wav, self.model_small), "CLI")
                    stop.assert_called_once_with(server)
                    cli.assert_called_once()
                    self.assertFalse(instance._servers)
                    if result == "dead":
                        http.assert_not_called()

    def test_listener_ownership_is_checked_before_opening_or_sending_audio(self):
        instance = self._runtime()
        server = self._server(self.model_small)
        with patch.object(instance, "_owns_listener", return_value=False), \
                patch.object(runtime.http.client, "HTTPConnection") as connection:
            with self.assertRaisesRegex(runtime._ServerError, "not owned"):
                instance._http_json(server, "/inference", b"private audio", "audio/wav")
            connection.assert_not_called()
        with patch.object(instance, "_owns_listener", side_effect=[True, False]), \
                patch.object(runtime.http.client, "HTTPConnection") as connection:
            with self.assertRaisesRegex(runtime._ServerError, "ownership changed"):
                instance._http_json(server, "/inference", b"private audio", "audio/wav")
            connection.return_value.request.assert_not_called()
            connection.return_value.close.assert_called()

    def test_http_rejects_redirects_wrong_type_identity_and_malformed_json(self):
        instance = self._runtime()
        server = self._server(self.model_small)
        for status, headers, body in (
            (302, {"Content-Type": "application/json", "Server": "whisper.cpp"}, b"{}"),
            (200, {"Content-Type": "text/html", "Server": "whisper.cpp"}, b"{}"),
            (200, {"Content-Type": "application/json", "Server": "other"}, b"{}"),
            (200, {"Content-Type": "application/json", "Server": "whisper.cpp"}, b"not-json"),
            (200, {"Content-Type": "application/json", "Server": "whisper.cpp"}, b'{"text":"a","text":"b"}'),
        ):
            with self.subTest(status=status, headers=headers, body=body), \
                    patch.object(instance, "_owns_listener", return_value=True), \
                    patch.object(runtime.http.client, "HTTPConnection") as connection:
                response = connection.return_value.getresponse.return_value
                response.status = status
                response.getheader.side_effect = lambda key, default="": headers.get(key, default)
                response.read.return_value = body
                with self.assertRaises((runtime._ServerError, ValueError)):
                    instance._http_json(server, "/inference", b"audio", "audio/wav")
                connection.assert_called_once_with("127.0.0.1", server.port, timeout=45.0)
                connection.return_value.close.assert_called()

    def test_context_exception_closes_pipes_and_waits_for_guardian(self):
        instance = self._runtime()
        server = self._server(self.model_small)
        with self.assertRaisesRegex(RuntimeError, "job failed"):
            with instance:
                instance._servers[str(self.model_small)] = server
                raise RuntimeError("job failed")
        server.guard.stdin.close.assert_called_once()
        server.guard.stdout.close.assert_called_once()
        self.assertTrue(server.guard.waited)
        self.assertEqual(instance.stats["server_stops"], 1)
        self.assertEqual(instance.stats["cleanup_failures"], 0)

    def test_guardian_reaps_child_when_parent_dies(self):
        child = _FakeGuard()
        with patch.object(runtime.signal, "signal"), \
                patch.object(runtime.os, "getppid", side_effect=[1234, 1]), \
                patch.object(runtime.subprocess, "Popen", return_value=child), \
                patch.object(runtime.sys, "stdout"):
            self.assertEqual(runtime._guard_main(["owned-whisper"], 1234), 0)
        self.assertTrue(child.terminated)
        self.assertTrue(child.waited)

    def test_resource_cleanup_only_closes_owned_servers(self):
        instance = self._runtime()
        owned = self._server(self.model_small)
        instance._servers[str(self.model_small.absolute())] = owned
        with patch.object(instance, "_stop_server") as stop:
            instance.close()
            instance.close()
        stop.assert_called_once_with(owned)
        self.assertTrue(instance._closed)

    def test_cli_failure_and_missing_model_remain_none(self):
        instance = self._runtime(enabled=False)
        with patch.object(runtime.subprocess, "run", return_value=_Completed(1, "bad")):
            self.assertIsNone(instance.transcribe(self.wav, self.model_small))
        missing = self.tempdir.name + "/missing.bin"
        with patch.object(instance, "_cli", return_value=None) as cli:
            self.assertIsNone(instance.transcribe(self.wav, missing))
        cli.assert_called_once_with(self.wav, missing)


if __name__ == "__main__":
    unittest.main()
