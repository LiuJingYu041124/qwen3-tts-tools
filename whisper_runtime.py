"""Task-owned, unprompted whisper.cpp ASR with a fail-closed CLI fallback.

The reviewed protocol is upstream v1.9.2 (commit 306c88f4d1286aec1bf96e544632897886af5501):
https://github.com/ggml-org/whisper.cpp/blob/v1.9.2/examples/server/server.cpp
https://github.com/ggml-org/whisper.cpp/blob/v1.9.2/examples/cli/cli.cpp
https://github.com/ggml-org/whisper.cpp/blob/v1.9.2/src/whisper.cpp

server.cpp:480-628 defines multipart fields; 804-833 copies defaults per request;
109 and 955 force no_context; whisper.cpp:6819-6924 clears results, reseeds each
sampler and clears both prompt histories on every call. CLI's beam/best_of are
5/5, NOT server's -1/2. token_timestamps=false also makes server's implicit
max_len=60 inert. No transcript, expected text or nonempty prompt is accepted.

Only the locally audited Homebrew arm64 1.9.2 binaries/library are enabled.
Other installations remain usable through the original CLI, without guessing
that their HTTP API or default decoder semantics are compatible.
"""

from __future__ import annotations

import atexit
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import select
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional


_POLICY = "whisper-task-server-v1-reviewed-1.9.2"
_KNOWN_BUILD = {
    "cli": "40bca494d49af736058eb3f33cbcebaa020eacf6d0087b623f334946e1ab2128",
    "server": "e140a952c9267a97c1892c2fab746c10d28f0ea25dc95976356fdd5719864c3a",
    "library": "1013710cf1962a05006988a354890778979d3582ccb1e3f4628f0ae6d9001198",
}
_CLI_TIMEOUT = 45.0
_START_TIMEOUT = 45.0
_REQUEST_TIMEOUT = 45.0
_MAX_RESPONSE = 1024 * 1024
_MAX_AUDIO = 64 * 1024 * 1024
_GUARD_FLAG = "--whisper-runtime-guard"

# no_context is not an HTTP field in this release. Its compiled-in true value
# is part of the binary allowlist above, rather than an ignored request option.
_REQUEST_FIELDS = {
    "offset_t": "0", "offset_n": "0", "duration": "0",
    "max_context": "-1", "max_len": "0", "audio_ctx": "0",
    "best_of": "5", "beam_size": "5",
    "word_thold": "0.01", "entropy_thold": "2.4",
    "logprob_thold": "-1.0", "no_speech_thold": "0.6",
    "temperature": "0.0", "temperature_inc": "0.2",
    "language": "zh", "detect_language": "false", "translate": "false",
    "prompt": "", "carry_initial_prompt": "false",
    "no_timestamps": "true", "token_timestamps": "false",
    "diarize": "false", "tinydiarize": "false", "split_on_word": "false",
    "debug_mode": "false", "suppress_nst": "false", "vad": "false",
    "response_format": "json", "no_language_probabilities": "true",
}


class _ServerError(RuntimeError):
    pass


def _path(value):
    return str(Path(value).expanduser().absolute())


def _executable(value):
    value = os.fspath(value)
    return _path(shutil.which(value) or value)


def _stat_signature(path):
    requested = _path(path)
    try:
        resolved = Path(requested).resolve(strict=True)
        stat = resolved.stat()
        if not resolved.is_file():
            raise OSError("not a regular file")
        return {
            "path": requested, "resolved": str(resolved),
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "inode": stat.st_ino,
            "device": stat.st_dev,
        }
    except (OSError, RuntimeError):
        return {"path": requested, "missing": True}


def _json_key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _ServerError("duplicate JSON field")
        result[key] = value
    return result


def _multipart(audio):
    boundary = "qwen-asr-" + os.urandom(18).hex()
    chunks = []
    for key, value in _REQUEST_FIELDS.items():
        chunks.append((
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"'
            f"\r\n\r\n{value}\r\n"
        ).encode("ascii"))
    chunks.extend([
        (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
         'filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode("ascii"),
        audio, f"\r\n--{boundary}--\r\n".encode("ascii"),
    ])
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _terminate_child(process):
    """Only the guardian that owns the Popen object signals the ASR process."""
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _guard_main(command, parent_pid):
    # The guardian survives a worker process-group kill. Parent reparenting is
    # also checked, so a forked worker retaining the pipe cannot hide its death.
    stopping = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda _signum, _frame: stopping.set())
    child = None
    try:
        if os.getppid() != parent_pid:
            return 1
        child = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, close_fds=True,
        )
        sys.stdout.write(json.dumps({"pid": child.pid}) + "\n")
        sys.stdout.flush()
        while not stopping.is_set() and os.getppid() == parent_pid:
            if child.poll() is not None:
                return 1
            ready, _, _ = select.select([sys.stdin.fileno()], [], [], 0.2)
            if ready and not os.read(sys.stdin.fileno(), 1):
                break
        return 0
    finally:
        if child is not None:
            _terminate_child(child)


@dataclass
class _Server:
    guard: object
    port: int
    prefix: str
    model: dict
    pid: int = 0
    closed: bool = False


class WhisperRuntime:
    """Serialize ASR calls and lazily own at most two model-specific servers.

    ``identity()`` is stable across lazy starts/fallbacks and includes strategy,
    binaries, shared-library signatures and relevant environment. An audio cache
    must additionally key the requested model's file signature (the worker
    already does this). Models are re-statted on every transcribe call.
    ``stats`` contains counts and seconds, never audio or recognized text.
    """

    def __init__(self, cli_path, server_path=None, enabled=True):
        self.cli_path = _executable(cli_path)
        self.server_path = _executable(
            server_path if server_path is not None
            else str(Path(self.cli_path).with_name("whisper-server"))
        )
        self.enabled = bool(enabled)
        self.stats = {
            "server_starts": 0, "server_start_failures": 0,
            "server_requests": 0, "server_failures": 0, "server_stops": 0,
            "cli_requests": 0, "cli_failures": 0, "fallbacks": 0,
            "disabled_models": 0, "cleanup_failures": 0,
            "server_start_seconds": 0.0, "server_request_seconds": 0.0,
            "cli_seconds": 0.0, "total_seconds": 0.0,
            "fallback_reasons": {},
        }
        self._owner_pid = os.getpid()
        self._lock = threading.RLock()
        self._servers = {}
        self._disabled_models = {}
        self._digest_cache = {}
        self._binaries_key = None
        self._closed = False
        self._lsof = shutil.which("lsof") or (
            "/usr/sbin/lsof" if Path("/usr/sbin/lsof").is_file() else None
        )
        atexit.register(self.close)

    def __enter__(self):
        if self._closed:
            raise RuntimeError("WhisperRuntime is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _signature(self, path):
        signature = _stat_signature(path)
        if signature.get("missing"):
            return signature
        key = _json_key(signature)
        cached = self._digest_cache.get(signature["path"])
        if cached is not None and cached[0] == key:
            return dict(cached[1])
        try:
            digest = hashlib.sha256()
            with open(signature["resolved"], "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if signature != _stat_signature(path):
                return {"path": signature["path"], "unstable": True}
            signature["sha256"] = digest.hexdigest()
            self._digest_cache[signature["path"]] = (key, dict(signature))
        except OSError:
            signature["unreadable"] = True
        return signature

    def _binary_identity(self):
        cli = self._signature(self.cli_path)
        server = self._signature(self.server_path)
        cli_install = Path(cli.get("resolved", self.cli_path)).parent.parent
        server_install = Path(server.get("resolved", self.server_path)).parent.parent
        libraries = {}
        if cli_install == server_install:
            library = cli_install / "lib" / "libwhisper.1.dylib"
            libraries["whisper"] = self._signature(library)
            # Homebrew's dynamically loaded ggml backends can change while the
            # executable remains untouched; they also invalidate resident models.
            if cli_install.parent.parent.name == "Cellar":
                ggml = cli_install.parent.parent.parent / "opt" / "ggml"
                for folder, pattern in (("lib", "*.dylib"), ("libexec", "*.so")):
                    for dependency in sorted((ggml / folder).glob(pattern)):
                        libraries[str(dependency)] = self._signature(dependency)
        environment = {
            name: os.environ.get(name, "") for name in (
                "WHISPER_ARG_DEVICE", "GGML_BACKEND_PATH", "GGML_METAL_PATH_RESOURCES",
                "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
                "DYLD_INSERT_LIBRARIES", "LD_LIBRARY_PATH", "LD_PRELOAD",
            )
        }
        return {"cli": cli, "server": server, "libraries": libraries,
                "environment": environment}

    def _compatibility(self, binaries):
        if not self.enabled or os.environ.get("QWEN3_TTS_ASR_BACKEND", "").lower() == "cli":
            return "cli_requested"
        if sys.platform != "darwin" or platform.machine() != "arm64":
            return "unreviewed_platform"
        if not self._lsof:
            return "listener_ownership_unavailable"
        if any(value for key, value in binaries["environment"].items()
               if key != "WHISPER_ARG_DEVICE"):
            return "library_override_unreviewed"
        signatures = {
            "cli": binaries["cli"], "server": binaries["server"],
            "library": binaries["libraries"].get("whisper", {}),
        }
        if any(signatures[name].get("sha256") != expected
               for name, expected in _KNOWN_BUILD.items()):
            return "unreviewed_binary"
        if not all(os.access(value, os.X_OK) for value in (self.cli_path, self.server_path)):
            return "binary_not_executable"
        return None

    def _refresh(self):
        binaries = self._binary_identity()
        key = _json_key(binaries)
        if self._binaries_key is not None and self._binaries_key != key:
            self._stop_all()
        self._binaries_key = key
        return binaries

    def identity(self):
        with self._lock:
            binaries = self._refresh()
            reason = self._compatibility(binaries)
            return {
                "policy": _POLICY, "backend": "cli" if reason else "server-with-cli-fallback",
                "compatibility": reason or "reviewed-homebrew-arm64-1.9.2",
                "binaries": binaries,
                "cli_arguments": ["-l", "zh", "-nt", "-np"],
                "request_fields": dict(_REQUEST_FIELDS),
                "no_context": True, "max_servers": 2,
                "threads": min(4, os.cpu_count() or 1), "processors": 1,
                "timeout_seconds": _CLI_TIMEOUT,
            }

    def _alive(self, instance):
        return not instance.closed and instance.guard.poll() is None

    def _owns_listener(self, instance):
        if not instance.pid or not self._alive(instance) or not self._lsof:
            return False
        result = subprocess.run(
            [self._lsof, "-nP", "-a", "-p", str(instance.pid),
             f"-iTCP:{instance.port}", "-sTCP:LISTEN", "-Fpn"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=2,
        )
        fields = result.stdout.splitlines()
        return (
            result.returncode == 0 and f"p{instance.pid}" in fields
            and f"n127.0.0.1:{instance.port}" in fields
            and self._alive(instance)
        )

    def _http_json(self, instance, suffix, body=None, content_type=None, timeout=None):
        if not self._owns_listener(instance):
            raise _ServerError("listener is not owned by this task")
        timeout = _REQUEST_TIMEOUT if timeout is None else timeout
        # HTTPConnection never consults HTTP_PROXY, HTTPS_PROXY, ALL_PROXY or
        # macOS system proxies, and never follows a redirect to another host.
        connection = http.client.HTTPConnection("127.0.0.1", instance.port, timeout=timeout)
        expired = threading.Event()

        def abort():
            expired.set()
            sock = connection.sock
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()

        timer = threading.Timer(timeout, abort)
        timer.daemon = True
        timer.start()
        try:
            connection.connect()
            if expired.is_set() or not self._owns_listener(instance):
                raise _ServerError("listener ownership changed")
            headers = {"Accept": "application/json", "Connection": "close"}
            if content_type:
                headers["Content-Type"] = content_type
            connection.request("POST" if body is not None else "GET",
                               instance.prefix + suffix, body=body, headers=headers)
            response = connection.getresponse()
            if response.status != 200:
                raise _ServerError("unexpected HTTP status")
            if response.getheader("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                raise _ServerError("unexpected response type")
            if response.getheader("Server", "") != "whisper.cpp":
                raise _ServerError("unexpected server identity")
            raw = response.read(_MAX_RESPONSE + 1)
            if expired.is_set():
                raise TimeoutError("Whisper request exceeded deadline")
            if len(raw) > _MAX_RESPONSE:
                raise _ServerError("response exceeds limit")
            result = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
            if not isinstance(result, dict) or not self._alive(instance):
                raise _ServerError("invalid result or server exited")
            return result
        finally:
            timer.cancel()
            connection.close()

    def _guard_pid(self, instance):
        deadline = time.monotonic() + 5
        data = bytearray()
        fd = instance.guard.stdout.fileno()
        while time.monotonic() < deadline and self._alive(instance):
            ready, _, _ = select.select([fd], [], [], max(0, deadline - time.monotonic()))
            if not ready:
                break
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > 4096:
                break
            if b"\n" in data:
                message = json.loads(data.split(b"\n", 1)[0])
                pid = message.get("pid")
                if type(pid) is int and pid > 1 and pid not in (os.getpid(), instance.guard.pid):
                    return pid
                break
        raise _ServerError("guardian did not identify its child")

    def _start_server(self, model_signature):
        started = time.monotonic()
        self.stats["server_starts"] += 1
        instance = None
        try:
            # Binding an ephemeral port is only a reservation, NOT evidence of
            # server ownership. lsof is checked before any HTTP/audio is sent.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
                if port == 8765:
                    raise _ServerError("reserved application port")
            prefix = "/qwen-asr-" + os.urandom(24).hex()
            command = [
                self.server_path, "-m", model_signature["resolved"],
                "--host", "127.0.0.1", "--port", str(port),
                "--request-path", prefix, "--inference-path", "/inference",
                "--public", os.devnull, "-l", "zh", "-nt",
                "-t", str(min(4, os.cpu_count() or 1)), "-p", "1",
                "-bo", "5", "-bs", "5", "-mc", "-1", "-ml", "0",
                "-ac", "0", "-wt", "0.01", "-et", "2.4",
                "-lpt", "-1.0", "-nth", "0.6",
            ]
            guard = subprocess.Popen(
                [sys.executable, "-I", str(Path(__file__).resolve()), _GUARD_FLAG,
                 str(os.getpid()), json.dumps(command)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                close_fds=True, start_new_session=True,
            )
            instance = _Server(guard, port, prefix, dict(model_signature))
            instance.pid = self._guard_pid(instance)
            deadline = started + _START_TIMEOUT
            while time.monotonic() < deadline:
                if not self._alive(instance):
                    raise _ServerError("server exited during startup")
                if self._owns_listener(instance):
                    remaining = max(0.01, deadline - time.monotonic())
                    try:
                        health = self._http_json(instance, "/health", timeout=min(2, remaining))
                    except (OSError, TimeoutError, _ServerError):
                        health = None
                    if health == {"status": "ok"}:
                        return instance
                time.sleep(0.1)
            raise TimeoutError("Whisper server startup timed out")
        except BaseException:
            self.stats["server_start_failures"] += 1
            if instance is not None:
                self._stop_server(instance)
            raise
        finally:
            self.stats["server_start_seconds"] += time.monotonic() - started

    def _stop_server(self, instance):
        if instance.closed:
            return
        instance.closed = True
        self.stats["server_stops"] += 1
        guard = instance.guard
        try:
            if guard.stdin is not None:
                guard.stdin.close()
            try:
                guard.wait(timeout=5)
            except subprocess.TimeoutExpired:
                guard.terminate()
                guard.wait(timeout=3)
        except (OSError, subprocess.SubprocessError):
            # Never SIGKILL the guardian: it must be able to reap its own child.
            # EOF and its parent-PID check remain active if shutdown is delayed.
            self.stats["cleanup_failures"] += 1
        finally:
            if guard.stdout is not None:
                guard.stdout.close()

    def _stop_all(self):
        instances, self._servers = list(self._servers.values()), {}
        for instance in instances:
            self._stop_server(instance)

    def close(self):
        if os.getpid() != self._owner_pid:
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop_all()
            atexit.unregister(self.close)

    def _cli(self, wav_path, model_path):
        started = time.monotonic()
        self.stats["cli_requests"] += 1
        try:
            result = subprocess.run(
                [self.cli_path, "-m", str(model_path), "-f", str(wav_path),
                 "-l", "zh", "-nt", "-np"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=_CLI_TIMEOUT,
            )
            if result.returncode != 0:
                self.stats["cli_failures"] += 1
                return None
            return result.stdout or ""
        except (OSError, subprocess.SubprocessError, UnicodeError):
            self.stats["cli_failures"] += 1
            return None
        finally:
            self.stats["cli_seconds"] += time.monotonic() - started

    def _fallback(self, reason, wav_path, model_path):
        self.stats["fallbacks"] += 1
        reasons = self.stats["fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        return self._cli(wav_path, model_path)

    def transcribe(self, wav_path, model_path) -> Optional[str]:
        with self._lock:
            if self._closed:
                raise RuntimeError("WhisperRuntime is closed")
            started = time.monotonic()
            try:
                if os.getpid() != self._owner_pid:
                    return self._fallback("forked_runtime", wav_path, model_path)
                binaries = self._refresh()
                reason = self._compatibility(binaries)
                if reason:
                    self._stop_all()
                    return self._fallback(reason, wav_path, model_path)
                model = self._signature(model_path)
                key = model["path"]
                instance = self._servers.get(key)
                if instance is not None and instance.model != model:
                    self._stop_server(self._servers.pop(key))
                    instance = None
                if model.get("missing"):
                    return self._fallback("model_missing", wav_path, model_path)
                if not model.get("sha256"):
                    return self._fallback("model_signature_unavailable", wav_path, model_path)
                if self._disabled_models.get(key) != model.get("sha256"):
                    self._disabled_models.pop(key, None)
                    self.stats["disabled_models"] = len(self._disabled_models)
                if self._disabled_models.get(key) == model.get("sha256"):
                    return self._fallback("model_server_disabled", wav_path, model_path)
                if instance is None and len(self._servers) >= 2:
                    return self._fallback("model_capacity", wav_path, model_path)
                try:
                    if instance is None:
                        instance = self._start_server(model)
                        self._servers[key] = instance
                    if not self._alive(instance):
                        raise _ServerError("server crashed")
                    with open(wav_path, "rb") as handle:
                        audio = handle.read(_MAX_AUDIO + 1)
                    if not audio or len(audio) > _MAX_AUDIO:
                        raise _ServerError("audio is empty or exceeds transport limit")
                    body, content_type = _multipart(audio)
                    request_started = time.monotonic()
                    self.stats["server_requests"] += 1
                    try:
                        result = self._http_json(instance, "/inference", body, content_type)
                    finally:
                        self.stats["server_request_seconds"] += time.monotonic() - request_started
                    text = result.get("text")
                    if ("error" in result or not isinstance(text, str) or not text.strip()
                            or "\ufffd" in text or any(
                                (ord(char) < 32 and char not in "\r\n\t")
                                or 0xD800 <= ord(char) <= 0xDFFF for char in text)):
                        raise _ServerError("empty or corrupt transcript")
                    if model != self._signature(model_path):
                        raise _ServerError("model changed during recognition")
                    current_binaries = self._binary_identity()
                    if self._binaries_key != _json_key(current_binaries):
                        raise _ServerError("binary changed during recognition")
                    return text
                except Exception:
                    self.stats["server_failures"] += 1
                    self._disabled_models[key] = model.get("sha256")
                    self.stats["disabled_models"] = len(self._disabled_models)
                    bad_instance = self._servers.pop(key, None)
                    if bad_instance is not None:
                        self._stop_server(bad_instance)
                    return self._fallback("server_failure", wav_path, model_path)
            finally:
                self.stats["total_seconds"] += time.monotonic() - started


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != _GUARD_FLAG:
        raise SystemExit("This module is imported by the TTS worker, not a public server command.")
    raise SystemExit(_guard_main(json.loads(sys.argv[3]), int(sys.argv[2])))
