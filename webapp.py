#!/usr/bin/env python3
import fcntl
import json
import os
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file as flask_send_file


def send_file(path, **kwargs):
    """Freeze mutable generated media before computing HTTP length/ranges.

    Writers may truncate/replace the published path during a request. A private
    disk-backed snapshot keeps headers and streamed bytes tied to one object.
    """
    import hashlib
    snapshot = tempfile.TemporaryFile(mode="w+b")
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                snapshot.write(chunk)
                digest.update(chunk)
        size = snapshot.tell()
        snapshot.seek(0)
        conditional = kwargs.pop("conditional", True)
        kwargs.setdefault("download_name", Path(path).name)
        kwargs.setdefault("etag", digest.hexdigest())
        response = flask_send_file(snapshot, conditional=False, **kwargs)
        response.content_length = size
        if conditional:
            response.make_conditional(request.environ, accept_ranges=True, complete_length=size)
        response.call_on_close(snapshot.close)
        return response
    except BaseException:
        snapshot.close()
        raise

import dubstudio as core
from document_import import ALLOWED_DOCUMENTS, DOCUMENT_MAX_BYTES, DocumentError
from script_direction import personality_emotion_instruction

JOBS = {}
JOBS_LOCK = threading.Lock()
MLX_LOCK_PATH = Path(os.environ.get("QWEN3_TTS_MLX_LOCK", Path(tempfile.gettempdir()) / "qwen3-tts-mlx.lock"))
MLX_QUEUE = deque()
MLX_QUEUE_LOCK = threading.Lock()
ALLOWED_VIDEO = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
ALLOWED_AUDIO = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
REFERENCE_MAX_BYTES = 64 * 1024 * 1024
REFERENCE_MIN_SECONDS = 1.5
REFERENCE_MAX_SECONDS = 60.0
REFERENCE_TRANSCRIPT_MAX_CHARS = 1000


def uploaded_suffix(filename):
    return Path(str(filename or "")).suffix.lower()


def reference_duration(path):
    media = core.probe(path)
    try:
        return float(media.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError):
        return 0.0


def project_has_running_job(pid):
    with JOBS_LOCK:
        return any(value.get("status") in {"queued", "running"} for key, value in JOBS.items() if key.startswith(f"{pid}:"))


def mlx_is_busy():
    try:
        with MLX_LOCK_PATH.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return False
    except BlockingIOError:
        return True
    except OSError:
        return False


@contextmanager
def queued_job_slot(key, kind):
    ticket = {"key": key, "kind": kind}
    with MLX_QUEUE_LOCK:
        MLX_QUEUE.append(ticket)
        position = len(MLX_QUEUE)
    with JOBS_LOCK:
        job = JOBS.get(key)
        if job:
            job.update(
                status="queued",
                queue_position=position,
                message=f"排队中：前面还有 {max(0, position - 1)} 个任务",
            )
    try:
        while True:
            with MLX_QUEUE_LOCK:
                is_first = bool(MLX_QUEUE) and MLX_QUEUE[0] is ticket
            if is_first and not mlx_is_busy():
                break
            with MLX_QUEUE_LOCK:
                position = next((index + 1 for index, item in enumerate(MLX_QUEUE) if item is ticket), 1)
            with JOBS_LOCK:
                job = JOBS.get(key)
                if job and job.get("status") == "queued":
                    job.update(
                        queue_position=position,
                        message=f"排队中：前面还有 {max(0, position - 1)} 个任务",
                    )
            time.sleep(0.25)
        update_generation_job(key, "running")
        yield
    finally:
        with MLX_QUEUE_LOCK:
            try:
                MLX_QUEUE.remove(ticket)
            except ValueError:
                pass


def project_jobs(pid):
    with JOBS_LOCK:
        return {
            key.split(":", 1)[1]: dict(value)
            for key, value in JOBS.items()
            if key.startswith(f"{pid}:")
        }


def update_generation_job(key, status):
    preview = key.split(":", 1)[1].startswith("preview-")
    message = (
        "排队中：等待其他语音任务完成，将自动开始"
        if status == "queued" else
        "正在生成试听" if preview else "正在生成配音"
    )
    with JOBS_LOCK:
        job = JOBS.get(key)
        if job and job.get("status") in {"queued", "running"}:
            job.update(status=status, message=message)
            if status == "running":
                job.pop("queue_position", None)


def reserve_job(key, message="正在执行"):
    with JOBS_LOCK:
        if JOBS.get(key, {}).get("status") in {"queued", "running"}:
            return False
        JOBS[key] = {"status": "queued", "message": f"排队中：{message}将在前序任务完成后开始", "queue_position": 0}
        return True


def invalidate_clone_artifacts(project, role_id, message):
    core.cleanup_role_preview_files(project["id"], role_id)
    core.cleanup_role_adopted_takes(project, role_id)
    core.cleanup_role_generation_files(project, role_id)
    core.cleanup_project_generation_files(project["id"], project)
    core.invalidate_tts_outputs(project, message)
    return project


def pipeline_worker(pid, resume=False):
    key = f"{pid}:pipeline"
    token = core.TTS_JOB_OBSERVER.set(lambda status: update_generation_job(key, status))
    try:
        with queued_job_slot(key, "整轨生成"):
            core.auto_pipeline(pid, resume=resume)
        JOBS[key] = {"status": "done", "message": "自动配音完成"}
    except Exception as exc:
        JOBS[key] = {"status": "failed", "message": str(exc), "trace": traceback.format_exc()}
    finally:
        core.TTS_JOB_OBSERVER.reset(token)


def start_auto_pipeline(pid, resume=False):
    key = f"{pid}:pipeline"
    if not reserve_job(key, "自动处理已启动"):
        return False
    threading.Thread(target=pipeline_worker, args=(pid, resume), daemon=True).start()
    return True


def direction_worker(pid):
    key = f"{pid}:direction"
    JOBS[key] = {"status": "running", "message": "正在读取上下文，分配情绪与音色"}
    try:
        core.refine_imported_direction(pid)
        JOBS[key] = {"status": "done", "message": "情绪与音色已分配"}
    except Exception as exc:
        JOBS[key] = {"status": "failed", "message": str(exc), "trace": traceback.format_exc()}


def start_script_direction(pid):
    key = f"{pid}:direction"
    if not reserve_job(key, "正在读取上下文，分配情绪与音色"):
        return False
    threading.Thread(target=direction_worker, args=(pid,), daemon=True).start()
    return True


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True

    def worker(pid, stage, fn, *args, **kwargs):
        key = f"{pid}:{stage}"
        token = core.TTS_JOB_OBSERVER.set(lambda status: update_generation_job(key, status))
        try:
            kind = "试听" if stage.startswith("preview-") else "语音任务"
            slot = queued_job_slot(key, kind) if stage in {"tts", "retry"} or stage.startswith("preview-") else nullcontext()
            with slot:
                result = fn(pid, *args, **kwargs)
            JOBS[key] = {"status": "done", "message": "完成", "result": result.name if isinstance(result, Path) else None}
        except Exception as exc:
            JOBS[key] = {"status": "failed", "message": str(exc), "trace": traceback.format_exc()}
            if not stage.startswith("preview"):
                try:
                    core.set_stage(pid, stage, "failed", str(exc))
                except Exception:
                    pass
        finally:
            core.TTS_JOB_OBSERVER.reset(token)

    @app.get("/")
    def home():
        return render_template("index.html", document_voice_modes=True)

    @app.get("/api/projects")
    def projects():
        out = []
        for directory in sorted(core.PROJECTS_DIR.glob("*"), reverse=True):
            project = core.read_json(directory / "project.json")
            if project:
                out.append(project)
        return jsonify(out)

    @app.post("/api/projects")
    def create():
        payload = request.get_json(force=True)
        project_type = payload.get("project_type", "video" if payload.get("source") else "voice")
        if project_type == "voice":
            return jsonify(core.create_voice_project(
                payload.get("name"),
                payload.get("target_lang", "zh"),
                payload.get("direction_brief", ""),
            ))
        return jsonify(
            core.create_project(
                payload["source"],
                payload.get("name"),
                payload.get("copy_source", False),
                payload.get("target_lang", "zh"),
                payload.get("tts_engine", "local"),
            )
        )

    @app.post("/api/auto")
    def auto_upload():
        video = request.files.get("video")
        if not video or not video.filename:
            return jsonify({"error": "请选择要上传的视频"}), 400
        suffix = uploaded_suffix(video.filename)
        if suffix not in ALLOWED_VIDEO:
            return jsonify({"error": "不支持的视频格式"}), 400
        with tempfile.NamedTemporaryFile(prefix="dub-upload-", suffix=suffix, delete=False) as temporary:
            video.save(temporary.name)
            temp_path = Path(temporary.name)
        try:
            project = core.create_project(
                temp_path,
                request.form.get("name") or Path(video.filename).stem,
                copy_source=True,
                target_lang=request.form.get("target_lang", "zh"),
                tts_engine=request.form.get("tts_engine", "local"),
            )
            project["original_source"] = video.filename
            core.save_project(project)
        finally:
            temp_path.unlink(missing_ok=True)
        start_auto_pipeline(project["id"])
        return jsonify(project), 202

    def save_uploaded_document():
        document = request.files.get("document")
        if not document or not document.filename:
            return None, jsonify({"error": "请选择要导入的文档"}), 400
        suffix = Path(document.filename).suffix.lower()
        if suffix not in ALLOWED_DOCUMENTS:
            return None, jsonify({"error": "不支持该文档格式。请使用 TXT、MD、DOCX、DOC、RTF、ODT 或 PDF。"}), 400
        with tempfile.NamedTemporaryFile(prefix="dub-script-", suffix=suffix, delete=False) as temporary:
            document.save(temporary.name)
            temp_path = Path(temporary.name)
        if temp_path.stat().st_size > DOCUMENT_MAX_BYTES:
            temp_path.unlink(missing_ok=True)
            return None, jsonify({"error": "文档超过 12 MB，请先删减后再导入。"}), 400
        return (document, temp_path), None, None

    @app.post("/api/script-upload")
    def script_upload():
        saved, error, status = save_uploaded_document()
        if error:
            return error, status
        document, temp_path = saved
        try:
            project = core.create_voice_project_from_document(
                temp_path,
                request.form.get("name") or Path(document.filename).stem,
                request.form.get("target_lang", "zh"),
                request.form.get("direction_brief", ""),
                voice_mode=request.form.get("voice_mode", "design"),
            )
            project["original_source"] = document.filename
            core.save_project(project)
        except DocumentError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            return jsonify({"error": "文档导入失败。请改用 TXT 或 DOCX，或检查文件是否损坏。"}), 400
        finally:
            temp_path.unlink(missing_ok=True)
        if (project.get("script_import") or {}).get("voice_mode") != "clone":
            start_script_direction(project["id"])
        return jsonify(project), 201

    @app.post("/api/projects/<pid>/script-upload")
    def script_upload_into_project(pid):
        saved, error, status = save_uploaded_document()
        if error:
            return error, status
        document, temp_path = saved
        try:
            replace = request.form.get("replace")
            project = core.import_document_into_project(
                pid,
                temp_path,
                document.filename,
                None if replace in (None, "") else replace.lower() in {"1", "true", "yes"},
                request.form.get("direction_brief"),
                voice_mode=request.form.get("voice_mode", "design"),
            )
        except DocumentError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            return jsonify({"error": "文档导入失败。请改用 TXT 或 DOCX，或检查文件是否损坏。"}), 400
        finally:
            temp_path.unlink(missing_ok=True)
        if (project.get("script_import") or {}).get("voice_mode") != "clone":
            start_script_direction(project["id"])
        return jsonify(project), 200

    @app.get("/api/projects/<pid>")
    def get_project(pid):
        project = core.load_project(pid)
        try:
            core.migrate_clone_transcripts(project)
        except RuntimeError:
            project = core.load_project(pid)
        normalized = core.normalize_project(project)
        # Natural-timeline generation computes real cue positions after synthesis.
        # Expose those generated positions without overwriting editable script timing.
        generated = core.read_json(core.project_dir(pid) / "status.json", []) or []
        generated_by_id = {
            int(row.get("id") or 0): row
            for row in generated
            if isinstance(row, dict) and int(row.get("id") or 0) > 0
        }
        for cue in normalized.get("cues", []):
            row = generated_by_id.get(int(cue.get("id") or 0))
            if not row:
                continue
            start = row.get("start")
            end = row.get("end")
            if start is not None and end is not None:
                cue["playback_start"] = float(start)
                cue["playback_end"] = float(end)
        normalized["blockers"] = core.cue_blockers(pid, normalized)
        normalized["review_queue"] = core.cue_review_queue(pid, normalized)
        try:
            progress = core.get_tts_progress(pid, normalized)
        except (FileNotFoundError, OSError, ValueError):
            progress = None
        if progress:
            normalized["tts_progress"] = progress
            if progress.get("status") == "running":
                normalized["pipeline"] = {
                    **(normalized.get("pipeline") or {}),
                    "status": "running",
                    "stage": progress.get("stage"),
                    "percent": progress.get("percent", 0),
                    "message": progress.get("message") or normalized.get("last_message", ""),
                }
        return jsonify(normalized)

    @app.delete("/api/projects/<pid>")
    def delete_project(pid):
        if project_has_running_job(pid):
            return jsonify({"error": "项目正在生成或试听，请等待任务结束后再删除。"}), 409
        try:
            project = core.load_project(pid)
            name = str(project.get("name") or "未命名项目")
            core.delete_project(pid)
        except FileNotFoundError:
            return jsonify({"error": "项目不存在或已经删除"}), 404
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 400
        with JOBS_LOCK:
            for key in [key for key in JOBS if key.startswith(f"{pid}:")]:
                JOBS.pop(key, None)
        return jsonify({"deleted": True, "id": pid, "name": name})

    @app.put("/api/projects/<pid>")
    def save(pid):
        return jsonify(core.save_manifest(pid, request.get_json(force=True)))

    @app.post("/api/projects/<pid>/roles/<role_id>/reference")
    def upload_reference(pid, role_id):
        audio = request.files.get("audio")
        if not audio or not audio.filename:
            return jsonify({"error": "请选择参考音频"}), 400
        suffix = uploaded_suffix(audio.filename)
        if suffix not in ALLOWED_AUDIO:
            return jsonify({"error": "不支持的音频格式。请使用 WAV、MP3、M4A、AAC、FLAC、OGG 或 OPUS。"}), 400
        consent = request.form.get("consent", "").lower() in {"1", "true", "yes", "on"}
        if not consent:
            return jsonify({"error": "上传克隆参考音频前必须确认已获得声音授权"}), 400
        project = core.normalize_project(core.load_project(pid))
        if project_has_running_job(pid):
            return jsonify({"error": "项目正在生成或试听，请等待任务结束后再替换参考音频。"}), 409
        if project.get("settings", {}).get("tts_engine", "local") != "local":
            return jsonify({"error": "授权克隆仅支持本地 Qwen3-TTS，请先将项目引擎切换为本地。"}), 400
        role = next((item for item in project.get("roles", []) if item.get("id") == role_id), None)
        if not role:
            return jsonify({"error": "角色不存在"}), 404
        reference_dir = core.project_dir(pid) / "references"
        reference_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        source_tmp = reference_dir / f".{role_id}-{token}-source{suffix}"
        normalized_tmp = reference_dir / f".{role_id}-{token}.wav"
        source_path = reference_dir / f"{role_id}-source{suffix}"
        normalized_path = reference_dir / f"{role_id}.wav"
        requested_text = request.form.get("ref_text", "").strip()
        backup_normalized = reference_dir / f".{role_id}-{token}-backup.wav"
        existing_source = next(iter(reference_dir.glob(f"{role_id}-source.*")), None)
        backup_source = reference_dir / f".{role_id}-{token}-backup-source{existing_source.suffix if existing_source else suffix}"
        promoted = False
        try:
            audio.save(source_tmp)
            if source_tmp.stat().st_size > REFERENCE_MAX_BYTES:
                raise RuntimeError("参考音频超过 64 MB，请先裁剪为 5–15 秒的清晰单人语音。")
            core.run(
                [
                    core.media_tool("ffmpeg"), "-y", "-v", "error", "-i", source_tmp,
                    "-vn", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", normalized_tmp,
                ]
            )
            duration = reference_duration(normalized_tmp)
            if duration < REFERENCE_MIN_SECONDS:
                raise RuntimeError("参考音频过短，请上传至少 1.5 秒、建议 5–15 秒的清晰单人语音。")
            if duration > REFERENCE_MAX_SECONDS:
                raise RuntimeError("参考音频超过 60 秒，请裁剪为建议的 5–15 秒清晰单人语音。")
            transcript = core.transcribe_reference_audio(
                normalized_tmp,
                project.get("settings", {}).get("target_lang", "auto"),
                project.get("settings", {}).get("whisper_model"),
            )
            reference_hash = core.file_sha256(normalized_tmp)
            if normalized_path.is_file():
                shutil.copy2(normalized_path, backup_normalized)
            if existing_source and existing_source.is_file():
                shutil.copy2(existing_source, backup_source)
            old_sources = list(reference_dir.glob(f"{role_id}-source.*"))
            normalized_tmp.replace(normalized_path)
            source_tmp.replace(source_path)
            promoted = True
            for old_source in old_sources:
                if old_source != source_path:
                    old_source.unlink(missing_ok=True)
            latest = core.normalize_project(core.load_project(pid))
            role = next((item for item in latest.get("roles", []) if item.get("id") == role_id), None)
            if not role:
                raise RuntimeError("音频处理期间角色已被删除，请重新上传")
            role.update({
                "mode": "clone",
                "ref_audio": str(normalized_path),
                "ref_name": audio.filename,
                "clone_consent": consent,
                "reference_revision": token,
                "reference_sha256": reference_hash,
                "ref_text_auto": transcript,
                "ref_text": transcript,
                "ref_text_source": "auto",
                "ref_text_requested": requested_text,
                "ref_text_warning": bool(requested_text and core.transcript_similarity(requested_text, transcript) < 0.65),
                "ref_text_confirmed": False,
                "ref_text_confirmed_revision": "",
            })
            message = "参考音频已更新，请播放核听并确认逐字稿后再生成"
            invalidate_clone_artifacts(latest, role_id, message)
            latest["last_message"] = message
            latest["pipeline"] = {"status": "pending", "stage": "script", "percent": 0 if latest.get("project_type") == "voice" else 42, "message": message}
            project = core.save_project(core.normalize_project(latest))
        except Exception as exc:
            source_tmp.unlink(missing_ok=True)
            normalized_tmp.unlink(missing_ok=True)
            if promoted:
                normalized_path.unlink(missing_ok=True)
                source_path.unlink(missing_ok=True)
                if backup_normalized.is_file():
                    backup_normalized.replace(normalized_path)
                if backup_source.is_file():
                    backup_source.replace(reference_dir / f"{role_id}-source{existing_source.suffix if existing_source else suffix}")
            message = str(exc) if isinstance(exc, RuntimeError) else "参考音频处理失败，请检查文件是否损坏或重试。"
            return jsonify({"error": message}), 400 if not promoted else 409
        finally:
            backup_normalized.unlink(missing_ok=True)
            backup_source.unlink(missing_ok=True)
        return jsonify(project)

    @app.get("/api/projects/<pid>/roles/<role_id>/reference")
    def get_reference(pid, role_id):
        project = core.normalize_project(core.load_project(pid))
        role = next((item for item in project.get("roles", []) if item.get("id") == role_id), None)
        if not role:
            return jsonify({"error": "角色不存在"}), 404
        ref_value = str(role.get("ref_audio") or "").strip()
        reference = Path(ref_value).expanduser() if ref_value else None
        if not reference or not reference.is_file():
            return jsonify({"error": "参考音频不存在"}), 404
        root = core.project_dir(pid).resolve()
        references = (root / "references").resolve()
        try:
            resolved = reference.resolve()
        except OSError:
            return jsonify({"error": "参考音频不存在"}), 404
        if resolved.parent != references or resolved.name != f"{role_id}.wav":
            return jsonify({"error": "参考音频路径无效"}), 403
        return send_file(resolved, conditional=True, mimetype="audio/wav")

    @app.put("/api/projects/<pid>/roles/<role_id>/reference-transcript")
    def confirm_reference_transcript(pid, role_id):
        if project_has_running_job(pid):
            return jsonify({"error": "项目正在生成或试听，请等待任务结束后再确认逐字稿。"}), 409
        payload = request.get_json(force=True) or {}
        transcript = str(payload.get("ref_text") or "").strip()
        requested_revision = str(payload.get("reference_revision") or "").strip()
        if not transcript:
            return jsonify({"error": "参考逐字稿不能为空，请按音频中实际说出的原话填写。"}), 400
        if len(transcript) > REFERENCE_TRANSCRIPT_MAX_CHARS:
            return jsonify({"error": f"参考逐字稿超过 {REFERENCE_TRANSCRIPT_MAX_CHARS} 字，请只填写参考音频中实际说出的内容。"}), 400
        project = core.normalize_project(core.load_project(pid))
        role = next((item for item in project.get("roles", []) if item.get("id") == role_id), None)
        if not role:
            return jsonify({"error": "角色不存在"}), 404
        if role.get("mode") != "clone":
            return jsonify({"error": "该角色当前不是授权克隆模式"}), 400
        ref_value = str(role.get("ref_audio") or "").strip()
        reference = Path(ref_value).expanduser() if ref_value else None
        if not reference or not reference.is_file():
            return jsonify({"error": "参考音频不存在，请重新上传。"}), 400
        current_revision = str(role.get("reference_revision") or "")
        if not requested_revision or requested_revision != current_revision:
            return jsonify({"error": "参考音频已被替换，请重新播放最新音频后再确认逐字稿。"}), 409
        auto_text = str(role.get("ref_text_auto") or "").strip()
        previous_text = str(role.get("ref_text") or "").strip()
        was_confirmed = bool(role.get("ref_text_confirmed")) and str(role.get("ref_text_confirmed_revision") or "") == current_revision
        changed = transcript != previous_text or not was_confirmed
        role.update({
            "ref_text": transcript,
            "ref_text_source": "auto" if auto_text and transcript == auto_text else "manual",
            "ref_text_confirmed": True,
            "ref_text_confirmed_revision": current_revision,
        })
        if changed:
            message = "参考逐字稿已确认，需要重新生成配音"
            invalidate_clone_artifacts(project, role_id, message)
        project["last_message"] = "参考逐字稿已确认，可以生成试听和整轨" if not changed else "参考逐字稿已确认，需要重新生成配音"
        if changed:
            project["pipeline"] = {"status": "pending", "stage": "script", "percent": 0 if project.get("project_type") == "voice" else 42, "message": project["last_message"]}
        core.save_project(project)
        return jsonify(project)

    @app.delete("/api/projects/<pid>/roles/<role_id>/reference")
    def delete_reference(pid, role_id):
        project = core.normalize_project(core.load_project(pid))
        if project_has_running_job(pid):
            return jsonify({"error": "项目正在生成或试听，请等待任务结束后再移除参考音频。"}), 409
        role = next((item for item in project.get("roles", []) if item.get("id") == role_id), None)
        if not role:
            return jsonify({"error": "角色不存在"}), 404
        core.cleanup_role_reference_files(pid, role_id)
        core.cleanup_role_adopted_takes(project, role_id)
        core.cleanup_role_generation_files(project, role_id)
        core.cleanup_project_generation_files(pid, project)
        role.update({
            "ref_audio": "", "ref_name": "", "reference_revision": "", "reference_sha256": "",
            "ref_text": "", "ref_text_auto": "", "ref_text_source": "", "ref_text_requested": "", "ref_text_warning": False,
            "ref_text_confirmed": False, "ref_text_confirmed_revision": "", "clone_consent": False, "mode": "design",
        })
        core.invalidate_tts_outputs(project, "参考音频已移除，需要重新生成配音")
        project["last_message"] = "参考音频已移除，需要重新生成配音"
        project["pipeline"] = {"status": "pending", "stage": "script", "percent": 0 if project.get("project_type") == "voice" else 42, "message": project["last_message"]}
        core.save_project(core.normalize_project(project))
        return jsonify(project)

    @app.delete("/api/projects/<pid>/roles/<role_id>")
    def delete_role(pid, role_id):
        project = core.normalize_project(core.load_project(pid))
        if project_has_running_job(pid):
            return jsonify({"error": "项目正在生成或试听，请等待任务结束后再删除角色。"}), 409
        roles = project.get("roles", [])
        if len(roles) <= 1:
            return jsonify({"error": "项目至少需要保留一个角色"}), 400
        role = next((item for item in roles if item.get("id") == role_id), None)
        if not role:
            return jsonify({"error": "角色不存在"}), 404
        replacement = next(item.get("id") for item in roles if item.get("id") != role_id)
        core.cleanup_role_reference_files(pid, role_id)
        core.cleanup_role_adopted_takes(project, role_id)
        core.cleanup_role_generation_files(project, role_id)
        core.cleanup_project_generation_files(pid, project)
        for cue in project.get("cues", []):
            if cue.get("role") == role_id:
                cue["role"] = replacement
        project["roles"] = [item for item in roles if item.get("id") != role_id]
        core.invalidate_tts_outputs(project, "角色已删除，需要重新生成配音")
        project["last_message"] = "角色已删除，需要重新生成配音"
        project["pipeline"] = {"status": "pending", "stage": "script", "percent": 0 if project.get("project_type") == "voice" else 42, "message": project["last_message"]}
        core.save_project(core.normalize_project(project))
        return jsonify(project)

    @app.post("/api/projects/<pid>/direct-role-emotions")
    def direct_role_emotions(pid):
        if project_has_running_job(pid):
            return jsonify({"error": "项目正在生成或试听，请等待任务结束后再按人物性格重算表演。"}), 409
        payload = request.get_json(force=True) or {}
        role_id = str(payload.get("role_id") or "").strip()
        project = core.normalize_project(core.load_project(pid))
        role = next((item for item in project.get("roles", []) if item.get("id") == role_id), None)
        if not role:
            return jsonify({"error": "角色不存在"}), 404
        if not role.get("personality"):
            return jsonify({"error": "请先填写人物性格"}), 400
        cue_ids = [
            cue.get("id") for cue in project.get("cues", [])
            if cue.get("role") == role_id and cue.get("enabled", True)
        ]
        if not cue_ids:
            return jsonify({"error": "这个角色还没有可处理的台词"}), 400
        try:
            _roles, cues = core.ollama_direct_script(
                project.get("roles") or [],
                project.get("cues") or [],
                project.get("settings", {}).get("ollama_model", core.OLLAMA_MODEL),
                {item.get("id") for item in project.get("roles", []) if item.get("id")},
                cue_ids,
                project.get("direction_brief") or "",
            )
        except Exception as exc:
            return jsonify({"error": f"人物性格导演失败：{exc}"}), 400
        changed_ids = {int(item) for item in cue_ids}
        updated_emotions = {
            int(cue["id"]): str(cue.get("emotion") or "自然")
            for cue in cues
            if cue.get("id") is not None and int(cue["id"]) in changed_ids
        }
        for cue in project.get("cues", []):
            if int(cue.get("id") or 0) in updated_emotions:
                cue["emotion"] = updated_emotions[int(cue["id"])]
        core.cleanup_role_preview_files(pid, role_id)
        core.cleanup_role_adopted_takes(project, role_id)
        core.cleanup_role_generation_files(project, role_id)
        core.cleanup_project_generation_files(pid, project)
        core.invalidate_tts_outputs(project, "已按人物性格重算该角色表演，需要重新生成配音")
        project["last_message"] = f"已按人物性格更新角色“{role.get('name') or '未命名'}”的 {len(updated_emotions)} 条表演"
        project["pipeline"] = {
            "status": "pending", "stage": "script",
            "percent": 0 if project.get("project_type") == "voice" else 42,
            "message": project["last_message"],
        }
        core.save_project(core.normalize_project(project))
        core.add_log(pid, project["last_message"])
        return jsonify(project)

    @app.post("/api/projects/<pid>/preview-voice")
    def preview_voice(pid):
        payload = request.get_json(force=True)
        role_id = payload.get("role_id")
        project = core.load_project(pid)
        core.migrate_clone_transcripts(project, {role_id})
        project = core.normalize_project(project)
        role = next((item for item in project.get("roles", []) if item.get("id") == role_id), None)
        if not role:
            return jsonify({"error": "角色不存在"}), 404
        try:
            core.validate_clone_roles(project, {role_id})
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 400
        emotion = str(payload.get("emotion") or "自然").strip() or "自然"
        personality_directed = bool(payload.get("personality_directed"))
        cue_id = payload.get("cue_id")
        if personality_directed and role.get("personality") and cue_id is None:
            emotion = personality_emotion_instruction(role.get("personality"), emotion)
        preview_nonce = str(payload.get("preview_nonce") or uuid.uuid4().hex)
        key = f"{pid}:preview-{role_id}"
        if not reserve_job(key, "正在准备试听"):
            return jsonify(JOBS[key]), 409
        try:
            threading.Thread(
                target=worker,
                args=(pid, f"preview-{role_id}", core.preview_voice, role_id, payload.get("text", ""), emotion, personality_directed, cue_id, preview_nonce),
                daemon=True,
            ).start()
        except Exception:
            with JOBS_LOCK:
                JOBS.pop(key, None)
            raise
        return jsonify(JOBS[key]), 202

    @app.get("/api/projects/<pid>/preview-voice/<role_id>")
    def preview_voice_file(pid, role_id):
        preview_dir = core.project_dir(pid) / "previews"
        requested = request.args.get("file", "")
        if requested:
            safe_name = Path(requested).name
            candidate = preview_dir / safe_name
            if candidate.is_file() and candidate.name.startswith(f"{role_id}-") and candidate.suffix == ".wav":
                return send_file(candidate, conditional=True, mimetype="audio/wav")
        files = sorted(preview_dir.glob(f"{role_id}-*.wav"), key=lambda path: path.stat().st_mtime, reverse=True) if preview_dir.exists() else []
        if not files:
            return jsonify({"error": "试听音频尚未生成"}), 404
        return send_file(files[0], conditional=True, mimetype="audio/wav")

    @app.get("/api/projects/<pid>/preview-voice/<role_id>/gate")
    def preview_voice_gate(pid, role_id):
        filename = Path(str(request.args.get("file") or "")).name
        if not filename or not filename.startswith(f"{role_id}-") or not filename.endswith(".wav"):
            return jsonify({"error": "试听文件无效"}), 400
        report = core.preview_gate_report(pid, filename)
        if not report:
            return jsonify({"gate_passed": True})
        return jsonify(report)

    @app.post("/api/projects/<pid>/cues/<int:cue_id>/adopt-take")
    def adopt_take(pid, cue_id):
        if project_has_running_job(pid):
            return jsonify({"error": "项目正在生成或试听，请等待任务结束后再采用试听版本。"}), 409
        payload = request.get_json(force=True) or {}
        try:
            project = core.adopt_preview_take(
                pid, cue_id, payload.get("file"),
                confirm_gate_rejection=bool(payload.get("confirm_gate_rejection")),
                review_note=payload.get("review_note") or "",
            )
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(project)

    @app.delete("/api/projects/<pid>/cues/<int:cue_id>/adopt-take")
    def discard_take(pid, cue_id):
        if project_has_running_job(pid):
            return jsonify({"error": "项目正在生成或试听，请等待任务结束后再取消采用。"}), 409
        try:
            project = core.discard_adopted_take(pid, cue_id)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(project)

    @app.get("/api/projects/<pid>/cues/<int:cue_id>/adopted-take")
    def adopted_take_file(pid, cue_id):
        project = core.normalize_project(core.load_project(pid))
        cue = next((item for item in project.get("cues", []) if int(item.get("id") or 0) == cue_id), None)
        take = core.cue_adopted_take(project, cue) if cue else None
        if not take:
            return jsonify({"error": "这句台词没有可用的已采用版本"}), 404
        return send_file(take["path"], conditional=True, mimetype="audio/wav")

    @app.post("/api/projects/<pid>/<stage>")
    def action(pid, stage):
        if stage == "auto":
            if not start_auto_pipeline(pid):
                return jsonify(JOBS[f"{pid}:pipeline"]), 409
            return jsonify(JOBS[f"{pid}:pipeline"]), 202
        if stage == "retry":
            project = core.load_project(pid)
            # Voice projects are already analyzed and scripted. Retrying them
            # must resume the TTS stage, not send a DOCX script to ffprobe.
            if project.get("project_type") == "voice":
                stage = "tts"
            else:
                if not start_auto_pipeline(pid, resume=True):
                    return jsonify(JOBS[f"{pid}:pipeline"]), 409
                return jsonify(JOBS[f"{pid}:pipeline"]), 202
        funcs = {
            "analyze": core.analyze,
            "translate": core.translate_project,
            "tts": core.generate_tts,
            "render": core.render,
            "qa": core.qa,
        }
        if stage not in funcs:
            return jsonify({"error": "未知操作"}), 404
        if stage == "tts":
            project = core.load_project(pid)
            core.migrate_clone_transcripts(project)
            project = core.normalize_project(project)
            try:
                core.validate_clone_roles(project)
            except RuntimeError as exc:
                return jsonify({"error": str(exc)}), 400
        key = f"{pid}:{stage}"
        if not reserve_job(key, "正在执行"):
            return jsonify(JOBS[key]), 409
        try:
            threading.Thread(target=worker, args=(pid, stage, funcs[stage]), daemon=True).start()
        except Exception:
            with JOBS_LOCK:
                JOBS.pop(key, None)
            raise
        return jsonify(JOBS[key]), 202

    @app.get("/api/projects/<pid>/jobs")
    def jobs(pid):
        return jsonify(project_jobs(pid))

    @app.get("/api/projects/<pid>/log")
    def log(pid):
        path = core.project_dir(pid) / "activity.log"
        return jsonify({"log": path.read_text() if path.exists() else ""})

    @app.get("/api/projects/<pid>/download")
    def download(pid):
        project = core.load_project(pid)
        if project.get("project_type") == "voice":
            if not core.voice_output_is_complete(pid, project):
                return jsonify({"error": "配音音轨尚未完整生成，请先完成整轨生成"}), 409
            output = Path(project["output_audio"])
        else:
            output = Path(project.get("output") or "")
        if not output.exists():
            return jsonify({"error": "输出文件不存在"}), 404
        download_name = f"{project.get('name','配音')}-配音音轨.wav" if project.get("project_type") == "voice" else output.name
        return send_file(output, as_attachment=True, download_name=download_name)

    @app.get("/api/projects/<pid>/cues/<int:cue_id>/download")
    def download_cue(pid, cue_id):
        project = core.normalize_project(core.load_project(pid))
        cue = next((item for item in project.get("cues", []) if int(item.get("id") or 0) == cue_id), None)
        if not cue:
            return jsonify({"error": "台词不存在"}), 404
        if cue.get("enabled") is False:
            return jsonify({"error": "这句台词当前未启用，无法导出"}), 409
        if project.get("stages", {}).get("tts") != "done":
            return jsonify({"error": "台词已修改或尚未生成，请先生成整轨后再导出此句"}), 409
        output = core.project_dir(pid) / "fit" / f"{cue_id:03d}.wav"
        if not output.is_file():
            return jsonify({"error": "这句台词的成品音频不存在，请重新生成整轨"}), 404
        role = next((item for item in project.get("roles", []) if item.get("id") == cue.get("role")), {})
        download_name = core.cue_export_filename(cue, role)
        return send_file(output, as_attachment=True, download_name=download_name, mimetype="audio/wav")
    @app.delete("/api/projects/<pid>/cues/batch")
    def batch_delete_cues(pid):
        payload = request.get_json(force=True, silent=True) or {}
        try:
            requested = {int(value) for value in payload.get("cue_ids", [])}
            project = core.normalize_project(core.load_project(pid), renumber_cues=False)
        except FileNotFoundError:
            return jsonify({"error": "项目不存在"}), 404
        except (TypeError, ValueError):
            return jsonify({"error": "台词编号格式错误"}), 400
        if not requested:
            return jsonify({"error": "请至少选择一条台词"}), 400
        existing = {int(cue.get("id") or 0) for cue in project.get("cues", [])}
        unknown = sorted(requested - existing)
        if unknown:
            return jsonify({"error": "包含不存在的台词编号", "unknown": unknown}), 400
        project["cues"] = [cue for cue in project.get("cues", []) if int(cue.get("id") or 0) not in requested]
        core.cleanup_project_generation_files(pid, project)
        project = core.normalize_project(project)
        core.invalidate_tts_outputs(project, f"已删除 {len(requested)} 条台词，需要重新生成配音")
        core.save_project(project)
        return jsonify({"deleted": len(requested), "project": project})

    @app.put("/api/projects/<pid>/pronunciation-lexicon")
    def update_pronunciation_lexicon(pid):
        payload = request.get_json(force=True, silent=True) or {}
        rules = payload.get("rules", [])
        if not isinstance(rules, list):
            return jsonify({"error": "发音词典格式错误"}), 400
        project = core.normalize_project(core.load_project(pid))
        project.setdefault("settings", {})["pronunciation_lexicon"] = rules
        project = core.normalize_project(project)
        core.cleanup_project_generation_files(pid, project)
        core.invalidate_tts_outputs(project, "发音词典已更新，需要重新生成配音")
        core.save_project(project)
        return jsonify(project)

    @app.post("/api/projects/<pid>/cues/batch-speed-gap")
    def batch_update_speed_gap(pid):
        payload = request.get_json(force=True, silent=True) or {}
        cue_ids = payload.get("cue_ids", [])
        speed = payload.get("speed")
        gap_after = payload.get("gap_after")
        selected_all = payload.get("selected_all", False)

        try:
            project = core.normalize_project(core.load_project(pid))
        except (FileNotFoundError, KeyError, ValueError):
            return jsonify({"error": "项目不存在"}), 404

        try:
            normalized_ids = {int(value) for value in cue_ids}
            normalized_speed = None if speed is None else max(0.5, min(2.0, float(speed)))
            normalized_gap = None if gap_after is None else max(0.0, min(10.0, float(gap_after)))
        except (TypeError, ValueError):
            return jsonify({"error": "语速、间隔或台词编号格式错误"}), 400

        if not selected_all and not normalized_ids:
            return jsonify({"error": "请至少选择一条台词"}), 400

        updated = 0
        for cue in project.get("cues", []):
            if selected_all or int(cue.get("id", 0)) in normalized_ids:
                if normalized_speed is not None:
                    cue["speed"] = normalized_speed
                if normalized_gap is not None:
                    cue["gap_after"] = normalized_gap
                updated += 1

        if not updated:
            return jsonify({"error": "没有找到可更新的台词"}), 404

        project.setdefault("stages", {})["tts"] = "pending"
        project["output_audio"] = ""
        project["last_message"] = f"已批量更新 {updated} 条台词，需要重新生成配音音轨"
        core.save_project(project)
        return jsonify({"updated": updated, "speed": normalized_speed, "gap_after": normalized_gap})


    @app.get("/api/projects/<pid>/preview")
    def preview(pid):
        project = core.load_project(pid)
        if project.get("project_type") == "voice":
            if not core.voice_output_is_complete(pid, project):
                return jsonify({"error": "配音音轨尚未完整生成，请先完成整轨生成"}), 409
            output = Path(project["output_audio"])
            return send_file(output, conditional=True, mimetype="audio/wav")
        output = Path(project.get("output") or "")
        if not output.exists():
            return jsonify({"error": "成片不存在"}), 404
        return send_file(output, conditional=True, mimetype="video/mp4")

    return app


if __name__ == "__main__":
    create_app().run(port=8765)
