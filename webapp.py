#!/usr/bin/env python3
import json
import tempfile
import threading
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

import dubstudio as core
from document_import import ALLOWED_DOCUMENTS, DOCUMENT_MAX_BYTES, DocumentError

JOBS = {}
JOBS_LOCK = threading.Lock()
ALLOWED_VIDEO = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
ALLOWED_AUDIO = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}


def reserve_job(key, message="正在执行"):
    with JOBS_LOCK:
        if JOBS.get(key, {}).get("status") == "running":
            return False
        JOBS[key] = {"status": "running", "message": message}
        return True


def pipeline_worker(pid, resume=False):
    key = f"{pid}:pipeline"
    JOBS[key] = {"status": "running", "message": "自动处理已启动"}
    try:
        core.auto_pipeline(pid, resume=resume)
        JOBS[key] = {"status": "done", "message": "自动配音完成"}
    except Exception as exc:
        JOBS[key] = {"status": "failed", "message": str(exc), "trace": traceback.format_exc()}


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

    def worker(pid, stage, fn, *args, **kwargs):
        key = f"{pid}:{stage}"
        with JOBS_LOCK:
            JOBS[key] = {"status": "running", "message": "正在执行"}
        try:
            result = fn(pid, *args, **kwargs)
            JOBS[key] = {"status": "done", "message": "完成", "result": result.name if isinstance(result, Path) else None}
        except Exception as exc:
            JOBS[key] = {"status": "failed", "message": str(exc), "trace": traceback.format_exc()}
            if not stage.startswith("preview"):
                try:
                    core.set_stage(pid, stage, "failed", str(exc))
                except Exception:
                    pass

    @app.get("/")
    def home():
        return render_template("index.html")

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
            return jsonify(core.create_voice_project(payload.get("name"), payload.get("target_lang", "zh")))
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
        suffix = Path(secure_filename(video.filename)).suffix.lower()
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
            )
            project["original_source"] = document.filename
            core.save_project(project)
        except DocumentError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            return jsonify({"error": "文档导入失败。请改用 TXT 或 DOCX，或检查文件是否损坏。"}), 400
        finally:
            temp_path.unlink(missing_ok=True)
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
            )
        except DocumentError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            return jsonify({"error": "文档导入失败。请改用 TXT 或 DOCX，或检查文件是否损坏。"}), 400
        finally:
            temp_path.unlink(missing_ok=True)
        start_script_direction(project["id"])
        return jsonify(project), 200

    @app.get("/api/projects/<pid>")
    def get_project(pid):
        return jsonify(core.load_project(pid))

    @app.put("/api/projects/<pid>")
    def save(pid):
        return jsonify(core.save_manifest(pid, request.get_json(force=True)))

    @app.post("/api/projects/<pid>/roles/<role_id>/reference")
    def upload_reference(pid, role_id):
        audio = request.files.get("audio")
        if not audio or not audio.filename:
            return jsonify({"error": "请选择参考音频"}), 400
        suffix = Path(secure_filename(audio.filename)).suffix.lower()
        if suffix not in ALLOWED_AUDIO:
            return jsonify({"error": "不支持的音频格式"}), 400
        consent = request.form.get("consent", "").lower() in {"1", "true", "yes", "on"}
        if not consent:
            return jsonify({"error": "上传克隆参考音频前必须确认已获得声音授权"}), 400
        project = core.normalize_project(core.load_project(pid))
        role = next((item for item in project.get("roles", []) if item.get("id") == role_id), None)
        if not role:
            return jsonify({"error": "角色不存在"}), 404
        reference_dir = core.project_dir(pid) / "references"
        reference_dir.mkdir(parents=True, exist_ok=True)
        source_path = reference_dir / f"{role_id}-source{suffix}"
        normalized_path = reference_dir / f"{role_id}.wav"
        audio.save(source_path)
        core.run(
            [
                core.media_tool("ffmpeg"), "-y", "-v", "error", "-i", source_path,
                "-vn", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", normalized_path,
            ]
        )
        role["mode"] = "clone"
        role["ref_audio"] = str(normalized_path)
        role["ref_name"] = audio.filename
        role["clone_consent"] = consent
        if "ref_text" in request.form:
            role["ref_text"] = request.form.get("ref_text", "").strip()
        project = core.save_manifest(pid, project)
        return jsonify(project)

    @app.delete("/api/projects/<pid>/roles/<role_id>/reference")
    def delete_reference(pid, role_id):
        project = core.normalize_project(core.load_project(pid))
        role = next((item for item in project.get("roles", []) if item.get("id") == role_id), None)
        if not role:
            return jsonify({"error": "角色不存在"}), 404
        ref_audio = Path(str(role.get("ref_audio") or ""))
        if ref_audio.exists() and core.project_dir(pid) in ref_audio.parents:
            ref_audio.unlink(missing_ok=True)
        role.update({"ref_audio": "", "ref_name": "", "ref_text": "", "clone_consent": False, "mode": "design"})
        project = core.save_manifest(pid, project)
        return jsonify(project)

    @app.post("/api/projects/<pid>/preview-voice")
    def preview_voice(pid):
        payload = request.get_json(force=True)
        role_id = payload.get("role_id")
        key = f"{pid}:preview-{role_id}"
        if not reserve_job(key):
            return jsonify(JOBS[key]), 409
        try:
            threading.Thread(
                target=worker,
                args=(pid, f"preview-{role_id}", core.preview_voice, role_id, payload.get("text", ""), payload.get("emotion", "自然")),
                daemon=True,
            ).start()
        except Exception:
            with JOBS_LOCK:
                JOBS.pop(key, None)
            raise
        return jsonify({"status": "running", "job": f"preview-{role_id}"}), 202

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

    @app.post("/api/projects/<pid>/<stage>")
    def action(pid, stage):
        if stage in ("auto", "retry"):
            if not start_auto_pipeline(pid, resume=stage == "retry"):
                return jsonify(JOBS[f"{pid}:pipeline"]), 409
            return jsonify({"status": "running"})
        funcs = {
            "analyze": core.analyze,
            "translate": core.translate_project,
            "tts": core.generate_tts,
            "render": core.render,
            "qa": core.qa,
        }
        if stage not in funcs:
            return jsonify({"error": "未知操作"}), 404
        key = f"{pid}:{stage}"
        if not reserve_job(key):
            return jsonify(JOBS[key]), 409
        try:
            threading.Thread(target=worker, args=(pid, stage, funcs[stage]), daemon=True).start()
        except Exception:
            with JOBS_LOCK:
                JOBS.pop(key, None)
            raise
        return jsonify({"status": "running"}), 202

    @app.get("/api/projects/<pid>/jobs")
    def jobs(pid):
        return jsonify({key.split(":", 1)[1]: value for key, value in JOBS.items() if key.startswith(pid + ":")})

    @app.get("/api/projects/<pid>/log")
    def log(pid):
        path = core.project_dir(pid) / "activity.log"
        return jsonify({"log": path.read_text() if path.exists() else ""})

    @app.get("/api/projects/<pid>/download")
    def download(pid):
        project = core.load_project(pid)
        output = Path(project.get("output_audio") or "") if project.get("project_type") == "voice" else Path(project.get("output") or "")
        if not output.exists():
            return jsonify({"error": "输出文件不存在"}), 404
        download_name = f"{project.get('name','配音')}-配音音轨.wav" if project.get("project_type") == "voice" else output.name
        return send_file(output, as_attachment=True, download_name=download_name)

    @app.get("/api/projects/<pid>/preview")
    def preview(pid):
        project = core.load_project(pid)
        if project.get("project_type") == "voice":
            output = Path(project.get("output_audio") or core.project_dir(pid) / "voice_timeline.wav")
            if not output.exists():
                return jsonify({"error": "配音音轨不存在"}), 404
            return send_file(output, conditional=True, mimetype="audio/wav")
        output = Path(project.get("output") or "")
        if not output.exists():
            return jsonify({"error": "成片不存在"}), 404
        return send_file(output, conditional=True, mimetype="video/mp4")

    return app


if __name__ == "__main__":
    create_app().run(port=8765)
