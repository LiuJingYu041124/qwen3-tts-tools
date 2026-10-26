import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dubstudio as core
import webapp


class AutoPipelineTests(unittest.TestCase):
    def test_auto_pipeline_runs_every_stage_in_order_and_marks_complete(self):
        calls = []
        fake_project = {
            "id": "p1",
            "stages": {},
            "cues": [{"id": 1, "text": "你好", "enabled": True}],
            "roles": [{"id": "role1", "name": "角色1"}],
        }

        def stage(name):
            def run(pid):
                calls.append(name)
                if name == "qa":
                    fake_project["qa"] = {"passed": True}
                return fake_project
            return run

        with patch.object(core, "load_project", return_value=fake_project), \
             patch.object(core, "save_project", side_effect=lambda project: project), \
             patch.object(core, "add_log"), \
             patch.object(core, "analyze", side_effect=stage("analyze")), \
             patch.object(core, "translate_project", side_effect=stage("translate")), \
             patch.object(core, "save_manifest", side_effect=lambda pid, payload: calls.append("save") or fake_project), \
             patch.object(core, "generate_tts", side_effect=stage("tts")), \
             patch.object(core, "render", side_effect=stage("render")), \
             patch.object(core, "qa", side_effect=stage("qa")):
            result = core.auto_pipeline("p1")

        self.assertEqual(calls, ["analyze", "translate", "save", "tts", "render", "qa"])
        self.assertEqual(result["pipeline"]["status"], "done")
        self.assertEqual(result["pipeline"]["percent"], 100)

    def test_upload_endpoint_creates_project_and_starts_pipeline(self):
        app = webapp.create_app()
        client = app.test_client()
        created = {"id": "uploaded-project", "name": "clip"}

        with patch.object(core, "create_project", return_value=created) as create, \
             patch.object(core, "save_project"), \
             patch.object(webapp, "start_auto_pipeline") as start:
            response = client.post(
                "/api/auto",
                data={"video": (io.BytesIO(b"fake-video"), "clip.mp4")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json()["id"], "uploaded-project")
        create.assert_called_once()
        args, kwargs = create.call_args
        self.assertEqual(Path(args[0]).suffix, ".mp4")
        self.assertTrue(kwargs["copy_source"])
        self.assertEqual(kwargs["target_lang"], "zh")
        self.assertEqual(kwargs["tts_engine"], "local")
        start.assert_called_once_with("uploaded-project")

    def test_generation_endpoint_reports_queued_before_worker_starts(self):
        app = webapp.create_app()
        client = app.test_client()
        webapp.JOBS.clear()

        class PendingThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

        project = {
            "id": "p1", "project_type": "voice",
            "settings": core.default_settings("zh", "local"),
            "roles": [{"id": "role1", "name": "旁白", "mode": "design", "description": "自然男声"}],
            "cues": [{"id": 1, "role": "role1", "text": "测试", "enabled": True, "start": 0.0, "end": 1.0}],
        }
        with patch.object(core, "load_project", return_value=project), \
             patch.object(webapp.threading, "Thread", PendingThread):
            response = client.post("/api/projects/p1/tts")

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json()["status"], "queued")
        self.assertIn("排队中", response.get_json()["message"])
        self.assertEqual(webapp.JOBS["p1:tts"]["status"], "queued")
        webapp.JOBS.clear()

    def test_personality_direction_endpoint_updates_only_selected_role(self):
        webapp.JOBS.clear()
        app = webapp.create_app()
        client = app.test_client()
        project = {
            "id": "p1", "project_type": "voice",
            "settings": {**core.default_settings("zh", "local"), "timeline_mode": "natural"},
            "stages": {"script": "ready", "tts": "done", "render": "done", "qa": "done"},
            "roles": [
                {"id": "role1", "name": "林深", "mode": "clone", "description": "授权声线", "personality": "冷静克制"},
                {"id": "role2", "name": "晚晴", "mode": "design", "description": "年轻女声", "personality": ""},
            ],
            "cues": [
                {"id": 1, "role": "role1", "text": "马上离开。", "emotion": "旧情绪", "enabled": True},
                {"id": 2, "role": "role2", "text": "等等。", "emotion": "保留情绪", "enabled": True},
            ],
        }
        directed = [
            {"id": 1, "role": "role1", "text": "马上离开。", "emotion": "冷静但急迫、重音明确", "enabled": True},
            project["cues"][1],
        ]
        with patch.object(core, "load_project", return_value=project), \
             patch.object(core, "ollama_direct_script", return_value=(project["roles"], directed)) as direct, \
             patch.object(core, "cleanup_role_preview_files"), \
             patch.object(core, "cleanup_role_adopted_takes"), \
             patch.object(core, "cleanup_role_generation_files"), \
             patch.object(core, "cleanup_project_generation_files"), \
             patch.object(core, "save_project", side_effect=lambda value: value), \
             patch.object(core, "add_log"):
            response = client.post("/api/projects/p1/direct-role-emotions", json={"role_id": "role1"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["cues"][0]["emotion"], "冷静但急迫、重音明确")
        self.assertEqual(response.get_json()["cues"][1]["emotion"], "保留情绪")
        self.assertEqual(set(direct.call_args.args[3]), {"role1", "role2"})
        self.assertEqual(direct.call_args.args[4], [1])

    def test_preview_endpoint_reserves_job_before_starting_thread(self):
        app = webapp.create_app()
        client = app.test_client()
        webapp.JOBS.clear()

        class PendingThread:
            starts = 0
            worker_args = None

            def __init__(self, *args, **kwargs):
                type(self).worker_args = kwargs.get("args")

            def start(self):
                type(self).starts += 1

        payload = {"role_id": "role1", "text": "测试试听", "emotion": "自然", "personality_directed": True}
        project = {
            "id": "p1", "project_type": "voice", "settings": core.default_settings("zh", "local"),
            "roles": [{
                "id": "role1", "name": "旁白", "mode": "design", "description": "自然男声",
                "personality": "冷静克制，紧张时先停顿再说。",
            }],
            "cues": [],
        }
        with patch.object(core, "load_project", return_value=project), \
             patch.object(webapp.threading, "Thread", PendingThread):
            first = client.post("/api/projects/p1/preview-voice", json=payload)
            second = client.post("/api/projects/p1/preview-voice", json=payload)

        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 409)
        self.assertEqual(PendingThread.starts, 1)
        self.assertIn("按人物性格", PendingThread.worker_args[5])
        self.assertIn("音量偏低", PendingThread.worker_args[5])
        self.assertTrue(PendingThread.worker_args[6])
        self.assertIsNone(PendingThread.worker_args[7])
        self.assertTrue(PendingThread.worker_args[8])
        webapp.JOBS.clear()

    def test_preview_endpoint_passes_cue_identity_without_rewriting_saved_emotion(self):
        app = webapp.create_app()
        client = app.test_client()
        webapp.JOBS.clear()

        class PendingThread:
            worker_args = None

            def __init__(self, *args, **kwargs):
                type(self).worker_args = kwargs.get("args")

            def start(self):
                pass

        project = {
            "id": "p1", "project_type": "voice", "settings": core.default_settings("zh", "local"),
            "roles": [{
                "id": "role1", "name": "旁白", "mode": "design", "description": "自然男声",
                "personality": "冷静克制，紧张时先停顿再说。",
            }],
            "cues": [{
                "id": 1, "start": 0, "end": 2, "role": "role1", "text": "马上离开。",
                "emotion": "冷静但急迫", "speed": 1, "gap_after": .25, "enabled": True,
            }],
        }
        payload = {
            "role_id": "role1", "text": "马上离开。", "emotion": "冷静但急迫",
            "personality_directed": True, "cue_id": 1, "preview_nonce": "nonce-1",
        }
        with patch.object(core, "load_project", return_value=project), \
             patch.object(webapp.threading, "Thread", PendingThread):
            response = client.post("/api/projects/p1/preview-voice", json=payload)

        self.assertEqual(response.status_code, 202)
        self.assertEqual(PendingThread.worker_args[5], "冷静但急迫")
        self.assertTrue(PendingThread.worker_args[6])
        self.assertEqual(PendingThread.worker_args[7], 1)
        self.assertEqual(PendingThread.worker_args[8], "nonce-1")
        webapp.JOBS.clear()

    def test_upload_endpoint_rejects_missing_video(self):
        app = webapp.create_app()
        response = app.test_client().post("/api/auto", data={}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("视频", response.get_json()["error"])

    def test_retry_resumes_from_failed_stage_without_repeating_completed_work(self):
        calls = []
        project = {
            "id": "p1",
            "stages": {"analyze": "done", "script": "ready", "tts": "failed", "render": "pending", "qa": "pending"},
            "pipeline": {"status": "failed", "stage": "tts", "percent": 48},
            "qa": {},
            "cues": [{"id": 1, "text": "你好", "enabled": True}],
            "roles": [{"id": "role1"}],
        }

        def run(name):
            def inner(pid):
                calls.append(name)
                project["stages"][name] = "done"
                if name == "qa":
                    project["qa"] = {"passed": True}
                return project
            return inner

        with patch.object(core, "load_project", return_value=project), \
             patch.object(core, "save_project", side_effect=lambda value: value), \
             patch.object(core, "add_log"), \
             patch.object(core, "analyze", side_effect=run("analyze")), \
             patch.object(core, "translate_project", side_effect=run("translate")), \
             patch.object(core, "save_manifest", side_effect=lambda *args: calls.append("save") or project), \
             patch.object(core, "generate_tts", side_effect=run("tts")), \
             patch.object(core, "render", side_effect=run("render")), \
             patch.object(core, "qa", side_effect=run("qa")):
            core.auto_pipeline("p1", resume=True)
        self.assertEqual(calls, ["tts", "render", "qa"])

    def test_auto_pipeline_for_voice_project_skips_media_stages(self):
        project = {
            "id": "p1",
            "project_type": "voice",
            "settings": core.default_settings("zh", "local"),
            "stages": {"script": "ready", "tts": "pending", "render": "pending", "qa": "pending"},
            "cues": [{"id": 1, "role": "role1", "text": "测试", "enabled": True}],
            "roles": [{"id": "role1", "name": "旁白", "mode": "design", "description": "自然男声"}],
        }
        with patch.object(core, "load_project", return_value=project), \
             patch.object(core, "generate_tts", return_value=project) as generate_tts, \
             patch.object(core, "analyze") as analyze, \
             patch.object(core, "translate_project") as translate:
            result = core.auto_pipeline("p1")
        self.assertIs(result, project)
        generate_tts.assert_called_once_with("p1", reuse=True)
        analyze.assert_not_called()
        translate.assert_not_called()

    def test_retry_for_voice_project_starts_tts_without_reanalyzing_document(self):
        app = webapp.create_app()
        client = app.test_client()
        webapp.JOBS.clear()

        class PendingThread:
            worker_args = None

            def __init__(self, *args, **kwargs):
                type(self).worker_args = kwargs.get("args")

            def start(self):
                pass

        project = {
            "id": "p1",
            "project_type": "voice",
            "settings": core.default_settings("zh", "local"),
            "stages": {"script": "ready", "tts": "failed", "render": "pending", "qa": "pending"},
            "roles": [{"id": "role1", "name": "旁白", "mode": "design", "description": "自然男声"}],
            "cues": [{"id": 1, "role": "role1", "text": "测试", "enabled": True, "start": 0.0, "end": 1.0}],
        }
        with patch.object(core, "load_project", return_value=project), \
             patch.object(webapp.threading, "Thread", PendingThread):
            response = client.post("/api/projects/p1/retry")

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json()["status"], "queued")
        self.assertEqual(PendingThread.worker_args[0:2], ("p1", "tts"))
        self.assertIs(PendingThread.worker_args[2], core.generate_tts)
        self.assertNotIn("pipeline", webapp.JOBS["p1:tts"])
        webapp.JOBS.clear()

    def test_auto_pipeline_marks_failed_stage_in_project_state(self):
        project = {
            "id": "p1",
            "settings": {"target_lang": "en", "tts_engine": "local"},
            "stages": {"analyze": "done", "script": "ready", "tts": "pending", "render": "pending", "qa": "pending"},
            "pipeline": {"status": "pending", "stage": "script", "percent": 42},
            "cues": [{"id": 1, "text": "Hello", "enabled": True}],
            "roles": [{"id": "role1"}],
        }
        with patch.object(core, "load_project", return_value=project), \
             patch.object(core, "save_project", side_effect=lambda value: value), \
             patch.object(core, "add_log"), \
             patch.object(core, "generate_tts", side_effect=RuntimeError("time fit failed")):
            with self.assertRaisesRegex(RuntimeError, "time fit failed"):
                core.auto_pipeline("p1", resume=True)
        self.assertEqual(project["stages"]["tts"], "failed")
        self.assertEqual(project["pipeline"]["status"], "failed")
        self.assertEqual(project["pipeline"]["stage"], "tts")

    def test_translation_payload_includes_spoken_length_budget(self):
        payload = core.translation_payload(
            [{"id": 7, "start": 2.0, "end": 3.0, "source_text": "家里就是个卖鱼的"}],
            "en",
        )
        self.assertEqual(payload[0]["slot_seconds"], 1.0)
        self.assertEqual(payload[0]["max_words_or_characters"], 3)
        self.assertEqual(payload[0]["source"], "家里就是个卖鱼的")


class VoiceStudioTests(unittest.TestCase):
    def test_create_voice_project_has_natural_timeline_and_design_role(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("广播剧", "zh")
            self.assertEqual(project["project_type"], "voice")
            self.assertEqual(project["settings"]["timeline_mode"], "natural")
            self.assertEqual(project["roles"][0]["mode"], "design")
            self.assertEqual(project["cues"][0]["role"], project["roles"][0]["id"])
            self.assertTrue((Path(directory) / project["id"] / "project.json").exists())

    def test_home_keeps_document_dialog_markup_and_templates_fresh(self):
        app = webapp.create_app()
        self.assertTrue(app.config["TEMPLATES_AUTO_RELOAD"])
        self.assertTrue(app.jinja_env.auto_reload)
        response = app.test_client().get("/")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="documentDialog"', html)
        self.assertIn('id="documentDirection"', html)
        self.assertIn('id="newProjectDirection"', html)
        self.assertIn('id="documentVoiceMode"', html)
        self.assertIn('id="createVoiceMode"', html)
        self.assertIn('value="clone">不生成音色，导入后配置声音克隆', html)
        self.assertIn('id="importedVoiceGuide"', html)
        self.assertIn('id="deleteProjectDialog"', html)
        self.assertIn('id="cueRoleFilters"', html)
        self.assertIn('id="cueSearch"', html)
        self.assertIn('/static/app.js?v=studio28', html)
        self.assertIn('/static/style.css?v=studio27', html)

    def test_document_voice_options_wait_for_backend_support(self):
        from flask import render_template
        app = webapp.create_app()
        with app.test_request_context("/"):
            html = render_template("index.html")
        self.assertNotIn('id="documentVoiceMode"', html)
        self.assertNotIn('id="createVoiceMode"', html)
        self.assertIn('id="documentDialog"', html)
        self.assertIn('id="createDialog"', html)

    def test_cue_search_rerender_does_not_force_focus_or_selection(self):
        script = Path(core.__file__).with_name("static").joinpath("app.js").read_text()
        start = script.index("function setCueSearch(value)")
        end = script.index("\n}\n", start) + 2
        handler = script[start:end]
        self.assertIn("renderCues();", handler)
        self.assertNotIn("focus()", handler)
        self.assertNotIn("setSelectionRange", handler)

    def test_role_deletion_saves_pending_project_edits_first(self):
        script = Path(core.__file__).with_name("static").joinpath("app.js").read_text()
        start = script.index("async function removeRole(index)")
        end = script.index("\n}\n", start) + 2
        handler = script[start:end]
        self.assertLess(handler.index("await saveProject(true)"), handler.index('method: "DELETE"'))

    def test_cue_download_returns_final_fitted_wav_with_readable_filename(self):
        app = webapp.create_app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("逐句导出", "zh")
            project = core.load_project(project["id"])
            project["roles"][0]["name"] = "旁白 / A"
            project["cues"][0].update({"text": "你好，世界！", "enabled": True})
            project["stages"]["tts"] = "done"
            core.save_project(project)
            fitted = core.project_dir(project["id"]) / "fit" / "001.wav"
            fitted.parent.mkdir(parents=True, exist_ok=True)
            fitted.write_bytes(b"final-fitted-wave")

            response = client.get(f"/api/projects/{project['id']}/cues/1/download")
            body = response.get_data()
            disposition = response.headers.get("Content-Disposition", "")
            response.close()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, b"final-fitted-wave")
        self.assertEqual(response.mimetype, "audio/wav")
        self.assertIn("attachment", disposition)
        self.assertIn("001-", disposition)
        self.assertIn(".wav", disposition)

    def test_cue_download_requires_current_generated_output(self):
        app = webapp.create_app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("待生成", "zh")
            response = client.get(f"/api/projects/{project['id']}/cues/1/download")

        self.assertEqual(response.status_code, 409)
        self.assertIn("先生成整轨", response.get_json()["error"])

    def test_cue_download_rejects_disabled_cue(self):
        app = webapp.create_app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("禁用台词", "zh")
            project = core.load_project(project["id"])
            project["cues"][0]["enabled"] = False
            project["stages"]["tts"] = "done"
            core.save_project(project)
            response = client.get(f"/api/projects/{project['id']}/cues/1/download")

        self.assertEqual(response.status_code, 409)
        self.assertIn("未启用", response.get_json()["error"])

    def test_delete_project_endpoint_removes_entire_project_directory(self):
        webapp.JOBS.clear()
        app = webapp.create_app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("待删除项目", "zh")
            root = core.project_dir(project["id"])
            (root / "previews").mkdir()
            (root / "takes").mkdir()
            (root / "previews" / "candidate.wav").write_bytes(b"preview")
            (root / "takes" / "locked.wav").write_bytes(b"take")
            (root / "voice_timeline.wav").write_bytes(b"output")

            response = client.delete(f"/api/projects/{project['id']}")

            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.get_json()["deleted"])
            self.assertEqual(response.get_json()["name"], "待删除项目")
            self.assertFalse(root.exists())

    def test_delete_project_endpoint_blocks_running_project(self):
        webapp.JOBS.clear()
        app = webapp.create_app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("运行中项目", "zh")
            root = core.project_dir(project["id"])
            webapp.JOBS[f"{project['id']}:tts"] = {"status": "running", "message": "生成中"}

            response = client.delete(f"/api/projects/{project['id']}")

            self.assertEqual(response.status_code, 409)
            self.assertIn("正在生成", response.get_json()["error"])
            self.assertTrue(root.exists())
        webapp.JOBS.clear()

    def test_delete_project_endpoint_returns_404_for_missing_project(self):
        webapp.JOBS.clear()
        app = webapp.create_app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            response = client.delete("/api/projects/not-found")
        self.assertEqual(response.status_code, 404)
        self.assertIn("不存在", response.get_json()["error"])

    def test_voice_project_api_uses_blank_project_creator(self):
        app = webapp.create_app()
        client = app.test_client()
        created = {"id": "voice-project", "project_type": "voice", "name": "旁白"}
        with patch.object(core, "create_voice_project", return_value=created) as create:
            response = client.post(
                "/api/projects",
                json={
                    "project_type": "voice",
                    "name": "旁白",
                    "target_lang": "zh",
                    "direction_brief": "纪录片旁白，沉稳可信",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["project_type"], "voice")
        create.assert_called_once_with("旁白", "zh", "纪录片旁白，沉稳可信")

    def test_role_personality_is_normalized_and_saved(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("性格测试", "zh")
            payload = core.load_project(project["id"])
            payload["roles"][0]["personality"] = "  外冷内热，责任感强。\n生气时不吼，音量更低、咬字更重。  "
            saved = core.save_manifest(project["id"], payload)
        self.assertEqual(saved["roles"][0]["personality"], "外冷内热，责任感强。 生气时不吼，音量更低、咬字更重。")

    def _build_generation_cache_project(self, directory):
        project = core.create_voice_project("逐句清理", "zh")
        project = core.load_project(project["id"])
        project["roles"].append({
            "id": "role2", "name": "角色2", "voice": "female_warm", "mode": "design",
            "description": "自然女声", "style": "自然女声", "personality": "",
        })
        project["cues"] = [
            {"id": 1, "start": 0, "end": 2, "text": "第一句", "role": "role1", "emotion": "自然", "enabled": True},
            {"id": 2, "start": 2, "end": 4, "text": "第二句", "role": "role1", "emotion": "自然", "enabled": True},
            {"id": 3, "start": 4, "end": 6, "text": "第三句", "role": "role2", "emotion": "自然", "enabled": True},
        ]
        project["stages"].update({"tts": "done", "render": "done", "qa": "done"})
        project["pipeline"] = {"status": "done", "stage": "complete", "percent": 100}
        project["qa"] = {"passed": True}
        root = core.project_dir(project["id"])
        artifact_paths = (
            root / "voice_timeline.wav", root / "status.json", root / "summary.json", root / "qa.json",
            root / "output-video.mp4",
        )
        project["output_audio"] = str(artifact_paths[0])
        project["output"] = str(artifact_paths[-1])
        for path in artifact_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"artifact-{path.name}".encode())
        cache_paths = {}
        for cue in project["cues"]:
            cue_id = int(cue["id"])
            role_id = cue["role"]
            paths = [
                root / "raw" / f"{cue_id:03d}-{role_id}.wav",
                root / "raw" / f"{cue_id:03d}-{role_id}.json",
                root / "raw" / f"{cue_id:03d}-{role_id}.failed.json",
                root / "clean" / f"{cue_id:03d}.wav",
                root / "clean" / f"{cue_id:03d}.json",
                root / "clean" / f"{cue_id:03d}.wav.json",
                root / "fit" / f"{cue_id:03d}.wav",
                root / "fit" / f"{cue_id:03d}.json",
                root / "fit" / f"{cue_id:03d}.wav.json",
            ]
            cache_paths[cue_id] = paths
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"cache-{cue_id}-{path.name}".encode())
        core.save_project(project)
        return project, root, cache_paths

    def test_adopting_one_cue_cleans_only_its_generation_cache_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project, root, cache_paths = self._build_generation_cache_project(directory)
            preview_dir = root / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview = preview_dir / "role1-candidate.wav"
            preview.write_bytes(b"candidate-wave")
            cue = project["cues"][0]
            signature = core.cue_take_signature(project, cue, project["roles"][0])
            core.write_json(preview.with_suffix(".json"), {
                "cue_id": cue["id"], "emotion": cue["emotion"], "signature": signature, "seed": 123456,
            })
            preserved = {
                path: path.read_bytes()
                for cue_id in (2, 3)
                for path in cache_paths[cue_id]
            }

            saved = core.adopt_preview_take(project["id"], cue["id"], preview.name)

            for path in cache_paths[1]:
                self.assertFalse(path.exists(), path)
            for path, contents in preserved.items():
                self.assertEqual(path.read_bytes(), contents, path)
            for filename in ("voice_timeline.wav", "status.json", "summary.json", "qa.json", "output-video.mp4"):
                self.assertFalse(root.joinpath(filename).exists())
            self.assertNotIn("output_audio", saved)
            self.assertNotIn("output", saved)
            self.assertNotIn("qa", saved)
            self.assertEqual(saved["stages"]["tts"], "pending")
            self.assertTrue(preview.exists())
            take = saved["cues"][0]["adopted_take"]
            self.assertEqual(core.adopted_take_path(project["id"], take).read_bytes(), b"candidate-wave")

    def test_discarding_one_cue_cleans_only_its_generation_cache_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project, root, cache_paths = self._build_generation_cache_project(directory)
            preview_dir = root / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview = preview_dir / "role1-candidate.wav"
            preview.write_bytes(b"candidate-wave")
            cue = project["cues"][0]
            signature = core.cue_take_signature(project, cue, project["roles"][0])
            core.write_json(preview.with_suffix(".json"), {
                "cue_id": cue["id"], "emotion": cue["emotion"], "signature": signature, "seed": 123456,
            })
            adopted = core.adopt_preview_take(project["id"], cue["id"], preview.name)
            take_path = core.adopted_take_path(project["id"], adopted["cues"][0]["adopted_take"])
            preserved = {
                path: path.read_bytes()
                for cue_id in (2, 3)
                for path in cache_paths[cue_id]
            }
            for path in cache_paths[1]:
                path.write_bytes(f"recreated-{path.name}".encode())
            artifacts = {
                "voice_timeline.wav": b"timeline-again", "status.json": b"status-again",
                "summary.json": b"summary-again", "qa.json": b"qa-again", "output-video.mp4": b"output-again",
            }
            for filename, contents in artifacts.items():
                (root / filename).write_bytes(contents)
            reloaded = core.load_project(project["id"])
            reloaded["output_audio"] = str(root / "voice_timeline.wav")
            reloaded["output"] = str(root / "output-video.mp4")
            reloaded["qa"] = {"passed": True}
            reloaded["stages"].update({"tts": "done", "render": "done", "qa": "done"})
            core.save_project(reloaded)

            saved = core.discard_adopted_take(project["id"], cue["id"])

            self.assertFalse(take_path.exists())
            for path in cache_paths[1]:
                self.assertFalse(path.exists(), path)
            for path, contents in preserved.items():
                self.assertEqual(path.read_bytes(), contents, path)
            for filename in artifacts:
                self.assertFalse((root / filename).exists())
            self.assertNotIn("output_audio", saved)
            self.assertNotIn("output", saved)
            self.assertNotIn("qa", saved)
            self.assertNotIn("adopted_take", saved["cues"][0])
            self.assertEqual(saved["stages"]["tts"], "pending")

    def test_discard_without_adopted_take_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project, root, cache_paths = self._build_generation_cache_project(directory)
            project_json = root / "project.json"
            before_project = project_json.read_bytes()
            before_files = {
                path: path.read_bytes()
                for paths in cache_paths.values()
                for path in paths
            }
            before_artifacts = {
                path: path.read_bytes()
                for path in (
                    root / "voice_timeline.wav", root / "status.json", root / "summary.json", root / "qa.json",
                    root / "output-video.mp4",
                )
            }

            returned = core.discard_adopted_take(project["id"], 1)

            self.assertEqual(project_json.read_bytes(), before_project)
            for path, contents in {**before_files, **before_artifacts}.items():
                self.assertEqual(path.read_bytes(), contents, path)
            self.assertNotIn("adopted_take", returned["cues"][0])

    def _stage_review_preview(self, project, root, gate_transcript="姐姐还在养病"):
        """Write a preview take the ASR gate rejected, as the worker would."""
        preview_dir = root / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview = preview_dir / "role1-review.wav"
        preview.write_bytes(b"reviewed-wave")
        cue = project["cues"][0]
        signature = core.cue_take_signature(project, cue, project["roles"][0])
        core.write_json(preview.with_suffix(".json"), {
            "cue_id": cue["id"], "emotion": cue["emotion"], "signature": signature, "seed": 4242,
        })
        core.write_json(preview_dir / "role1-review.review.json", {
            "gate_passed": False,
            "gate_label": "clone-pronunciation:clone-name-pair-杰:clear-name-pair:930101",
            "gate_transcript": gate_transcript,
            "gate_quality": {"passed": False, "length_ok": False, "phonetic_ok": False, "similarity": 0.81},
            "expected_text": cue["text"],
        })
        return preview, cue

    def test_reviewed_take_requires_explicit_human_confirmation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project, root, _cache = self._build_generation_cache_project(directory)
            preview, cue = self._stage_review_preview(project, root)

            with self.assertRaisesRegex(RuntimeError, "请先完整听完"):
                core.adopt_preview_take(project["id"], cue["id"], preview.name)
            self.assertNotIn("adopted_take", core.load_project(project["id"])["cues"][0])

            saved = core.adopt_preview_take(
                project["id"], cue["id"], preview.name,
                confirm_gate_rejection=True, review_note="整句都听完了，名字没问题",
            )
            take = saved["cues"][0]["adopted_take"]
            self.assertTrue(take["reviewed"])
            self.assertEqual(take["review_transcript"], "姐姐还在养病")
            self.assertEqual(take["review_expected_text"], cue["text"])
            self.assertIn("clone-name-pair-杰", take["review_gate_reason"])
            self.assertEqual(take["review_note"], "整句都听完了，名字没问题")
            self.assertTrue(take["reviewed_at"])
            self.assertEqual(core.adopted_take_path(project["id"], take).read_bytes(), b"reviewed-wave")
            self.assertIn("人工复听确认", saved["last_message"])

    def test_confirming_a_clean_preview_needs_no_review_audit(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project, root, _cache = self._build_generation_cache_project(directory)
            preview_dir = root / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview = preview_dir / "role1-clean.wav"
            preview.write_bytes(b"clean-wave")
            cue = project["cues"][0]
            core.write_json(preview.with_suffix(".json"), {
                "cue_id": cue["id"], "emotion": cue["emotion"],
                "signature": core.cue_take_signature(project, cue, project["roles"][0]), "seed": 7,
            })
            self.assertEqual(core.preview_gate_report(project["id"], preview.name), {})

            saved = core.adopt_preview_take(project["id"], cue["id"], preview.name)

            take = saved["cues"][0]["adopted_take"]
            self.assertNotIn("reviewed", take)
            self.assertIn("已采用台词1", saved["last_message"])

    def test_reviewed_take_survives_script_edits_that_invalidate_a_plain_take(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project, root, _cache = self._build_generation_cache_project(directory)
            preview, cue = self._stage_review_preview(project, root)
            core.adopt_preview_take(project["id"], cue["id"], preview.name, confirm_gate_rejection=True)

            payload = core.load_project(project["id"])
            payload["cues"][0]["text"] = "第一句改过了"
            saved = core.save_manifest(project["id"], payload)

            reviewed_cue = saved["cues"][0]
            self.assertIn("adopted_take", reviewed_cue)
            self.assertIsNone(core.cue_adopted_take(saved, reviewed_cue))
            self.assertIsNotNone(core.cue_adopted_take(saved, reviewed_cue, audit_reviewed=True))
            self.assertTrue(core.adopted_take_path(saved["id"], reviewed_cue["adopted_take"]).is_file())
            self.assertNotIn(
                int(reviewed_cue["id"]),
                [row["cue_id"] for row in core.cue_blockers(saved["id"], saved)],
            )

    def test_plain_take_is_dropped_when_the_script_changes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project, root, _cache = self._build_generation_cache_project(directory)
            preview_dir = root / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview = preview_dir / "role1-clean.wav"
            preview.write_bytes(b"clean-wave")
            cue = project["cues"][0]
            core.write_json(preview.with_suffix(".json"), {
                "cue_id": cue["id"], "emotion": cue["emotion"],
                "signature": core.cue_take_signature(project, cue, project["roles"][0]), "seed": 7,
            })
            adopted = core.adopt_preview_take(project["id"], cue["id"], preview.name)
            take_path = core.adopted_take_path(project["id"], adopted["cues"][0]["adopted_take"])

            payload = core.load_project(project["id"])
            payload["cues"][0]["text"] = "第一句改过了"
            saved = core.save_manifest(project["id"], payload)

            self.assertNotIn("adopted_take", saved["cues"][0])
            self.assertFalse(take_path.exists())

    def test_cue_blockers_report_why_the_track_cannot_publish(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project, root, _cache = self._build_generation_cache_project(directory)
            raw = root / "raw"
            (raw / "001-role1.wav").write_bytes(b"accepted-raw")
            (raw / "002-role1.wav").unlink()
            (raw / "003-role2.wav").unlink()
            core.write_json(raw / "002-role1.failed.json", {
                "cue_id": 2, "role": "role1", "mode": "clone",
                "error": "台词2完整性校验失败，未保存缺字音频：姐姐还在养病",
                "timestamp": "2026-09-17T15:00:00",
            })

            blockers = core.cue_blockers(project["id"], core.load_project(project["id"]))

            self.assertEqual([row["cue_id"] for row in blockers], [2, 3])
            self.assertEqual(blockers[0]["error"], "台词2完整性校验失败，未保存缺字音频：姐姐还在养病")
            self.assertEqual(blockers[0]["mode"], "clone")
            self.assertEqual(blockers[1]["error"], "这次生成没有产出可用音频")

    def test_cue_blockers_stay_empty_before_the_first_generation_attempt(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("还没生成过", "zh")
            project = core.load_project(project["id"])
            project["cues"] = [
                {"id": 1, "start": 0, "end": 2, "text": "杰杰还在养病", "role": "role1", "emotion": "自然"},
            ]
            project["stages"]["tts"] = "pending"
            core.save_project(project)

            self.assertEqual(core.cue_blockers(project["id"]), [])

    def test_cue_blockers_ignore_cues_with_a_reviewed_take(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project, root, _cache = self._build_generation_cache_project(directory)
            preview, cue = self._stage_review_preview(project, root)
            core.adopt_preview_take(project["id"], cue["id"], preview.name, confirm_gate_rejection=True)

            blockers = core.cue_blockers(project["id"], core.load_project(project["id"]))

            self.assertNotIn(cue["id"], [row["cue_id"] for row in blockers])

    def test_review_queue_lists_published_cues_the_gate_flagged(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project, root, _cache = self._build_generation_cache_project(directory)
            core.write_json(root / "status.json", [
                {"id": 1, "role": "role1", "text": "第一句", "verified": True, "needs_review": False},
                {"id": 2, "role": "role1", "text": "杰杰还在养病", "verified": False, "needs_review": True,
                 "gate_transcript": "姐姐还在养病", "gate_reason": "clone-pronunciation:clone-name-pair-杰"},
                {"id": 3, "role": "role2", "text": "第三句", "verified": False, "needs_review": True,
                 "gate_transcript": "第三句啊", "degraded": True},
            ])

            rows = core.cue_review_queue(project["id"], core.load_project(project["id"]))

            self.assertEqual([row["cue_id"] for row in rows], [2, 3])
            self.assertEqual(rows[0]["transcript"], "姐姐还在养病")
            self.assertEqual(rows[0]["reason"], "clone-pronunciation:clone-name-pair-杰")
            self.assertFalse(rows[0]["degraded"])
            self.assertTrue(rows[1]["degraded"])

    def test_review_queue_is_empty_when_every_cue_passed(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project, root, _cache = self._build_generation_cache_project(directory)
            core.write_json(root / "status.json", [
                {"id": 1, "role": "role1", "text": "第一句", "verified": True, "needs_review": False},
            ])

            self.assertEqual(core.cue_review_queue(project["id"]), [])

    def test_adopting_a_flagged_cue_clears_it_from_the_review_queue(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project, root, _cache = self._build_generation_cache_project(directory)
            preview, cue = self._stage_review_preview(project, root)
            core.write_json(root / "status.json", [
                {"id": 2, "role": "role2", "text": "第二句", "needs_review": True, "gate_transcript": "第二句啊"},
                {"id": cue["id"], "role": "role1", "text": cue["text"], "needs_review": True,
                 "gate_transcript": "姐妹是你救了我吗谢谢你"},
            ])
            self.assertEqual([row["cue_id"] for row in core.cue_review_queue(project["id"])], [cue["id"], 2])

            core.adopt_preview_take(project["id"], cue["id"], preview.name, confirm_gate_rejection=True)

            # Adopting invalidates the published track, so the stale review list goes
            # with it; the next build re-grades the cue from the adopted audio.
            self.assertEqual(core.cue_review_queue(project["id"]), [])

    def test_role_generation_cleanup_cleans_all_role_cues_only(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project, _root, cache_paths = self._build_generation_cache_project(directory)
            preserved = {path: path.read_bytes() for path in cache_paths[3]}

            core.cleanup_role_generation_files(project, "role1")

            for cue_id in (1, 2):
                for path in cache_paths[cue_id]:
                    self.assertFalse(path.exists(), path)
            for path, contents in preserved.items():
                self.assertEqual(path.read_bytes(), contents, path)

    def test_adopt_preview_take_persists_locked_audio_and_manifest_metadata(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("可信 Take", "zh")
            project = core.load_project(project["id"])
            cue = project["cues"][0]
            cue.update({"text": "马上离开。", "emotion": "冷静但急迫"})
            core.save_project(project)
            preview_dir = core.project_dir(project["id"]) / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview = preview_dir / f"{cue['role']}-candidate.wav"
            preview.write_bytes(b"candidate-wave")
            role = project["roles"][0]
            signature = core.cue_take_signature(project, cue, role)
            core.write_json(preview.with_suffix(".json"), {
                "cue_id": cue["id"], "emotion": cue["emotion"], "signature": signature, "seed": 123456,
            })

            saved = core.adopt_preview_take(project["id"], cue["id"], preview.name)
            take = saved["cues"][0]["adopted_take"]
            take_path = core.adopted_take_path(project["id"], take)

            self.assertEqual(take["signature"], signature)
            self.assertEqual(take["seed"], 123456)
            self.assertEqual(take_path.read_bytes(), b"candidate-wave")
            self.assertEqual(core.load_project(project["id"])["cues"][0]["adopted_take"]["id"], take["id"])

    def test_cue_adopted_take_rejects_tampered_audio(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("可信 Take", "zh")
            project = core.load_project(project["id"])
            cue = project["cues"][0]
            cue.update({"text": "锁定版本", "emotion": "自然"})
            takes = core.project_dir(project["id"]) / "takes"
            takes.mkdir(parents=True, exist_ok=True)
            take_path = takes / "cue-001-locked.wav"
            take_path.write_bytes(b"original")
            cue["adopted_take"] = {
                "id": "locked", "file": take_path.name,
                "signature": core.cue_take_signature(project, cue, project["roles"][0]),
                "sha256": core.file_sha256(take_path), "seed": 1,
            }
            core.save_project(project)
            take_path.write_bytes(b"tampered")
            reloaded = core.load_project(project["id"])

            self.assertIsNone(core.cue_adopted_take(reloaded, reloaded["cues"][0]))

    def test_adopt_take_endpoint_rejects_preview_from_another_cue(self):
        webapp.JOBS.clear()
        app = webapp.create_app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("可信 Take", "zh")
            project = core.load_project(project["id"])
            project["cues"] = [
                {**project["cues"][0], "id": 1, "text": "第一句", "emotion": "自然"},
                {**project["cues"][0], "id": 2, "text": "第二句", "emotion": "自然"},
            ]
            core.save_project(project)
            preview_dir = core.project_dir(project["id"]) / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            role = project["roles"][0]
            preview = preview_dir / f"{role['id']}-candidate.wav"
            preview.write_bytes(b"candidate-wave")
            core.write_json(preview.with_suffix(".json"), {
                "cue_id": 1,
                "emotion": "自然",
                "signature": core.cue_take_signature(project, project["cues"][0], role),
                "seed": 123,
            })

            response = client.post(
                f"/api/projects/{project['id']}/cues/2/adopt-take",
                json={"file": preview.name},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("不属于", response.get_json()["error"])

    def test_save_manifest_keeps_take_for_postprocess_changes_and_invalidates_generation_changes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("可信 Take", "zh")
            project = core.load_project(project["id"])
            cue = project["cues"][0]
            cue.update({"text": "第一版", "emotion": "自然"})
            takes = core.project_dir(project["id"]) / "takes"
            takes.mkdir(parents=True, exist_ok=True)
            take_file = "cue-001-existing.wav"
            (takes / take_file).write_bytes(b"locked")
            cue["adopted_take"] = {
                "id": "existing", "file": take_file,
                "signature": core.cue_take_signature(project, cue, project["roles"][0]),
                "seed": 10,
            }
            core.save_project(project)

            same_payload = core.load_project(project["id"])
            same_saved = core.save_manifest(project["id"], same_payload)
            speed_payload = core.load_project(project["id"])
            speed_payload["cues"][0]["speed"] = 1.2
            speed_saved = core.save_manifest(project["id"], speed_payload)
            disabled_payload = core.load_project(project["id"])
            disabled_payload["cues"][0]["enabled"] = False
            disabled_saved = core.save_manifest(project["id"], disabled_payload)
            changed_payload = core.load_project(project["id"])
            changed_payload["cues"][0]["text"] = "第二版"
            changed_saved = core.save_manifest(project["id"], changed_payload)

        self.assertIn("adopted_take", same_saved["cues"][0])
        self.assertIn("adopted_take", speed_saved["cues"][0])
        self.assertIn("adopted_take", disabled_saved["cues"][0])
        self.assertNotIn("adopted_take", changed_saved["cues"][0])
        self.assertFalse((takes / take_file).exists())

    def test_save_manifest_keeps_take_with_original_cue_when_reordered(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("可信 Take", "zh")
            project = core.load_project(project["id"])
            base = project["cues"][0]
            project["cues"] = [
                {**base, "id": 1, "text": "第一句"},
                {**base, "id": 2, "text": "第二句"},
            ]
            takes = core.project_dir(project["id"]) / "takes"
            takes.mkdir(parents=True, exist_ok=True)
            for cue in project["cues"]:
                take_file = f"cue-{cue['id']:03d}-take.wav"
                take_path = takes / take_file
                take_path.write_bytes(cue["text"].encode())
                cue["adopted_take"] = {
                    "id": cue["text"], "file": take_file,
                    "signature": core.cue_take_signature(project, cue, project["roles"][0]),
                    "sha256": core.file_sha256(take_path), "seed": cue["id"],
                }
            core.save_project(project)

            payload = core.load_project(project["id"])
            payload["cues"] = [payload["cues"][1], payload["cues"][0]]
            saved = core.save_manifest(project["id"], payload)
            take_validity = [bool(core.cue_adopted_take(saved, cue)) for cue in saved["cues"]]

        self.assertEqual([(cue["id"], cue["text"], cue["adopted_take"]["id"]) for cue in saved["cues"]], [
            (1, "第二句", "第二句"), (2, "第一句", "第一句"),
        ])
        self.assertEqual(take_validity, [True, True])

    def test_save_manifest_deleting_a_cue_keeps_surviving_take_and_removes_orphan(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("可信 Take", "zh")
            project = core.load_project(project["id"])
            base = project["cues"][0]
            project["cues"] = [
                {**base, "id": 1, "text": "第一句"},
                {**base, "id": 2, "text": "第二句"},
            ]
            takes = core.project_dir(project["id"]) / "takes"
            takes.mkdir(parents=True, exist_ok=True)
            for cue in project["cues"]:
                take_file = f"cue-{cue['id']:03d}-take.wav"
                take_path = takes / take_file
                take_path.write_bytes(cue["text"].encode())
                cue["adopted_take"] = {
                    "id": cue["text"], "file": take_file,
                    "signature": core.cue_take_signature(project, cue, project["roles"][0]),
                    "sha256": core.file_sha256(take_path), "seed": cue["id"],
                }
            core.save_project(project)

            payload = core.load_project(project["id"])
            payload["cues"].pop(0)
            saved = core.save_manifest(project["id"], payload)
            files = sorted(path.name for path in takes.glob("*.wav"))
            surviving_take_is_valid = bool(core.cue_adopted_take(saved, saved["cues"][0]))

        self.assertEqual([(cue["id"], cue["text"], cue["adopted_take"]["id"]) for cue in saved["cues"]], [
            (1, "第二句", "第二句"),
        ])
        self.assertEqual(files, ["cue-002-take.wav"])
        self.assertTrue(surviving_take_is_valid)

    def test_clone_role_requires_consent_and_reference_audio(self):
        project = {
            "roles": [{"id": "role1", "name": "角色", "mode": "clone", "clone_consent": False, "ref_audio": ""}],
            "cues": [{"role": "role1", "enabled": True}],
        }
        with self.assertRaisesRegex(RuntimeError, "授权"):
            core.validate_clone_roles(project)
        project["roles"][0]["clone_consent"] = True
        with self.assertRaisesRegex(RuntimeError, "参考音频"):
            core.validate_clone_roles(project)

    def test_clone_role_requires_confirmed_transcript_for_current_reference(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as reference:
            project = {
                "roles": [{
                    "id": "role1", "name": "角色", "mode": "clone", "clone_consent": True,
                    "ref_audio": reference.name, "ref_text": "",
                }],
                "cues": [{"role": "role1", "enabled": True}],
            }
            with self.assertRaisesRegex(RuntimeError, "逐字稿"):
                core.validate_clone_roles(project)
            project["roles"][0].update({
                "ref_text": "这是准确参考文本", "reference_revision": "rev-1",
                "ref_text_confirmed": False, "ref_text_confirmed_revision": "",
            })
            with self.assertRaisesRegex(RuntimeError, "核听确认"):
                core.validate_clone_roles(project)
            project["roles"][0].update({"ref_text_confirmed": True, "ref_text_confirmed_revision": "rev-1"})
            core.validate_clone_roles(project)
            project["roles"][0]["reference_revision"] = "rev-2"
            with self.assertRaisesRegex(RuntimeError, "核听确认"):
                core.validate_clone_roles(project)

    def test_reference_upload_binds_normalized_audio_and_consent(self):
        webapp.JOBS.clear()
        app = webapp.create_app()
        client = app.test_client()
        project = {
            "id": "p1", "project_type": "voice", "settings": {},
            "roles": [{"id": "role1", "name": "角色", "mode": "design"}], "cues": [],
        }
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(core, "load_project", return_value=project), \
             patch.object(core, "project_dir", return_value=Path(directory)), \
             patch.object(core, "media_tool", return_value="ffmpeg"), \
             patch.object(core, "run", side_effect=lambda command: Path(command[-1]).write_bytes(b"wav")) as run, \
             patch.object(webapp, "reference_duration", return_value=6.0), \
             patch.object(core, "file_sha256", return_value="audio-hash"), \
             patch.object(core, "transcribe_reference_audio", return_value="参考音频真实台词") as transcribe, \
             patch.object(core, "save_manifest", side_effect=lambda pid, value: value), \
             patch.object(core, "save_project", side_effect=lambda value: value):
            response = client.post(
                "/api/projects/p1/roles/role1/reference",
                data={
                    "audio": (io.BytesIO(b"fake-audio"), "reference.mp3"),
                    "consent": "true",
                    "ref_text": "这是音色描述，不是逐字稿",
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        role = response.get_json()["roles"][0]
        self.assertEqual(role["mode"], "clone")
        self.assertTrue(role["clone_consent"])
        self.assertEqual(role["ref_text"], "参考音频真实台词")
        self.assertEqual(role["ref_text_auto"], "参考音频真实台词")
        self.assertEqual(role["ref_text_source"], "auto")
        self.assertFalse(role["ref_text_confirmed"])
        self.assertEqual(role["ref_text_confirmed_revision"], "")
        self.assertEqual(role["ref_text_requested"], "这是音色描述，不是逐字稿")
        self.assertTrue(role["ref_text_warning"])
        self.assertTrue(role["ref_audio"].endswith("role1.wav"))
        self.assertTrue(role["reference_revision"])
        self.assertEqual(role["reference_sha256"], "audio-hash")
        run.assert_called_once()
        transcribe.assert_called_once()

    def test_reference_upload_accepts_chinese_filename_extension(self):
        webapp.JOBS.clear()
        app = webapp.create_app()
        client = app.test_client()
        project = {
            "id": "p1", "project_type": "voice", "settings": {"tts_engine": "local"},
            "roles": [{"id": "role1", "name": "角色", "mode": "design"}], "cues": [],
        }
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(core, "load_project", return_value=project), \
             patch.object(core, "project_dir", return_value=Path(directory)), \
             patch.object(core, "media_tool", return_value="ffmpeg"), \
             patch.object(core, "run", side_effect=lambda command: Path(command[-1]).write_bytes(b"wav")), \
             patch.object(webapp, "reference_duration", return_value=6.0), \
             patch.object(core, "file_sha256", return_value="audio-hash"), \
             patch.object(core, "transcribe_reference_audio", return_value="准确逐字稿"), \
             patch.object(core, "save_manifest", side_effect=lambda pid, value: value), \
             patch.object(core, "save_project", side_effect=lambda value: value):
            response = client.post(
                "/api/projects/p1/roles/role1/reference",
                data={"audio": (io.BytesIO(b"fake-audio"), "参考音频.mp3"), "consent": "true"},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["roles"][0]["ref_name"], "参考音频.mp3")

    def test_reference_replacement_changes_revision_and_invalidates_output(self):
        webapp.JOBS.clear()
        app = webapp.create_app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            root = Path(directory) / "p1"
            reference_dir = root / "references"
            reference_dir.mkdir(parents=True)
            (reference_dir / "role1.wav").write_bytes(b"old-wave")
            (reference_dir / "role1-source.mp3").write_bytes(b"old-source")
            project = {
                "id": "p1", "name": "克隆测试", "project_type": "voice",
                "stages": {"script": "ready", "tts": "done", "render": "done", "qa": "done"},
                "settings": core.default_settings("zh", "local"),
                "roles": [{
                    "id": "role1", "name": "旁白", "mode": "clone", "clone_consent": True,
                    "ref_audio": str(reference_dir / "role1.wav"), "ref_name": "参考音频.mp3",
                    "ref_text": "旧逐字稿", "ref_text_auto": "旧逐字稿", "ref_text_source": "auto",
                    "reference_revision": "old", "ref_text_confirmed": True, "ref_text_confirmed_revision": "old",
                    "description": "授权声线",
                }],
                "cues": [{
                    "id": 1, "start": 0, "end": 2, "text": "测试", "source_text": "", "role": "role1",
                    "emotion": "自然", "enabled": True, "speed": 1, "gap_after": .25,
                }],
                "pipeline": {"status": "done"}, "output_audio": str(root / "voice_timeline.wav"),
            }
            (root / "voice_timeline.wav").write_bytes(b"old-output")
            previews = root / "previews"
            raw = root / "raw"
            clean = root / "clean"
            fit = root / "fit"
            for directory_path in (previews, raw, clean, fit):
                directory_path.mkdir()
            for path in (
                previews / "role1-old.wav", raw / "001-role1.wav", raw / "001-role1.json",
                clean / "001.wav", fit / "001.wav", root / "summary.json", root / "status.json", root / "qa.json",
            ):
                path.write_bytes(b"old")
            core.save_project(project)
            with patch.object(core, "media_tool", return_value="ffmpeg"), \
                 patch.object(core, "run", side_effect=lambda command: Path(command[-1]).write_bytes(b"new-wave")), \
                 patch.object(webapp, "reference_duration", return_value=6.0), \
                 patch.object(core, "file_sha256", return_value="new-hash"), \
                 patch.object(core, "transcribe_reference_audio", return_value="新逐字稿"):
                response = client.post(
                    "/api/projects/p1/roles/role1/reference",
                    data={"audio": (io.BytesIO(b"new-audio"), "参考音频.mp3"), "consent": "true"},
                    content_type="multipart/form-data",
                )
        self.assertEqual(response.status_code, 200)
        role = response.get_json()["roles"][0]
        self.assertNotEqual(role["reference_revision"], "old")
        self.assertEqual(role["reference_sha256"], "new-hash")
        self.assertFalse(role["ref_text_confirmed"])
        self.assertEqual(role["ref_text_confirmed_revision"], "")
        self.assertEqual(response.get_json()["stages"]["tts"], "pending")
        self.assertNotIn("output_audio", response.get_json())
        self.assertFalse((root / "voice_timeline.wav").exists())
        self.assertFalse((root / "previews" / "role1-old.wav").exists())
        self.assertFalse((root / "raw" / "001-role1.wav").exists())
        self.assertFalse((root / "raw" / "001-role1.json").exists())
        self.assertFalse((root / "clean" / "001.wav").exists())
        self.assertFalse((root / "fit" / "001.wav").exists())
        self.assertFalse((root / "qa.json").exists())

    def test_delete_reference_cleans_all_role_authorization_files(self):
        webapp.JOBS.clear()
        app = webapp.create_app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            root = Path(directory) / "p1"
            references = root / "references"
            previews = root / "previews"
            references.mkdir(parents=True)
            previews.mkdir()
            for path in (references / "role1.wav", references / "role1-source.mp3", previews / "role1-test.wav", previews / "role1-test.json"):
                path.write_bytes(b"data")
            project = {
                "id": "p1", "name": "克隆测试", "project_type": "voice",
                "stages": {"script": "ready", "tts": "done", "render": "done", "qa": "done"},
                "settings": core.default_settings("zh", "local"),
                "roles": [{
                    "id": "role1", "name": "旁白", "mode": "clone", "clone_consent": True,
                    "ref_audio": str(references / "role1.wav"), "ref_text": "逐字稿", "ref_text_source": "auto",
                    "description": "授权声线",
                }],
                "cues": [], "pipeline": {"status": "done"},
            }
            core.save_project(project)
            response = client.delete("/api/projects/p1/roles/role1/reference")
            remaining = list(references.glob("role1*")) + list(previews.glob("role1-*"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(remaining, [])
        self.assertEqual(response.get_json()["roles"][0]["mode"], "design")

    def test_delete_role_cleans_reference_files_and_reassigns_cues(self):
        webapp.JOBS.clear()
        app = webapp.create_app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            root = Path(directory) / "p1"
            references = root / "references"
            previews = root / "previews"
            references.mkdir(parents=True)
            previews.mkdir()
            (references / "role2.wav").write_bytes(b"audio")
            (references / "role2-source.mp3").write_bytes(b"source")
            (previews / "role2-old.wav").write_bytes(b"preview")
            project = {
                "id": "p1", "name": "角色清理", "project_type": "voice",
                "stages": {"script": "ready", "tts": "done", "render": "done", "qa": "done"},
                "settings": core.default_settings("zh", "local"),
                "roles": [
                    {"id": "role1", "name": "保留", "mode": "design", "description": "自然声线"},
                    {"id": "role2", "name": "删除", "mode": "clone", "clone_consent": True,
                     "ref_audio": str(references / "role2.wav"), "ref_text": "逐字稿", "ref_text_source": "auto"},
                ],
                "cues": [{
                    "id": 1, "start": 0, "end": 2, "text": "测试", "source_text": "", "role": "role2",
                    "emotion": "自然", "enabled": True, "speed": 1, "gap_after": .25,
                }],
                "pipeline": {"status": "done"},
            }
            core.save_project(project)
            response = client.delete("/api/projects/p1/roles/role2")
            remaining = list(references.glob("role2*")) + list(previews.glob("role2-*"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(remaining, [])
        self.assertEqual([role["id"] for role in response.get_json()["roles"]], ["role1"])
        self.assertEqual(response.get_json()["cues"][0]["role"], "role1")

    def test_transcript_similarity_distinguishes_transcript_from_voice_description(self):
        self.assertLess(core.transcript_similarity("外国人讲中文的口音", "年轻人你调的是这个精神灯还是这个阴神灯"), 0.65)
        self.assertGreater(core.transcript_similarity("年轻人，你调的是这个神经灯还是这个阴神灯？", "年轻人你调的是这个精神灯还是这个阴神灯"), 0.9)

    def test_auto_reference_transcript_survives_later_project_save(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            root = Path(directory) / "p1"
            root.mkdir()
            reference = root / "references" / "role1.wav"
            reference.parent.mkdir()
            reference.write_bytes(b"wav")
            project = {
                "id": "p1", "name": "克隆测试", "project_type": "voice",
                "stages": {"script": "ready", "tts": "pending", "render": "pending", "qa": "pending"},
                "settings": core.default_settings("zh", "local"),
                "roles": [{
                    "id": "role1", "name": "旁白", "mode": "clone", "clone_consent": True,
                    "ref_audio": str(reference), "ref_text": "人工确认的真实台词", "ref_text_auto": "自动识别的真实台词",
                    "ref_text_source": "manual", "ref_text_requested": "错误的音色描述", "reference_revision": "rev-1",
                    "ref_text_warning": True, "ref_text_confirmed": True, "ref_text_confirmed_revision": "rev-1",
                    "description": "授权声线",
                }],
                "cues": [{
                    "id": 1, "start": 0, "end": 2, "text": "第一次", "source_text": "",
                    "role": "role1", "emotion": "自然", "enabled": True, "speed": 1, "gap_after": .25,
                }],
                "pipeline": {},
            }
            core.save_project(project)
            stale_payload = core.load_project("p1")
            stale_payload["roles"][0]["ref_text"] = "伪造的旧客户端稿"
            stale_payload["roles"][0]["ref_text_auto"] = "伪造自动稿"
            stale_payload["roles"][0]["ref_text_source"] = "auto"
            stale_payload["roles"][0]["ref_text_confirmed"] = False
            stale_payload["roles"][0]["ref_text_confirmed_revision"] = "stale"
            stale_payload["cues"][0]["text"] = "第二次"
            saved = core.save_manifest("p1", stale_payload)
        role = saved["roles"][0]
        self.assertEqual(role["mode"], "clone")
        self.assertTrue(role["clone_consent"])
        self.assertEqual(role["ref_audio"], str(reference))
        self.assertEqual(role["ref_text"], "人工确认的真实台词")
        self.assertEqual(role["ref_text_auto"], "自动识别的真实台词")
        self.assertEqual(role["ref_text_source"], "manual")
        self.assertTrue(role["ref_text_warning"])
        self.assertTrue(role["ref_text_confirmed"])
        self.assertEqual(role["ref_text_confirmed_revision"], "rev-1")

    def test_legacy_clone_reference_is_upgraded_to_pending_auto_transcript(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as reference, \
             patch.object(core, "transcribe_reference_audio", return_value="自动识别逐字稿") as transcribe:
            project = {
                "settings": {"target_lang": "zh", "whisper_model": "model.bin"},
                "roles": [{
                    "id": "role1", "mode": "clone", "clone_consent": True,
                    "ref_audio": reference.name, "ref_text": "错误音色描述",
                }],
            }
            changed = core.ensure_clone_transcripts(project)
            changed_again = core.ensure_clone_transcripts(project)
        self.assertTrue(changed)
        self.assertFalse(changed_again)
        self.assertEqual(project["roles"][0]["ref_text"], "自动识别逐字稿")
        self.assertEqual(project["roles"][0]["ref_text_auto"], "自动识别逐字稿")
        self.assertEqual(project["roles"][0]["ref_text_source"], "auto")
        self.assertTrue(project["roles"][0]["ref_text_warning"])
        self.assertTrue(project["roles"][0]["reference_revision"])
        self.assertFalse(project["roles"][0]["ref_text_confirmed"])
        self.assertEqual(project["roles"][0]["ref_text_confirmed_revision"], "")
        transcribe.assert_called_once()

    def test_get_project_stays_available_when_legacy_clone_transcription_fails(self):
        webapp.JOBS.clear()
        client = webapp.create_app().test_client()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("旧克隆项目", "zh")
            project = core.load_project(project["id"])
            reference = core.project_dir(project["id"]) / "references" / "role1.wav"
            reference.parent.mkdir(parents=True, exist_ok=True)
            reference.write_bytes(b"wav")
            role = project["roles"][0]
            role.update({"mode": "clone", "clone_consent": True, "ref_audio": str(reference), "ref_text": ""})
            for field in ("ref_text_auto", "ref_text_confirmed", "ref_text_confirmed_revision", "reference_revision"):
                role.pop(field, None)
            core.save_project(project)

            with patch.object(core, "transcribe_reference_audio", side_effect=RuntimeError("offline")):
                response = client.get(f"/api/projects/{project['id']}")

        self.assertEqual(response.status_code, 200)
        loaded_role = response.get_json()["roles"][0]
        self.assertEqual(loaded_role["mode"], "clone")
        self.assertFalse(loaded_role["ref_text_confirmed"])

    def test_confirm_reference_transcript_updates_manual_text_and_invalidates_artifacts(self):
        webapp.JOBS.clear()
        app = webapp.create_app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            root = Path(directory) / "p1"
            reference_dir = root / "references"
            previews = root / "previews"
            raw = root / "raw"
            clean = root / "clean"
            fit = root / "fit"
            for directory_path in (reference_dir, previews, raw, clean, fit):
                directory_path.mkdir(parents=True, exist_ok=True)
            reference = reference_dir / "role1.wav"
            reference.write_bytes(b"wav")
            for path in (
                previews / "role1-old.wav", previews / "role1-old.json",
                raw / "001-role1.wav", raw / "001-role1.json", clean / "001.wav", fit / "001.wav",
                root / "voice_timeline.wav", root / "summary.json", root / "status.json", root / "qa.json",
            ):
                path.write_bytes(b"old")
            project = {
                "id": "p1", "name": "确认测试", "project_type": "voice",
                "stages": {"script": "ready", "tts": "done", "render": "done", "qa": "done"},
                "settings": core.default_settings("zh", "local"),
                "roles": [{
                    "id": "role1", "name": "旁白", "mode": "clone", "clone_consent": True,
                    "ref_audio": str(reference), "ref_text": "自动识别稿", "ref_text_auto": "自动识别稿",
                    "ref_text_source": "auto", "reference_revision": "rev-1",
                    "ref_text_confirmed": False, "ref_text_confirmed_revision": "",
                }],
                "cues": [{
                    "id": 1, "start": 0, "end": 2, "text": "测试", "source_text": "", "role": "role1",
                    "emotion": "自然", "enabled": True, "speed": 1, "gap_after": .25,
                }],
                "pipeline": {"status": "done"}, "output_audio": str(root / "voice_timeline.wav"),
                "qa": {"passed": True},
            }
            core.save_project(project)
            response = client.put(
                "/api/projects/p1/roles/role1/reference-transcript",
                json={"ref_text": "人工校正稿", "reference_revision": "rev-1"},
            )
            saved = core.load_project("p1")
        self.assertEqual(response.status_code, 200)
        role = response.get_json()["roles"][0]
        self.assertEqual(role["ref_text"], "人工校正稿")
        self.assertEqual(role["ref_text_auto"], "自动识别稿")
        self.assertEqual(role["ref_text_source"], "manual")
        self.assertTrue(role["ref_text_confirmed"])
        self.assertEqual(role["ref_text_confirmed_revision"], "rev-1")
        self.assertEqual(saved["stages"]["tts"], "pending")
        self.assertNotIn("output_audio", saved)
        self.assertNotIn("qa", saved)
        self.assertFalse((root / "previews" / "role1-old.wav").exists())
        self.assertFalse((root / "raw" / "001-role1.wav").exists())
        self.assertFalse((root / "voice_timeline.wav").exists())

    def test_confirm_reference_transcript_rejects_more_than_1000_characters(self):
        webapp.JOBS.clear()
        app = webapp.create_app()
        client = app.test_client()
        project = {
            "id": "p1", "project_type": "voice", "settings": core.default_settings("zh", "local"),
            "roles": [{
                "id": "role1", "name": "旁白", "mode": "clone", "clone_consent": True,
                "ref_audio": "/tmp/role1.wav", "ref_text": "自动稿", "reference_revision": "rev-1",
                "ref_text_confirmed": False, "ref_text_confirmed_revision": "",
            }],
            "cues": [],
        }
        with patch.object(core, "load_project", return_value=project):
            response = client.put(
                "/api/projects/p1/roles/role1/reference-transcript",
                json={"ref_text": "字" * 1001, "reference_revision": "rev-1"},
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("1000", response.get_json()["error"])

    def test_confirm_reference_transcript_rejects_stale_revision(self):
        webapp.JOBS.clear()
        app = webapp.create_app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            root = Path(directory) / "p1"
            references = root / "references"
            references.mkdir(parents=True)
            reference = references / "role1.wav"
            reference.write_bytes(b"wav")
            project = {
                "id": "p1", "project_type": "voice", "settings": core.default_settings("zh", "local"),
                "roles": [{
                    "id": "role1", "name": "旁白", "mode": "clone", "clone_consent": True,
                    "ref_audio": str(reference), "ref_text": "自动稿", "ref_text_auto": "自动稿",
                    "ref_text_source": "auto", "reference_revision": "current",
                    "ref_text_confirmed": False, "ref_text_confirmed_revision": "",
                }],
                "cues": [], "stages": {"script": "ready", "tts": "pending", "render": "pending", "qa": "pending"},
            }
            core.save_project(project)
            response = client.put(
                "/api/projects/p1/roles/role1/reference-transcript",
                json={"ref_text": "旧稿", "reference_revision": "stale"},
            )
        self.assertEqual(response.status_code, 409)
        self.assertIn("已被替换", response.get_json()["error"])

    def test_preview_and_tts_endpoints_reject_unconfirmed_clone_transcript(self):
        webapp.JOBS.clear()
        app = webapp.create_app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            root = Path(directory) / "p1"
            references = root / "references"
            references.mkdir(parents=True)
            reference = references / "role1.wav"
            reference.write_bytes(b"wav")
            project = {
                "id": "p1", "project_type": "voice", "settings": core.default_settings("zh", "local"),
                "roles": [{
                    "id": "role1", "name": "旁白", "mode": "clone", "clone_consent": True,
                    "ref_audio": str(reference), "ref_text": "自动稿", "ref_text_auto": "自动稿",
                    "ref_text_source": "auto", "reference_revision": "rev-1",
                    "ref_text_confirmed": False, "ref_text_confirmed_revision": "",
                }],
                "cues": [{"id": 1, "start": 0, "end": 2, "text": "测试", "role": "role1", "enabled": True}],
                "stages": {"script": "ready", "tts": "pending", "render": "pending", "qa": "pending"},
            }
            core.save_project(project)
            preview = client.post("/api/projects/p1/preview-voice", json={"role_id": "role1", "text": "试听"})
            tts = client.post("/api/projects/p1/tts")
        self.assertEqual(preview.status_code, 400)
        self.assertEqual(tts.status_code, 400)
        self.assertIn("核听确认", preview.get_json()["error"])
        self.assertIn("核听确认", tts.get_json()["error"])
        self.assertEqual(webapp.JOBS, {})

    def test_reference_audio_get_route_serves_only_normalized_role_file(self):
        app = webapp.create_app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            root = Path(directory) / "p1"
            references = root / "references"
            references.mkdir(parents=True)
            reference = references / "role1.wav"
            reference.write_bytes(b"normalized-wave")
            core.save_project({
                "id": "p1", "project_type": "voice", "settings": core.default_settings("zh", "local"),
                "roles": [{
                    "id": "role1", "name": "旁白", "mode": "clone", "ref_audio": str(reference),
                    "ref_text": "自动稿", "ref_text_auto": "自动稿", "reference_revision": "rev-1",
                }],
                "cues": [], "stages": {},
            })
            response = client.get("/api/projects/p1/roles/role1/reference")
            status_code = response.status_code
            data = response.get_data()
            mimetype = response.mimetype
            response.close()
        self.assertEqual(status_code, 200)
        self.assertEqual(data, b"normalized-wave")
        self.assertEqual(mimetype, "audio/wav")

    def test_gemini_generation_rejects_clone_roles_instead_of_ignoring_them(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as reference, tempfile.TemporaryDirectory() as directory:
            project = {
                "id": "p1", "name": "视频", "project_type": "video", "source": "video.mp4",
                "stages": {"script": "ready", "tts": "pending", "render": "pending", "qa": "pending"},
                "settings": {**core.default_settings("zh", "gemini"), "tts_engine": "gemini"},
                "roles": [{
                    "id": "role1", "name": "授权角色", "mode": "clone", "clone_consent": True,
                    "ref_audio": reference.name, "ref_text": "准确逐字稿", "ref_text_source": "auto",
                }],
                "cues": [{
                    "id": 1, "start": 0, "end": 2, "text": "测试", "source_text": "", "role": "role1",
                    "emotion": "自然", "enabled": True, "speed": 1, "gap_after": .25,
                }],
                "pipeline": {},
            }
            with patch.object(core, "load_project", return_value=project), \
                 patch.object(core, "project_dir", return_value=Path(directory)), \
                 patch.object(core, "project_duration", return_value=2.0):
                with self.assertRaisesRegex(RuntimeError, "Gemini TTS 不支持授权克隆"):
                    core.generate_tts("p1")

    def test_generate_tts_counts_unique_status_ids_and_cleans_partial_output(self):
        project = {
            "id": "p1", "name": "整轨状态不误报", "project_type": "voice",
            "stages": {"script": "ready", "tts": "pending", "render": "pending", "qa": "pending"},
            "settings": {
                "target_lang": "zh", "tts_engine": "local", "timeline_mode": "natural", "default_gap": .3,
                "voice_design_model": "design-model", "voice_clone_model": "clone-model",
            },
            "roles": [{"id": "role1", "name": "旁白", "mode": "design", "description": "自然男声"}],
            "cues": [
                {"id": 1, "start": 0, "end": 3, "text": "第一句", "source_text": "", "role": "role1", "emotion": "自然", "speed": 1, "gap_after": .3},
                {"id": 2, "start": 3, "end": 6, "text": "第二句", "source_text": "", "role": "role1", "emotion": "自然", "speed": 1, "gap_after": .3},
            ],
            "pipeline": {},
        }
        process = type("Process", (), {"returncode": 0, "stdout": "DONE"})()
        status = [{"id": 1, "role": "role1", "start": 0, "duration": 0.8}]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(core, "load_project", side_effect=lambda pid: project), \
             patch.object(core, "project_dir", return_value=Path(directory)), \
             patch.object(core, "save_project", side_effect=lambda value: value), \
             patch.object(core, "add_log"), \
             patch.object(core, "set_stage"), \
             patch.object(core, "ffmpeg_full", return_value="ffmpeg"), \
             patch.object(core, "media_tool", return_value="ffprobe"), \
             patch.object(core.subprocess, "run", return_value=process):
            root = Path(directory)
            (root / "fit").mkdir()
            (root / "fit" / "001.wav").write_bytes(b"audio")
            (root / "fit" / "002.wav").write_bytes(b"stale-audio")
            (root / "voice_timeline.wav").write_bytes(b"partial-timeline")
            (root / "summary.json").write_text(json.dumps({"cues": 2}), encoding="utf-8")

            def publish_incomplete_status(*_args, **_kwargs):
                (root / "status.json").write_text(json.dumps(status), encoding="utf-8")
                return process

            with patch.object(core.subprocess, "run", side_effect=publish_incomplete_status):
                with self.assertRaisesRegex(RuntimeError, r"配音未完成：1/2，这些台词没生成出音频 \[2\]"):
                    core.generate_tts("p1")

            self.assertFalse((root / "voice_timeline.wav").exists())
            self.assertFalse((root / "summary.json").exists())

    def test_failed_tts_clears_stale_output_and_output_pointer(self):
        project = {
            "id": "p1", "name": "失败清理", "project_type": "voice",
            "stages": {"script": "ready", "tts": "pending", "render": "pending", "qa": "pending"},
            "settings": {"target_lang": "zh", "tts_engine": "local", "timeline_mode": "natural"},
            "roles": [{"id": "role1", "name": "旁白", "mode": "design", "description": "自然男声"}],
            "cues": [{"id": 1, "start": 0, "end": 3, "text": "你好", "role": "role1", "enabled": True}],
            "pipeline": {},
        }
        process = type("Process", (), {"returncode": 1, "stdout": "worker failed"})()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "voice_timeline.wav").write_bytes(b"stale")
            (root / "summary.json").write_text("{}", encoding="utf-8")
            project["output_audio"] = str(root / "voice_timeline.wav")
            with patch.object(core, "load_project", side_effect=lambda pid: project), \
                 patch.object(core, "project_dir", return_value=root), \
                 patch.object(core, "save_project", side_effect=lambda value: value), \
                 patch.object(core, "add_log"), \
                 patch.object(core, "set_stage"), \
                 patch.object(core, "ffmpeg_full", return_value="ffmpeg"), \
                 patch.object(core, "media_tool", return_value="ffprobe"), \
                 patch.object(core.subprocess, "run", return_value=process):
                with self.assertRaisesRegex(RuntimeError, "Qwen3-TTS 失败"):
                    core.generate_tts("p1")
            self.assertFalse((root / "voice_timeline.wav").exists())
            self.assertFalse((root / "summary.json").exists())
            self.assertNotIn("output_audio", project)

    def test_generate_tts_reads_worker_status_file_before_counting_generated_cues(self):
        project = {
            "id": "p1", "name": "整轨状态", "project_type": "voice",
            "stages": {"script": "ready", "tts": "pending", "render": "pending", "qa": "pending"},
            "settings": {
                "target_lang": "zh", "tts_engine": "local", "timeline_mode": "natural", "default_gap": .3,
                "voice_design_model": "design-model", "voice_clone_model": "clone-model",
            },
            "roles": [{"id": "role1", "name": "旁白", "mode": "design", "description": "自然男声"}],
            "cues": [{
                "id": 1, "start": 0, "end": 3, "text": "你好", "source_text": "", "role": "role1",
                "emotion": "自然", "speed": 1, "gap_after": .3,
            }],
            "pipeline": {},
        }
        process = type("Process", (), {"returncode": 0, "stdout": "DONE"})()
        status = [{"id": 1, "role": "role1", "start": 0, "duration": 0.8}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fit").mkdir()
            (root / "fit" / "001.wav").write_bytes(b"audio")
            with patch.object(core, "load_project", side_effect=lambda pid: project), \
                 patch.object(core, "project_dir", return_value=root), \
                 patch.object(core, "save_project", side_effect=lambda value: value), \
                 patch.object(core, "add_log"), \
                 patch.object(core, "set_stage"), \
                 patch.object(core, "ffmpeg_full", return_value="ffmpeg"), \
                 patch.object(core, "media_tool", return_value="ffprobe"):
                def run_worker(*_args, **_kwargs):
                    (root / "status.json").write_text(json.dumps(status), encoding="utf-8")
                    return process

                with patch.object(core.subprocess, "run", side_effect=run_worker) as run:
                    result = core.generate_tts("p1")
        self.assertEqual(result["stages"]["tts"], "done")
        self.assertEqual(result["pipeline"]["status"], "done")
        self.assertIn("--timeline-mode", run.call_args.args[0])

    def test_voice_output_is_complete_requires_all_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("输出门禁", "zh")
            project = core.load_project(project["id"])
            project["cues"] = [
                {"id": 1, "text": "第一句", "role": "role1", "enabled": True},
                {"id": 2, "text": "第二句", "role": "role1", "enabled": True},
            ]
            project["stages"]["tts"] = "done"
            root = core.project_dir(project["id"])
            root.joinpath("voice_timeline.wav").write_bytes(b"timeline")
            root.joinpath("summary.json").write_text(json.dumps({"cues": 2}), encoding="utf-8")
            root.joinpath("status.json").write_text(json.dumps([{"id": 1}, {"id": 2}]), encoding="utf-8")
            root.joinpath("fit").mkdir()
            root.joinpath("fit", "001.wav").write_bytes(b"one")
            root.joinpath("fit", "002.wav").write_bytes(b"two")
            project["output_audio"] = str(root / "voice_timeline.wav")
            core.save_project(project)
            self.assertTrue(core.voice_output_is_complete(project["id"], project))
            root.joinpath("fit", "002.wav").unlink()
            self.assertFalse(core.voice_output_is_complete(project["id"], project))

    def test_generate_tts_passes_both_models_and_natural_mode_to_worker(self):
        project = {
            "id": "p1", "name": "脚本", "project_type": "voice",
            "stages": {"script": "ready", "tts": "pending", "render": "pending", "qa": "pending"},
            "settings": {
                "target_lang": "zh", "tts_engine": "local", "timeline_mode": "natural", "default_gap": .3,
                "voice_design_model": "design-model", "voice_clone_model": "clone-model",
            },
            "roles": [{"id": "role1", "name": "旁白", "mode": "design", "description": "自然男声"}],
            "cues": [{
                "id": 1, "start": 0, "end": 3, "text": "你好", "source_text": "", "role": "role1",
                "emotion": "自然", "enabled": True, "speed": 1, "gap_after": .3,
            }],
            "pipeline": {},
        }
        process = type("Process", (), {"returncode": 0, "stdout": "DONE"})()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fit").mkdir()
            (root / "fit" / "001.wav").write_bytes(b"audio")
            with patch.object(core, "load_project", side_effect=lambda pid: project), \
                 patch.object(core, "project_dir", return_value=root), \
                 patch.object(core, "save_project", side_effect=lambda value: value), \
                 patch.object(core, "add_log"), \
                 patch.object(core, "set_stage"), \
                 patch.object(core, "ffmpeg_full", return_value="ffmpeg"), \
                 patch.object(core, "media_tool", return_value="ffprobe"):
                def run_worker(*_args, **_kwargs):
                    (root / "status.json").write_text('[{"id": 1}]', encoding="utf-8")
                    return process

                with patch.object(core.subprocess, "run", side_effect=run_worker) as run:
                    core.generate_tts("p1")
        command = run.call_args.args[0]
        self.assertIn("--timeline-mode", command)
        self.assertIn("natural", command)
        self.assertIn("--design-model", command)
        self.assertIn("design-model", command)
        self.assertIn("--clone-model", command)
        self.assertIn("clone-model", command)

    def test_generate_tts_cleans_partial_output_when_worker_reports_incomplete_status(self):
        project = {
            "id": "p1", "name": "整轨失败清理", "project_type": "voice",
            "stages": {"script": "ready", "tts": "pending", "render": "pending", "qa": "pending"},
            "settings": {
                "target_lang": "zh", "tts_engine": "local", "timeline_mode": "natural", "default_gap": .3,
                "voice_design_model": "design-model", "voice_clone_model": "clone-model",
            },
            "roles": [{"id": "role1", "name": "旁白", "mode": "design", "description": "自然男声"}],
            "cues": [
                {"id": 1, "start": 0, "end": 3, "text": "第一句", "role": "role1", "enabled": True},
                {"id": 2, "start": 3, "end": 6, "text": "第二句", "role": "role1", "enabled": True},
            ],
            "pipeline": {},
        }
        process = type("Process", (), {"returncode": 0, "stdout": "DONE"})()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fit").mkdir()
            (root / "fit" / "001.wav").write_bytes(b"audio")
            (root / "voice_timeline.wav").write_bytes(b"partial")
            (root / "summary.json").write_text(json.dumps({"cues": 1}), encoding="utf-8")

            def run_worker(*_args, **_kwargs):
                (root / "status.json").write_text(json.dumps([{"id": 1}]), encoding="utf-8")
                return process

            with patch.object(core, "load_project", side_effect=lambda pid: project), \
                 patch.object(core, "project_dir", return_value=root), \
                 patch.object(core, "save_project", side_effect=lambda value: value), \
                 patch.object(core, "add_log"), \
                 patch.object(core, "set_stage"), \
                 patch.object(core, "ffmpeg_full", return_value="ffmpeg"), \
                 patch.object(core, "media_tool", return_value="ffprobe"), \
                 patch.object(core.subprocess, "run", side_effect=run_worker):
                with self.assertRaisesRegex(RuntimeError, r"配音未完成：1/2，这些台词没生成出音频 \[2\]"):
                    core.generate_tts("p1")

            self.assertFalse((root / "voice_timeline.wav").exists())
            self.assertFalse((root / "summary.json").exists())
            self.assertEqual(project["stages"]["tts"], "failed")


class DocumentImportTests(unittest.TestCase):
    def test_parse_script_keeps_paragraph_order_and_defaults_to_narrator(self):
        from document_import import parse_script
        items = parse_script("夜色压低了城市的呼吸。\n\n我们必须在天亮前抵达。")
        self.assertEqual([item["role_name"] for item in items], ["旁白", "旁白"])
        self.assertEqual(items[0]["text"], "夜色压低了城市的呼吸。")
        self.assertEqual(items[1]["text"], "我们必须在天亮前抵达。")

    def test_parse_script_creates_roles_only_for_explicit_speaker_lines(self):
        from document_import import parse_script
        items = parse_script(
            "旁白：夜色压低了城市的呼吸。\n"
            "【林深】：我们必须在天亮前抵达。\n"
            "注意：这不是角色。\n"
            "第一章：故事开始。"
        )
        self.assertEqual([(item["role_name"], item["text"]) for item in items], [
            ("旁白", "夜色压低了城市的呼吸。"),
            ("林深", "我们必须在天亮前抵达。"),
            ("旁白", "注意：这不是角色。"),
            ("旁白", "第一章：故事开始。"),
        ])

    def test_parse_script_handles_word_line_separators_and_scene_directions(self):
        from document_import import parse_script
        items = parse_script(
            "《森林火灾》整理版文案\u2028\u2028"
            "小雪：森林着火了！\u2028"
            "△浓烟扑面，小雪吓得后退。\u2028"
            "浩宇：别怕，我马上来！"
        )
        self.assertEqual([item["role_name"] for item in items], ["旁白", "小雪", "浩宇"])
        self.assertEqual(items[1]["text"], "森林着火了！")
        self.assertNotIn("浓烟扑面", "".join(item["text"] for item in items))
        self.assertIn("浓烟扑面", items[1]["scene_after"])
        self.assertIn("浓烟扑面", items[2]["scene_before"])

    def test_parse_script_treats_shot_notes_as_context_instead_of_roles(self):
        from document_import import parse_script
        items = parse_script(
            "画面内容：\n"
            "【关于诡异世界的笔记：不能吃食物了】\n"
            "转场——\n"
            "【小黎父亲录音】：小黎，爸爸想你了。"
        )
        self.assertEqual([item["role_name"] for item in items], ["小黎父亲录音"])
        self.assertEqual(items[0]["text"], "小黎，爸爸想你了。")
        self.assertIn("诡异世界的笔记", items[0]["scene_before"])
        self.assertIn("转场", items[0]["scene_before"])

    def test_parse_script_splits_inline_scene_from_group_dialogue(self):
        from document_import import parse_script
        items = parse_script("△两人一起挥手。合：拜拜——！")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["role_name"], "两人齐声")
        self.assertEqual(items[0]["text"], "拜拜——！")
        self.assertEqual(items[0]["scene_before"], "两人一起挥手。")

    def test_parse_script_marker_before_group_dialogue_does_not_create_fake_scene(self):
        from document_import import parse_script
        items = parse_script("△合：快跑！")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["role_name"], "两人齐声")
        self.assertEqual(items[0]["text"], "快跑！")
        self.assertEqual(items[0]["scene_before"], "")

    def test_parse_script_marker_before_regular_role_keeps_dialogue(self):
        from document_import import parse_script
        items = parse_script("△小雪：森林着火了！")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["role_name"], "小雪")
        self.assertEqual(items[0]["text"], "森林着火了！")
        self.assertEqual(items[0]["scene_before"], "")

    def test_parse_script_keeps_dialogue_with_parenthesized_stage_direction(self):
        from document_import import parse_script
        items = parse_script("△小雪（低声）：门外有人。")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["role_name"], "小雪")
        self.assertEqual(items[0]["text"], "门外有人。")
        self.assertEqual(items[0]["scene_before"], "低声")

    def test_scene_labels_need_a_delimiter_and_do_not_swallow_narration(self):
        from document_import import parse_script
        items = parse_script("场景里的雪越来越大。\n书本从架子上掉了下来。\n场景：雪原中央。\n小雪：快走！")
        self.assertEqual([item["role_name"] for item in items], ["旁白", "小雪"])
        self.assertIn("场景里的雪越来越大。", items[0]["text"])
        self.assertIn("书本从架子上掉了下来。", items[0]["text"])
        self.assertEqual(items[1]["scene_before"], "雪原中央。")

    def test_scene_prefix_labels_are_generated_consistently(self):
        from document_import import SCENE_LABELS, scene_direction
        for label in SCENE_LABELS:
            self.assertEqual(scene_direction(label), label)
            self.assertEqual(scene_direction(f"{label}：测试说明。"), "测试说明。")

    def test_scene_context_reaches_cue_emotion_direction(self):
        from document_import import materialize_script, parse_script
        from script_direction import direct_script

        items = parse_script(
            "小雪：先别出声。\n"
            "△怪物在门外徘徊，脚步越来越近。\n"
            "浩宇：我们从后门走。"
        )
        roles, cues = materialize_script(items, voice_templates=core.VOICE_TEMPLATES)
        _, directed_cues, _ = direct_script(roles, cues, voice_templates=core.VOICE_TEMPLATES)
        self.assertIn("怪物在门外", directed_cues[0]["scene_after"])
        self.assertIn("怪物在门外", directed_cues[1]["scene_before"])
        self.assertTrue(all(any(token in cue["emotion"] for token in ("紧张", "压低", "警惕")) for cue in directed_cues))

    def test_materialize_script_splits_long_paragraphs_and_builds_roles(self):
        from document_import import materialize_script
        long_line = "第一句。" + ("很长的说明。" * 80)
        roles, cues = materialize_script([
            {"role_name": "旁白", "text": "短句。"},
            {"role_name": "林深", "text": long_line},
        ], voice_templates={"male_adult": "自然男声"})
        self.assertEqual([role["name"] for role in roles], ["旁白", "林深"])
        self.assertGreaterEqual(len(cues), 3)
        self.assertTrue(all(len(cue["text"]) <= 280 for cue in cues))
        self.assertEqual(cues[0]["role"], "role1")
        self.assertEqual(cues[1]["role"], "role2")

    def test_materialize_script_keeps_scene_context_on_every_split_piece(self):
        from document_import import materialize_script
        long_line = "第一句。" + ("很长的说明。" * 80)
        _, cues = materialize_script([
            {
                "role_name": "小雪",
                "text": long_line,
                "scene_before": "怪物正在逼近。",
                "scene_after": "出口终于打开。",
            }
        ])
        self.assertGreater(len(cues), 1)
        self.assertTrue(all(cue["scene_before"] == "怪物正在逼近。" for cue in cues))
        self.assertTrue(all(cue["scene_after"] == "出口终于打开。" for cue in cues))

    def test_extract_document_reads_txt_and_docx(self):
        from document_import import extract_document
        with tempfile.TemporaryDirectory() as directory:
            txt = Path(directory) / "script.txt"
            txt.write_text("第一段。\n\n第二段。", encoding="utf-8")
            extracted = extract_document(txt)
            self.assertIn("第一段。", extracted["text"])
            gb = Path(directory) / "gb.txt"
            gb.write_bytes("夜色来了。".encode("gb18030"))
            self.assertIn("夜色来了。", extract_document(gb)["text"])
            markdown = Path(directory) / "script.md"
            markdown.write_text("# 标题\n\n**加粗**的旁白。", encoding="utf-8")
            self.assertIn("加粗的旁白。", extract_document(markdown)["text"].replace(" ", ""))
            docx = Path(directory) / "script.docx"
            self._write_docx(docx, ["角色A：出发。", "角色B：跟上。"])
            extracted = extract_document(docx)
            self.assertIn("角色A：出发。", extracted["text"])
            self.assertEqual(extracted["format"], "docx")

    def test_extract_document_rejects_empty_and_scanned_pdf(self):
        from document_import import DocumentError, extract_document
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "empty.txt"
            empty.write_text("   \n", encoding="utf-8")
            with self.assertRaisesRegex(DocumentError, "可配音的文字"):
                extract_document(empty)
            pdf = Path(directory) / "blank.pdf"
            pdf.write_bytes(self._blank_pdf())
            with self.assertRaisesRegex(DocumentError, "扫描版 PDF|可提取文字"):
                extract_document(pdf)

    def test_script_upload_creates_voice_project_without_starting_pipeline(self):
        import webapp
        app = webapp.create_app()
        client = app.test_client()
        created = {
            "id": "doc-project",
            "project_type": "voice",
            "name": "夜戏",
            "cues": [{"id": 1}],
            "script_import": {"cue_count": 2, "role_count": 1},
        }
        with patch.object(core, "create_voice_project_from_document", return_value=created) as create, \
             patch.object(core, "save_project") as save, \
             patch.object(webapp, "start_auto_pipeline") as start, \
             patch.object(webapp, "start_script_direction") as direction:
            response = client.post(
                "/api/script-upload",
                data={
                    "document": (io.BytesIO("旁白：夜色来了。".encode("utf-8")), "夜戏.txt"),
                    "name": "夜戏",
                    "target_lang": "zh",
                    "direction_brief": "少儿故事，亲切生动",
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["id"], "doc-project")
        create.assert_called_once()
        self.assertEqual(create.call_args.args[1], "夜戏")
        self.assertEqual(create.call_args.args[3], "少儿故事，亲切生动")
        self.assertEqual(create.call_args.kwargs["voice_mode"], "design")
        save.assert_called_once()
        start.assert_not_called()
        direction.assert_called_once_with("doc-project")

    def test_clone_script_upload_skips_direction_and_generation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)), \
             patch.object(webapp, "start_script_direction") as direction, \
             patch.object(webapp, "start_auto_pipeline") as pipeline, \
             patch.object(core, "ollama_direct_script") as ollama, \
             patch("script_direction.infer_voice_key") as design_voice, \
             patch.object(core, "generate_tts") as tts:
            client = webapp.create_app().test_client()
            response = client.post(
                "/api/script-upload",
                data={
                    "document": (io.BytesIO("旁白：月亮升起来了。\n林深：我们马上出发！".encode("utf-8")), "月光.txt"),
                    "voice_mode": "clone",
                    "direction_brief": "温暖绘本",
                },
                content_type="multipart/form-data",
            )
            self.assertEqual(response.status_code, 201)
            project = response.get_json()
            self.assertEqual(project["original_source"], "月光.txt")
            self.assertEqual(project["direction_brief"], "温暖绘本")
            self.assertEqual(project["script_import"]["voice_mode"], "clone")
            self.assertTrue(project["script_import"]["direction"]["refine_skipped"])
            self.assertFalse(project["script_import"]["direction"]["refined"])
            self.assertEqual([role["name"] for role in project["roles"]], ["旁白", "林深"])
            self.assertEqual([cue["text"] for cue in project["cues"]], ["月亮升起来了。", "我们马上出发！"])
            self.assertTrue(all(cue["emotion"] for cue in project["cues"]))
            self.assertEqual(project["stages"]["tts"], "pending")
            self.assertEqual(project["pipeline"]["status"], "pending")
            self.assertIn("已跳过", project["pipeline"]["message"])
            loaded = client.get(f"/api/projects/{project['id']}").get_json()
            for role in loaded["roles"]:
                self.assertEqual(role["mode"], "clone")
                self.assertEqual(role["voice"], "custom")
                self.assertFalse(role["clone_consent"])
                self.assertFalse(role["ref_audio"])
                self.assertFalse(role["ref_text"])
                self.assertFalse(role["ref_text_confirmed"])
            with self.assertRaisesRegex(RuntimeError, "授权"):
                core.validate_clone_roles(loaded)
            direction.assert_not_called()
            pipeline.assert_not_called()
            ollama.assert_not_called()
            design_voice.assert_not_called()
            tts.assert_not_called()

    def test_clone_script_upload_replaces_placeholder_without_background_job(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)), \
             patch.object(webapp, "start_script_direction") as direction, \
             patch.object(webapp, "start_auto_pipeline") as pipeline:
            project = core.create_voice_project("空白", "zh")
            response = webapp.create_app().test_client().post(
                f"/api/projects/{project['id']}/script-upload",
                data={
                    "document": (io.BytesIO("林深：新的一句。".encode("utf-8")), "新文稿.txt"),
                    "voice_mode": "clone",
                },
                content_type="multipart/form-data",
            )
            self.assertEqual(response.status_code, 200)
            updated = response.get_json()
            self.assertEqual(updated["id"], project["id"])
            self.assertEqual([cue["text"] for cue in updated["cues"]], ["新的一句。"])
            self.assertEqual([role["name"] for role in updated["roles"]], ["林深"])
            self.assertEqual(updated["roles"][0]["mode"], "clone")
            self.assertEqual(updated["script_import"]["voice_mode"], "clone")
            direction.assert_not_called()
            pipeline.assert_not_called()

    def test_clone_import_append_preserves_existing_voice_modes_and_references(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("已有角色", "zh")
            reference = Path(directory) / project["id"] / "reference.wav"
            reference.write_bytes(b"reference-audio")
            project["roles"].append({
                "id": "role2", "name": "林深", "mode": "clone", "voice": "custom",
                "description": "保留声线", "style": "", "clone_consent": True,
                "ref_audio": str(reference), "ref_text": "已经确认的参考台词。",
                "reference_revision": "existing-revision", "ref_text_confirmed": True,
                "ref_text_confirmed_revision": "existing-revision",
            })
            project["cues"][0].update(text="旧台词。", emotion="用户指定情绪")
            core.normalize_project(project)
            core.save_project(project)
            existing_roles = json.loads(json.dumps(project["roles"]))
            source = Path(directory) / "more.txt"
            source.write_text("旁白：新旁白。\n林深：新台词。\n小雪：快跟上！", encoding="utf-8")
            updated = core.import_document_into_project(project["id"], source, voice_mode="clone")
            self.assertEqual(updated["roles"][:2], existing_roles)
            self.assertEqual(updated["roles"][2]["name"], "小雪")
            self.assertEqual(updated["roles"][2]["mode"], "clone")
            self.assertFalse(updated["roles"][2]["clone_consent"])
            self.assertEqual([cue["role"] for cue in updated["cues"]], ["role1", "role1", "role2", "role3"])
            self.assertEqual(updated["cues"][0]["emotion"], "用户指定情绪")
            self.assertEqual(updated["script_import"]["cue_count"], 3)
            self.assertEqual(updated["script_import"]["direction"]["cue_ids"], [2, 3, 4])
            self.assertTrue(updated["script_import"]["roles"][1]["reused"])

    def test_document_upload_rejects_invalid_voice_mode_without_mutating_project(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)), \
             patch.object(webapp, "start_script_direction") as direction, \
             patch.object(core, "create_voice_project", wraps=core.create_voice_project) as create:
            project = core.create_voice_project("保留原项目", "zh")
            create.reset_mock()
            original = core.load_project(project["id"])
            client = webapp.create_app().test_client()
            for url in ("/api/script-upload", f"/api/projects/{project['id']}/script-upload"):
                with self.subTest(url=url):
                    response = client.post(
                        url,
                        data={
                            "document": (io.BytesIO("旁白：新台词。".encode("utf-8")), "invalid.txt"),
                            "voice_mode": "unsupported",
                        },
                        content_type="multipart/form-data",
                    )
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("导入方式", response.get_json()["error"])
            create.assert_not_called()
            direction.assert_not_called()
            self.assertEqual(core.load_project(project["id"]), original)

    def test_script_upload_returns_document_errors(self):
        from document_import import DocumentError
        app = webapp.create_app()
        with patch.object(core, "create_voice_project_from_document", side_effect=DocumentError("这是扫描版 PDF 或没有可提取文字。请改用 Word、TXT，或先做文字识别。")):
            response = app.test_client().post(
                "/api/script-upload",
                data={"document": (io.BytesIO(b"%PDF-1.4"), "scan.pdf")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("扫描版 PDF", response.get_json()["error"])

    def test_script_upload_rejects_missing_and_unsupported_files(self):
        app = webapp.create_app()
        missing = app.test_client().post("/api/script-upload", data={}, content_type="multipart/form-data")
        self.assertEqual(missing.status_code, 400)
        self.assertIn("文档", missing.get_json()["error"])
        unsupported = app.test_client().post(
            "/api/script-upload",
            data={"document": (io.BytesIO(b"nope"), "notes.xlsx")},
            content_type="multipart/form-data",
        )
        self.assertEqual(unsupported.status_code, 400)
        self.assertIn("格式", unsupported.get_json()["error"])

    def test_create_voice_project_from_document_copies_source_and_cues(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            source = Path(directory) / "radio.txt"
            source.write_text("旁白：夜色压低了城市的呼吸。\n林深：我们必须在天亮前抵达。", encoding="utf-8")
            project = core.create_voice_project_from_document(source, "广播剧", "zh")
            self.assertEqual(project["project_type"], "voice")
            self.assertEqual([role["name"] for role in project["roles"]], ["旁白", "林深"])
            self.assertEqual([cue["text"] for cue in project["cues"]], [
                "夜色压低了城市的呼吸。",
                "我们必须在天亮前抵达。",
            ])
            self.assertTrue(Path(project["source"]).exists())
            self.assertEqual(project["original_source"], "radio.txt")
            self.assertEqual(project["script_import"]["cue_count"], 2)
            self.assertEqual(project["script_import"]["role_count"], 2)
            self.assertEqual(
                [(role["name"], role["cue_count"]) for role in project["script_import"]["roles"]],
                [("旁白", 1), ("林深", 1)],
            )
            self.assertNotEqual(project["roles"][0]["voice"], project["roles"][1]["voice"])
            self.assertNotEqual(project["cues"][0]["emotion"], project["cues"][1]["emotion"])
            self.assertNotEqual(project["cues"][0]["emotion"], "自然")

    def test_document_direction_is_persisted_for_new_and_existing_projects(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            source = Path(directory) / "children.txt"
            source.write_text("旁白：月亮升起来了。\n小雪：我们出发吧！", encoding="utf-8")
            created = core.create_voice_project_from_document(
                source,
                "月光故事",
                "zh",
                "少儿故事：亲切、生动、咬字清楚",
            )
            self.assertEqual(created["direction_brief"], "少儿故事：亲切、生动、咬字清楚")

            more = Path(directory) / "more.txt"
            more.write_text("小雪：快跟上！", encoding="utf-8")
            updated = core.import_document_into_project(
                created["id"],
                more,
                direction_brief="温暖绘本：柔和、有陪伴感",
            )
            self.assertEqual(updated["direction_brief"], "温暖绘本：柔和、有陪伴感")

    def test_import_document_into_placeholder_project_replaces_starter_cue(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("空白", "zh")
            source = Path(directory) / "into.txt"
            source.write_text("旁白：夜色来了。\n林深：跟上。", encoding="utf-8")
            updated = core.import_document_into_project(project["id"], source)
            self.assertEqual([role["name"] for role in updated["roles"]], ["旁白", "林深"])
            self.assertEqual([cue["text"] for cue in updated["cues"]], ["夜色来了。", "跟上。"])
            self.assertTrue(all(cue["emotion"] and cue["emotion"] != "自然" for cue in updated["cues"]))

    def test_import_document_into_existing_project_appends_cues(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("已有台词", "zh")
            project["cues"][0]["text"] = "原来的第一句。"
            project["cues"][0]["emotion"] = "用户指定的情绪"
            core.save_project(project)
            source = Path(directory) / "more.txt"
            source.write_text("林深：新的一句。", encoding="utf-8")
            updated = core.import_document_into_project(project["id"], source)
            self.assertEqual([cue["text"] for cue in updated["cues"]], ["原来的第一句。", "新的一句。"])
            self.assertEqual([role["name"] for role in updated["roles"]], ["旁白", "林深"])
            self.assertEqual(updated["cues"][0]["emotion"], "用户指定的情绪")
            self.assertNotEqual(updated["cues"][1]["emotion"], "自然")
            self.assertEqual(updated["script_import"]["cue_count"], 1)
            self.assertEqual(updated["script_import"]["role_count"], 1)
            self.assertEqual(updated["script_import"]["roles"][0]["name"], "林深")
            self.assertEqual(updated["script_import"]["roles"][0]["cue_count"], 1)

    def test_import_document_reuses_existing_clone_role_and_reports_it(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("已有克隆角色", "zh")
            reference = Path(directory) / project["id"] / "references" / "role1.wav"
            reference.parent.mkdir()
            reference.write_bytes(b"wav")
            project["roles"][0].update({
                "name": "林深", "mode": "clone", "clone_consent": True,
                "ref_audio": str(reference), "ref_text": "已经确认的参考台词",
                "reference_revision": "rev-existing", "ref_text_confirmed": True,
                "ref_text_confirmed_revision": "rev-existing",
            })
            project["cues"][0]["text"] = "旧台词。"
            core.save_project(project)
            source = Path(directory) / "more.txt"
            source.write_text("林深：这是新导入的台词。", encoding="utf-8")
            updated = core.import_document_into_project(project["id"], source)

        self.assertEqual(len(updated["roles"]), 1)
        role = updated["roles"][0]
        self.assertEqual(role["id"], "role1")
        self.assertEqual(role["mode"], "clone")
        self.assertEqual(role["reference_revision"], "rev-existing")
        self.assertEqual(updated["cues"][-1]["role"], "role1")
        self.assertEqual(updated["script_import"]["roles"], [{
            "id": "role1", "name": "林深", "cue_count": 1, "reused": True,
        }])

    def test_project_script_upload_endpoint_uses_current_project(self):
        app = webapp.create_app()
        updated = {"id": "p1", "cues": [{"id": 1}], "script_import": {"cue_count": 1}}
        with patch.object(core, "import_document_into_project", return_value=updated) as import_into, \
             patch.object(webapp, "start_script_direction") as direction:
            response = app.test_client().post(
                "/api/projects/p1/script-upload",
                data={
                    "document": (io.BytesIO("旁白：夜色来了。".encode("utf-8")), "夜戏.txt"),
                    "direction_brief": "悬疑广播剧，克制紧张",
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        import_into.assert_called_once()
        self.assertEqual(import_into.call_args.args[0], "p1")
        self.assertEqual(import_into.call_args.args[4], "悬疑广播剧，克制紧张")
        self.assertEqual(import_into.call_args.kwargs["voice_mode"], "design")
        direction.assert_called_once_with("p1")

    def _write_docx(self, path, paragraphs):
        body = "".join(
            f'<w:p><w:r><w:t xml:space="preserve">{paragraph}</w:t></w:r></w:p>'
            for paragraph in paragraphs
        )
        document = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{body}</w:body></w:document>"
        )
        import zipfile
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"></Types>')
            archive.writestr("word/document.xml", document)

    def _blank_pdf(self):
        return (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000052 00000 n \n0000000101 00000 n \n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n168\n%%EOF\n"
        )


class ScriptDirectionTests(unittest.TestCase):
    def test_direct_script_assigns_distinct_voices_and_emotions(self):
        from script_direction import direct_script
        roles = [
            {"id": "role1", "name": "旁白", "mode": "design", "voice": "male_adult", "description": "旧旁白"},
            {"id": "role2", "name": "林深", "mode": "design", "voice": "male_adult", "description": "旧男声"},
            {"id": "role3", "name": "晚晴", "mode": "design", "voice": "male_adult", "description": "旧女声"},
        ]
        cues = [
            {"id": 1, "role": "role1", "text": "夜色压低了城市的呼吸。"},
            {"id": 2, "role": "role2", "text": "我们必须马上出发！"},
            {"id": 3, "role": "role3", "text": "等等，你听到了什么？"},
            {"id": 4, "role": "role2", "text": "闭嘴，别回头。"},
            {"id": 5, "role": "role2", "text": "跟上。"},
        ]
        directed_roles, directed_cues, meta = direct_script(roles, cues, voice_templates=core.VOICE_TEMPLATES)
        voices = [role["voice"] for role in directed_roles]
        self.assertEqual(len(set(voices)), 3)
        self.assertEqual(directed_roles[2]["voice"], "female_warm")
        emotions = [cue["emotion"] for cue in directed_cues]
        self.assertGreaterEqual(len(set(emotions)), 4)
        self.assertTrue(all(emotion != "自然" for emotion in emotions))
        self.assertIn("催促", directed_cues[1]["emotion"])
        self.assertTrue(any(token in directed_cues[3]["emotion"] for token in ("怒", "愤怒")))
        self.assertIn("催促", directed_cues[4]["emotion"])
        self.assertFalse(meta["refined"])

    def test_obvious_character_gender_is_respected(self):
        from script_direction import direct_script
        roles = [
            {"id": "role1", "name": "小雪", "mode": "design", "voice": "male_adult", "description": "旧声线"},
            {"id": "role2", "name": "浩宇", "mode": "design", "voice": "female_warm", "description": "旧声线"},
            {"id": "role3", "name": "小宇", "mode": "design", "voice": "female_warm", "description": "旧声线"},
            {"id": "role4", "name": "奇奇", "mode": "design", "voice": "male_young", "description": "旧声线"},
            {"id": "role5", "name": "李娜", "mode": "design", "voice": "male_young", "description": "旧声线"},
            {"id": "role6", "name": "王军", "mode": "design", "voice": "female_warm", "description": "旧声线"},
            {"id": "role7", "name": "小红", "mode": "design", "voice": "male_adult", "description": "旧声线"},
            {"id": "role8", "name": "小敏", "mode": "design", "voice": "male_adult", "description": "旧声线"},
        ]
        cues = [
            {"id": 1, "role": "role1", "text": "我来帮你。"},
            {"id": 2, "role": "role2", "text": "马上出发。"},
            {"id": 3, "role": "role3", "text": "别担心。"},
            {"id": 4, "role": "role4", "text": "森林着火了！"},
            {"id": 5, "role": "role5", "text": "我来照顾伤员。"},
            {"id": 6, "role": "role6", "text": "我负责开路。"},
            {"id": 7, "role": "role7", "text": "我来帮忙。"},
            {"id": 8, "role": "role8", "text": "大家小心。"},
        ]
        directed_roles, _, _ = direct_script(roles, cues, voice_templates=core.VOICE_TEMPLATES)
        by_name = {role["name"]: role["voice"] for role in directed_roles}
        female_voices = {"female_warm", "female_brave", "girl_soft", "girl_lively"}
        male_voices = {"male_adult", "male_young"}
        self.assertIn(by_name["小雪"], female_voices)
        self.assertIn(by_name["奇奇"], female_voices)
        self.assertIn(by_name["李娜"], female_voices)
        self.assertIn(by_name["小红"], female_voices)
        self.assertIn(by_name["小敏"], female_voices)
        self.assertIn(by_name["浩宇"], male_voices)
        self.assertIn(by_name["小宇"], male_voices)
        self.assertIn(by_name["王军"], male_voices)

    def test_compound_role_labels_do_not_force_gender_from_internal_titles(self):
        from script_direction import infer_name_gender
        self.assertIsNone(infer_name_gender("小梅姐夫"))
        self.assertIsNone(infer_name_gender("王女士助理"))
        self.assertEqual(infer_name_gender("小雪姐姐"), "female")
        self.assertEqual(infer_name_gender("浩宇哥哥"), "male")

    def test_punctuation_and_scene_context_drive_emotion(self):
        from script_direction import direct_script
        roles = [
            {"id": "role1", "name": "小雪", "mode": "design", "voice": "female_warm", "description": "旧声线"},
            {"id": "role2", "name": "浩宇", "mode": "design", "voice": "male_adult", "description": "旧声线"},
        ]
        cues = [
            {
                "id": 1,
                "role": "role1",
                "text": "森林着火了！",
                "scene_after": "浓烟扑面，小雪吓得后退。",
            },
            {
                "id": 2,
                "role": "role2",
                "text": "先别出声。",
                "scene_before": "怪物在门外徘徊，脚步越来越近。",
            },
            {"id": 3, "role": "role1", "text": "你真的没事吗？"},
            {"id": 4, "role": "role2", "text": "呼……他终于走了。"},
        ]
        _, directed_cues, _ = direct_script(roles, cues, voice_templates=core.VOICE_TEMPLATES)
        self.assertTrue(any(token in directed_cues[0]["emotion"] for token in ("惊", "慌", "急")))
        self.assertTrue(any(token in directed_cues[1]["emotion"] for token in ("紧张", "压低", "警惕", "小心")))
        self.assertTrue(any(token in directed_cues[2]["emotion"] for token in ("担心", "关切", "疑问", "探询")))
        self.assertIn("松一口气", directed_cues[3]["emotion"])

    def test_rule_direction_marks_personality_as_stable_reaction_constraint(self):
        from script_direction import direct_script
        roles = [{
            "id": "role1", "name": "林深", "mode": "clone", "voice": "custom",
            "description": "授权声线", "personality": "外冷内热，遇险先压住慌张；生气时不吼，音量更低、咬字更重。",
        }]
        cues = [{"id": 1, "role": "role1", "text": "闭嘴，马上离开这里！"}]
        _, directed, _ = direct_script(roles, cues, voice_templates=core.VOICE_TEMPLATES)
        self.assertIn("按人物性格", directed[0]["emotion"])
        self.assertIn("音量偏低", directed[0]["emotion"])
        self.assertIn("不脱离", directed[0]["emotion"])

    def test_personality_emotion_instruction_is_idempotent(self):
        from script_direction import personality_emotion_instruction
        personality = "冷静克制，紧张时先停顿再说。"
        first = personality_emotion_instruction(personality, "惊慌、声音发紧")
        second = personality_emotion_instruction(personality, first)
        self.assertEqual(second, first)

    def test_words_containing_relief_substrings_do_not_imply_relief(self):
        from script_direction import assign_cue_emotions
        cues = [
            {"id": 1, "role": "role1", "text": "快呼叫消防员！"},
            {"id": 2, "role": "role1", "text": "呼啸的风越来越大。"},
            {"id": 3, "role": "role1", "text": "夜色压低了城市的呼吸。"},
            {"id": 4, "role": "role1", "text": "我还好奇他去了哪里。"},
            {"id": 5, "role": "role1", "text": "这个办法的总算式很复杂。"},
        ]
        directed = assign_cue_emotions(cues)
        self.assertTrue(all("松一口气" not in cue["emotion"] for cue in directed))

    def test_current_danger_dialogue_beats_later_safe_scene(self):
        from script_direction import assign_cue_emotions
        cue = {
            "id": 1,
            "role": "role1",
            "text": "森林着火了！",
            "scene_after": "消防车赶到，火焰终于被扑灭。",
        }
        emotion = assign_cue_emotions([cue])[0]["emotion"]
        self.assertTrue(any(token in emotion for token in ("惊慌", "急促", "发紧")))

    def test_model_direction_cannot_flip_an_obvious_character_gender(self):
        from script_direction import apply_direction
        roles = [
            {"id": "role1", "name": "小雪", "mode": "design", "voice": "female_warm", "description": "年轻女声"},
            {"id": "role2", "name": "浩宇", "mode": "design", "voice": "male_young", "description": "年轻男声"},
            {"id": "role3", "name": "小女孩", "mode": "design", "voice": "girl_soft", "description": "小女孩声音"},
            {"id": "role4", "name": "男童", "mode": "design", "voice": "male_young", "description": "小男孩声音"},
        ]
        next_roles, _ = apply_direction(
            roles,
            [],
            [
                {"id": "role1", "voice": "male_adult", "description": "成年男声"},
                {"id": "role2", "voice": "female_warm", "description": "年轻女声"},
                {"id": "role3", "voice": "male_adult", "description": "成年男声"},
                {"id": "role4", "voice": "female_warm", "description": "年轻女声"},
            ],
            [],
        )
        by_name = {role["name"]: role for role in next_roles}
        self.assertIn(by_name["小雪"]["voice"], {"female_warm", "female_brave", "girl_soft", "girl_lively"})
        self.assertIn("女", by_name["小雪"]["description"])
        self.assertIn(by_name["浩宇"]["voice"], {"male_adult", "male_young"})
        self.assertIn("男", by_name["浩宇"]["description"])
        self.assertIn(by_name["小女孩"]["voice"], {"female_warm", "female_brave", "girl_soft", "girl_lively"})
        self.assertIn("女", by_name["小女孩"]["description"])
        self.assertIn(by_name["男童"]["voice"], {"male_adult", "male_young"})
        self.assertIn("男", by_name["男童"]["description"])

    def test_clone_roles_keep_uploaded_voice(self):
        from script_direction import apply_direction, direct_script
        roles = [
            {"id": "role1", "name": "旁白", "mode": "clone", "voice": "custom", "description": "授权声线", "ref_audio": "/tmp/a.wav"},
            {"id": "role2", "name": "晚晴", "mode": "design", "voice": "male_adult", "description": "待分配"},
        ]
        cues = [
            {"id": 1, "role": "role1", "text": "夜色来了。"},
            {"id": 2, "role": "role2", "text": "我跟你走。"},
        ]
        directed_roles, directed_cues, _ = direct_script(roles, cues, voice_templates=core.VOICE_TEMPLATES)
        self.assertEqual(directed_roles[0]["description"], "授权声线")
        self.assertEqual(directed_roles[0]["voice"], "custom")
        self.assertEqual(directed_roles[1]["voice"], "female_warm")
        next_roles, next_cues = apply_direction(
            roles,
            cues,
            [{"id": "role1", "voice": "female_brave", "description": "不该覆盖"}, {"id": "role2", "voice": "girl_soft", "description": "新女声"}],
            [{"id": 1, "emotion": "低沉旁白"}, {"id": 2, "emotion": "温柔回应"}],
        )
        self.assertEqual(next_roles[0]["description"], "授权声线")
        self.assertEqual(next_roles[1]["voice"], "girl_soft")
        self.assertEqual(next_cues[1]["emotion"], "温柔回应")

    def test_refine_imported_direction_uses_context_and_skips_user_edits(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            source = Path(directory) / "radio.txt"
            source.write_text("旁白：夜色来了。\n林深：我们必须马上出发。\n晚晴：等等。", encoding="utf-8")
            project = core.create_voice_project_from_document(source, "广播剧", "zh", "少儿冒险，清楚生动")
            project["cues"][2]["emotion"] = "用户改过的情绪"
            core.save_project(project)
            received_direction = []

            def fake_ollama(roles, cues, model, locked_role_ids=None, cue_ids=None, direction_brief=""):
                received_direction.append(direction_brief)
                return (
                    [{"id": "role1", "voice": "male_adult", "description": "沉稳旁白"},
                     {"id": "role2", "voice": "male_young", "description": "年轻男声"},
                     {"id": "role3", "voice": "female_warm", "description": "温暖女声"}],
                    [{"id": 1, "emotion": "压低声音、叙述"}, {"id": 2, "emotion": "急迫催促"}, {"id": 3, "emotion": "被覆盖的情绪"}],
                )

            with patch.object(core, "ollama_direct_script", side_effect=fake_ollama):
                updated = core.refine_imported_direction(project["id"])
            self.assertEqual(received_direction, ["少儿冒险，清楚生动"])
            self.assertEqual(updated["cues"][0]["emotion"], "压低声音、叙述")
            self.assertEqual(updated["cues"][1]["emotion"], "急迫催促")
            self.assertEqual(updated["cues"][2]["emotion"], "用户改过的情绪")
            self.assertTrue(updated["script_import"]["direction"]["refined"])

    def test_ollama_prompt_includes_role_personality_and_reaction_rules(self):
        import json as json_module
        from script_direction import _ollama_direction_batch

        roles = [{
            "id": "role1", "name": "林深", "mode": "clone",
            "personality": "外冷内热；紧张时先沉默半拍，生气时压低音量。",
        }]
        cues = [{"id": 1, "role": "role1", "text": "你再说一遍？"}]
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json_module.dumps({"response": '{"roles":[],"cues":[{"id":1,"emotion":"冷怒、停顿后压低声音"}]}'}, ensure_ascii=False).encode()

        def fake_urlopen(req, timeout=None):
            captured["body"] = json_module.loads(req.data.decode())
            return FakeResponse()

        with patch("script_direction.urllib.request.urlopen", side_effect=fake_urlopen):
            _ollama_direction_batch(roles, cues, "qwen-test", {"role1"}, [1], True)
        prompt = captured["body"]["prompt"]
        self.assertIn(roles[0]["personality"], prompt)
        self.assertIn("情绪外露程度", prompt)
        self.assertIn("不要把性格原文复制", prompt)

    def test_ollama_prompt_includes_document_direction(self):
        import json as json_module
        from script_direction import _ollama_direction_batch

        roles = [{"id": "role1", "name": "旁白", "mode": "design"}]
        cues = [{"id": 1, "role": "role1", "text": "月亮升起来了。"}]
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json_module.dumps({"response": '{"roles":[],"cues":[{"id":1,"emotion":"温柔讲述"}]}'}, ensure_ascii=False).encode()

        def fake_urlopen(req, timeout=None):
            captured.update(json_module.loads(req.data.decode()))
            return FakeResponse()

        with patch("script_direction.urllib.request.urlopen", side_effect=fake_urlopen):
            _ollama_direction_batch(
                roles,
                cues,
                "qwen-test",
                set(),
                [1],
                True,
                "少儿故事：亲切、生动、咬字清楚",
            )
        self.assertIn("少儿故事：亲切、生动、咬字清楚", captured["prompt"])
        self.assertIn("全局硬约束", captured["prompt"])

    def test_refine_imported_direction_falls_back_when_ollama_fails(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            source = Path(directory) / "radio.txt"
            source.write_text("旁白：夜色来了。\n林深：跟上。", encoding="utf-8")
            project = core.create_voice_project_from_document(source, "广播剧", "zh")
            original = [cue["emotion"] for cue in project["cues"]]
            with patch.object(core, "ollama_direct_script", side_effect=RuntimeError("未安装 Ollama")):
                updated = core.refine_imported_direction(project["id"])
            self.assertEqual([cue["emotion"] for cue in updated["cues"]], original)
            self.assertFalse(updated["script_import"]["direction"]["refined"])
            self.assertIn("未安装 Ollama", updated["last_message"])


class VoiceConsistencyTests(unittest.TestCase):
    def test_design_role_every_line_keeps_voice_design_for_per_cue_emotion(self):
        from voice_consistency import cue_generation_plan
        cues = [
            {"id": 1, "role": "role1", "text": "这是平静的第一句。", "emotion": "平静", "enabled": True},
            {"id": 2, "role": "role1", "text": "后面突然着火了！", "emotion": "惊慌", "enabled": True},
            {"id": 3, "role": "role2", "text": "另一角色仍走造声。", "emotion": "坚定", "enabled": True},
        ]
        voices = {
            "role1": {"id": "role1", "mode": "design", "description": "沉稳男声"},
            "role2": {"id": "role2", "mode": "design", "description": "年轻女声"},
        }
        plan = cue_generation_plan(cues, voices)
        self.assertTrue(all(item["mode"] == "design" for item in plan))
        self.assertEqual(plan[1]["cue"]["emotion"], "惊慌")
        self.assertNotIn("ref_audio", plan[1]["voice"])

    def test_voice_design_generation_passes_each_cue_emotion_to_model(self):
        from voice_consistency import generate_cue_results

        class FakeModel:
            def __init__(self):
                self.calls = []

            def generate_voice_design(self, **kwargs):
                self.calls.append(kwargs)
                result = type("Result", (), {"audio": [0.0, 0.1], "sample_rate": 24000})()
                return [result]

        model = FakeModel()
        voice = {"mode": "design", "description": "年轻女声，清亮坚定"}
        first = {"id": 1, "role": "role1", "text": "先别出声。", "emotion": "紧张、压低声音、保持警惕"}
        second = {"id": 2, "role": "role1", "text": "我们成功了！", "emotion": "兴奋、带着惊喜和笑意"}
        generate_cue_results(model, "design", voice, first, "Chinese", {}, {})
        generate_cue_results(model, "design", voice, second, "Chinese", {}, {})
        self.assertEqual(len(model.calls), 2)
        self.assertIn(first["emotion"], model.calls[0]["instruct"])
        self.assertIn(second["emotion"], model.calls[1]["instruct"])
        self.assertNotEqual(model.calls[0]["instruct"], model.calls[1]["instruct"])

    def test_voice_design_cache_key_changes_with_cue_emotion(self):
        from voice_consistency import generation_cache_key
        voice = {"mode": "design", "description": "年轻女声，清亮坚定"}
        first = {"id": 1, "role": "role1", "text": "原来是你。", "emotion": "惊恐、急促"}
        second = {**first, "emotion": "松一口气、声音缓下来"}
        first_key = generation_cache_key(first, voice, "Chinese", "design-model", {"temperature": 0.7})
        second_key = generation_cache_key(second, voice, "Chinese", "design-model", {"temperature": 0.7})
        self.assertNotEqual(first_key, second_key)

    def test_voice_design_instruction_includes_project_direction(self):
        from voice_consistency import voice_design_instruction
        voice = {
            "mode": "design",
            "description": "温暖年轻女声",
            "project_direction": "少儿故事：亲切、生动、咬字清楚",
        }
        cue = {"text": "我们一起出发吧！", "emotion": "开心、充满期待"}
        instruction = voice_design_instruction(voice, cue)
        self.assertIn(voice["project_direction"], instruction)
        self.assertIn(cue["emotion"], instruction)

    def test_voice_design_instruction_includes_role_personality(self):
        from voice_consistency import voice_design_instruction
        voice = {
            "mode": "design",
            "description": "低沉克制的年轻男声",
            "personality": "外冷内热；生气时不吼，越愤怒音量越低。",
        }
        cue = {"text": "你再说一遍？", "emotion": "冷怒、停顿后压低声音"}
        instruction = voice_design_instruction(voice, cue)
        self.assertIn(voice["personality"], instruction)
        self.assertIn("情绪外露程度", instruction)
        self.assertIn(cue["emotion"], instruction)

    def test_voice_design_cache_key_changes_with_personality(self):
        from voice_consistency import generation_cache_key
        cue = {"id": 1, "role": "role1", "text": "原来是你。", "emotion": "惊讶"}
        first_voice = {"mode": "clone", "description": "授权声线", "personality": "外向直接"}
        second_voice = {**first_voice, "personality": "内敛克制"}
        first_key = generation_cache_key(cue, first_voice, "Chinese", "clone-model", {"max_tokens": 96})
        second_key = generation_cache_key(cue, second_voice, "Chinese", "clone-model", {"max_tokens": 96})
        self.assertNotEqual(first_key, second_key)

    def test_voice_design_cache_key_changes_with_project_direction(self):
        from voice_consistency import generation_cache_key
        cue = {"id": 1, "role": "role1", "text": "原来是你。", "emotion": "惊讶"}
        first_voice = {"mode": "design", "description": "年轻女声", "project_direction": "少儿故事"}
        second_voice = {**first_voice, "project_direction": "悬疑广播剧"}
        first_key = generation_cache_key(cue, first_voice, "Chinese", "design-model", {"temperature": 0.7})
        second_key = generation_cache_key(cue, second_voice, "Chinese", "design-model", {"temperature": 0.7})
        self.assertNotEqual(first_key, second_key)

    def test_saving_project_direction_invalidates_existing_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("方向测试", "zh")
            root = Path(directory) / project["id"]
            output = root / "voice_timeline.wav"
            output.write_bytes(b"old-output")
            project["output_audio"] = str(output)
            project["stages"].update({"tts": "done", "render": "done", "qa": "done"})
            core.save_project(project)
            payload = core.load_project(project["id"])
            payload["direction_brief"] = "悬疑广播剧：克制紧张，留出停顿"
            saved = core.save_manifest(project["id"], payload)
        self.assertEqual(saved["direction_brief"], "悬疑广播剧：克制紧张，留出停顿")
        self.assertNotIn("output_audio", saved)
        self.assertEqual(saved["stages"]["tts"], "pending")

    def test_saving_role_personality_invalidates_existing_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("人格缓存测试", "zh")
            root = Path(directory) / project["id"]
            output = root / "voice_timeline.wav"
            output.write_bytes(b"old-output")
            project["output_audio"] = str(output)
            project["stages"].update({"tts": "done", "render": "done", "qa": "done"})
            core.save_project(project)
            payload = core.load_project(project["id"])
            payload["roles"][0]["personality"] = "冷静克制，紧张时先停顿再说。"
            saved = core.save_manifest(project["id"], payload)
        self.assertEqual(saved["roles"][0]["personality"], "冷静克制，紧张时先停顿再说。")
        self.assertNotIn("output_audio", saved)
        self.assertEqual(saved["stages"]["tts"], "pending")

    def test_authorized_clone_roles_keep_uploaded_reference(self):
        from voice_consistency import cue_generation_plan
        cues = [
            {"id": 1, "role": "role1", "text": "第一句", "enabled": True},
            {"id": 2, "role": "role1", "text": "第二句", "enabled": True},
        ]
        voices = {
            "role1": {
                "id": "role1", "mode": "clone", "clone_consent": True,
                "ref_audio": "/tmp/user.wav", "ref_text": "用户参考",
            }
        }
        plan = cue_generation_plan(cues, voices)
        self.assertTrue(all(item["mode"] == "clone" for item in plan))
        self.assertEqual(plan[1]["voice"]["ref_audio"], "/tmp/user.wav")
        self.assertEqual(plan[1]["voice"]["ref_text"], "用户参考")

    def test_imported_roles_route_each_cue_to_its_own_confirmed_reference(self):
        from voice_consistency import cue_generation_plan, generate_cue_results

        class FakeModel:
            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return []

            def generate_voice_design(self, **kwargs):
                raise AssertionError("文稿中的克隆角色不应调用 VoiceDesign")

        with tempfile.NamedTemporaryFile(suffix=".wav") as first, tempfile.NamedTemporaryFile(suffix=".wav") as second:
            voices = {
                "role1": {
                    "id": "role1", "mode": "clone", "clone_consent": True,
                    "ref_audio": first.name, "ref_text": "角色甲参考台词",
                    "reference_revision": "rev-a", "ref_text_confirmed": True,
                    "ref_text_confirmed_revision": "rev-a",
                },
                "role2": {
                    "id": "role2", "mode": "clone", "clone_consent": True,
                    "ref_audio": second.name, "ref_text": "角色乙参考台词",
                    "reference_revision": "rev-b", "ref_text_confirmed": True,
                    "ref_text_confirmed_revision": "rev-b",
                },
            }
            cues = [
                {"id": 1, "role": "role1", "text": "甲的新台词"},
                {"id": 2, "role": "role2", "text": "乙的新台词"},
            ]
            model = FakeModel()
            for item in cue_generation_plan(cues, voices):
                generate_cue_results(model, item["mode"], item["voice"], item["cue"], "Chinese", {}, {})

        self.assertEqual([call["ref_audio"] for call in model.calls], [first.name, second.name])
        self.assertEqual([call["ref_text"] for call in model.calls], ["角色甲参考台词", "角色乙参考台词"])
        self.assertEqual([call["text"] for call in model.calls], ["甲的新台词", "乙的新台词"])

    def test_clone_personality_adapts_punctuation_without_changing_words(self):
        from voice_consistency import clone_performance_text
        voice = {"mode": "clone", "personality": "外冷内热，生气时不吼，越愤怒音量越低。"}
        cue = {"text": "你马上给我离开！！！", "emotion": "愤怒、咬字更重，但压住音量"}
        adapted = clone_performance_text(voice, cue)
        self.assertEqual(adapted.rstrip("。！？…"), "你马上给我离开")
        self.assertEqual(adapted[-1], "。")

    def test_clone_personality_adapts_sampling_profile(self):
        from voice_consistency import clone_performance_options
        base = {"temperature": .72, "top_k": 36, "top_p": .9}
        restrained = clone_performance_options(
            {"personality": "冷静克制，紧张时先停顿"},
            {"emotion": "惊慌但压住声音"},
            base,
        )
        expressive = clone_performance_options(
            {"personality": "外向活泼，情绪鲜明"},
            {"emotion": "兴奋欢呼"},
            base,
        )
        self.assertLess(restrained["temperature"], base["temperature"])
        self.assertGreater(expressive["temperature"], base["temperature"])

    def test_clone_fallback_emotion_changes_sampling_profile(self):
        from voice_consistency import clone_performance_options
        base = {"temperature": .72, "top_k": 36, "top_p": .9}
        name_clear = clone_performance_options(
            {},
            {"emotion": "自然、清晰、平稳，名字两个音节分开完整发音"},
            base,
        )
        restrained_boundary = clone_performance_options(
            {},
            {"emotion": "自然、清晰、克制，句首语气词只发一次"},
            base,
        )
        self.assertEqual(name_clear["temperature"], .66)
        self.assertEqual(name_clear["top_k"], 30)
        self.assertEqual(name_clear["top_p"], .86)
        self.assertEqual(restrained_boundary["temperature"], .66)
        self.assertEqual(restrained_boundary["top_k"], 30)
        self.assertEqual(restrained_boundary["top_p"], .86)
        expressive_laugh = clone_performance_options(
            {},
            {"emotion": "轻快、含着笑，句首两个嘿清晰发音"},
            base,
        )
        self.assertEqual(expressive_laugh["temperature"], .78)
        self.assertEqual(expressive_laugh["top_k"], 42)
        self.assertEqual(expressive_laugh["top_p"], .92)

    def test_clone_generation_still_uses_authorized_confirmed_reference(self):
        from voice_consistency import generate_cue_results

        class FakeModel:
            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return []

            def generate_voice_design(self, **kwargs):
                raise AssertionError("授权克隆不应调用 VoiceDesign")

        with tempfile.NamedTemporaryFile(suffix=".wav") as reference:
            model = FakeModel()
            generate_cue_results(
                model,
                "clone",
                {
                    "clone_consent": True,
                    "ref_audio": reference.name,
                    "ref_text": "准确参考文本",
                    "reference_revision": "rev-1",
                    "ref_text_confirmed": True,
                    "ref_text_confirmed_revision": "rev-1",
                },
                {"id": 1, "text": "新的台词"},
                "Chinese",
                {"temperature": 0.7, "max_tokens": 96},
                {},
            )
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["text"], "新的台词")
        self.assertEqual(model.calls[0]["ref_audio"], reference.name)
        self.assertEqual(model.calls[0]["ref_text"], "准确参考文本")
        self.assertEqual(model.calls[0]["lang_code"], "Chinese")
        self.assertEqual(model.calls[0]["max_tokens"], 96)

    def test_two_clone_generations_both_use_same_authorized_reference(self):
        from voice_consistency import generate_cue_results

        class FakeModel:
            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return []

        with tempfile.NamedTemporaryFile(suffix=".wav") as reference:
            model = FakeModel()
            voice = {
                "mode": "clone", "clone_consent": True, "ref_audio": reference.name,
                "ref_text": "人工确认的真实逐字稿", "ref_text_auto": "自动识别的真实逐字稿", "ref_text_source": "manual",
                "reference_revision": "rev-1", "ref_text_confirmed": True, "ref_text_confirmed_revision": "rev-1",
            }
            for cue_id, text in ((1, "第一次生成"), (2, "第二次生成")):
                generate_cue_results(
                    model, "clone", voice, {"id": cue_id, "text": text}, "Chinese",
                    {"temperature": 0.7}, {},
                )
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(all(call["ref_audio"] == reference.name for call in model.calls))
        self.assertTrue(all(call["ref_text"] == "人工确认的真实逐字稿" for call in model.calls))

    def test_disabled_design_cues_do_not_change_per_line_generation_mode(self):
        from voice_consistency import cue_generation_plan
        cues = [
            {"id": 1, "role": "role1", "text": "禁用这句", "enabled": False},
            {"id": 2, "role": "role1", "text": "真正的第一句", "emotion": "平静", "enabled": True},
            {"id": 3, "role": "role1", "text": "后续惊呼", "emotion": "惊讶", "enabled": True},
        ]
        voices = {"role1": {"id": "role1", "mode": "design", "description": "男声"}}
        plan = cue_generation_plan(cues, voices)
        self.assertTrue(all(item["mode"] == "design" for item in plan))
        self.assertEqual(plan[2]["cue"]["emotion"], "惊讶")

    def test_clone_cache_key_changes_when_reference_revision_changes(self):
        from voice_consistency import generation_cache_key
        cue = {"id": 1, "role": "role1", "text": "同一句台词", "emotion": "自然"}
        voice = {
            "mode": "clone", "description": "授权声线", "ref_text": "准确逐字稿",
            "reference_revision": "first",
        }
        first = generation_cache_key(cue, voice, "Chinese", "clone-model", {"temperature": 0.7})
        second = generation_cache_key(cue, {**voice, "reference_revision": "second"}, "Chinese", "clone-model", {"temperature": 0.7})
        self.assertNotEqual(first, second)

    def test_clone_cache_key_changes_when_confirmed_transcript_changes(self):
        from voice_consistency import generation_cache_key
        cue = {"id": 1, "role": "role1", "text": "同一句台词", "emotion": "自然"}
        voice = {"mode": "clone", "reference_revision": "rev-1", "ref_text": "自动稿"}
        first = generation_cache_key(cue, voice, "Chinese", "clone-model", {"max_tokens": 96})
        second = generation_cache_key(cue, {**voice, "ref_text": "人工校正稿"}, "Chinese", "clone-model", {"max_tokens": 96})
        self.assertNotEqual(first, second)

    def test_clone_dynamic_max_tokens_changes_with_text_and_enters_cache_identity(self):
        from voice_consistency import cue_generation_cache_key, cue_generation_options
        short = {"id": 1, "role": "role1", "text": "走吧。", "start": 0, "end": 2}
        long = {**short, "text": "我们现在必须穿过这片森林，在天黑之前赶到安全的营地。", "end": 8}
        base = {"temperature": .72, "top_k": 36, "top_p": .9, "repetition_penalty": 1.2}
        short_options, short_limits = cue_generation_options("clone", short, "Chinese", "natural", base)
        long_options, long_limits = cue_generation_options("clone", long, "Chinese", "natural", base)
        self.assertLess(short_options["max_tokens"], long_options["max_tokens"])
        self.assertLess(short_limits["max_seconds"], long_limits["max_seconds"])
        voice = {"mode": "clone", "reference_revision": "rev-1", "ref_text": "确认稿"}
        first = cue_generation_cache_key(short, voice, "Chinese", "clone-model", short_options, limits=short_limits)
        second = cue_generation_cache_key(short, voice, "Chinese", "clone-model", short_options, limits={**short_limits, "policy": "new-policy"})
        self.assertNotEqual(first, second)

    def test_fixed_timeline_can_tighten_clone_duration_budget(self):
        from voice_consistency import cue_generation_options
        cue = {
            "id": 1, "role": "role1", "text": "这是一句相对较长、但时间槽非常短的台词。",
            "start": 0, "end": .5,
        }
        _, natural = cue_generation_options("clone", cue, "Chinese", "natural", {})
        _, fixed = cue_generation_options("clone", cue, "Chinese", "fixed", {})
        self.assertLessEqual(fixed["max_seconds"], natural["max_seconds"])

    def test_clone_duration_guard_accepts_normal_short_audio_and_rejects_runaway(self):
        from voice_consistency import clone_generation_limits, validate_clone_output_duration
        limits = clone_generation_limits("年轻人，你调的是这个精神灯还是这个阴神灯？", "Chinese")
        self.assertIs(validate_clone_output_duration("测试", min(4.0, limits["max_seconds"]), limits=limits), limits)
        with self.assertRaisesRegex(RuntimeError, "30.78.*时长异常|时长异常.*30.78"):
            validate_clone_output_duration("测试", 30.782, limits=limits)

    def test_design_generation_options_are_unchanged_by_clone_guard(self):
        from voice_consistency import cue_generation_options
        options, limits = cue_generation_options(
            "design", {"text": "测试"}, "Chinese", "natural",
            clone_options={"max_tokens": 1}, design_options={"temperature": .7, "max_tokens": 1400},
        )
        self.assertEqual(options, {"temperature": .7, "max_tokens": 1400})
        self.assertIsNone(limits)

    def test_raw_cache_policy_rejects_metadata_from_an_older_generation_contract(self):
        source = Path(core.__file__).with_name("tts_worker.py").read_text()
        policy = re.search(r'^RAW_CACHE_POLICY = "([^"]+)"$', source, re.MULTILINE)
        self.assertIsNotNone(policy)
        self.assertTrue(policy.group(1).startswith("raw-generation-v"))
        self.assertIn('"raw_cache_policy": RAW_CACHE_POLICY', source)

    def test_design_line_cache_identity_depends_on_its_own_emotion_not_first_line(self):
        from voice_consistency import cue_generation_plan
        voices = {"role1": {"id": "role1", "mode": "design", "description": "男声"}}
        first = cue_generation_plan([
            {"id": 1, "role": "role1", "text": "旧的第一句", "enabled": True},
            {"id": 2, "role": "role1", "text": "后续台词", "emotion": "紧张", "enabled": True},
        ], voices)
        changed = cue_generation_plan([
            {"id": 1, "role": "role1", "text": "新的第一句", "enabled": True},
            {"id": 2, "role": "role1", "text": "后续台词", "emotion": "紧张", "enabled": True},
        ], voices)
        self.assertEqual(first[1]["voice"], changed[1]["voice"])
        self.assertEqual(changed[1]["cue"]["emotion"], "紧张")


if __name__ == "__main__":
    unittest.main()
