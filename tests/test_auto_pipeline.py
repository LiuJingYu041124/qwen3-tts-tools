import io
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
             patch.object(core, "save_project", side_effect=lambda p: p), \
             patch.object(core, "add_log"), \
             patch.object(core, "analyze", side_effect=stage("analyze")), \
             patch.object(core, "ollama_translate", side_effect=stage("translate")), \
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
             patch.object(core, "save_project") as save, \
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
        start.assert_called_once_with("uploaded-project")

    def test_upload_endpoint_rejects_missing_video(self):
        app = webapp.create_app()
        response = app.test_client().post("/api/auto", data={}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("视频", response.get_json()["error"])

    def test_retry_resumes_from_failed_stage_without_repeating_completed_work(self):
        calls = []
        project = {
            "id": "p1", "stages": {"analyze": "done", "script": "ready", "tts": "failed", "render": "pending", "qa": "pending"},
            "pipeline": {"status": "failed", "stage": "tts", "percent": 48},
            "qa": {}, "cues": [{"id": 1, "text": "你好", "enabled": True}], "roles": [{"id": "role1"}],
        }
        def run(name):
            def inner(pid):
                calls.append(name)
                project["stages"][name] = "done"
                if name == "qa": project["qa"] = {"passed": True}
                return project
            return inner
        with patch.object(core, "load_project", return_value=project), \
             patch.object(core, "save_project", side_effect=lambda p: p), patch.object(core, "add_log"), \
             patch.object(core, "analyze", side_effect=run("analyze")), patch.object(core, "ollama_translate", side_effect=run("translate")), \
             patch.object(core, "save_manifest", side_effect=lambda *x: calls.append("save") or project), \
             patch.object(core, "generate_tts", side_effect=run("tts")), patch.object(core, "render", side_effect=run("render")), \
             patch.object(core, "qa", side_effect=run("qa")):
            core.auto_pipeline("p1", resume=True)
        self.assertEqual(calls, ["tts", "render", "qa"])


if __name__ == "__main__":
    unittest.main()
