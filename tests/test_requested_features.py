import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dubstudio as core
import webapp


class RequestedFeatureTests(unittest.TestCase):
    def test_apply_pronunciation_lexicon_prefers_longest_terms(self):
        rules = [{"source": "重庆", "spoken": "崇庆"}, {"source": "重", "spoken": "虫"}]
        self.assertEqual(core.apply_pronunciation_lexicon("重庆银行", rules), "崇庆银行")

    def test_spoken_text_quality_detects_missing_final_character(self):
        result = core.spoken_text_quality("这是一直延伸的吗？", "这是一直延伸的")
        self.assertFalse(result["passed"])
        self.assertFalse(result["final_ok"])

    def test_spoken_text_quality_accepts_punctuation_and_script_variants(self):
        result = core.spoken_text_quality("这是一直延伸的吗？", "這是一直延伸的嗎")
        self.assertTrue(result["passed"])
        self.assertTrue(result["final_ok"])

    def test_batch_delete_cues_removes_selected_and_invalidates_outputs(self):
        app = webapp.create_app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("批量删除", "zh")
            project = core.load_project(project["id"])
            project["cues"] = [
                {"id": 1, "text": "甲", "role": "role1", "enabled": True},
                {"id": 2, "text": "乙", "role": "role1", "enabled": True},
                {"id": 3, "text": "丙", "role": "role1", "enabled": True},
            ]
            project["stages"]["tts"] = "done"
            project["output_audio"] = str(core.project_dir(project["id"]) / "voice_timeline.wav")
            core.save_project(project)
            root = core.project_dir(project["id"])
            (root / "fit").mkdir()
            (root / "fit" / "002.wav").write_bytes(b"x")
            (root / "voice_timeline.wav").write_bytes(b"mix")

            response = client.delete(
                f"/api/projects/{project['id']}/cues/batch",
                json={"cue_ids": [2, 3]},
            )
            updated = response.get_json()

            self.assertEqual(response.status_code, 200)
            self.assertEqual(updated["deleted"], 2)
            saved = core.load_project(project["id"])
            self.assertEqual([cue["id"] for cue in saved["cues"]], [1])
            self.assertEqual(saved["stages"]["tts"], "pending")
            self.assertFalse((root / "voice_timeline.wav").exists())

    def test_batch_delete_rejects_unknown_cue_ids(self):
        app = webapp.create_app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, "PROJECTS_DIR", Path(directory)):
            project = core.create_voice_project("校验删除", "zh")
            response = client.delete(
                f"/api/projects/{project['id']}/cues/batch",
                json={"cue_ids": [999]},
            )
            self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
