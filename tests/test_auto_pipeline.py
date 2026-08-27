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

    def test_preview_endpoint_reserves_job_before_starting_thread(self):
        app = webapp.create_app()
        client = app.test_client()
        webapp.JOBS.clear()

        class PendingThread:
            starts = 0

            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                type(self).starts += 1

        payload = {"role_id": "role1", "text": "测试试听", "emotion": "自然"}
        with patch.object(webapp.threading, "Thread", PendingThread):
            first = client.post("/api/projects/p1/preview-voice", json=payload)
            second = client.post("/api/projects/p1/preview-voice", json=payload)

        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 409)
        self.assertEqual(PendingThread.starts, 1)
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

    def test_voice_project_api_uses_blank_project_creator(self):
        app = webapp.create_app()
        client = app.test_client()
        created = {"id": "voice-project", "project_type": "voice", "name": "旁白"}
        with patch.object(core, "create_voice_project", return_value=created) as create:
            response = client.post(
                "/api/projects",
                json={"project_type": "voice", "name": "旁白", "target_lang": "zh"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["project_type"], "voice")
        create.assert_called_once_with("旁白", "zh")

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

    def test_clone_role_accepts_reference_with_or_without_transcript(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as reference:
            project = {
                "roles": [{
                    "id": "role1", "name": "角色", "mode": "clone", "clone_consent": True,
                    "ref_audio": reference.name, "ref_text": "",
                }],
                "cues": [{"role": "role1", "enabled": True}],
            }
            core.validate_clone_roles(project)
            project["roles"][0]["ref_text"] = "这是准确参考文本"
            core.validate_clone_roles(project)

    def test_reference_upload_binds_normalized_audio_and_consent(self):
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
             patch.object(core, "run") as run, \
             patch.object(core, "save_manifest", side_effect=lambda pid, value: value):
            response = client.post(
                "/api/projects/p1/roles/role1/reference",
                data={
                    "audio": (io.BytesIO(b"fake-audio"), "reference.mp3"),
                    "consent": "true",
                    "ref_text": "参考文本",
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        role = response.get_json()["roles"][0]
        self.assertEqual(role["mode"], "clone")
        self.assertTrue(role["clone_consent"])
        self.assertEqual(role["ref_text"], "参考文本")
        self.assertTrue(role["ref_audio"].endswith("role1.wav"))
        run.assert_called_once()

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
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(core, "load_project", side_effect=lambda pid: project), \
             patch.object(core, "project_dir", return_value=Path(directory)), \
             patch.object(core, "save_project", side_effect=lambda value: value), \
             patch.object(core, "add_log"), \
             patch.object(core, "set_stage"), \
             patch.object(core, "ffmpeg_full", return_value="ffmpeg"), \
             patch.object(core, "media_tool", return_value="ffprobe"), \
             patch.object(core.subprocess, "run", return_value=process) as run:
            core.generate_tts("p1")
        command = run.call_args.args[0]
        self.assertIn("--timeline-mode", command)
        self.assertIn("natural", command)
        self.assertIn("--design-model", command)
        self.assertIn("design-model", command)
        self.assertIn("--clone-model", command)
        self.assertIn("clone-model", command)


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
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["id"], "doc-project")
        create.assert_called_once()
        self.assertEqual(create.call_args.args[1], "夜戏")
        save.assert_called_once()
        start.assert_not_called()
        direction.assert_called_once_with("doc-project")

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
            self.assertNotEqual(project["roles"][0]["voice"], project["roles"][1]["voice"])
            self.assertNotEqual(project["cues"][0]["emotion"], project["cues"][1]["emotion"])
            self.assertNotEqual(project["cues"][0]["emotion"], "自然")

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

    def test_project_script_upload_endpoint_uses_current_project(self):
        app = webapp.create_app()
        updated = {"id": "p1", "cues": [{"id": 1}], "script_import": {"cue_count": 1}}
        with patch.object(core, "import_document_into_project", return_value=updated) as import_into, \
             patch.object(webapp, "start_script_direction") as direction:
            response = app.test_client().post(
                "/api/projects/p1/script-upload",
                data={"document": (io.BytesIO("旁白：夜色来了。".encode("utf-8")), "夜戏.txt")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        import_into.assert_called_once()
        self.assertEqual(import_into.call_args.args[0], "p1")
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
            project = core.create_voice_project_from_document(source, "广播剧", "zh")
            project["cues"][2]["emotion"] = "用户改过的情绪"
            core.save_project(project)

            def fake_ollama(roles, cues, model, locked_role_ids=None, cue_ids=None):
                return (
                    [{"id": "role1", "voice": "male_adult", "description": "沉稳旁白"},
                     {"id": "role2", "voice": "male_young", "description": "年轻男声"},
                     {"id": "role3", "voice": "female_warm", "description": "温暖女声"}],
                    [{"id": 1, "emotion": "压低声音、叙述"}, {"id": 2, "emotion": "急迫催促"}, {"id": 3, "emotion": "被覆盖的情绪"}],
                )

            with patch.object(core, "ollama_direct_script", side_effect=fake_ollama):
                updated = core.refine_imported_direction(project["id"])
            self.assertEqual(updated["cues"][0]["emotion"], "压低声音、叙述")
            self.assertEqual(updated["cues"][1]["emotion"], "急迫催促")
            self.assertEqual(updated["cues"][2]["emotion"], "用户改过的情绪")
            self.assertTrue(updated["script_import"]["direction"]["refined"])

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

    def test_clone_generation_still_uses_authorized_reference(self):
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
                },
                {"id": 1, "text": "新的台词"},
                "Chinese",
                {"temperature": 0.7},
                {},
            )
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["ref_audio"], reference.name)
        self.assertEqual(model.calls[0]["ref_text"], "准确参考文本")
        self.assertEqual(model.calls[0]["lang_code"], "Chinese")

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
