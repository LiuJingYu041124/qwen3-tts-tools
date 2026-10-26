import json
import stat
import tempfile
import unittest
import weakref
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

import tts_worker as w


class WorkCacheTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.cli = self.root / "whisper-cli"
        self.model = self.root / "small.bin"
        self.cli.write_bytes(b"cli")
        self.model.write_bytes(b"small")
        self.stack.enter_context(patch.object(w, "WHISPER_CLI", str(self.cli)))
        self.stack.enter_context(patch.object(w, "WHISPER_MODEL", str(self.model)))
        self.audio = np.full(2400, .1, dtype=np.float32)
        self.cue = {"id": 1, "role": "role1", "text": "你好。"}
        self.voice = {"id": "role1", "mode": "design", "description": "清晰自然"}

    def response(self, text="你好。", code=0):
        return SimpleNamespace(returncode=code, stdout=text)

    def result(self, audio=None):
        return iter([SimpleNamespace(
            audio=self.audio.copy() if audio is None else audio,
            sample_rate=24000, token_count=12,
        )])

    def selected(self, transcript="你好。", cue=None):
        take = (self.audio.copy(), 24000, {}, None, 12)
        with patch.object(w, "_generate_audio_candidate", return_value=take), \
                patch.object(w, "transcribe_full_cue", return_value=transcript):
            return w.generate_audio(None, "design", self.voice, cue or self.cue, "Chinese", 7,
                                    review_sink={})

    def leaf(self, seed=7, model=None, voice=None, cue=None, options=None, mode="design"):
        return w.generate_seeded_audio(
            model, mode, self.voice if voice is None else voice,
            self.cue if cue is None else cue, "Chinese", {},
            {"temperature": .7} if options is None else options, seed,
        )

    def mock_medium_model(self):
        medium = Path(w.__file__).parent / "models" / "ggml-medium.bin"
        real_stat = Path.stat

        def model_stat(path, *args, **kwargs):
            if path == medium:
                return SimpleNamespace(st_mode=stat.S_IFREG, st_size=1533763059, st_mtime_ns=1)
            return real_stat(path, *args, **kwargs)

        self.stack.enter_context(patch.object(Path, "stat", model_stat))

    def incomplete_review_selection(self):
        take = (self.audio.copy(), 24000, {}, None, 12)
        with patch.object(w, "_generate_audio_candidate", side_effect=[take, RuntimeError("no second take")]), \
                patch.object(w, "design_pronunciation_variants", return_value=[]):
            return w.generate_audio(None, "design", self.voice, self.cue, "Chinese", 7,
                                    review_sink={}, fast_review=True)

    def test_partial_asr_reviews_are_retried_after_boundary_or_medium_recovers(self):
        self.mock_medium_model()
        for missing in (None, ""):
            cases = [
                ("boundary-first", ["你", missing, "你"], ["你", "你好。", "你好。"]),
                ("boundary-second", ["你", "你好。", missing, "你"], ["你", "你好。", "你好。"]),
                ("medium-first", ["你", "你", missing], ["你", "你", "你好。", "你好。", "你好。"]),
                ("medium-second", ["你", "你", "你好。", missing], ["你", "你", "你好。", "你好。", "你好。"]),
                ("medium-third", ["你", "你", "你好。", "你好。", missing], ["你", "你", "你好。", "你好。", "你好。"]),
            ]
            for label, initial, recovered in cases:
                with self.subTest(view=label, missing=missing), \
                        patch.object(w, "transcribe_audio_array", side_effect=initial + recovered) as transcribe:
                    selected = self.incomplete_review_selection()
                    self.assertTrue(selected[5]["best_effort"])
                    self.assertNotIn("_generation_qa", selected[5])
                    self.assertEqual(transcribe.call_count, len(initial))
                    self.assertEqual(w.selected_candidate_transcript(selected, self.cue["text"]), ("你好。", False))
                    self.assertEqual(transcribe.call_count, len(initial) + len(recovered))

    def test_cli_failure_during_review_does_not_freeze_failed_grade(self):
        self.mock_medium_model()
        for failure in (self.response(code=1), self.response(""), w.subprocess.TimeoutExpired("whisper", 45)):
            responses = [self.response("你"), failure, self.response("你"),
                         self.response("你"), self.response(), self.response()]
            with self.subTest(failure=failure), patch.object(w.subprocess, "run", side_effect=responses) as run:
                selected = self.incomplete_review_selection()
                self.assertNotIn("_generation_qa", selected[5])
                self.assertEqual(w.selected_candidate_transcript(selected, self.cue["text"]), ("你好。", False))
                self.assertEqual(run.call_count, 6)

    def test_completed_failed_review_remains_reusable(self):
        self.mock_medium_model()
        with patch.object(w, "transcribe_audio_array", side_effect=["你", "你", "你"]) as transcribe:
            selected = self.incomplete_review_selection()
            self.assertEqual(w.selected_candidate_transcript(selected, self.cue["text"]), ("你", True))
            self.assertEqual(transcribe.call_count, 3)

    def test_incomplete_review_does_not_poison_later_complete_review(self):
        self.mock_medium_model()
        take = (self.audio.copy(), 24000, {}, None, 12)
        with patch.object(w, "_generate_audio_candidate", return_value=take) as candidate, \
                patch.object(w, "transcribe_audio_array", side_effect=["你", None, "你", "你好。"]) as transcribe:
            selected = w.generate_audio(None, "design", self.voice, self.cue, "Chinese", 7)
            self.assertEqual(w.selected_candidate_transcript(selected, self.cue["text"]), ("你好。", True))
            self.assertEqual(candidate.call_count, 2)
            self.assertEqual(transcribe.call_count, 4)

    def test_incomplete_review_removes_prior_receipt_for_same_audio(self):
        self.mock_medium_model()
        take = (self.audio.copy(), 24000, {}, None, 12)
        with patch.object(w, "_generate_audio_candidate", return_value=take), \
                patch.object(w, "design_pronunciation_variants", return_value=[]), \
                patch.object(w, "transcribe_audio_array", side_effect=["你", "你", "你", "你", None, "你"]):
            selected = w.generate_audio(None, "design", self.voice, self.cue, "Chinese", 7,
                                        review_sink={}, fast_review=True)
        self.assertTrue(selected[5]["best_effort"])
        self.assertNotIn("_generation_qa", selected[5])
        with patch.object(w, "transcribe_audio_array", side_effect=["你", "你好。", "你好。"]):
            self.assertEqual(w.selected_candidate_transcript(selected, self.cue["text"]), ("你好。", False))

    def test_context_exit_releases_pcm_even_with_retained_exception(self):
        for fail in (False, True):
            retained_errors = []
            with self.subTest(fail=fail):
                try:
                    with w.cue_work_cache() as cache:
                        cache.remember_transcript("key", "你好。")
                        cache.reviews["key"] = "你好。"
                        cache.remember_candidate("key", (self.audio, 24000, 12))
                        samples = weakref.ref(cache.candidates["key"][0])
                        cache.stats["candidate_executions"] = 1
                        if fail:
                            raise RuntimeError("retain traceback")
                except RuntimeError as exc:
                    retained_errors.append(exc)
                self.assertIsNone(w._CUE_WORK_CACHE.get())
                self.assertFalse(cache.transcripts)
                self.assertFalse(cache.reviews)
                self.assertFalse(cache.candidates)
                self.assertEqual(cache.audio_bytes, 0)
                self.assertEqual(cache.stats["candidate_executions"], 1)
                self.assertIsNone(samples())
                self.assertEqual(len(retained_errors), int(fail))

    def test_successful_retry_releases_all_cached_pcm_without_damaging_selected_audio(self):
        caches, samples, errors = [], [], []

        def candidate(model, mode, voice, cue, language, seed, timeline_mode):
            audio, rate, tokens = self.leaf(seed=seed)
            cache = w._CUE_WORK_CACHE.get()
            caches.append(cache)
            samples.extend(weakref.ref(take[0]) for take in cache.candidates.values())
            if len(caches) == 1:
                try:
                    raise RuntimeError("retry once")
                except RuntimeError as exc:
                    errors.append(exc)
                    raise
            return audio, rate, {}, None, tokens

        with patch.object(w, "generate_cue_results", side_effect=lambda *args: self.result()), \
                patch.object(w, "_generate_audio_candidate", side_effect=candidate), \
                patch.object(w, "transcribe_full_cue", return_value="你好。"):
            selected = w.generate_audio(None, "design", self.voice, self.cue, "Chinese", 7)
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(caches), 2)
        self.assertTrue(all(not cache.candidates and cache.audio_bytes == 0 for cache in caches))
        self.assertTrue(all(sample() is None for sample in samples))
        np.testing.assert_array_equal(selected[0], self.audio)
        self.assertEqual(selected[5]["_generation_qa"]["transcript"], "你好。")
        self.assertEqual(selected[5]["work_cache"]["candidate_executions"], 2)

    def test_exact_pcm_is_reused_and_context_does_not_leak(self):
        with patch.object(w.subprocess, "run", return_value=self.response()) as run:
            with w.cue_work_cache() as cache:
                self.assertEqual(w.transcribe_audio_array(self.audio, 24000), "你好。")
                self.assertEqual(w.transcribe_audio_array(self.audio.copy(), 24000), "你好。")
                self.assertEqual(cache.stats["asr_cache_hits"], 1)
                self.assertEqual(cache.stats["asr_executions"], 1)
            self.assertIsNone(w._CUE_WORK_CACHE.get())
            with w.cue_work_cache():
                w.transcribe_audio_array(self.audio, 24000)
            self.assertEqual(run.call_count, 2)

    def test_pcm_rate_shape_dtype_boundary_and_model_are_distinct(self):
        other_model = self.root / "medium.bin"
        other_model.write_bytes(b"medium")
        changed = self.audio.copy()
        changed[0] += .01
        calls = [
            (self.audio, 24000, None), (changed, 24000, None),
            (self.audio, 16000, None), (self.audio[:, None], 24000, None),
            (self.audio.astype(np.float64), 24000, None),
            (np.pad(self.audio, (0, 240)), 24000, None),
            (self.audio, 24000, other_model),
        ]
        with w.cue_work_cache(), patch.object(w.subprocess, "run", return_value=self.response()) as run:
            for audio, rate, model in calls:
                w.transcribe_audio_array(audio, rate, model)
            self.assertEqual(run.call_count, len(calls))
            self.model.write_bytes(b"changed-small-model")
            w.transcribe_audio_array(self.audio, 24000)
            self.cli.write_bytes(b"changed-cli")
            w.transcribe_audio_array(self.audio, 24000)
            self.assertEqual(run.call_count, len(calls) + 2)

    def test_failures_and_empty_transcripts_are_not_cached(self):
        for response in (self.response(code=1), self.response(""), w.subprocess.TimeoutExpired("whisper", 45)):
            with self.subTest(response=response), w.cue_work_cache() as cache, \
                    patch.object(w.subprocess, "run", side_effect=[response, self.response()]) as run:
                self.assertFalse(w.transcribe_audio_array(self.audio, 24000))
                self.assertEqual(w.transcribe_audio_array(self.audio, 24000), "你好。")
                self.assertEqual(run.call_count, 2)
                self.assertEqual(cache.stats["asr_cache_hits"], 0)

    def test_job_runtime_logs_only_after_cleanup_and_restores_context_on_error(self):
        import whisper_runtime
        for fail in (False, True):
            with self.subTest(fail=fail):
                runtime = whisper_runtime.WhisperRuntime(self.cli, enabled=False)
                outer = object()
                token = w._ASR_RUNTIME.set(outer)
                try:
                    def output(*args, **kwargs):
                        self.assertTrue(runtime._closed)
                        self.assertIs(w._ASR_RUNTIME.get(), outer)
                        self.assertTrue(args[0].startswith("ASR_RUNTIME "))

                    with patch.object(whisper_runtime, "WhisperRuntime", return_value=runtime), \
                            patch("builtins.print", side_effect=output) as printed:
                        try:
                            with w.asr_job_runtime() as active:
                                self.assertIs(active, runtime)
                                self.assertIs(w._ASR_RUNTIME.get(), runtime)
                                if fail:
                                    raise RuntimeError("job failed")
                        except RuntimeError as exc:
                            self.assertTrue(fail)
                            self.assertEqual(str(exc), "job failed")
                    printed.assert_called_once()
                finally:
                    w._ASR_RUNTIME.reset(token)
                    runtime.close()

    def test_job_runtime_reuses_pcm_but_never_feeds_expected_text_to_asr(self):
        from unittest.mock import Mock
        runtime = Mock()
        runtime.identity.return_value = {"policy": "test-server", "binary": "server"}
        runtime.transcribe.return_value = " 你 好。 \n"
        token = w._ASR_RUNTIME.set(runtime)
        self.addCleanup(w._ASR_RUNTIME.reset, token)
        with w.cue_work_cache() as cache, patch.object(w.subprocess, "run") as cli:
            self.assertEqual(w.transcribe_audio_array(self.audio, 24000), "你好。")
            self.assertEqual(w.transcribe_audio_array(self.audio.copy(), 24000), "你好。")
            self.assertEqual(cache.stats["asr_executions"], 1)
            self.assertEqual(cache.stats["asr_cache_hits"], 1)
        cli.assert_not_called()
        runtime.transcribe.assert_called_once()
        args = runtime.transcribe.call_args.args
        self.assertEqual(len(args), 2)
        self.assertEqual(str(args[1]), str(self.model))
        self.assertFalse(Path(args[0]).exists())

    def test_backend_identity_invalidates_pcm_and_file_asr_reuse(self):
        from unittest.mock import Mock
        runtime = Mock()
        runtime.identity.return_value = {"policy": "test-server-v1"}
        runtime.transcribe.return_value = "你好。"
        path = self.root / "clean.wav"
        wavfile.write(path, 24000, self.audio)
        token = w._ASR_RUNTIME.set(runtime)
        self.addCleanup(w._ASR_RUNTIME.reset, token)
        with w.cue_work_cache():
            w.transcribe_audio_array(self.audio, 24000)
            shared = {}
            first, state = w.validate_or_reuse_audio_file(path, self.cue, {}, shared)
            self.assertEqual(state, "checked")
            runtime.identity.return_value = {"policy": "test-server-v2"}
            self.assertIsNone(w.cached_file_asr({"file_asr": first}, path, self.cue))
            _, state = w.validate_or_reuse_audio_file(path, self.cue, {"file_asr": first}, shared)
            self.assertEqual(state, "checked")
            self.assertEqual(runtime.transcribe.call_count, 2)

    def test_runtime_failure_does_not_leave_cached_text_or_temp_audio(self):
        from unittest.mock import Mock
        runtime = Mock()
        runtime.identity.return_value = {"policy": "test-server"}
        runtime.transcribe.side_effect = [None, "你好。"]
        token = w._ASR_RUNTIME.set(runtime)
        self.addCleanup(w._ASR_RUNTIME.reset, token)
        with w.cue_work_cache() as cache:
            self.assertIsNone(w.transcribe_audio_array(self.audio, 24000))
            self.assertFalse(cache.transcripts)
            self.assertEqual(w.transcribe_audio_array(self.audio, 24000), "你好。")
        self.assertTrue(all(not Path(call.args[0]).exists() for call in runtime.transcribe.call_args_list))

    def test_cached_transcript_does_not_cache_a_verdict_for_different_text(self):
        long_audio = np.full(24 * 24000, .1, dtype=np.float32)
        with w.cue_work_cache(), patch.object(w.subprocess, "run", return_value=self.response()) as run:
            first = w.transcribe_full_cue(long_audio, 24000, "你好。")
            second = w.transcribe_full_cue(long_audio, 24000, "你好吗？")
            self.assertTrue(w.cue_asr_quality("你好。", first)["passed"])
            self.assertFalse(w.cue_asr_quality("你好吗？", second)["passed"])
            self.assertEqual(run.call_count, 1)

    def test_selected_qa_is_reused_only_for_identical_input(self):
        selected = self.selected()
        with patch.object(w, "transcribe_full_cue", return_value="重新识别") as transcribe:
            self.assertEqual(w.selected_candidate_transcript(selected, "你好。"), ("你好。", True))
            transcribe.assert_not_called()
            for expected in ("你好", "你好吗？"):
                self.assertEqual(w.selected_candidate_transcript(selected, expected), ("重新识别", False))
            changed = selected[0].copy()
            changed[0] += .01
            for take in ((changed, *selected[1:]), (selected[0], 16000, *selected[2:]),
                         (selected[0].astype(np.float64), *selected[1:]), selected[:5]):
                self.assertEqual(w.selected_candidate_transcript(take, "你好。"), ("重新识别", False))
            self.model.write_bytes(b"new-model")
            self.assertEqual(w.selected_candidate_transcript(selected, "你好。"), ("重新识别", False))

    def test_selected_qa_invalidates_on_validator_policy_change(self):
        selected = self.selected()
        with patch.object(w, "FILE_ASR_CACHE_POLICY", "changed-policy"), \
                patch.object(w, "transcribe_full_cue", return_value="重新识别") as transcribe:
            self.assertEqual(w.selected_candidate_transcript(selected, "你好。"), ("重新识别", False))
            transcribe.assert_called_once()

    def test_selection_reuses_its_own_review_not_last_rejected_candidate(self):
        seen = []

        def candidate(*args):
            audio = np.full(2400, (len(seen) + 1) / 100, dtype=np.float32)
            seen.append(audio)
            return audio, 24000, {}, None, 12

        def transcript(audio, *_):
            return "你好呀" if np.array_equal(audio, seen[0]) else "错"

        with patch.object(w, "_generate_audio_candidate", side_effect=candidate), \
                patch.object(w, "transcribe_full_cue", side_effect=transcript):
            selected = w.generate_audio(None, "design", self.voice, self.cue, "Chinese", 7, review_sink={})
        self.assertTrue(selected[5]["best_effort"])
        np.testing.assert_array_equal(selected[0], seen[0])
        with patch.object(w, "transcribe_full_cue", side_effect=AssertionError("must reuse selected review")):
            self.assertEqual(w.selected_candidate_transcript(selected, self.cue["text"]), ("你好呀", True))
        self.assertFalse(w.cue_asr_quality(self.cue["text"], "你好呀")["passed"])

    def test_load_uses_verified_review_and_does_not_persist_ephemeral_receipt(self):
        raw = self.root / "raw.wav"
        meta = self.root / "raw.json"
        take = (self.audio.copy(), 24000, {}, None, 12)
        with patch.object(w, "load_model", return_value=object()), \
                patch.object(w, "_generate_audio_candidate", return_value=take), \
                patch.object(w, "transcribe_full_cue", return_value="你好。") as transcribe:
            w.load_and_generate("test", "design", [(self.cue, self.voice, raw, meta, "fp", {})],
                                "Chinese", {"design": "test"}, "natural")
        self.assertEqual(transcribe.call_count, 1)
        saved = json.loads(meta.read_text())
        self.assertTrue(saved["generation_qa_reused"])
        self.assertTrue(saved["asr_quality"]["passed"])
        self.assertNotIn("_generation_qa", saved)

    def test_changed_postprocessed_audio_still_requires_file_asr(self):
        self.selected()
        path = self.root / "clean.wav"
        wavfile.write(path, 24000, np.pad(self.audio, (0, 3360)))
        with patch.object(w, "transcribe_full_cue", return_value="你好。") as transcribe:
            record = w._validate_audio_file_record(path, self.cue)
        transcribe.assert_called_once()
        self.assertTrue(record["passed"])

    def test_overlapping_seed_retries_keep_five_unique_candidates(self):
        with w.cue_work_cache() as cache, \
                patch.object(w, "generate_cue_results", side_effect=lambda *args: self.result()) as generate:
            for outer in (0, 7919, 15838):
                for inner in (0, 7919, 15838):
                    self.leaf(seed=outer + inner)
        self.assertEqual(generate.call_count, 5)
        self.assertEqual(cache.stats["candidate_cache_hits"], 4)
        self.assertEqual(cache.stats["candidate_executions"], 5)

    def test_leaf_cache_preserves_audio_and_does_not_skip_validation(self):
        with w.cue_work_cache(), \
                patch.object(w, "generate_cue_results", side_effect=lambda *args: self.result()) as generate, \
                patch.object(w, "transcribe_audio_array", return_value="你好啊。") as transcribe:
            cue = dict(self.cue, text="你好啊。")
            first = w._generate_audio_candidate(None, "design", self.voice, cue, "Chinese", 7)
            second = w._generate_audio_candidate(None, "design", self.voice, cue, "Chinese", 7)
            np.testing.assert_array_equal(first[0], second[0])
            self.assertEqual(transcribe.call_count, 2)
            self.assertEqual(generate.call_count, 1)

    def test_leaf_cache_does_not_bypass_clone_length_guard(self):
        with w.cue_work_cache() as cache, \
                patch.object(w, "generate_cue_results", side_effect=lambda *args: self.result()) as generate, \
                patch.object(w, "validate_clone_output_duration", side_effect=RuntimeError("too long")) as guard:
            for _ in range(2):
                with self.assertRaises(w.CandidateRejected):
                    w._generate_audio_candidate(None, "clone", self.voice, self.cue, "Chinese", 7)
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(guard.call_count, 2)
            self.assertEqual(cache.stats["candidate_cache_hits"], 1)

    def test_unserializable_input_bypasses_cache_without_losing_generation(self):
        with w.cue_work_cache() as cache, \
                patch.object(w, "generate_cue_results", side_effect=lambda *args: self.result()) as generate:
            for _ in range(2):
                self.leaf(cue=dict(self.cue, extra=object()))
            self.assertEqual(generate.call_count, 2)
            self.assertFalse(cache.candidates)

    def test_cache_does_not_reuse_a_later_asr_failure_as_an_earlier_success(self):
        selected = list(self.selected())
        selected[5]["_generation_qa"]["transcript"] = None
        with patch.object(w, "transcribe_full_cue", return_value=None) as transcribe:
            self.assertEqual(w.selected_candidate_transcript(tuple(selected), self.cue["text"]), (None, False))
            transcribe.assert_called_once()

    def test_generation_identity_separates_voice_text_options_model_and_seed(self):
        with w.cue_work_cache(), \
                patch.object(w, "generate_cue_results", side_effect=lambda *args: self.result()) as generate:
            self.leaf()
            self.leaf(seed=8)
            self.leaf(voice=dict(self.voice, description="低沉"))
            self.leaf(cue=dict(self.cue, emotion="兴奋"))
            self.leaf(cue=dict(self.cue, text="再见。"))
            self.leaf(options={"temperature": .3})
            self.leaf(model=object())
            self.assertEqual(generate.call_count, 7)

    def test_clone_reference_content_invalidates_candidate(self):
        reference = self.root / "ref.wav"
        reference.write_bytes(b"original")
        voice = dict(self.voice, mode="clone", ref_audio=str(reference))
        with w.cue_work_cache(), \
                patch.object(w, "generate_cue_results", side_effect=lambda *args: self.result()) as generate:
            self.leaf(voice=voice, mode="clone")
            reference.write_bytes(b"modified")
            self.leaf(voice=voice, mode="clone")
            self.assertEqual(generate.call_count, 2)

    def test_cache_copy_and_memory_limits(self):
        with w.cue_work_cache() as cache, \
                patch.object(w, "generate_cue_results", side_effect=lambda *args: self.result()) as generate:
            first = self.leaf()
            first[0][0] = .9
            second = self.leaf()
            self.assertAlmostEqual(float(second[0][0]), .1)
            second[0][0] = .8
            self.assertAlmostEqual(float(self.leaf()[0][0]), .1)
            self.assertEqual(generate.call_count, 1)
            cache.max_audio_bytes = self.audio.nbytes
            self.leaf(seed=8)
            self.assertLessEqual(cache.audio_bytes, cache.max_audio_bytes)
            self.leaf()
            self.assertEqual(generate.call_count, 3)
            cache.max_entries = 1
            cache.remember_transcript("one", "一")
            cache.remember_transcript("two", "二")
            self.assertEqual(list(cache.transcripts), ["two"])

    def test_generation_errors_are_not_cached_and_context_resets_on_failure(self):
        with self.assertRaises(RuntimeError):
            with w.cue_work_cache(), patch.object(w, "generate_cue_results", side_effect=RuntimeError("failed")):
                self.leaf()
        self.assertIsNone(w._CUE_WORK_CACHE.get())
        with w.cue_work_cache(), \
                patch.object(w, "generate_cue_results", side_effect=[RuntimeError("failed"), self.result()]) as generate:
            with self.assertRaises(RuntimeError):
                self.leaf()
            self.leaf()
            self.assertEqual(generate.call_count, 2)

    def test_nested_contexts_do_not_mix_cues(self):
        with w.cue_work_cache() as outer:
            outer.remember_transcript("key", "outside")
            with w.cue_work_cache() as inner:
                self.assertNotIn("key", inner.transcripts)
            self.assertIs(w._CUE_WORK_CACHE.get(), outer)
        self.assertIsNone(w._CUE_WORK_CACHE.get())


if __name__ == "__main__":
    unittest.main()
