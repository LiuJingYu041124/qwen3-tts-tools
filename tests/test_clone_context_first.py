import os
import unittest
from unittest.mock import patch

import numpy as np

import tts_worker as w


class CloneContextFirstTests(unittest.TestCase):
    def setUp(self):
        self.cue = {"id": 6, "role": "role3", "text": "小海，我们出发。", "emotion": "自然"}
        self.voice = {"id": "role3", "mode": "clone"}
        self.take = (np.full(24000, .1, dtype=np.float32), 24000, {}, {"max_seconds": 8}, 12)
        self.env = patch.dict(os.environ, {"QWEN3_TTS_CLONE_CONTEXT_FIRST": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def generate(self, **kwargs):
        return w.generate_audio(None, "clone", self.voice, self.cue, "Chinese", 930862,
                                "natural", **kwargs)

    def test_success_uses_one_full_context_candidate_without_mutating_input(self):
        original = dict(self.cue)
        with patch.object(w, "_generate_audio_candidate", return_value=self.take) as generate, \
                patch.object(w, "transcribe_full_cue", return_value=self.cue["text"]) as transcribe:
            result = self.generate()
        generate.assert_called_once()
        self.assertTrue(generate.call_args.args[3]["_context_review"])
        self.assertEqual(generate.call_args.args[3]["text"], self.cue["text"])
        self.assertEqual(generate.call_args.args[5], 930862)
        transcribe.assert_called_once()
        self.assertEqual(self.cue, original)
        self.assertEqual(result[5]["generation_strategy"], "clone-context-first")
        self.assertEqual(result[5]["_generation_qa"]["transcript"], self.cue["text"])

    def test_miss_returns_to_original_segmented_candidate_and_seed(self):
        with patch.object(w, "_generate_audio_candidate", return_value=self.take) as generate, \
                patch.object(w, "transcribe_full_cue", side_effect=["我们出发", self.cue["text"]]):
            self.generate()
        self.assertEqual(generate.call_count, 2)
        self.assertTrue(generate.call_args_list[0].args[3]["_context_review"])
        self.assertNotIn("_context_review", generate.call_args_list[1].args[3])
        self.assertEqual([call.args[5] for call in generate.call_args_list], [930862, 930862])

    def test_context_candidate_is_not_repeated_after_segmented_retries(self):
        with patch.object(w, "_generate_audio_candidate", return_value=self.take) as generate, \
                patch.object(w, "transcribe_full_cue", side_effect=["我们出发"] * 4 + [self.cue["text"]]):
            self.generate()
        calls = generate.call_args_list
        self.assertEqual([(bool(c.args[3].get("_context_review")), c.args[5]) for c in calls], [
            (True, 930862), (False, 930862), (False, 938781), (False, 946700), (True, 938781),
        ])

    def test_context_duration_rejection_still_uses_segmented_fallback(self):
        with patch.object(w, "_generate_audio_candidate", side_effect=[w.CandidateRejected("时长异常"), self.take]) as generate, \
                patch.object(w, "transcribe_full_cue", return_value=self.cue["text"]):
            selected = self.generate(review_sink={})
        self.assertEqual(generate.call_count, 2)
        self.assertNotIn("best_effort", selected[5])

    def test_all_rejected_candidates_keep_review_grade_and_original_strategy_budget(self):
        with patch.object(w, "_generate_audio_candidate", return_value=self.take) as generate, \
                patch.object(w, "transcribe_full_cue", return_value="我们出发"), \
                patch.object(w, "clone_pronunciation_variants", return_value=[]):
            result = self.generate(review_sink={})
        self.assertEqual(generate.call_count, 6)
        self.assertTrue(result[5]["best_effort"])
        self.assertFalse(result[5]["gate_quality"]["passed"])
        self.assertEqual(result[5]["gate_quality"], w.cue_asr_quality(self.cue["text"], "我们出发"))

    def test_no_asr_is_never_treated_as_a_pass(self):
        with patch.object(w, "_generate_audio_candidate", return_value=self.take), \
                patch.object(w, "transcribe_full_cue", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "ASR未返回结果"):
                self.generate(review_sink={})

    def test_guard_terminal_and_full_cue_checks_remain_on_context_path(self):
        cue = dict(self.cue, text="小海，我们出发吧。")
        audio = self.take[:2] + (self.take[4],)
        with patch.object(w, "generate_seeded_audio", return_value=audio) as leaf, \
                patch.object(w, "terminal_particle_present", return_value=(True, cue["text"])) as terminal, \
                patch.object(w, "validate_clone_output_duration") as guard, \
                patch.object(w, "transcribe_full_cue", return_value=cue["text"]) as full:
            result = w.generate_audio(None, "clone", self.voice, cue, "zh", 930862, "natural")
        leaf.assert_called_once()
        terminal.assert_called_once()
        guard.assert_called_once()
        full.assert_called_once()
        self.assertEqual(guard.call_args.args[0], cue["text"])
        self.assertEqual(result[5]["generation_strategy"], "clone-context-first")

    def test_eligibility_preserves_fixed_long_single_and_non_chinese_paths(self):
        self.assertTrue(w.clone_context_first(self.cue, "Chinese", "natural"))
        for cue, language, timeline in [
            (self.cue, "Chinese", "fixed"), (self.cue, "English", "natural"),
            (dict(self.cue, text="我们出发。"), "Chinese", "natural"),
            (dict(self.cue, text="你好，" + "天气不错" * 12), "Chinese", "natural"),
            (dict(self.cue, text="，。"), "Chinese", "natural"),
            (dict(self.cue, _context_review=True), "Chinese", "natural"),
            (dict(self.cue, _generation_strategy="clone-name-segmented"), "Chinese", "natural"),
        ]:
            with self.subTest(cue=cue, language=language, timeline=timeline):
                self.assertFalse(w.clone_context_first(cue, language, timeline))

    def test_design_generation_order_is_unchanged(self):
        cue = dict(self.cue, text="这是很长的一段旁白，这部分还是要按照原来的方式生成。")
        with patch.object(w, "_generate_audio_candidate", return_value=self.take) as generate, \
                patch.object(w, "transcribe_full_cue", return_value=cue["text"]):
            w.generate_audio(None, "design", {}, cue, "Chinese", 930862, "natural")
        self.assertNotIn("_context_review", generate.call_args.args[3])

    def test_switch_does_not_invalidate_previously_approved_raw_cache(self):
        models = {"clone": "test-clone", "design": "test-design"}
        original = w.raw_cache_key(self.cue, self.voice, "Chinese", models, "natural")
        with patch.dict(os.environ, {"QWEN3_TTS_CLONE_CONTEXT_FIRST": "0"}):
            self.assertFalse(w.clone_context_first(self.cue, "Chinese", "natural"))
            self.assertEqual(w.raw_cache_key(self.cue, self.voice, "Chinese", models, "natural"), original)
            with patch.object(w, "_generate_audio_candidate", return_value=self.take) as generate, \
                    patch.object(w, "transcribe_full_cue", return_value=self.cue["text"]):
                self.generate()
            self.assertNotIn("_context_review", generate.call_args.args[3])


if __name__ == "__main__":
    unittest.main()
