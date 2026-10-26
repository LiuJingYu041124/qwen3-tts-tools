import unittest
from unittest.mock import patch
import tts_worker as w

class TraditionalASRTests(unittest.TestCase):
    def test_reported_traditional_phrase(self):
        self.assertTrue(w.cue_asr_quality('环境勉强有几分相似，', '環境勉強有幾分相似。')['passed'])

    def test_traditional_missing_each_character_rejected(self):
        expected = '环境勉强有几分相似'
        actual = '環境勉強有幾分相似'
        for i in range(len(actual)):
            with self.subTest(i=i):
                self.assertFalse(w.cue_asr_quality(expected, actual[:i]+actual[i+1:])['passed'])

    def test_traditional_phrase_homophone(self):
        self.assertTrue(w.cue_asr_quality('环境勉强有几分相似', '環境勉強有幾分相四')['passed'])

    def test_wrong_tone_still_rejected(self):
        self.assertFalse(w.cue_asr_quality('环境勉强有几分相似', '環境勉強有幾分相思')['passed'])

    def test_terminal_particle_normalizes_script_and_trailing_punctuation(self):
        cases = [
            ('吗', '方塊上刻的太陽圖案這一關會和太陽有關嗎?'),
            ('吗', '方塊上課的太陽圖案這一關會和太陽有關嗎?'),
            ('吗', '这一关会和太阳有关吗？'),
            ('吗', '這一關會和太陽有關嗎？！……」  \n'),
            ('嗎', '这一关会和太阳有关吗？'),
            ('吗', '可以嘛？'),
            ('呢', '他們在哪兒呢？！”'),
            ('吧', '我們走吧……」'),
            ('啊', '快跑啊！！！'),
            ('呀', '真的呀～？！'),
            ('嘛', '你說嘛？！'),
        ]
        for particle, transcript in cases:
            with self.subTest(particle=particle, transcript=transcript), \
                    patch.object(w, 'transcribe_audio_array', return_value=transcript):
                accepted, actual = w.terminal_particle_present(None, 24000, particle)
                self.assertIs(accepted, True)
                self.assertEqual(actual, transcript)

    def test_terminal_particle_must_be_the_last_spoken_character(self):
        for transcript in ('你好吗走', '你好嗎不', '这是猫？', '這是媽？',
                           '是呢？', '還有關？', '', '？！……  '):
            with self.subTest(transcript=transcript), \
                    patch.object(w, 'transcribe_audio_array', return_value=transcript):
                accepted, actual = w.terminal_particle_present(None, 24000, '吗')
                self.assertIs(accepted, False)
                self.assertEqual(actual, transcript)

    def test_terminal_asr_unavailable_remains_unknown(self):
        with patch.object(w, 'transcribe_audio_array', return_value=None):
            self.assertEqual(w.terminal_particle_present(None, 24000, '吗'), (None, ''))

    def test_traditional_terminal_does_not_weaken_full_cue_validation(self):
        expected = '方块上刻的太阳图案……这一关会和太阳有关吗？'
        complete = '方塊上刻的太陽圖案這一關會和太陽有關嗎?'
        self.assertTrue(w.cue_asr_quality(expected, complete)['passed'])
        for transcript in (
            complete.replace('太陽', '太'),
            complete.replace('刻', '可'),
            complete.replace('嗎', ''),
            complete.replace('嗎', '貓'),
            complete.replace('嗎', '嗎嗎'),
        ):
            with self.subTest(transcript=transcript):
                self.assertFalse(w.cue_asr_quality(expected, transcript)['passed'])

if __name__ == '__main__': unittest.main()
