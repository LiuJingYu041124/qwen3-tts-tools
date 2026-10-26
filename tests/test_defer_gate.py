import json, sys, unittest, tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0, '/Users/administrator/qwen3-tts')
import tts_worker as w
import numpy as np

class FullCueGate(unittest.TestCase):
    def exercise(self, transcript, text='巫妖，使用十字陨星', expected=None):
        cue={'id':2,'role':'role2','text':text}
        if expected is not None: cue['expected_text']=expected
        with patch.object(w,'generation_options',return_value=({},{})), patch.object(w,'collect_audio',return_value=(np.zeros(24000,dtype=np.float32),24000,1)), patch.object(w,'generate_cue_results',return_value=[]), patch.object(w,'transcribe_audio_array',return_value=transcript):
            return w.generate_audio(None,'design',{},cue,'Chinese',970001)
    def test_deferred_bad_full_sentence_rejected(self):
        with self.assertRaises(RuntimeError): self.exercise('无忧使用十字陨星')
    def test_asr_unavailable_rejected(self):
        with self.assertRaises(RuntimeError): self.exercise(None)
    def test_asr_empty_rejected(self):
        with self.assertRaises(RuntimeError): self.exercise('')
    def test_single_sentence_wrong_tone_rejected(self):
        with self.assertRaises(RuntimeError): self.exercise('无忧','巫妖')
    def test_expected_text_not_generation_text(self):
        with self.assertRaises(RuntimeError): self.exercise('无忧','无忧',expected='巫妖')
    def test_correct_full_sentence_accepted(self):
        self.assertEqual(self.exercise('巫妖使用十字陨星')[1],24000)
    def test_processed_wrong_audio_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fit.wav'; w.wavfile.write(p,24000,np.zeros(24000,dtype=np.float32))
            with patch.object(w,'transcribe_audio_array',return_value='乌妖使用十字蕴星'):
                with self.assertRaises(RuntimeError): w.validate_audio_file(p,{'id':2,'text':'巫妖，使用十字陨星'})
    def test_adopted_bad_take_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'takes').mkdir()
            w.wavfile.write(root/'takes/bad.wav',24000,np.zeros(24000,dtype=np.float32))
            cue={'id':2,'text':'巫妖，使用十字陨星','adopted_take':{'file':'bad.wav'}}
            with patch.object(w,'transcribe_audio_array',return_value='无忧使用十字陨星'):
                with self.assertRaises(RuntimeError): w.adopted_take_source(root,cue)
    def test_preview_keeps_the_best_take_and_reports_the_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'preview.wav'
            with patch.object(w,'read_json',return_value={'voice':{'id':'role2','mode':'design'},'text':'巫妖，使用十字陨星'}), patch.object(w,'load_model',return_value=None), patch.object(w,'generation_options',return_value=({},{})), patch.object(w,'collect_audio',return_value=(np.zeros(24000,dtype=np.float32),24000,1)), patch.object(w,'generate_cue_results',return_value=[]), patch.object(w,'transcribe_audio_array',return_value='无忧使用十字陨星'), patch.object(w,'clean_audio',side_effect=lambda raw,dest: dest.write_bytes(b'take')):
                w.build_preview(SimpleNamespace(preview_request='unused',preview_output=str(out),design_model='test',clone_model='test'))
            # A preview exists to be listened to, so it comes back with audio and an
            # explicit verdict rather than nothing at all.
            self.assertTrue(out.exists())
            report=json.loads(out.with_suffix('.review.json').read_text())
            self.assertFalse(report['gate_passed'])
            self.assertEqual(report['gate_transcript'],'无忧使用十字陨星')
            self.assertEqual(report['expected_text'],'巫妖，使用十字陨星')
            self.assertFalse(out.with_suffix('.raw.wav').exists())
if __name__=='__main__': unittest.main(verbosity=2)
