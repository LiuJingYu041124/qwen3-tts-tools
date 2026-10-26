import sys,unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,'/Users/administrator/qwen3-tts')
import tts_worker as w
class BoundaryReview(unittest.TestCase):
 def review(self,texts):
  with patch.object(w,'transcribe_audio_array',side_effect=texts):
   return w.transcribe_full_cue(np.ones(24000,dtype=np.float32),24000,'巫妖，使用十字陨星')
 def test_two_exact_tone_agreements_recover_boundary_miss(self):
  self.assertTrue(w.cue_asr_quality('巫妖使用十字陨星',self.review(['使用十字允星','乌妖使用十字允星','巫妖使用十字允星']))['passed'])
 def test_single_lucky_review_cannot_pass(self):
  self.assertEqual(self.review(['使用十字允星','巫妖使用十字允星','乌妖使用十字蕴星','使用十字允星']),'使用十字允星')
 def test_missing_word_stays_rejected(self):
  self.assertEqual(self.review(['使用十字允星']*3),'使用十字允星')
 def test_unavailable_fails_closed(self):
  self.assertIsNone(self.review([None]))
if __name__=='__main__':unittest.main()
