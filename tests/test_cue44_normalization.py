import sys,unittest
sys.path.insert(0,'/Users/administrator/qwen3-tts')
from tts_worker import cue_asr_quality
class NormalizationTests(unittest.TestCase):
 def test_numeric_format(self):
  self.assertTrue(cue_asr_quality('我要直接打阿乐一个三比零','我要直接打阿樂，一個３比０。')['passed'])
 def test_negative(self):
  target='我要直接打阿乐一个三比零'
  for s in ['我要打阿乐一个3比0','我要直接打阿乐3比0','我要直接打阿乐一个3比','我要直接打阿乐一个4比0','我要直接打阿了一个3比0','我要直接打阿勒一个3比0','我要直接打阿乐一个3比6']:
   with self.subTest(s=s):self.assertFalse(cue_asr_quality(target,s)['passed'])
 def test_general_numbers(self):
  for a,b in [('有十二个','有12个'),('一百零五','105'),('三点五','3.5'),('二零二六','2026')]:
   # Ambiguous year/digit strings are not silently treated as cardinals.
   if a=='二零二六':self.assertFalse(cue_asr_quality(a,b)['passed'])
   else:self.assertTrue(cue_asr_quality(a,b)['passed'])
if __name__=='__main__':unittest.main()
