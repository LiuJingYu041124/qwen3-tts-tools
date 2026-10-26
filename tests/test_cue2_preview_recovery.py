import sys,unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,'/Users/administrator/qwen3-tts');import tts_worker as w
class PreviewRecovery(unittest.TestCase):
 def test_clone_exhausted_random_seeds_try_project_role_seed(self):
  result=(np.zeros(24000,dtype=np.float32),24000,{}, {},1)
  with patch.object(w,'_generate_audio_candidate',return_value=result) as gen,patch.object(w,'transcribe_full_cue',side_effect=['无忧']*4+['巫妖使用十字陨星']):
   selected=w.generate_audio(None,'clone',{'id':'role2'},{'id':1,'role':'role2','text':'巫妖，使用十字陨星'},'zh',970001)
   self.assertEqual(selected[:5],result)
   self.assertEqual(selected[5]['_generation_qa']['transcript'],'巫妖使用十字陨星')
   self.assertEqual(gen.call_args.args[5],932194)
if __name__=='__main__':unittest.main()
