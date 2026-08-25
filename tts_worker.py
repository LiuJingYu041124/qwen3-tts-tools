#!/usr/bin/env python3
import argparse,gc,json,subprocess
from pathlib import Path
import mlx.core as mx
import numpy as np
from scipy.io import wavfile
from mlx_audio.tts.utils import load_model

MODEL='mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit'
FF='/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg'
SR=24000

def dur(p):
 return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(p)],text=True).strip())
def atempo_chain(x):
 out=[]
 while x>2: out.append('atempo=2.0'); x/=2
 while x<.5: out.append('atempo=0.5'); x/=.5
 out.append(f'atempo={x:.8f}'); return ','.join(out)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--duration',type=float,required=True); ap.add_argument('--reuse',action='store_true'); a=ap.parse_args()
 root=Path(a.root); raw=root/'raw'; clean=root/'clean'; fit=root/'fit'
 for p in (raw,clean,fit): p.mkdir(parents=True,exist_ok=True)
 rec=json.loads((root/'manifest.json').read_text()); voices=json.loads((root/'voices.json').read_text())
 model=load_model(MODEL); status=[]
 for r in rec:
  i=int(r['id']); role=r['role']; rp=raw/f'{i:03d}-{role}.wav'
  if not (a.reuse and rp.exists() and dur(rp)>.1):
   mx.random.seed(930000+i); parts=[]; sr=None
   slot=r['end']-r['start']; pace='语速紧凑，句尾不要拖长。' if slot<1.3 else '语速自然，停顿简短。'
   ins=voices[role]+' 当前台词表演要求：'+r.get('emotion','自然')+'。'+pace+'严格只说给定中文台词，不增加解释。'
   for z in model.generate(text=r['text'],instruct=ins,lang_code='chinese',temperature=.70,top_k=36,top_p=.9,repetition_penalty=1.18,max_tokens=700,verbose=False):
    parts.append(np.asarray(z.audio,dtype=np.float32).reshape(-1)); sr=z.sample_rate
   if not parts: raise RuntimeError(f'cue {i} 没有生成音频')
   wavfile.write(rp,sr,np.clip(np.concatenate(parts),-1,1)); gc.collect(); mx.clear_cache()
  cp=clean/f'{i:03d}.wav'
  subprocess.run([FF,'-y','-v','error','-i',str(rp),'-af','silenceremove=start_periods=1:start_duration=0.02:start_threshold=-46dB:stop_periods=-1:stop_duration=0.06:stop_threshold=-46dB,areverse,silenceremove=start_periods=1:start_duration=0.02:start_threshold=-46dB,areverse','-ar',str(SR),'-ac','1','-c:a','pcm_f32le',str(cp)],check=True)
  slot=max(.18,r['end']-r['start']-.035); cd=dur(cp); required=cd/slot
  max_tempo=1.45 if slot>=.85 else (1.85 if slot>=.60 else 2.5)
  applied=max(.86,min(max_tempo,required)); post=cd/applied
  if post>slot+.012: raise RuntimeError(f'台词{i}过长：清理后{cd:.2f}s，槽位{slot:.2f}s，需要{required:.2f}倍，允许{max_tempo:.2f}倍。请缩短中文或调时间。')
  # Always fill the complete cue slot. `pad_dur` was unreliable with some
  # FFmpeg builds; whole_dur guarantees the fitted WAV ends at the cue boundary.
  fp=fit/f'{i:03d}.wav'; filters=[atempo_chain(applied),f'apad=whole_dur={slot:.6f}',f'atrim=duration={slot:.6f}','afade=t=in:st=0:d=0.006']
  subprocess.run([FF,'-y','-v','error','-i',str(cp),'-af',','.join(filters),'-ar',str(SR),'-ac','1','-c:a','pcm_f32le',str(fp)],check=True)
  x={**r,'raw_duration':cd,'tempo':applied,'duration':dur(fp),'end_error':(r['start']+dur(fp))-r['end']}
  status.append(x); (root/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)); print(i,role,r['text'],f'tempo={applied:.3f}',flush=True)
 timeline=np.zeros(round(a.duration*SR),np.float32)
 for r in status:
  _,x=wavfile.read(fit/f"{r['id']:03d}.wav"); x=np.asarray(x,np.float32); b=round(r['start']*SR); e=min(len(timeline),b+len(x)); timeline[b:e]+=x[:e-b]
 peak=float(np.max(np.abs(timeline))) if len(timeline) else 0
 if peak>.92: timeline*=.92/peak
 wavfile.write(root/'voice_timeline.wav',SR,timeline)
 summary={'cues':len(status),'roles':len(set(x['role'] for x in status)),'max_tempo':max(x['tempo'] for x in status),'peak':float(np.max(np.abs(timeline)))}
 (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)); print('DONE',json.dumps(summary,ensure_ascii=False))
if __name__=='__main__': main()
