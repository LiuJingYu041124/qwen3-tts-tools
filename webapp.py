#!/usr/bin/env python3
import json,threading,traceback,tempfile
from pathlib import Path
from flask import Flask,jsonify,render_template,request,send_file
from werkzeug.utils import secure_filename
import dubstudio as core

JOBS={}
ALLOWED_VIDEO={'.mp4','.mov','.mkv','.webm','.m4v','.avi'}

def pipeline_worker(pid,resume=False):
 key=f'{pid}:pipeline'; JOBS[key]={'status':'running','message':'自动处理已启动'}
 try:
  core.auto_pipeline(pid,resume=resume)
  JOBS[key]={'status':'done','message':'自动配音完成'}
 except Exception as e:
  JOBS[key]={'status':'failed','message':str(e),'trace':traceback.format_exc()}

def start_auto_pipeline(pid,resume=False):
 key=f'{pid}:pipeline'
 if JOBS.get(key,{}).get('status')=='running': return False
 threading.Thread(target=pipeline_worker,args=(pid,resume),daemon=True).start()
 return True

def create_app():
 app=Flask(__name__,template_folder='templates',static_folder='static')
 def worker(pid,stage,fn):
  key=f'{pid}:{stage}'; JOBS[key]={'status':'running','message':'正在执行'}
  try: fn(pid); JOBS[key]={'status':'done','message':'完成'}
  except Exception as e:
   JOBS[key]={'status':'failed','message':str(e),'trace':traceback.format_exc()}; core.set_stage(pid,stage,'failed',str(e))
 @app.get('/')
 def home(): return render_template('index.html')
 @app.get('/api/projects')
 def projects():
  out=[]
  for d in sorted(core.PROJECTS_DIR.glob('*'),reverse=True):
   p=core.read_json(d/'project.json')
   if p: out.append(p)
  return jsonify(out)
 @app.post('/api/projects')
 def create():
  x=request.get_json(force=True); return jsonify(core.create_project(x['source'],x.get('name'),x.get('copy_source',False)))
 @app.post('/api/auto')
 def auto_upload():
  video=request.files.get('video')
  if not video or not video.filename: return jsonify({'error':'请选择要上传的视频'}),400
  suffix=Path(secure_filename(video.filename)).suffix.lower()
  if suffix not in ALLOWED_VIDEO: return jsonify({'error':'不支持的视频格式'}),400
  with tempfile.NamedTemporaryFile(prefix='dub-upload-',suffix=suffix,delete=False) as tmp:
   video.save(tmp.name); temp_path=Path(tmp.name)
  try:
   p=core.create_project(temp_path,request.form.get('name') or Path(video.filename).stem,copy_source=True,
                         target_lang=request.form.get('target_lang','zh'),
                         tts_engine=request.form.get('tts_engine','gemini'))
   p['original_source']=video.filename
   core.save_project(p)
  finally:
   temp_path.unlink(missing_ok=True)
  start_auto_pipeline(p['id'])
  return jsonify(p),202
 @app.get('/api/projects/<pid>')
 def get_project(pid): return jsonify(core.load_project(pid))
 @app.put('/api/projects/<pid>')
 def save(pid): return jsonify(core.save_manifest(pid,request.get_json(force=True)))
 @app.post('/api/projects/<pid>/<stage>')
 def action(pid,stage):
  if stage in ('auto','retry'):
   if not start_auto_pipeline(pid,resume=stage=='retry'): return jsonify(JOBS[f'{pid}:pipeline']),409
   return jsonify({'status':'running'})
  funcs={'analyze':core.analyze,'translate':core.gemini_analyze,'tts':core.generate_tts,'render':core.render,'qa':core.qa}
  if stage not in funcs: return jsonify({'error':'未知操作'}),404
  key=f'{pid}:{stage}'
  if JOBS.get(key,{}).get('status')=='running': return jsonify(JOBS[key]),409
  threading.Thread(target=worker,args=(pid,stage,funcs[stage]),daemon=True).start()
  return jsonify({'status':'running'})
 @app.get('/api/projects/<pid>/jobs')
 def jobs(pid): return jsonify({k.split(':',1)[1]:v for k,v in JOBS.items() if k.startswith(pid+':')})
 @app.get('/api/projects/<pid>/log')
 def log(pid):
  p=core.project_dir(pid)/'activity.log'; return jsonify({'log':p.read_text() if p.exists() else ''})
 @app.get('/api/projects/<pid>/download')
 def download(pid):
  p=core.load_project(pid); out=Path(p.get('output',''))
  if not out.exists(): return jsonify({'error':'成片不存在'}),404
  return send_file(out,as_attachment=True,download_name=out.name)
 @app.get('/api/projects/<pid>/preview')
 def preview(pid):
  p=core.load_project(pid); out=Path(p.get('output',''))
  if not out.exists(): return jsonify({'error':'成片不存在'}),404
  return send_file(out,conditional=True,mimetype='video/mp4')
 return app

if __name__=='__main__': create_app().run(port=8765)
