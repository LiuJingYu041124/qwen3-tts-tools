#!/usr/bin/env python3
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time, uuid, urllib.request
from datetime import datetime
from pathlib import Path

from document_import import DocumentError, extract_document, materialize_script, parse_script
from script_direction import apply_direction, direct_script, locked_roles, ollama_direct_script
APP_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = APP_DIR / "projects"
DEFAULT_WHISPER = Path(os.environ.get("WHISPER_MODEL", APP_DIR / "models" / "ggml-small.bin")).expanduser()
DEFAULT_TTS_PYTHON = Path(os.environ.get(
    "QWEN3_TTS_PYTHON", Path.home() / ".venvs" / "mlx-qwen3-tts" / "bin" / "python"
)).expanduser()
DEFAULT_TTS_MODEL = os.environ.get(
    "QWEN3_TTS_VOICE_DESIGN_MODEL",
    os.environ.get("QWEN3_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"),
)
DEFAULT_CLONE_MODEL = os.environ.get(
    "QWEN3_TTS_VOICE_CLONE_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
)
TTS_WORKER = APP_DIR / "tts_worker.py"
GEMINI_TTS_WORKER = APP_DIR / "gemini_tts_worker.py"
GEMINI_KEY_FILE = APP_DIR / "gemini_key.txt"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")

# 分析端点（mcfan.tv OpenAI 兼容接口，key 读自 ~/.hermes/.env）
MCFAN_BASE = "https://api.mcfan.tv/v1"
MCFAN_MODEL = "gemini-3.1-pro-low"

TARGET_LANGS = {
    "zh": {"name": "中文", "prompt": "简体中文，自然口语", "suffix": "中文多角色配音"},
    "en": {"name": "英语", "prompt": "natural spoken English", "suffix": "English dub"},
    "ja": {"name": "日语", "prompt": "自然な日本語の口語", "suffix": "日本語吹替"},
    "ko": {"name": "韩语", "prompt": "자연스러운 한국어 구어체", "suffix": "한국어 더빙"},
}

# Gemini TTS 30 个预置音色（官方说明）
GEMINI_VOICES = {
    "Zephyr": "明亮", "Puck": "欢快", "Charon": "低沉", "Kore": "坚定", "Fenrir": "易激动",
    "Leda": "年轻", "Orus": "坚定", "Aoede": "轻快", "Callirrhoe": "随和", "Autonoe": "明亮",
    "Enceladus": "气声低沉", "Iapetus": "清晰", "Umbriel": "随和", "Algieba": "平稳", "Despina": "平稳",
    "Erinome": "清晰", "Algenib": "沙哑", "Rasalgethi": "知性", "Laomedeia": "欢快", "Achernar": "柔和",
    "Alnilam": "坚定", "Schedar": "平稳", "Gacrux": "成熟", "Pulcherrima": "前卫", "Achird": "友好",
    "Zubenelgenubi": "随性", "Vindemiatrix": "温柔", "Sadachbia": "活泼", "Sadaltager": "博学", "Sulafat": "温暖",
}

VOICE_TEMPLATES = {
    "male_adult": "三十五岁左右的中国成年男性声音，中低音，沉稳可靠，略带粗犷，普通话标准，情绪自然鲜明，不要播音腔。",
    "male_young": "二十岁左右的中国年轻男性声音，音色明亮自然，有少年感和好奇心，普通话标准，不要儿童声。",
    "female_warm": "二十五岁左右的中国年轻女性声音，温暖爽朗，音色清亮自然，有活力，普通话标准。",
    "female_brave": "二十五岁左右的中国年轻女性冒险者声音，有力量、略带英气，情绪鲜明，普通话标准。",
    "girl_soft": "一个可爱的中国小女孩声音，年龄感约六到八岁，音色软萌清甜，普通话标准，不要成人腔。",
    "girl_lively": "一个活泼勇敢的中国小女孩声音，年龄感约八到十岁，音色明亮俏皮，普通话标准。",
}


def run(cmd, cwd=None, capture=True, check=True):
    p = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True,
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.PIPE if capture else None)
    if check and p.returncode:
        raise RuntimeError(f"命令失败({p.returncode}): {' '.join(map(str, cmd))}\n{p.stderr or ''}")
    return p


def tool(name):
    p = shutil.which(name)
    if not p:
        raise RuntimeError(f"缺少命令：{name}")
    return p


def media_tool(name):
    configured=os.environ.get(name.upper())
    if configured and Path(configured).exists(): return configured
    full=Path("/opt/homebrew/opt/ffmpeg-full/bin")/name
    return str(full if full.exists() else tool(name))


def ffmpeg_full(): return media_tool("ffmpeg")


def now(): return datetime.now().isoformat(timespec="seconds")

def read_json(p, default=None):
    try: return json.loads(Path(p).read_text())
    except Exception: return default

def write_json(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(p)


def project_dir(pid):
    p = PROJECTS_DIR / pid
    if not p.exists(): raise FileNotFoundError(pid)
    return p

def load_project(pid): return read_json(project_dir(pid) / "project.json", {})
def save_project(p):
    p["updated_at"] = now(); write_json(PROJECTS_DIR / p["id"] / "project.json", p)
    return p

def add_log(pid, message):
    p = project_dir(pid)
    with (p / "activity.log").open("a") as f: f.write(f"[{now()}] {message}\n")

def set_stage(pid, stage, status, message=""):
    p = load_project(pid); p.setdefault("stages", {})[stage] = status
    if message: p["last_message"] = message
    save_project(p); add_log(pid, f"{stage}: {status} {message}")


def set_pipeline(pid, status, stage, percent, message):
    p=load_project(pid)
    p["pipeline"]={"status":status,"stage":stage,"percent":int(percent),"message":message,"updated_at":now()}
    p["last_message"]=message
    save_project(p); add_log(pid,f"auto/{stage}: {status} {percent}% {message}")
    return p


def default_settings(target_lang="zh", tts_engine="local"):
    return {
        "whisper_model": str(DEFAULT_WHISPER),
        "ollama_model": OLLAMA_MODEL,
        "target_lang": target_lang,
        "tts_engine": tts_engine,
        "voice_design_model": DEFAULT_TTS_MODEL,
        "voice_clone_model": DEFAULT_CLONE_MODEL,
        "timeline_mode": "fixed",
        "default_gap": 0.25,
        "cover_y": 752,
        "cover_h": 72,
        "subtitle_font_size": 32,
        "source_gain": 1.0,
        "voice_gain": 1.55,
        "duck_threshold": 0.008,
        "keep_source_audio": True,
    }


def create_project(source, name=None, copy_source=False, target_lang="zh", tts_engine="local"):
    src = Path(source).expanduser().resolve()
    if not src.exists(): raise FileNotFoundError(src)
    if target_lang not in TARGET_LANGS: target_lang = "zh"
    if tts_engine not in ("local", "gemini"): tts_engine = "local"
    pid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    root = PROJECTS_DIR / pid; root.mkdir(parents=True)
    dst = root / ("source" + src.suffix.lower()) if copy_source else src
    if copy_source: shutil.copy2(src, dst)
    data = {
        "id": pid, "name": name or src.stem, "project_type": "video",
        "source": str(dst), "original_source": str(src),
        "created_at": now(), "updated_at": now(),
        "stages": {"analyze":"pending","script":"pending","tts":"pending","render":"pending","qa":"pending"},
        "settings": default_settings(target_lang, tts_engine),
        "roles": [], "cues": [],
        "pipeline": {"status":"pending","stage":"upload","percent":0,"message":f"视频已上传，目标语言：{TARGET_LANGS[target_lang]['name']}，等待自动处理"}
    }
    save_project(data); add_log(pid, f"创建视频项目，源文件：{src}")
    return data


def create_voice_project(name="未命名配音", target_lang="zh"):
    if target_lang not in TARGET_LANGS: target_lang = "zh"
    pid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    root = PROJECTS_DIR / pid
    root.mkdir(parents=True)
    role = {
        "id": "role1",
        "name": "旁白",
        "voice": "male_adult",
        "mode": "design",
        "description": VOICE_TEMPLATES["male_adult"],
        "style": VOICE_TEMPLATES["male_adult"],
        "clone_consent": False,
        "ref_audio": "",
        "ref_text": "",
    }
    data = {
        "id": pid,
        "name": (name or "未命名配音").strip(),
        "project_type": "voice",
        "source": "",
        "original_source": "空白脚本",
        "created_at": now(),
        "updated_at": now(),
        "stages": {"analyze":"skipped","script":"ready","tts":"pending","render":"pending","qa":"pending"},
        "settings": {**default_settings(target_lang, "local"), "timeline_mode": "natural", "keep_source_audio": False},
        "roles": [role],
        "cues": [{
            "id": 1,
            "start": 0.0,
            "end": 3.0,
            "source_text": "",
            "text": "在这里输入第一句台词。",
            "role": "role1",
            "emotion": "自然、真诚",
            "speed": 1.0,
            "gap_after": 0.25,
            "enabled": True,
        }],
        "pipeline": {"status":"pending","stage":"script","percent":0,"message":"脚本项目已创建，请编辑角色与台词"},
    }
    save_project(data)
    add_log(pid, "创建空白配音项目")
    return data


def direction_baseline(roles, cues, cue_ids=None, locked_role_ids=None):
    wanted = None if cue_ids is None else {int(item) for item in cue_ids}
    locked = {str(item) for item in (locked_role_ids or ())}
    return {
        "cues": {
            str(cue["id"]): {
                "text": str(cue.get("text") or ""),
                "emotion": str(cue.get("emotion") or ""),
            }
            for cue in cues
            if wanted is None or int(cue["id"]) in wanted
        },
        "roles": {
            role["id"]: {
                "voice": role.get("voice"),
                "description": str(role.get("description") or ""),
            }
            for role in roles
            if role.get("id") and str(role.get("id")) not in locked
        },
    }


def apply_imported_script(project, source, extracted, roles, cues, original_name=None, direction=None):
    root = project_dir(project["id"])
    src = Path(source)
    dest = root / f"script{src.suffix.lower() or '.txt'}"
    shutil.copy2(src, dest)
    (root / "script.txt").write_text(extracted["text"], encoding="utf-8")
    project["source"] = str(dest)
    project["original_source"] = original_name or src.name
    warning_text = f"；{extracted['warnings'][0]}" if extracted.get("warnings") else ""
    direction_data: dict[str, object] = dict(direction or {"source": "rules", "refined": False})
    direction_data["baseline"] = direction_baseline(
        roles,
        cues,
        direction_data.get("cue_ids"),
        direction_data.get("locked_role_ids"),
    )
    project["script_import"] = {
        "format": extracted["format"],
        "pages": extracted["pages"],
        "warnings": extracted["warnings"],
        "cue_count": len(cues),
        "role_count": len(roles),
        "direction": direction_data,
    }
    project["pipeline"] = {
        "status": "pending",
        "stage": "script",
        "percent": 0,
        "message": f"已从文档导入 {len(cues)} 条台词、{len(roles)} 个角色，并分配了情绪与音色{warning_text}",
    }
    project["last_message"] = project["pipeline"]["message"]
    project["stages"]["script"] = "ready"
    project["stages"]["tts"] = "pending"
    project["stages"]["qa"] = "pending"
    if project.get("project_type") == "voice":
        project["stages"]["render"] = "pending"
        project.pop("output_audio", None)
    save_project(project)
    add_log(project["id"], f"从文档导入脚本：{project['original_source']}，{len(cues)}条台词，{len(roles)}个角色")
    return project


def create_voice_project_from_document(source, name=None, target_lang="zh"):
    src = Path(source).expanduser().resolve()
    extracted = extract_document(src)
    items = parse_script(extracted["text"])
    settings = default_settings(target_lang, "local")
    roles, cues = materialize_script(items, default_gap=settings["default_gap"], voice_templates=VOICE_TEMPLATES)
    roles, cues, direction = direct_script(roles, cues, voice_templates=VOICE_TEMPLATES)
    project_name = (name or src.stem or "未命名配音").strip()
    project = create_voice_project(project_name, target_lang)
    project["name"] = project_name
    project["roles"] = roles
    project["cues"] = cues
    return apply_imported_script(project, src, extracted, roles, cues, src.name, direction)


def is_placeholder_script(project):
    cues = project.get("cues") or []
    if not cues:
        return True
    if len(cues) != 1:
        return False
    text = str(cues[0].get("text") or "").strip()
    return text in {"", "在这里输入第一句台词。"}


def merge_imported_roles(existing_roles, incoming_roles):
    merged = list(existing_roles or [])
    ids = {role.get("id") for role in merged}
    by_name = {str(role.get("name") or "").strip(): role for role in merged}
    mapping = {}
    for role in incoming_roles:
        name = str(role.get("name") or "旁白").strip() or "旁白"
        if name in by_name:
            mapping[role["id"]] = by_name[name]["id"]
            continue
        role_id = role["id"]
        index = 1
        while role_id in ids:
            index += 1
            role_id = f"role{index}"
        copy = dict(role)
        copy["id"] = role_id
        merged.append(copy)
        ids.add(role_id)
        by_name[name] = copy
        mapping[role["id"]] = role_id
    return merged, mapping


def import_document_into_project(pid, source, original_name=None, replace=None):
    project = normalize_project(load_project(pid))
    if project.get("project_type") != "voice":
        raise DocumentError("视频项目请继续用导入视频。文档脚本只能导入到脚本配音项目。")
    src = Path(source).expanduser().resolve()
    extracted = extract_document(src)
    items = parse_script(extracted["text"])
    roles, cues = materialize_script(
        items,
        default_gap=project["settings"].get("default_gap", 0.25),
        voice_templates=VOICE_TEMPLATES,
    )
    roles, cues, direction = direct_script(roles, cues, voice_templates=VOICE_TEMPLATES)
    should_replace = is_placeholder_script(project) if replace is None else bool(replace)
    if should_replace:
        project["roles"] = roles
        project["cues"] = cues
    else:
        existing_ids = {role.get("id") for role in project.get("roles") or []}
        merged_roles, mapping = merge_imported_roles(project.get("roles"), roles)
        start_id = max((int(cue.get("id") or 0) for cue in project.get("cues") or []), default=0)
        last = (project.get("cues") or [{}])[-1]
        cursor = float(last.get("end") or 0) + float(last.get("gap_after") or 0)
        appended = []
        for cue in cues:
            start_id += 1
            duration = max(0.2, float(cue.get("end") or 0) - float(cue.get("start") or 0))
            appended.append({
                **cue,
                "id": start_id,
                "role": mapping.get(cue["role"], cue["role"]),
                "start": round(cursor, 3),
                "end": round(cursor + duration, 3),
            })
            cursor += duration + float(cue.get("gap_after") or 0)
        project["roles"] = merged_roles
        project["cues"] = list(project.get("cues") or []) + appended
        direction["locked_role_ids"] = sorted(str(item) for item in existing_ids if item)
        direction["cue_ids"] = [cue["id"] for cue in appended]
    return apply_imported_script(project, src, extracted, project["roles"], project["cues"], original_name or src.name, direction)


def refine_imported_direction(pid, cue_ids=None, locked_role_ids=None):
    project = normalize_project(load_project(pid))
    if project.get("project_type") != "voice":
        return project
    direction = ((project.get("script_import") or {}).get("direction") or {})
    target_ids = cue_ids if cue_ids is not None else direction.get("cue_ids")
    locked = locked_roles(project.get("roles"), locked_role_ids or direction.get("locked_role_ids"))
    text_snapshot = {int(cue["id"]): str(cue.get("text") or "") for cue in project.get("cues") or []}
    emotion_snapshot = {int(cue["id"]): str(cue.get("emotion") or "") for cue in project.get("cues") or []}
    voice_snapshot = {
        role.get("id"): (role.get("voice"), str(role.get("description") or ""))
        for role in project.get("roles") or []
        if role.get("id")
    }
    set_pipeline(pid, "running", "script", 28, "正在读取上下文，给每个角色和每句台词分配音色与情绪")
    try:
        model = project.get("settings", {}).get("ollama_model", OLLAMA_MODEL)
        new_roles, new_cues = ollama_direct_script(
            project.get("roles") or [],
            project.get("cues") or [],
            model,
            locked,
            target_ids,
        )
        project = normalize_project(load_project(pid))
        wanted = None if target_ids is None else {int(item) for item in target_ids}
        baseline = ((project.get("script_import") or {}).get("direction") or {}).get("baseline") or {}
        baseline_cues = baseline.get("cues") or {}
        baseline_roles = baseline.get("roles") or {}
        current_roles = {role.get("id"): role for role in project.get("roles") or []}
        current_cues = {int(cue["id"]): cue for cue in project.get("cues") or []}

        def unchanged_role(role_id):
            current = current_roles.get(role_id) or {}
            now = (current.get("voice"), str(current.get("description") or ""))
            if now != voice_snapshot.get(role_id):
                return False
            imported = baseline_roles.get(role_id)
            if not imported:
                return True
            return now == (imported.get("voice"), str(imported.get("description") or ""))

        def unchanged_cue(cue_id):
            current = current_cues.get(int(cue_id)) or {}
            text = str(current.get("text") or "")
            emotion = str(current.get("emotion") or "")
            if text != text_snapshot.get(int(cue_id)) or emotion != emotion_snapshot.get(int(cue_id)):
                return False
            imported = baseline_cues.get(str(cue_id)) or baseline_cues.get(cue_id)
            if not imported:
                return True
            return text == str(imported.get("text") or "") and emotion == str(imported.get("emotion") or "")

        role_payload = [
            {"id": role["id"], "voice": role.get("voice"), "description": role.get("description")}
            for role in new_roles
            if role.get("id") not in locked and unchanged_role(role.get("id"))
        ]
        cue_payload = [
            {"id": cue["id"], "emotion": cue.get("emotion")}
            for cue in new_cues
            if (wanted is None or int(cue["id"]) in wanted) and unchanged_cue(cue["id"])
        ]
        project["roles"], project["cues"] = apply_direction(
            project.get("roles") or [],
            project.get("cues") or [],
            role_payload,
            cue_payload,
            locked,
        )
        direction = dict(((project.get("script_import") or {}).get("direction") or {}))
        direction["source"] = "ollama"
        direction["refined"] = True
        direction.pop("refine_error", None)
        project.setdefault("script_import", {})["direction"] = direction
        message = f"已按上下文分配 {len(project['roles'])} 个角色音色、{len(project['cues'])} 条情绪，可继续编辑"
        project["stages"]["script"] = "ready"
        project["pipeline"] = {"status": "pending", "stage": "script", "percent": 0, "message": message}
        project["last_message"] = message
        save_project(project)
        add_log(pid, message)
        return project
    except Exception as exc:
        project = normalize_project(load_project(pid))
        message = f"已用规则分配情绪与音色。上下文精修未完成：{exc}"
        project.setdefault("script_import", {}).setdefault("direction", {})
        project["script_import"]["direction"]["refined"] = False
        project["script_import"]["direction"]["refine_error"] = str(exc)
        project["pipeline"] = {"status": "pending", "stage": "script", "percent": 0, "message": message}
        project["last_message"] = message
        save_project(project)
        add_log(pid, message)
        return project


def probe(source):
    out = run([media_tool("ffprobe"), "-v","error","-show_entries",
               "format=duration,size:stream=index,codec_type,codec_name,width,height,sample_rate,channels",
               "-of","json", source]).stdout
    return json.loads(out)


def parse_whisper_json(path):
    d = read_json(path, {})
    cues=[]; idx=1
    for seg in d.get("transcription", []):
        start = seg.get("offsets",{}).get("from",0)/1000
        end = seg.get("offsets",{}).get("to",0)/1000
        text = (seg.get("text") or "").strip()
        if not text: continue
        cues.append({"id":idx,"start":round(start,3),"end":round(end,3),"source_text":text,
                     "text":"","role":"role1","emotion":"自然、符合画面情境","enabled":True})
        idx += 1
    return cues


def analyze(pid):
    root=project_dir(pid); p=load_project(pid); set_stage(pid,"analyze","running","正在检查媒体并本地转写")
    src=p["source"]; media=probe(src); p["media"]=media; save_project(p)
    audio=root/"source.wav"
    run([media_tool("ffmpeg"),"-y","-v","error","-i",src,"-vn","-ar","16000","-ac","1","-c:a","pcm_s16le",audio])
    configured=Path(p["settings"].get("whisper_model",DEFAULT_WHISPER)).expanduser()
    model=configured if configured.exists() else DEFAULT_WHISPER
    if not model.exists():
        raise RuntimeError(f"Whisper 模型不存在：{model}。请重新运行启动器自动下载。")
    if configured != model:
        p["settings"]["whisper_model"]=str(model); save_project(p)
    prefix=root/"transcript"
    run([tool("whisper-cli"),"-m",model,"-f",audio,"-l","auto","-ojf","-osrt","-of",prefix,"-np"])
    cues=parse_whisper_json(root/"transcript.json")
    if not cues:
        # some whisper.cpp versions use .json with merged structure; fall back to SRT
        cues=parse_srt(root/"transcript.srt")
    p=load_project(pid); p["cues"]=cues
    if not p.get("roles"):
        p["roles"]=[{"id":"role1","name":"角色1","voice":"male_adult","description":VOICE_TEMPLATES["male_adult"]}]
    p["stages"]["analyze"]="done"; p["stages"]["script"]="needs_review"; p["last_message"]=f"识别出 {len(cues)} 个对白段，请拆句、翻译并分配角色"
    save_project(p); add_log(pid,p["last_message"]); return p


def parse_srt(path):
    txt=Path(path).read_text(errors="replace") if Path(path).exists() else ""
    cues=[]
    for block in re.split(r"\n\s*\n",txt.strip()):
        lines=block.splitlines()
        if len(lines)<3 or "-->" not in lines[1]: continue
        a,b=[x.strip() for x in lines[1].split("-->")]
        def sec(x):
            h,m,s=re.split("[:,]",x); return int(h)*3600+int(m)*60+float(s)
        cues.append({"id":len(cues)+1,"start":sec(a),"end":sec(b),"source_text":" ".join(lines[2:]),
                     "text":"","role":"role1","emotion":"自然、符合画面情境","enabled":True})
    return cues


def mcfan_api_key():
    import os
    k = os.environ.get("HERMES_CUSTOM_API_MCFAN_TV_API_KEY", "").strip()
    if k: return k
    cfg = Path.home() / ".hermes" / "config.yaml"
    if cfg.exists():
        in_section = False
        for line in cfg.read_text().splitlines():
            s = line.strip()
            if "api.mcfan.tv" in line: in_section = True
            elif in_section and s.startswith("api_key:"):
                return s.split(":", 1)[1].strip().strip('"').strip("'")
            elif in_section and s and not line.startswith((" ", "\t")):
                in_section = False
    raise RuntimeError("找不到 mcfan.tv API Key（~/.hermes/config.yaml 的 custom_providers.mcfan_tv.api_key）")


def speech_budget(cue, language):
    seconds = max(0.2, float(cue.get("end", 0)) - float(cue.get("start", 0)))
    rates = {"zh": 4.2, "ja": 5.0, "ko": 4.5, "en": 2.7}
    minimum = 2 if language == "en" else 3
    return max(minimum, round(seconds * rates.get(language, 4.0)))


def translation_payload(cues, language):
    return [
        {
            "id": cue["id"],
            "start": cue["start"],
            "end": cue["end"],
            "slot_seconds": round(max(0.2, float(cue["end"]) - float(cue["start"])), 3),
            "max_words_or_characters": speech_budget(cue, language),
            "source": cue["source_text"],
        }
        for cue in cues
    ]


def extract_json_list(raw):
    """从模型输出里容错地提取 JSON 数组（复用 ollama_translate 的解析套路）。"""
    cleaned = re.sub(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|.)", "", raw)
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.S).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.M).strip()
    decoder = json.JSONDecoder()
    try:
        outer = json.loads(cleaned)
        if isinstance(outer, list): return outer
        if isinstance(outer, dict):
            if isinstance(outer.get("items"), list): return outer["items"]
            for value in outer.values():
                if isinstance(value, list): return value
    except (json.JSONDecodeError, TypeError):
        pass
    for pos, ch in enumerate(cleaned):
        if ch != "[": continue
        try:
            value, _ = decoder.raw_decode(cleaned[pos:])
            if isinstance(value, list) and all(isinstance(x, dict) and "id" in x for x in value):
                return value
        except json.JSONDecodeError:
            pass
    return None


def gemini_analyze(pid):
    """云端 Gemini 分析：翻译台词到目标语言 + 角色识别 + 情绪 + Gemini TTS 音色分配。"""
    p = load_project(pid)
    cues = [c for c in p.get("cues", []) if c.get("enabled", True)]
    if not cues: raise RuntimeError("没有台词")
    lang = p["settings"].get("target_lang", "zh")
    lang_info = TARGET_LANGS.get(lang, TARGET_LANGS["zh"])
    payload = translation_payload(cues, lang)
    voice_list = "、".join(f"{k}({v})" for k, v in GEMINI_VOICES.items())
    prompt = (
        "你是专业的多角色视频配音导演。把输入 JSON 里的对白翻译成"
        f"{lang_info['name']}（{lang_info['prompt']}），并根据上下文判断说话角色、情绪，"
        "再给每个角色从 Gemini TTS 音色库中分配最贴合人物特征的音色。\n"
        "要求：\n"
        "1. 严格保持每条 id，不合并、不新增事实；同一角色的 speaker 名称必须一致。\n"
        "2. 每项的 slot_seconds 是可用时长，max_words_or_characters 是硬预算：英语按单词数，中文/日语/韩语按字符数；"
        "text 不得超出预算。优先保留核心含义，删去称谓、重复、主语和可由画面表达的信息，使用演员能一口说完的短句。\n"
        "3. emotion 用中文写表演要求（如：害怕颤抖、愤怒咆哮、平静冷酷）。\n"
        "4. voice 只能从以下音色中选（按角色性别、年龄、性格匹配，不同角色尽量不同音色）：\n"
        f"   {voice_list}\n"
        "5. style 是该角色的整体声音风格描述（中文，用于指导 TTS 表演，如：低沉平稳的中年男声，冷静中带着算计）。\n"
        "输出：纯 JSON 数组，每项含 id、text、emotion、speaker、voice、style。不要 Markdown，不要解释。\n"
        "输入：\n" + json.dumps(payload, ensure_ascii=False)
    )
    body = {
        "model": MCFAN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    last_err = None
    r = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                MCFAN_BASE + "/chat/completions",
                data=json.dumps(body, ensure_ascii=False).encode(),
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + mcfan_api_key()})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode())
            r = data["choices"][0]["message"]["content"]
            break
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    if r is None:
        raise RuntimeError("云端 Gemini 分析失败（已重试3次）：" + str(last_err))
    translated = extract_json_list(r)
    if translated is None:
        (project_dir(pid) / "translate-raw.log").write_text(r)
        raise RuntimeError("云端模型未返回可解析的 JSON，原始输出已保存为 translate-raw.log")
    by = {int(x["id"]): x for x in translated if "id" in x}
    expected = {int(c["id"]) for c in cues}
    if set(by) != expected or any(not str(by[i].get("text", "")).strip() for i in expected):
        (project_dir(pid) / "translate-raw.log").write_text(r)
        raise RuntimeError(f"云端模型返回的台词不完整（需要{len(expected)}条，得到{len(by)}条），未覆盖现有台词。")
    speaker_roles = {}; roles = []
    for c in p["cues"]:
        if c["id"] in by:
            item = by[c["id"]]
            c["text"] = item.get("text", "").strip()
            c["emotion"] = item.get("emotion", "自然")
            speaker = str(item.get("speaker") or "角色1").strip()
            if speaker not in speaker_roles:
                voice = str(item.get("voice", "Puck")).strip()
                if voice not in GEMINI_VOICES: voice = "Puck"
                style = str(item.get("style") or f"{GEMINI_VOICES[voice]}的声音，情绪自然").strip()
                rid = f"role{len(speaker_roles) + 1}"; speaker_roles[speaker] = rid
                roles.append({"id": rid, "name": speaker, "voice": voice,
                              "description": style, "gemini_voice": voice, "style": style})
            c["role"] = speaker_roles[speaker]
    if roles: p["roles"] = roles
    p["stages"]["script"] = "needs_review"
    p["last_message"] = f"云端分析完成：{len(roles)}个角色，目标语言{lang_info['name']}"
    save_project(p); add_log(pid, p["last_message"]); return p


def ollama_translate(pid):
    p=load_project(pid); cues=[c for c in p.get("cues",[]) if c.get("enabled",True)]
    if not cues: raise RuntimeError("没有台词")
    if not shutil.which("ollama"): raise RuntimeError("未安装 Ollama")
    lang=TARGET_LANGS.get(p.get("settings",{}).get("target_lang","zh"),TARGET_LANGS["zh"])
    language=p.get("settings",{}).get("target_lang","zh")
    payload=translation_payload(cues,language)
    prompt=(f"你是专业的视频配音导演。把输入 JSON 里的对白翻译或改写成{lang['name']}（{lang['prompt']}），"
            "保持原意、人物语气和上下文，并自动判断说话角色、情绪与声音类型。"
            "严格保持每条 id，不合并、不新增事实；同一角色的 speaker 名称必须一致；"
            "每项 slot_seconds 是可用时长，max_words_or_characters 是硬预算：英语按单词数，中文、日语、韩语按字符数，"
            "text 不得超出预算。优先保留核心含义，删除重复、称谓、主语和可由画面表达的信息，输出演员能一口说完的短句。"
            "voice 只能选 male_adult、male_young、female_warm、"
            "female_brave、girl_soft、girl_lively。每项必须包含 id、text、emotion、speaker、voice。"
            "不要 Markdown，不要思考过程。/no_think\n输入：\n"+json.dumps(payload,ensure_ascii=False))
    schema={"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{
        "id":{"type":"integer"},"text":{"type":"string"},"emotion":{"type":"string"},
        "speaker":{"type":"string"},"voice":{"type":"string","enum":list(VOICE_TEMPLATES)}},
        "required":["id","text","emotion","speaker","voice"]}}},"required":["items"]}
    max_tokens=min(8192,max(1600,len(cues)*120))
    body={"model":p["settings"].get("ollama_model",OLLAMA_MODEL),"prompt":prompt,"stream":False,
          "think":False,"keep_alive":0,"format":schema,
          "options":{"temperature":0.2,"num_predict":max_tokens}}
    try:
        req=urllib.request.Request("http://127.0.0.1:11434/api/generate",
            data=json.dumps(body,ensure_ascii=False).encode(),headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=600) as response:
            api_result=json.loads(response.read().decode())
        r=(api_result.get("response") or "").strip()
    except TimeoutError:
        raise RuntimeError("Ollama 翻译超过10分钟，已停止。可以在网页中手动填写台词，或稍后重试。")
    except Exception as e:
        raise RuntimeError("Ollama翻译失败："+str(e))
    # Normalize terminal controls and Chinese smart quotes before JSON parsing.
    cleaned=re.sub(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|.)", "", r)
    cleaned=cleaned.replace("“",'"').replace("”",'"').replace("‘","'").replace("’","'")
    cleaned=re.sub(r"<think>.*?</think>","",cleaned,flags=re.S).strip()
    decoder=json.JSONDecoder(); translated=None
    try:
        outer=json.loads(cleaned)
        if isinstance(outer,list): translated=outer
        elif isinstance(outer,str):
            inner=json.loads(outer)
            if isinstance(inner,list): translated=inner
        elif isinstance(outer,dict):
            if isinstance(outer.get("items"),list): translated=outer["items"]
            for value in outer.values():
                if isinstance(value,list): translated=value; break
    except (json.JSONDecodeError,TypeError):
        pass
    for pos,ch in enumerate(cleaned):
        if translated is not None: break
        if ch != "[": continue
        try:
            value,_=decoder.raw_decode(cleaned[pos:])
            if isinstance(value,list) and all(isinstance(x,dict) and "id" in x for x in value):
                translated=value; break
        except json.JSONDecodeError:
            pass
    if translated is None:
        (project_dir(pid)/"translate-raw.log").write_text(r)
        raise RuntimeError("本地模型未返回可解析的JSON，原始输出已保存为 translate-raw.log")
    by={int(x["id"]):x for x in translated}
    expected={int(c["id"]) for c in cues}
    if set(by) != expected or any(not str(by[i].get("text","")).strip() for i in expected):
        (project_dir(pid)/"translate-raw.log").write_text(r)
        raise RuntimeError(f"本地模型返回的台词不完整（需要{len(expected)}条，得到{len(by)}条），未覆盖现有台词。")
    speaker_roles={}; roles=[]
    for c in p["cues"]:
        if c["id"] in by:
            item=by[c["id"]]
            c["text"]=item.get("text","").strip(); c["emotion"]=item.get("emotion","自然")
            speaker=str(item.get("speaker") or "角色1").strip()
            if speaker not in speaker_roles:
                voice=item.get("voice") if item.get("voice") in VOICE_TEMPLATES else "male_adult"
                rid=f"role{len(speaker_roles)+1}"; speaker_roles[speaker]=rid
                roles.append({"id":rid,"name":speaker,"voice":voice,"description":VOICE_TEMPLATES[voice]})
            c["role"]=speaker_roles[speaker]
    if roles: p["roles"]=roles
    p["stages"]["script"]="needs_review"; p["last_message"]=f"本地翻译和角色分析已完成：{len(roles)}个角色"
    save_project(p); add_log(pid,p["last_message"]); return p


def translate_project(pid):
    engine=load_project(pid).get("settings",{}).get("tts_engine","local")
    return gemini_analyze(pid) if engine=="gemini" else ollama_translate(pid)


def auto_pipeline(pid, resume=False):
    """Run the entire local dubbing workflow after one video upload."""
    p0=load_project(pid)
    lang=TARGET_LANGS.get(p0.get("settings",{}).get("target_lang","zh"),TARGET_LANGS["zh"])["name"]
    engine=p0.get("settings",{}).get("tts_engine","local")
    translate_fn=translate_project
    translate_message=(f"云端 Gemini 正在翻译为{lang}并分配角色音色" if engine=="gemini"
                       else f"本地 Ollama 正在翻译为{lang}并分析角色与情绪")
    tts_message=(f"正在用 Gemini TTS 生成{lang}多角色配音" if engine=="gemini"
                 else f"正在用本地 Qwen3-TTS BF16 生成{lang}多角色配音")
    steps=[
        ("analyze",8,"正在分析视频并识别对白",analyze),
        ("translate",28,translate_message,translate_fn),
        ("script",42,"正在确认自动台词清单",None),
        ("tts",48,tts_message,generate_tts),
        ("render",82,"正在同步字幕、混音并合成视频",render),
        ("qa",95,"正在检查成片",qa),
    ]
    try:
        for stage,percent,message,fn in steps:
            p=load_project(pid)
            state=p.get("stages",{})
            complete=(stage=="analyze" and state.get("analyze")=="done") or (stage in ("translate","script") and state.get("script") in ("ready","done")) or (stage in ("tts","render","qa") and state.get(stage)=="done")
            if resume and complete: continue
            set_pipeline(pid,"running",stage,percent,message)
            if stage=="script":
                p=load_project(pid)
                save_manifest(pid,p)
            else:
                fn(pid)
        p=load_project(pid)
        if not p.get("qa",{}).get("passed"):
            raise RuntimeError("成片自动验收未通过")
        warning_count=int(p.get("qa",{}).get("fit_warnings",0) or 0)
        warning_text=f"；{warning_count}句因时间槽较短已加速，建议复听" if warning_count else ""
        set_pipeline(pid,"done","complete",100,f"自动配音完成，可以预览或下载成片{warning_text}")
        return load_project(pid)
    except Exception as e:
        p=load_project(pid)
        current=p.get("pipeline",{}).get("stage","unknown")
        if current in {"analyze", "tts", "render", "qa"}:
            p.setdefault("stages", {})[current] = "failed"
            save_project(p)
        set_pipeline(pid,"failed",current,p.get("pipeline",{}).get("percent",0),f"自动处理失败：{e}")
        raise


def normalize_project(p):
    p.setdefault("project_type", "video" if p.get("source") else "voice")
    settings = p.setdefault("settings", {})
    had_timeline_mode = "timeline_mode" in settings
    defaults = default_settings(settings.get("target_lang", "zh"), settings.get("tts_engine", "local"))
    for key, value in defaults.items(): settings.setdefault(key, value)
    if p["project_type"] == "voice" and not had_timeline_mode: settings["timeline_mode"] = "natural"
    role_ids={r["id"] for r in p.get("roles",[])}
    natural = settings.get("timeline_mode") == "natural"
    cursor = 0.0
    for i,c in enumerate(p.get("cues",[]),1):
        c["id"]=i
        c["enabled"]=bool(c.get("enabled",True)); c["text"]=str(c.get("text","")).strip()
        c["source_text"]=str(c.get("source_text","")).strip(); c["emotion"]=str(c.get("emotion","自然")).strip()
        c["scene_before"]=str(c.get("scene_before","")).strip()
        c["scene_after"]=str(c.get("scene_after","")).strip()
        c["speed"]=max(.5,min(2.0,float(c.get("speed",1.0) or 1.0)))
        c["gap_after"]=max(0.0,min(10.0,float(c.get("gap_after",settings.get("default_gap",.25)) or 0)))
        if c.get("role") not in role_ids and role_ids: c["role"]=next(iter(role_ids))
        if natural:
            c["start"]=float(c.get("start",cursor) or cursor)
            c["end"]=max(c["start"]+.2,float(c.get("end",c["start"]+3) or c["start"]+3))
            cursor=c["end"]+c["gap_after"]
        else:
            c["start"]=float(c["start"]); c["end"]=float(c["end"])
            if c["end"]<=c["start"]: raise ValueError(f"台词{i}结束时间必须大于开始时间")
    for r in p.get("roles",[]):
        r["mode"] = "clone" if r.get("mode") == "clone" else "design"
        if not r.get("description"): r["description"]=VOICE_TEMPLATES.get(r.get("voice"),VOICE_TEMPLATES["male_adult"])
        r.setdefault("style", r["description"])
        r.setdefault("ref_audio", "")
        r.setdefault("ref_text", "")
        r["clone_consent"] = bool(r.get("clone_consent", False))
    return p


def save_manifest(pid, payload):
    p=load_project(pid)
    before=json.dumps({"roles":p.get("roles",[]),"cues":p.get("cues",[]),"settings":p.get("settings",{})},ensure_ascii=False,sort_keys=True)
    for key in ("roles","cues","settings","name"):
        if key in payload: p[key]=payload[key]
    p=normalize_project(p)
    after=json.dumps({"roles":p.get("roles",[]),"cues":p.get("cues",[]),"settings":p.get("settings",{})},ensure_ascii=False,sort_keys=True)
    p["stages"]["script"]="ready"; p["last_message"]="台词清单已保存"
    if before != after:
        p["stages"]["tts"]="pending"; p["stages"]["qa"]="pending"
        if p.get("project_type")=="voice":
            p["stages"]["render"]="pending"; p.pop("output_audio",None)
            p["pipeline"]={"status":"pending","stage":"script","percent":0,"message":"修改已保存，需要重新生成配音音轨"}
        else:
            p["stages"]["render"]="pending"; p.pop("output",None); p.pop("output_audio",None)
            p["pipeline"]={"status":"pending","stage":"script","percent":42,"message":"修改已保存，需要重新生成视频成片"}
    save_project(p); write_json(project_dir(pid)/"manifest.json",[c for c in p["cues"] if c.get("enabled",True)])
    add_log(pid,p["last_message"]); return p


def enabled_cues(p):
    return [c for c in p.get("cues", []) if c.get("enabled", True)]


def project_duration(p):
    media_duration = p.get("media", {}).get("format", {}).get("duration")
    if media_duration:
        return float(media_duration)
    summary = read_json(project_dir(p["id"]) / "summary.json", {})
    if summary.get("duration"):
        return float(summary["duration"])
    return max((float(c.get("end", 0)) for c in enabled_cues(p)), default=0.0)


def validate_clone_roles(p, role_ids=None):
    selected = role_ids or {c.get("role") for c in enabled_cues(p)}
    for role in p.get("roles", []):
        if role.get("id") not in selected or role.get("mode") != "clone": continue
        if not role.get("clone_consent"):
            raise RuntimeError(f"角色“{role.get('name','未命名')}”尚未确认已获得声音授权")
        ref_value = str(role.get("ref_audio") or "").strip()
        ref = Path(ref_value).expanduser() if ref_value else None
        if not ref or not ref.is_file():
            raise RuntimeError(f"角色“{role.get('name','未命名')}”缺少克隆参考音频")


def generate_tts(pid, reuse=True):
    p=normalize_project(load_project(pid)); root=project_dir(pid)
    cues=enabled_cues(p)
    if not cues or any(not c["text"] for c in cues): raise RuntimeError("存在空白台词")
    if not p.get("roles"): raise RuntimeError("至少需要一个角色")
    engine=p["settings"].get("tts_engine","local")
    natural_voice = p.get("project_type") == "voice" and p["settings"].get("timeline_mode") == "natural"
    duration=0.0 if natural_voice else project_duration(p)
    if engine=="gemini":
        if p.get("project_type") == "voice":
            raise RuntimeError("纯配音项目当前使用本地 Qwen3-TTS，以便支持造声和克隆混合")
        write_json(root/"manifest.json",cues)
        voices={r["id"]:{"voice":r.get("gemini_voice") or r.get("voice") or "Puck",
                         "style":r.get("style") or r.get("description") or "自然、符合情境地朗读"}
                for r in p["roles"]}
        write_json(root/"voices.json",voices)
        if not GEMINI_KEY_FILE.exists(): raise RuntimeError(f"缺少 Gemini API Key 文件：{GEMINI_KEY_FILE}")
        set_stage(pid,"tts","running","正在用 Gemini 云端生成角色配音")
        cmd=[sys.executable,str(GEMINI_TTS_WORKER),"--root",str(root),"--duration",str(duration)]
        if reuse: cmd.append("--reuse")
        proc=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (root/"tts.log").write_text(proc.stdout)
        if proc.returncode: raise RuntimeError("Gemini TTS失败，详见 tts.log\n"+proc.stdout[-1500:])
    else:
        validate_clone_roles(p)
        write_json(root/"manifest.json",cues)
        write_json(root/"voices.json",{r["id"]:r for r in p["roles"]})
        if not DEFAULT_TTS_PYTHON.exists():
            raise RuntimeError(f"Qwen3-TTS Python 环境不存在：{DEFAULT_TTS_PYTHON}")
        used_roles={c["role"] for c in cues}
        modes=set()
        for role in p["roles"]:
            if role["id"] not in used_roles: continue
            modes.add("克隆" if role.get("mode")=="clone" else "逐句情绪造声")
        set_stage(pid,"tts","running",f"正在用本地 Qwen3-TTS BF16 生成{' + '.join(sorted(modes))}配音")
        env=os.environ.copy(); env["HF_HUB_OFFLINE"]="1"; env["FFMPEG"]=ffmpeg_full(); env["FFPROBE"]=media_tool("ffprobe")
        cmd=[
            str(DEFAULT_TTS_PYTHON),str(TTS_WORKER),"--root",str(root),"--duration",str(duration),
            "--timeline-mode",p["settings"].get("timeline_mode","fixed"),
            "--gap",str(p["settings"].get("default_gap",.25)),
            "--design-model",p["settings"].get("voice_design_model",DEFAULT_TTS_MODEL),
            "--clone-model",p["settings"].get("voice_clone_model",DEFAULT_CLONE_MODEL),
            "--language",p["settings"].get("target_lang","zh"),
        ]
        if reuse: cmd.append("--reuse")
        proc=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
        (root/"tts.log").write_text(proc.stdout)
        if proc.returncode: raise RuntimeError("Qwen3-TTS 失败，详见 tts.log\n"+proc.stdout[-2000:])
    p=load_project(pid); p["stages"]["tts"]="done"; p["stages"]["render"]="done" if p.get("project_type")=="voice" else p["stages"].get("render","pending")
    p["output_audio"]=str(root/"voice_timeline.wav")
    summary=read_json(root/"summary.json",{}) or {}
    warning_count=int(summary.get("fit_warnings",0) or 0)
    warning_text=f"；{warning_count}句因时间槽较短已加速，请复听" if warning_count else ""
    p["last_message"]=f"配音音轨已生成（{'Gemini云端' if engine=='gemini' else '本地 Qwen3-TTS'}）{warning_text}"
    p["pipeline"]={"status":"done" if p.get("project_type")=="voice" else "running","stage":"tts","percent":100 if p.get("project_type")=="voice" else 76,"message":p["last_message"]}
    save_project(p); add_log(pid,p["last_message"]); return p


def preview_voice(pid, role_id, text, emotion="自然"):
    p=normalize_project(load_project(pid)); root=project_dir(pid)
    role=next((r for r in p.get("roles",[]) if r.get("id")==role_id),None)
    if not role: raise RuntimeError("角色不存在")
    if not str(text or "").strip(): raise RuntimeError("请输入试听台词")
    validate_clone_roles(p,{role_id})
    preview_dir=root/"previews"; preview_dir.mkdir(parents=True,exist_ok=True)
    preview_role=dict(role)
    ref_value=str(preview_role.get("ref_audio") or "").strip(); ref_path=Path(ref_value).expanduser() if ref_value else None
    ref_signature={"size":ref_path.stat().st_size,"mtime_ns":ref_path.stat().st_mtime_ns} if ref_path and ref_path.is_file() else None
    signature=json.dumps({"role":preview_role,"reference":ref_signature,"text":text,"emotion":emotion,"lang":p["settings"].get("target_lang","zh")},ensure_ascii=False,sort_keys=True)
    token=hashlib.sha256(signature.encode()).hexdigest()[:16]
    output=preview_dir/f"{role_id}-{token}.wav"
    if output.exists(): return output
    request_path=preview_dir/f"{role_id}-{token}.json"
    write_json(request_path,{"voice":preview_role,"text":text,"emotion":emotion,"language":p["settings"].get("target_lang","zh"),
                             "design_model":p["settings"].get("voice_design_model",DEFAULT_TTS_MODEL),
                             "clone_model":p["settings"].get("voice_clone_model",DEFAULT_CLONE_MODEL)})
    env=os.environ.copy(); env["HF_HUB_OFFLINE"]="1"; env["FFMPEG"]=ffmpeg_full(); env["FFPROBE"]=media_tool("ffprobe")
    cmd=[str(DEFAULT_TTS_PYTHON),str(TTS_WORKER),"--preview-request",str(request_path),"--preview-output",str(output),
         "--design-model",p["settings"].get("voice_design_model",DEFAULT_TTS_MODEL),
         "--clone-model",p["settings"].get("voice_clone_model",DEFAULT_CLONE_MODEL)]
    proc=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
    (preview_dir/f"{role_id}-{token}.log").write_text(proc.stdout)
    if proc.returncode: raise RuntimeError("试听生成失败\n"+proc.stdout[-1600:])
    return output


def ass_time(v):
    h=int(v//3600); m=int(v%3600//60); s=v%60; return f"{h}:{m:02d}:{s:05.2f}"
def srt_time(v):
    h=int(v//3600); m=int(v%3600//60); s=v%60; return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".",",")

def subtitle_layout(p):
    video=next((x for x in p.get("media",{}).get("streams",[]) if x.get("codec_type")=="video"),{})
    width=int(video.get("width") or 720); height=int(video.get("height") or 1280)
    size=max(14,min(int(p["settings"].get("subtitle_font_size",32)),max(14,height//10)))
    h=min(height,max(size+18,min(int(p["settings"].get("cover_h",72)),height)))
    default_y=max(0,height-h-max(18,height//30))
    requested_y=int(p["settings"].get("cover_y",default_y))
    y=requested_y if 0<=requested_y<=height-h else default_y
    return width,height,size,y,h


def build_subtitles(p, root):
    cues=[c for c in p["cues"] if c.get("enabled",True)]
    srt="\n\n".join(f"{i}\n{srt_time(c['start'])} --> {srt_time(c['end'])}\n{c['text']}" for i,c in enumerate(cues,1))+"\n"
    (root/"chinese.srt").write_text(srt)
    width,height,size,y,h=subtitle_layout(p)
    margin=max(0,height-y-h+int((h-size)/2)-2)
    head=f"""[Script Info]\nScriptType: v4.00+\nPlayResX: {width}\nPlayResY: {height}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Chinese,PingFang SC,{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,28,28,{margin},1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"""
    body="".join(f"Dialogue: 0,{ass_time(c['start'])},{ass_time(c['end'])},Chinese,,0,0,0,,{c['text']}\n" for c in cues)
    (root/"chinese.ass").write_text(head+body)


def render(pid):
    p=normalize_project(load_project(pid)); root=project_dir(pid)
    if p.get("project_type") == "voice":
        if not (root/"voice_timeline.wav").exists(): raise RuntimeError("请先生成配音")
        p["output_audio"]=str(root/"voice_timeline.wav"); p["stages"]["render"]="done"; p["last_message"]="配音音轨已就绪"
        save_project(p); add_log(pid,p["last_message"]); return p
    build_subtitles(p,root)
    if not (root/"voice_timeline.wav").exists(): raise RuntimeError("请先生成配音")
    set_stage(pid,"render","running","正在合成最终视频")
    s=p["settings"]; width,_,_,y,h=subtitle_layout(p)
    cues=[c for c in p["cues"] if c.get("enabled",True)]
    enables="+".join(f"between(t,{c['start']:.3f},{c['end']:.3f})" for c in cues) or "0"
    out=root/(re.sub(r"[^\w\u4e00-\u9fff.-]+","_",p["name"])+"-"+TARGET_LANGS.get(p["settings"].get("target_lang","zh"),TARGET_LANGS["zh"])["suffix"]+".mp4")
    ff=ffmpeg_full(); src=p["source"]
    subtitle_path=str(root/"chinese.ass").replace("\\","\\\\").replace(":","\\:").replace("'","\\'")
    vf=f"drawbox=x=0:y={y}:w={width}:h={h}:color=black:t=fill:enable='{enables}',ass=filename='{subtitle_path}'"
    has_source_audio=any(x.get("codec_type")=="audio" for x in p.get("media",{}).get("streams",[]))
    if s.get("keep_source_audio",True) and has_source_audio:
        fc=f"[0:v]{vf}[v];[1:a]volume={float(s.get('voice_gain',1.55))},asplit=2[voice][sc];[0:a]volume={float(s.get('source_gain',1.0))}[src];[src][sc]sidechaincompress=threshold={float(s.get('duck_threshold',.008))}:ratio=16:attack=8:release=180:makeup=1[duck];[duck][voice]amix=inputs=2:duration=first:dropout_transition=0,alimiter=limit=0.95[a]"
    else:
        fc=f"[0:v]{vf}[v];[1:a]volume={float(s.get('voice_gain',1.55))},alimiter=limit=.95[a]"
    run([ff,"-y","-v","error","-i",src,"-i",root/"voice_timeline.wav","-filter_complex",fc,"-map","[v]","-map","[a]","-c:v","libx264","-preset","medium","-crf","18","-pix_fmt","yuv420p","-c:a","aac","-b:a","192k","-movflags","+faststart",out],capture=True)
    p=load_project(pid); p["output"]=str(out); p["stages"]["render"]="done"; p["last_message"]="最终视频已生成"; save_project(p); add_log(pid,p["last_message"]); return p


def qa(pid):
    p=load_project(pid); root=project_dir(pid)
    if p.get("project_type") == "voice":
        out=Path(p.get("output_audio") or root/"voice_timeline.wav")
        if not out.exists(): raise RuntimeError("没有配音音轨")
        set_stage(pid,"qa","running","正在检查配音音轨")
        ff=ffmpeg_full(); dec=run([ff,"-v","error","-i",out,"-f","null","-"],check=False)
        status=read_json(root/"status.json",[]); cues=enabled_cues(p)
        report={"decode_exit":dec.returncode,"decode_errors":(dec.stderr or "").strip(),"cues":len(cues),"generated":len(status),
                "roles":len(p.get("roles",[])),"passed":dec.returncode==0 and len(cues)==len(status)}
        write_json(root/"qa.json",report); p=load_project(pid); p["qa"]=report; p["stages"]["qa"]="done" if report["passed"] else "failed"
        p["last_message"]="音轨验收通过" if report["passed"] else "音轨验收发现问题"; save_project(p); add_log(pid,p["last_message"]); return p
    out=Path(p.get("output",""))
    if not out.exists(): raise RuntimeError("没有成片")
    set_stage(pid,"qa","running","正在完整解码并检查音量")
    ff=ffmpeg_full()
    dec=run([ff,"-v","error","-i",out,"-f","null","-"],check=False)
    vol=run([ff,"-v","info","-i",out,"-af","volumedetect","-f","null","-"],check=False)
    mean=re.search(r"mean_volume:\s*([^\n]+)",vol.stderr or ""); peak=re.search(r"max_volume:\s*([^\n]+)",vol.stderr or "")
    status=read_json(root/"status.json",[]); cues=[c for c in p["cues"] if c.get("enabled",True)]
    report={"decode_exit":dec.returncode,"decode_errors":(dec.stderr or "").strip(),"cues":len(cues),"generated":len(status),
            "roles":len(p.get("roles",[])),"max_tempo":max([x.get("tempo",1) for x in status] or [1]),
            "fit_warnings":sum(bool(x.get("fit_warning")) for x in status),
            "mean_volume":mean.group(1).strip() if mean else None,"max_volume":peak.group(1).strip() if peak else None,
            "passed":dec.returncode==0 and len(cues)==len(status)}
    write_json(root/"qa.json",report)
    p=load_project(pid); p["qa"]=report; p["stages"]["qa"]="done" if report["passed"] else "failed"
    warning_text=f"；{report['fit_warnings']}句时间槽加速，建议复听" if report.get("fit_warnings") else ""
    p["last_message"]=("验收通过"+warning_text) if report["passed"] else "验收发现问题"
    save_project(p); add_log(pid,p["last_message"]); return p


def main():
    ap=argparse.ArgumentParser(description="本地中文多角色视频配音工具")
    sub=ap.add_subparsers(dest="cmd",required=True)
    c=sub.add_parser("create"); c.add_argument("source"); c.add_argument("--name"); c.add_argument("--copy",action="store_true")
    for x in ("analyze","translate","tts","render","qa"): q=sub.add_parser(x); q.add_argument("project")
    s=sub.add_parser("serve"); s.add_argument("--host",default="127.0.0.1"); s.add_argument("--port",type=int,default=8765); s.add_argument("--no-browser",action="store_true")
    args=ap.parse_args(); PROJECTS_DIR.mkdir(parents=True,exist_ok=True)
    if args.cmd=="create": print(json.dumps(create_project(args.source,args.name,args.copy),ensure_ascii=False,indent=2))
    elif args.cmd=="analyze": print(json.dumps(analyze(args.project),ensure_ascii=False,indent=2))
    elif args.cmd=="translate": print(json.dumps(translate_project(args.project),ensure_ascii=False,indent=2))
    elif args.cmd=="tts": print(json.dumps(generate_tts(args.project),ensure_ascii=False,indent=2))
    elif args.cmd=="render": print(json.dumps(render(args.project),ensure_ascii=False,indent=2))
    elif args.cmd=="qa": print(json.dumps(qa(args.project),ensure_ascii=False,indent=2))
    elif args.cmd=="serve":
        from webapp import create_app
        app=create_app()
        if not args.no_browser:
            import threading, webbrowser
            threading.Timer(1.0,lambda:webbrowser.open(f"http://{args.host}:{args.port}")).start()
        app.run(host=args.host,port=args.port,debug=False,threaded=True)

if __name__=="__main__": main()
