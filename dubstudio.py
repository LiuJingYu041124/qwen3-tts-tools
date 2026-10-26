#!/usr/bin/env python3
import argparse, contextvars, hashlib, json, os, re, shutil, subprocess, sys, time, uuid, urllib.request
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from document_import import DocumentError, extract_document, materialize_script, parse_script
from script_direction import (
    apply_direction, direct_script, locked_roles, normalize_direction_brief,
    normalize_role_personality, ollama_direct_script,
)
from voice_consistency import CLONE_GENERATION_POLICY
APP_DIR = Path(__file__).resolve().parent
TRUSTED_TAKE_POLICY = "trusted-take-v1"
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
TTS_JOB_OBSERVER = contextvars.ContextVar("tts_job_observer", default=None)
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


def delete_project(pid):
    root = project_dir(pid).resolve()
    projects_root = PROJECTS_DIR.expanduser().resolve()
    if root.parent != projects_root or not (root / "project.json").is_file():
        raise RuntimeError("项目目录无效，未执行删除")
    shutil.rmtree(root)


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


def create_voice_project(name="未命名配音", target_lang="zh", direction_brief=""):
    if target_lang not in TARGET_LANGS: target_lang = "zh"
    pid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    root = PROJECTS_DIR / pid
    root.mkdir(parents=True)
    direction_brief = normalize_direction_brief(direction_brief)
    role = {
        "id": "role1",
        "name": "旁白",
        "voice": "male_adult",
        "mode": "design",
        "description": VOICE_TEMPLATES["male_adult"],
        "style": VOICE_TEMPLATES["male_adult"],
        "personality": "",
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
        "direction_brief": direction_brief,
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


def build_imported_role_summary(roles, cues, imported_role_ids=None):
    imported = None if imported_role_ids is None else {str(item) for item in imported_role_ids}
    cue_counts = {}
    for cue in cues or []:
        role_id = str(cue.get("role") or "")
        cue_counts[role_id] = cue_counts.get(role_id, 0) + 1
    return [
        {
            "id": str(role.get("id") or ""),
            "name": str(role.get("name") or "未命名角色"),
            "cue_count": cue_counts.get(str(role.get("id") or ""), 0),
            "reused": bool(role.get("ref_audio")),
        }
        for role in roles or []
        if role.get("id") and (imported is None or str(role.get("id")) in imported)
    ]


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


def normalize_import_voice_mode(value):
    mode = str(value or "design").strip()
    if mode not in {"design", "clone"}:
        raise DocumentError("不支持的导入方式，请选择自动设计音色或导入后配置声音克隆。")
    return mode


def prepare_imported_script(items, default_gap, voice_mode):
    roles, cues = materialize_script(items, default_gap=default_gap, voice_templates=VOICE_TEMPLATES)
    if voice_mode == "clone":
        for role in roles:
            role.update(mode="clone", voice="custom", description="音色以授权参考音频为准。", style="")
    roles, cues, direction = direct_script(roles, cues, voice_templates=VOICE_TEMPLATES)
    if voice_mode == "clone":
        direction["refine_skipped"] = True
    return roles, cues, direction


def apply_imported_script(project, source, extracted, roles, cues, original_name=None, direction=None, voice_mode="design"):
    root = project_dir(project["id"])
    src = Path(source)
    dest = root / f"script{src.suffix.lower() or '.txt'}"
    shutil.copy2(src, dest)
    (root / "script.txt").write_text(extracted["text"], encoding="utf-8")
    project["source"] = str(dest)
    project["original_source"] = original_name or src.name
    warning_text = f"；{extracted['warnings'][0]}" if extracted.get("warnings") else ""
    direction_data: dict[str, object] = dict(direction or {"source": "rules", "refined": False})
    direction_data["brief"] = project.get("direction_brief") or ""
    direction_data["baseline"] = direction_baseline(
        roles,
        cues,
        direction_data.get("cue_ids"),
        direction_data.get("locked_role_ids"),
    )
    imported_cue_ids = direction_data.get("cue_ids")
    if imported_cue_ids is None:
        imported_cues = list(cues)
    else:
        wanted_cues = {int(item) for item in imported_cue_ids}
        imported_cues = [cue for cue in cues if int(cue.get("id") or 0) in wanted_cues]
    imported_role_ids = {str(cue.get("role") or "") for cue in imported_cues if cue.get("role")}
    imported_roles = build_imported_role_summary(roles, imported_cues, imported_role_ids)
    project["script_import"] = {
        "voice_mode": voice_mode,
        "format": extracted["format"],
        "pages": extracted["pages"],
        "warnings": extracted["warnings"],
        "cue_count": len(imported_cues),
        "role_count": len(imported_roles),
        "roles": imported_roles,
        "direction": direction_data,
    }
    setup_message = (
        "；已跳过自动音色设计与上下文精修，请上传授权参考音频并确认逐字稿"
        if voice_mode == "clone" else "，并分配了情绪与音色"
    )
    project["pipeline"] = {
        "status": "pending",
        "stage": "script",
        "percent": 0,
        "message": f"已从文档导入 {len(imported_cues)} 条台词、{len(imported_roles)} 个角色{setup_message}{warning_text}",
    }
    project["last_message"] = project["pipeline"]["message"]
    project["stages"]["script"] = "ready"
    project["stages"]["tts"] = "pending"
    project["stages"]["qa"] = "pending"
    if project.get("project_type") == "voice":
        project["stages"]["render"] = "pending"
        project.pop("output_audio", None)
    save_project(project)
    add_log(project["id"], f"从文档导入脚本：{project['original_source']}，{len(imported_cues)}条台词，{len(imported_roles)}个角色")
    return project


def create_voice_project_from_document(source, name=None, target_lang="zh", direction_brief="", voice_mode="design"):
    voice_mode = normalize_import_voice_mode(voice_mode)
    src = Path(source).expanduser().resolve()
    extracted = extract_document(src)
    items = parse_script(extracted["text"])
    settings = default_settings(target_lang, "local")
    direction_brief = normalize_direction_brief(direction_brief)
    roles, cues, direction = prepare_imported_script(items, settings["default_gap"], voice_mode)
    project_name = (name or src.stem or "未命名配音").strip()
    project = create_voice_project(project_name, target_lang, direction_brief)
    project["name"] = project_name
    project["roles"] = roles
    project["cues"] = cues
    return apply_imported_script(project, src, extracted, roles, cues, src.name, direction, voice_mode)


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


def import_document_into_project(pid, source, original_name=None, replace=None, direction_brief=None, voice_mode="design"):
    voice_mode = normalize_import_voice_mode(voice_mode)
    project = normalize_project(load_project(pid))
    if project.get("project_type") != "voice":
        raise DocumentError("视频项目请继续用导入视频。文档脚本只能导入到脚本配音项目。")
    if direction_brief is not None:
        project["direction_brief"] = normalize_direction_brief(direction_brief)
    src = Path(source).expanduser().resolve()
    extracted = extract_document(src)
    items = parse_script(extracted["text"])
    roles, cues, direction = prepare_imported_script(
        items, project["settings"].get("default_gap", 0.25), voice_mode,
    )
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
    return apply_imported_script(
        project, src, extracted, project["roles"], project["cues"], original_name or src.name, direction, voice_mode,
    )


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
            project.get("direction_brief") or "",
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
        direction["brief"] = project.get("direction_brief") or ""
        direction.pop("refine_error", None)
        project.setdefault("script_import", {})["direction"] = direction
        imported_cue_ids = direction.get("cue_ids")
        wanted_imported_cues = None if imported_cue_ids is None else {int(item) for item in imported_cue_ids}
        imported_cues = list(project.get("cues") or []) if wanted_imported_cues is None else [
            cue for cue in project.get("cues") or [] if int(cue.get("id") or 0) in wanted_imported_cues
        ]
        imported_role_ids = {str(cue.get("role") or "") for cue in imported_cues if cue.get("role")}
        project.setdefault("script_import", {})["roles"] = build_imported_role_summary(
            project.get("roles") or [], imported_cues, imported_role_ids
        )
        project["script_import"]["cue_count"] = len(imported_cues)
        project["script_import"]["role_count"] = len(project["script_import"]["roles"])
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


def transcribe_reference_audio(audio, language="auto", model=None):
    configured = Path(model or DEFAULT_WHISPER).expanduser()
    if not configured.is_file():
        raise RuntimeError(f"Whisper 模型不存在：{configured}。请重新运行启动器自动下载。")
    process = run(
        [tool("whisper-cli"), "-m", configured, "-f", audio, "-l", language or "auto", "-nt", "-np"],
        check=False,
    )
    if process.returncode:
        raise RuntimeError("参考音频自动转写失败")
    transcript = re.sub(r"\s+", "", process.stdout or "").strip()
    if not transcript:
        raise RuntimeError("没有从参考音频中识别出清晰人声")
    return transcript


def normalize_transcript_text(value):
    return re.sub(r"[^\w\u4e00-\u9fff]", "", str(value or "")).lower()


def transcript_similarity(left, right):
    a = normalize_transcript_text(left)
    b = normalize_transcript_text(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def apply_pronunciation_lexicon(text, rules):
    value = str(text or "")
    normalized = []
    for item in rules or []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        spoken = str(item.get("spoken") or "").strip()
        if source and spoken:
            normalized.append((source, spoken))
    for source, spoken in sorted(normalized, key=lambda pair: len(pair[0]), reverse=True):
        value = value.replace(source, spoken)
    return value


def spoken_text_quality(expected, transcript):
    wanted = normalize_transcript_text(expected)
    actual = normalize_transcript_text(transcript)
    traditional_map = str.maketrans("這嗎裏後為與個們來說時會還從對發應麼於過聲長開關問間", "这吗里后为与个们来说时会还从对发应么于过声长开关问间")
    wanted = wanted.translate(traditional_map)
    actual = actual.translate(traditional_map)
    ratio = SequenceMatcher(None, wanted, actual).ratio() if wanted and actual else 0.0
    final_ok = bool(wanted and actual and wanted[-1] == actual[-1])
    passed = ratio >= 0.82 and final_ok
    return {"passed": passed, "similarity": round(ratio, 4), "final_ok": final_ok, "transcript": str(transcript or "").strip()}


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cue_take_signature(project, cue, role=None, emotion=None):
    role = role or next(
        (item for item in project.get("roles", []) if item.get("id") == cue.get("role")),
        {},
    )
    settings = project.get("settings", {})
    mode = "clone" if role.get("mode") == "clone" else "design"
    payload = {
        "text": str(cue.get("text") or "").strip(),
        "emotion": str(emotion if emotion is not None else cue.get("emotion") or "自然").strip(),
        "role": cue.get("role"),
        "language": settings.get("target_lang", "zh"),
        "mode": mode,
        "model": settings.get("voice_clone_model", DEFAULT_CLONE_MODEL) if mode == "clone" else settings.get("voice_design_model", DEFAULT_TTS_MODEL),
        "voice": role.get("voice") or "",
        "description": role.get("description") or role.get("style") or "",
        "personality": role.get("personality") or "",
        "project_direction": project.get("direction_brief") or "",
        "clone_consent": bool(role.get("clone_consent")) if mode == "clone" else False,
        "ref_text": (role.get("ref_text") or "") if mode == "clone" else "",
        "ref_text_confirmed": bool(role.get("ref_text_confirmed")) if mode == "clone" else False,
        "ref_text_confirmed_revision": str(role.get("ref_text_confirmed_revision") or "") if mode == "clone" else "",
        "reference_revision": (role.get("reference_revision") or role.get("reference_sha256") or "") if mode == "clone" else "",
        "reference_sha256": str(role.get("reference_sha256") or "") if mode == "clone" else "",
        "clone_generation_policy": CLONE_GENERATION_POLICY if mode == "clone" else "",
        "trusted_take_policy": TRUSTED_TAKE_POLICY,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def adopted_take_path(pid, take):
    filename = Path(str((take or {}).get("file") or "")).name
    if not filename or filename != str((take or {}).get("file") or "") or not filename.endswith(".wav"):
        return None
    takes = (project_dir(pid).resolve() / "takes").resolve()
    candidate = (takes / filename).resolve()
    if candidate.parent != takes:
        return None
    return candidate


def cue_adopted_take(project, cue, require_file=True, audit_reviewed=False):
    take = cue.get("adopted_take")
    if not isinstance(take, dict):
        return None
    role = next((item for item in project.get("roles", []) if item.get("id") == cue.get("role")), None)
    if not role or str(take.get("signature") or "") != cue_take_signature(project, cue, role):
        # A line the user already listened to and confirmed keeps its audio even while
        # the script is edited. Dropping it here would silently discard the sign-off and
        # make the confirmed line fail the whole track again on the next build.
        if not (audit_reviewed and take.get("reviewed")):
            return None
    path = adopted_take_path(project.get("id"), take) if project.get("id") else None
    if require_file and (not path or not path.is_file()):
        return None
    if require_file and str(take.get("sha256") or "") and file_sha256(path) != str(take.get("sha256") or ""):
        return None
    return {**take, "path": path}


def cleanup_cue_adopted_take(pid, cue, keep_path=None):
    take = cue.pop("adopted_take", None)
    path = adopted_take_path(pid, take) if take else None
    if path and (not keep_path or path.resolve() != Path(keep_path).resolve()):
        path.unlink(missing_ok=True)
    return bool(take)


def cleanup_role_adopted_takes(project, role_id):
    changed = False
    for cue in project.get("cues", []):
        if cue.get("role") == role_id:
            changed = cleanup_cue_adopted_take(project["id"], cue) or changed
    return changed


def cleanup_project_adopted_takes(project):
    changed = False
    for cue in project.get("cues", []):
        changed = cleanup_cue_adopted_take(project["id"], cue) or changed
    return changed


def invalidate_tts_outputs(project, message="参考音频已更新，需要重新生成配音"):
    project.setdefault("stages", {})["tts"] = "pending"
    project["stages"]["render"] = "pending"
    project["stages"]["qa"] = "pending"
    project.pop("output_audio", None)
    project.pop("output", None)
    project.pop("qa", None)
    percent = 0 if project.get("project_type") == "voice" else 42
    project["pipeline"] = {"status": "pending", "stage": "script", "percent": percent, "message": message}
    project["last_message"] = message
    return project


def cleanup_role_preview_files(pid, role_id):
    previews = (project_dir(pid).resolve() / "previews").resolve()
    for path in previews.glob(f"{role_id}-*") if previews.exists() else ():
        try:
            resolved = path.resolve()
            if resolved.is_file() and resolved.parent == previews:
                resolved.unlink(missing_ok=True)
        except OSError:
            continue


def cleanup_cue_generation_files(project, cue):
    root = project_dir(project["id"]).resolve()
    try:
        cue_id = int(cue.get("id"))
    except (TypeError, ValueError):
        return
    raw_dir = root / "raw"
    for path in raw_dir.glob(f"{cue_id:03d}-*") if raw_dir.exists() else ():
        if path.is_file():
            path.unlink(missing_ok=True)
    for path in (
        root / "clean" / f"{cue_id:03d}.wav",
        root / "clean" / f"{cue_id:03d}.json",
        root / "clean" / f"{cue_id:03d}.wav.json",
        root / "fit" / f"{cue_id:03d}.wav",
        root / "fit" / f"{cue_id:03d}.json",
        root / "fit" / f"{cue_id:03d}.wav.json",
    ):
        path.unlink(missing_ok=True)


def cleanup_role_generation_files(project, role_id):
    for cue in project.get("cues", []):
        if cue.get("role") == role_id:
            cleanup_cue_generation_files(project, cue)


def cleanup_project_generation_files(pid, project=None):
    root = project_dir(pid).resolve()
    project = project or load_project(pid)
    for filename in ("voice_timeline.wav", "summary.json", "status.json", "qa.json", "chinese.srt", "chinese.ass"):
        (root / filename).unlink(missing_ok=True)
    for field in ("output_audio", "output"):
        value = str(project.get(field) or "").strip()
        if not value:
            continue
        try:
            resolved = Path(value).expanduser().resolve()
            if resolved.is_file() and root in resolved.parents:
                resolved.unlink(missing_ok=True)
        except OSError:
            continue


def cleanup_role_reference_files(pid, role_id):
    root = project_dir(pid).resolve()
    references = (root / "references").resolve()
    candidates = []
    if references.exists():
        candidates.extend((references / f"{role_id}.wav",))
        candidates.extend(references.glob(f"{role_id}-source.*"))
        candidates.extend(references.glob(f".{role_id}-*"))
    for path in candidates:
        try:
            resolved = path.resolve()
            if resolved.is_file() and resolved.parent == references:
                resolved.unlink(missing_ok=True)
        except OSError:
            continue
    cleanup_role_preview_files(pid, role_id)


def cleanup_orphan_adopted_take_files(project):
    if not project.get("id"):
        return
    takes = project_dir(project["id"]).resolve() / "takes"
    if not takes.exists():
        return
    used = {
        str(cue.get("adopted_take", {}).get("file") or "")
        for cue in project.get("cues", [])
        if isinstance(cue.get("adopted_take"), dict)
    }
    for path in takes.glob("*.wav"):
        if path.name not in used:
            path.unlink(missing_ok=True)


def ensure_clone_transcripts(project, role_ids=None):
    selected = None if role_ids is None else {str(item) for item in role_ids}
    settings = project.get("settings", {})
    changed = False
    for role in project.get("roles", []):
        if role.get("mode") != "clone" or (selected is not None and str(role.get("id")) not in selected):
            continue
        ref_value = str(role.get("ref_audio") or "").strip()
        reference = Path(ref_value).expanduser() if ref_value else None
        revision = str(role.get("reference_revision") or "")
        if not revision and reference and reference.is_file():
            role["reference_revision"] = uuid.uuid4().hex
            if not role.get("reference_sha256"):
                role["reference_sha256"] = file_sha256(reference)
            revision = role["reference_revision"]
            changed = True
        confirmed_revision = str(role.get("ref_text_confirmed_revision") or "")
        confirmed = bool(role.get("ref_text_confirmed")) and bool(revision) and confirmed_revision == revision
        if bool(role.get("ref_text_confirmed")) != confirmed:
            role["ref_text_confirmed"] = confirmed
            changed = True
        if role.get("ref_text_confirmed_revision") != (confirmed_revision if confirmed else ""):
            role["ref_text_confirmed_revision"] = confirmed_revision if confirmed else ""
            changed = True
        existing_text = str(role.get("ref_text") or "").strip()
        auto_text = str(role.get("ref_text_auto") or "").strip()
        source = str(role.get("ref_text_source") or "")
        if confirmed:
            if not auto_text and source == "auto" and existing_text:
                role["ref_text_auto"] = existing_text
                changed = True
            continue
        if auto_text:
            if not existing_text:
                role["ref_text"] = auto_text
                changed = True
            continue
        if existing_text and source in {"auto", "manual"}:
            if source == "auto":
                role["ref_text_auto"] = existing_text
                changed = True
            continue
        if not reference or not reference.is_file():
            continue
        requested = existing_text
        try:
            transcript = transcribe_reference_audio(
                reference,
                settings.get("target_lang", "auto"),
                settings.get("whisper_model"),
            )
        except RuntimeError:
            if requested:
                role["ref_text_auto"] = requested
                role["ref_text_source"] = "auto"
                changed = True
                continue
            raise
        role["ref_text_auto"] = transcript
        role["ref_text"] = transcript
        role["ref_text_source"] = "auto"
        role["ref_text_requested"] = requested
        role["ref_text_warning"] = bool(requested and transcript_similarity(requested, transcript) < 0.65)
        role["ref_text_confirmed"] = False
        role["ref_text_confirmed_revision"] = ""
        changed = True
    return changed


def migrate_clone_transcripts(project, role_ids=None):
    selected = None if role_ids is None else {str(item) for item in role_ids}
    migration_role_ids = {
        str(role.get("id"))
        for role in project.get("roles", [])
        if role.get("mode") == "clone"
        and role.get("ref_audio")
        and (selected is None or str(role.get("id")) in selected)
        and (
            any(field not in role for field in ("ref_text_auto", "ref_text_confirmed", "ref_text_confirmed_revision", "reference_revision"))
            or (
                role.get("ref_text_confirmed")
                and (
                    not role.get("reference_revision")
                    or str(role.get("ref_text_confirmed_revision") or "") != str(role.get("reference_revision") or "")
                )
            )
        )
    }
    normalize_project(project)
    changed = ensure_clone_transcripts(project, role_ids)
    pending_role_ids = [
        str(role.get("id"))
        for role in project.get("roles", [])
        if role.get("mode") == "clone"
        and (selected is None or str(role.get("id")) in selected)
        and not (
            role.get("ref_text_confirmed")
            and role.get("reference_revision")
            and str(role.get("ref_text_confirmed_revision") or "") == str(role.get("reference_revision") or "")
        )
    ]
    invalidated = bool(set(pending_role_ids) & migration_role_ids)
    if changed and pending_role_ids:
        invalidated = True
    if invalidated and project.get("id"):
        for role_id in pending_role_ids:
            cleanup_role_preview_files(project["id"], role_id)
            cleanup_role_generation_files(project, role_id)
        cleanup_project_generation_files(project["id"], project)
        invalidate_tts_outputs(project, "旧项目参考逐字稿需要重新核听确认")
    if changed or migration_role_ids or invalidated:
        save_project(project)
        return True
    return False


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
    if p0.get("project_type") == "voice":
        # A voice project has no media source to analyze or translate. Its
        # script is already authoritative; retry only the TTS stage.
        return generate_tts(pid, reuse=True)
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
        if current == "tts" and p.get("project_type") == "voice":
            p = clear_partial_voice_output(pid, p)
        if current in {"analyze", "tts", "render", "qa"}:
            p.setdefault("stages", {})[current] = "failed"
            save_project(p)
        set_pipeline(pid,"failed",current,p.get("pipeline",{}).get("percent",0),f"自动处理失败：{e}")
        raise


def normalize_project(p, renumber_cues=True):
    p.setdefault("project_type", "video" if p.get("source") else "voice")
    p["direction_brief"] = normalize_direction_brief(p.get("direction_brief"))
    settings = p.setdefault("settings", {})
    had_timeline_mode = "timeline_mode" in settings
    defaults = default_settings(settings.get("target_lang", "zh"), settings.get("tts_engine", "local"))
    for key, value in defaults.items(): settings.setdefault(key, value)
    if p["project_type"] == "voice" and not had_timeline_mode: settings["timeline_mode"] = "natural"
    pronunciation = []
    for item in settings.get("pronunciation_lexicon", []) or []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        spoken = str(item.get("spoken") or "").strip()
        if source and spoken and len(source) <= 80 and len(spoken) <= 120:
            pronunciation.append({"source": source, "spoken": spoken})
    settings["pronunciation_lexicon"] = pronunciation[:200]
    role_ids={r["id"] for r in p.get("roles",[])}
    natural = settings.get("timeline_mode") == "natural"
    cursor = 0.0
    for i,c in enumerate(p.get("cues",[]),1):
        if renumber_cues:
            c["id"]=i
        c["enabled"]=bool(c.get("enabled",True)); c["text"]=str(c.get("text","")).strip()
        c["source_text"]=str(c.get("source_text","")).strip(); c["emotion"]=str(c.get("emotion","自然")).strip()
        c["scene_before"]=str(c.get("scene_before","")).strip()
        c["scene_after"]=str(c.get("scene_after","")).strip()
        c["speed"]=max(.5,min(2.0,float(c.get("speed",1.0) or 1.0)))
        c["gap_after"]=max(0.0,min(10.0,float(c.get("gap_after",settings.get("default_gap",.25)) or 0)))
        take = c.get("adopted_take")
        if isinstance(take, dict):
            filename = Path(str(take.get("file") or "")).name
            signature = str(take.get("signature") or "")
            reviewed = bool(take.get("reviewed"))
            keep = bool(filename) and filename == str(take.get("file") or "") and filename.endswith(".wav")
            if keep and (signature or reviewed):
                c["adopted_take"] = {
                    "id": str(take.get("id") or Path(filename).stem),
                    "file": filename,
                    "signature": signature,
                    "created_at": str(take.get("created_at") or ""),
                    "adopted_at": str(take.get("adopted_at") or ""),
                    "sha256": str(take.get("sha256") or ""),
                    "seed": int(take.get("seed") or 0),
                    "reviewed": reviewed,
                    "reviewed_at": str(take.get("reviewed_at") or ""),
                    "review_transcript": str(take.get("review_transcript") or ""),
                    "review_expected_text": str(take.get("review_expected_text") or ""),
                    "review_gate_reason": str(take.get("review_gate_reason") or ""),
                    "review_note": str(take.get("review_note") or ""),
                }
            else:
                c.pop("adopted_take", None)
        else:
            c.pop("adopted_take", None)
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
        r["personality"] = normalize_role_personality(r.get("personality"))
        r.setdefault("ref_audio", "")
        r.setdefault("ref_text", "")
        default_ref_source = "" if r.get("mode") == "clone" and r.get("ref_audio") else ("manual" if r.get("ref_text") else "")
        r.setdefault("ref_text_source", default_ref_source)
        r.setdefault("ref_text_auto", "")
        r.setdefault("ref_text_confirmed", False)
        r.setdefault("ref_text_confirmed_revision", "")
        r.setdefault("reference_revision", "")
        r.setdefault("reference_sha256", "")
        current_revision = str(r.get("reference_revision") or "")
        confirmed_revision = str(r.get("ref_text_confirmed_revision") or "")
        r["ref_text_confirmed"] = bool(r.get("ref_text_confirmed")) and bool(current_revision) and confirmed_revision == current_revision
        if not r["ref_text_confirmed"]:
            r["ref_text_confirmed_revision"] = ""
        r["clone_consent"] = bool(r.get("clone_consent", False))
    return p


def save_manifest(pid, payload):
    p=load_project(pid)
    before=json.dumps({"roles":p.get("roles",[]),"cues":p.get("cues",[]),"settings":p.get("settings",{}),"direction_brief":normalize_direction_brief(p.get("direction_brief"))},ensure_ascii=False,sort_keys=True)
    stored_roles = {role.get("id"): role for role in p.get("roles", [])}
    stored_cues = {int(cue.get("id")): cue for cue in p.get("cues", []) if cue.get("id") is not None}
    incoming_cues = payload.get("cues") if isinstance(payload.get("cues"), list) else None
    if incoming_cues is not None:
        for cue in incoming_cues:
            stored = stored_cues.get(int(cue.get("id") or 0)) or {}
            if stored.get("adopted_take"):
                cue["adopted_take"] = stored["adopted_take"]
            else:
                cue.pop("adopted_take", None)
    for key in ("roles","cues","settings","name","direction_brief"):
        if key in payload: p[key]=payload[key]
    protected_clone_fields = (
        "ref_audio", "ref_name", "reference_revision", "reference_sha256", "ref_text", "ref_text_auto",
        "ref_text_source", "ref_text_requested", "ref_text_warning", "ref_text_confirmed", "ref_text_confirmed_revision",
    )
    for role in p.get("roles", []):
        stored = stored_roles.get(role.get("id")) or {}
        if stored.get("ref_audio"):
            for field in protected_clone_fields:
                role[field] = stored.get(field, False if field in {"ref_text_warning", "ref_text_confirmed"} else "")
            role["clone_consent"] = bool(stored.get("clone_consent", False))
            continue
        for field in protected_clone_fields:
            role[field] = False if field in {"ref_text_warning", "ref_text_confirmed"} else ""
    p=normalize_project(p, renumber_cues=False)
    for cue in p.get("cues", []):
        take = cue.get("adopted_take")
        if take and cue_adopted_take(p, cue, audit_reviewed=True) is None:
            cleanup_cue_adopted_take(pid, cue)
    cleanup_orphan_adopted_take_files(p)
    for index, cue in enumerate(p.get("cues", []), 1):
        cue["id"] = index
    after=json.dumps({"roles":p.get("roles",[]),"cues":p.get("cues",[]),"settings":p.get("settings",{}),"direction_brief":p.get("direction_brief","")},ensure_ascii=False,sort_keys=True)
    p["stages"]["script"]="ready"; p["last_message"]="台词清单已保存"
    if before != after:
        message = "修改已保存，需要重新生成配音音轨" if p.get("project_type")=="voice" else "修改已保存，需要重新生成视频成片"
        invalidate_tts_outputs(p, message)
    save_project(p); write_json(project_dir(pid)/"manifest.json",[c for c in p["cues"] if c.get("enabled",True)])
    add_log(pid,p["last_message"]); return p


def enabled_cues(p):
    return [c for c in p.get("cues", []) if c.get("enabled", True)]


def clear_partial_voice_output(pid, project=None):
    root = project_dir(pid).resolve()
    for filename in ("voice_timeline.wav", "summary.json"):
        (root / filename).unlink(missing_ok=True)
    project = project or load_project(pid)
    project.pop("output_audio", None)
    return project


def cue_blockers(pid, project=None):
    """Cues with no audio at all. Everything else is publishable, graded or not."""
    project = project or load_project(pid)
    root = project_dir(pid)
    raw_dir = root / "raw"
    stages = project.get("stages", {})
    # Before the first synthesis attempt every cue legitimately has no audio; that is
    # "not generated yet", not "blocked", and must not be reported as a failure.
    if not _tts_attempted(root, stages):
        return []
    rows = []
    for cue in project.get("cues", []):
        if cue.get("enabled", True) is False:
            continue
        cue_id = int(cue.get("id") or 0)
        role = str(cue.get("role") or "")
        if cue_adopted_take(project, cue, audit_reviewed=True):
            continue
        failure = read_json(raw_dir / f"{cue_id:03d}-{role}.failed.json", {}) or {}
        if not failure and (raw_dir / f"{cue_id:03d}-{role}.wav").is_file():
            continue
        rows.append({
            "cue_id": cue_id,
            "role": role,
            "text": str(cue.get("text") or ""),
            "error": str((failure or {}).get("error") or "这次生成没有产出可用音频"),
            "mode": str((failure or {}).get("mode") or ""),
            "failed_at": str((failure or {}).get("timestamp") or ""),
        })
    return rows


def cue_review_queue(pid, project=None):
    """Published cues whose automatic check did not pass, so the user can listen.

    These never stop the track from being built; they are the lines worth a second
    pair of ears.
    """
    project = project or load_project(pid)
    root = project_dir(pid)
    rows = []
    for row in read_json(root / "status.json", []) or []:
        if not isinstance(row, dict) or not row.get("needs_review"):
            continue
        rows.append({
            "cue_id": int(row.get("id") or 0),
            "role": str(row.get("role") or ""),
            "text": str(row.get("text") or ""),
            "transcript": str(row.get("gate_transcript") or ""),
            "reason": str(row.get("gate_reason") or ""),
            "degraded": bool(row.get("degraded")),
        })
    rows.sort(key=lambda item: item["cue_id"])
    return rows


def _tts_attempted(root, stages):
    if (root / "status.json").is_file():
        return True
    if any((root / "raw").glob("*.failed.json")):
        return True
    return str((stages or {}).get("tts") or "") in {"failed", "done", "running"}


def voice_output_is_complete(pid, project=None):
    project = project or load_project(pid)
    if project.get("project_type") != "voice" or project.get("stages", {}).get("tts") != "done":
        return False
    root = project_dir(pid).resolve()
    output_value = str(project.get("output_audio") or "").strip()
    if not output_value:
        return False
    output = Path(output_value).expanduser().resolve()
    if output != (root / "voice_timeline.wav").resolve() or not output.is_file():
        return False
    status = read_json(root / "status.json", [])
    summary = read_json(root / "summary.json", {})
    if not isinstance(status, list) or not isinstance(summary, dict):
        return False
    expected_ids = {int(cue["id"]) for cue in enabled_cues(project)}
    status_ids = set()
    for row in status:
        if not isinstance(row, dict):
            continue
        try:
            status_ids.add(int(row["id"]))
        except (KeyError, TypeError, ValueError):
            continue
    try:
        summary_count = int(summary.get("cues", -1))
    except (TypeError, ValueError):
        return False
    if status_ids != expected_ids or summary_count != len(expected_ids):
        return False
    return all((root / "fit" / f"{cue_id:03d}.wav").is_file() for cue_id in expected_ids)


def cue_export_filename(cue, role=None):
    cue_id = max(0, int(cue.get("id") or 0))
    role_name = str((role or {}).get("name") or cue.get("role") or "角色").strip()
    text = re.sub(r"\s+", " ", str(cue.get("text") or "")).strip()
    safe_role = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", role_name).strip("._") or "角色"
    safe_text = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", text).strip("._") or "台词"
    return f"{cue_id:03d}-{safe_role[:24]}-{safe_text[:48]}.wav"


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
        if not str(role.get("ref_text") or "").strip():
            raise RuntimeError(f"角色“{role.get('name','未命名')}”缺少参考音频逐字稿")
        revision = str(role.get("reference_revision") or "")
        if not role.get("ref_text_confirmed") or not revision or str(role.get("ref_text_confirmed_revision") or "") != revision:
            raise RuntimeError(f"角色“{role.get('name','未命名')}”的参考逐字稿尚未核听确认")



TTS_PROGRESS = {}


def start_tts_progress(pid, total):
    """Create the web-process side of a progress snapshot for one TTS run.

    The actual generator is a child process, so the detailed counters are read
    from its append-only tts.log by get_tts_progress(). This snapshot only owns
    the run clock and the total cue count.
    """
    TTS_PROGRESS[pid] = {
        "status": "running",
        "total": max(0, int(total or 0)),
        "started_at": time.time(),
    }


def track_tts_progress(pid, current, total, stage="tts"):
    """Keep the legacy in-memory progress helper compatible with old callers."""
    info = TTS_PROGRESS.setdefault(pid, {"started_at": time.time()})
    info.update({
        "status": "running",
        "current": int(current or 0),
        "total": max(0, int(total or 0)),
        "stage": stage,
        "updated_at": time.time(),
    })
    return info


def _progress_cue_maps(project, total):
    cues = enabled_cues(project or {})
    by_id = {}
    for index, cue in enumerate(cues, 1):
        try:
            cue_id = int(cue.get("id") or 0)
        except (TypeError, ValueError):
            continue
        by_id[cue_id] = {
            "index": index,
            "role": str(cue.get("role") or ""),
            "text": str(cue.get("text") or ""),
        }
    return by_id, max(int(total or 0), len(by_id))


def _progress_seconds(value):
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def get_tts_progress(pid, project=None):
    """Parse the current worker log into a small, UI-ready progress snapshot."""
    project = project or load_project(pid)
    root = project_dir(pid)
    log_path = root / "tts.log"
    tracked_run = pid in TTS_PROGRESS
    pipeline_state = str((project.get("pipeline") or {}).get("status") or "")
    tts_state = str((project.get("stages") or {}).get("tts") or "")
    if not tracked_run and tts_state not in {"running", "done", "failed"} and pipeline_state not in {"running", "done", "failed"}:
        return None
    info = dict(TTS_PROGRESS.get(pid) or {})
    if not info and log_path.exists():
        try:
            stat = log_path.stat()
            started_at = getattr(stat, "st_birthtime", 0.0) or stat.st_mtime
        except OSError:
            started_at = time.time()
        info = {"started_at": started_at}

    cue_map, total = _progress_cue_maps(project, info.get("total", 0))
    if not total and not log_path.exists():
        return None
    info["total"] = total
    if "started_at" not in info:
        info["started_at"] = time.time()

    raw_ready = set()
    reused = set()
    processed = set()
    failed = set()
    needs_review = set()
    current_cue_id = None
    current_index = 0
    current_role = ""
    current_state = ""
    stage = "preparing"
    stage_mode = "estimated"
    candidate_executions = 0
    asr_executions = 0
    last_event = ""
    terminal_summary = None

    if log_path.exists():
        try:
            lines = log_path.read_text(errors="replace").splitlines()
        except OSError:
            lines = []
        for line in lines:
            last_event = line[:240]
            if line.startswith("MLX_QUEUE_WAIT"):
                stage = "queue"
                stage_mode = "indeterminate"
            elif line.startswith("MLX_QUEUE_ACQUIRED"):
                stage = "model_load"
                stage_mode = "indeterminate"
            elif line.startswith("LOAD "):
                stage = "model_load"
                stage_mode = "indeterminate"
            elif line.startswith("PROGRESS_STAGE "):
                stage_match = re.search(r"stage=([^ ]+)", line)
                if stage_match:
                    stage = stage_match.group(1)
                total_match = re.search(r"total=(\d+)", line)
                if total_match:
                    total = max(total, int(total_match.group(1)))
                state_match = re.search(r"state=([^ ]+)", line)
                stage_mode = "indeterminate" if state_match and state_match.group(1) in {"running", "loading"} else "estimated"
            elif line.startswith("PROGRESS_CUE "):
                cue_match = re.search(r"cue=(\d+)", line)
                if cue_match:
                    current_cue_id = int(cue_match.group(1))
                    cue_info = cue_map.get(current_cue_id, {})
                    current_index = int(cue_info.get("index") or 0)
                    current_role = str(cue_info.get("role") or "")
                phase_match = re.search(r"phase=([^ ]+)", line)
                if phase_match:
                    stage = phase_match.group(1)
                state_match = re.search(r"state=([^ ]+)", line)
                if state_match:
                    current_state = state_match.group(1)
                if stage == "generation" and current_state in {"done", "reused"} and current_cue_id is not None:
                    (reused if current_state == "reused" else raw_ready).add(current_cue_id)
                elif stage == "postprocess" and current_state == "done" and current_cue_id is not None:
                    processed.add(current_cue_id)
                elif current_state == "failed" and current_cue_id is not None:
                    failed.add(current_cue_id)
            elif line.startswith("CUE_START "):
                stage = "postprocess"
                stage_mode = "measured"
                match = re.search(r"index=(\d+)/(\d+)\s+cue=(\d+)", line)
                if match:
                    current_index = int(match.group(1))
                    current_cue_id = int(match.group(3))
                    cue_info = cue_map.get(current_cue_id, {})
                    current_role = str(cue_info.get("role") or "")
                    total = max(total, int(match.group(2)))
            elif line.startswith("GENERATED "):
                match = re.search(r"GENERATED\s+(\d+)", line)
                if match:
                    raw_ready.add(int(match.group(1)))
                stage = "generation"
                stage_mode = "measured"
            elif line.startswith("ADOPTED_TAKE "):
                match = re.search(r"ADOPTED_TAKE\s+(\d+)", line)
                if match:
                    reused.add(int(match.group(1)))
                stage = "generation"
                stage_mode = "measured"
            elif line.startswith("CUE_WORK_CACHE "):
                match = re.search(r"asr_executions=(\d+).*candidate_executions=(\d+)", line)
                if match:
                    asr_executions += int(match.group(1))
                    candidate_executions += int(match.group(2))
            elif line.startswith("GENERATION_FAILED_CONTINUE "):
                match = re.search(r"cue=(\d+)", line)
                if match:
                    failed.add(int(match.group(1)))
            elif line.startswith("POSTPROCESS_SKIP_MISSING_RAW "):
                match = re.search(r"cue=(\d+)", line)
                if match:
                    failed.add(int(match.group(1)))
            elif line.startswith("NEEDS_REVIEW "):
                match = re.search(r"cue=(\d+)", line)
                if match:
                    needs_review.add(int(match.group(1)))
            elif line.startswith("INCOMPLETE "):
                stage = "postprocess"
                stage_mode = "measured"
            elif line.startswith("DONE "):
                try:
                    terminal_summary = json.loads(line[5:])
                except (TypeError, ValueError, json.JSONDecodeError):
                    terminal_summary = None
                stage = "complete"
                stage_mode = "measured"

    if isinstance(terminal_summary, dict):
        total = max(total, int(terminal_summary.get("cues") or 0))
        raw_ready.update(range(1, total + 1))
        processed.update(range(1, total + 1))
        needs_review.update(int(value) for value in terminal_summary.get("needs_review_ids", []) or [] if str(value).isdigit())
        info["status"] = "done"
    else:
        pipeline = project.get("pipeline") or {}
        if project.get("stages", {}).get("tts") == "failed" or pipeline.get("status") == "failed":
            info["status"] = "failed"
        elif project.get("stages", {}).get("tts") == "done":
            info["status"] = "done"
        elif pipeline.get("status") == "queued":
            info["status"] = "queued"
        else:
            info["status"] = "running"

    if info["status"] == "queued":
        stage = "queue"
        stage_label = "等待语音资源"
        stage_current = 0
        percent = 0
        progress_mode = "indeterminate"
    elif stage == "complete" or info["status"] == "done":
        stage = "complete"
        stage_label = "生成完成"
        stage_current = total
        percent = 100
        progress_mode = "measured"
    elif stage == "queue":
        stage_label = "等待语音资源"
        stage_current = 0
        percent = 0
        progress_mode = "indeterminate"
    elif stage == "model_load":
        stage_label = "加载语音模型"
        stage_current = len(raw_ready | reused)
        percent = 5
        progress_mode = "indeterminate"
    elif stage == "postprocess":
        stage = "postprocess"
        stage_label = "后处理与语音校验"
        stage_current = len(processed)
        percent = 70 + round(25 * stage_current / total) if total else 70
        progress_mode = "measured"
    elif stage in {"timeline", "publish"}:
        stage_label = "合成音轨"
        stage_current = total
        percent = 96
        progress_mode = "indeterminate"
    else:
        stage = "generation"
        stage_label = "逐句生成与自动校验"
        stage_current = len(raw_ready | reused)
        percent = 10 + round(60 * stage_current / total) if total else 10
        progress_mode = "measured"

    completed = len(processed) if stage == "postprocess" else len(raw_ready | reused)
    if stage == "complete":
        completed = total
    elapsed = max(0.0, time.time() - float(info.get("started_at") or time.time()))
    remaining = 0
    if completed > 0 and total > completed and stage in {"generation", "postprocess"}:
        remaining = round((elapsed / completed) * (total - completed))

    cue_info = cue_map.get(current_cue_id or 0, {})
    if current_index <= 0:
        current_index = int(cue_info.get("index") or 0)
    if not current_role:
        current_role = str(cue_info.get("role") or "")
    current_text = str(cue_info.get("text") or "")
    if info["status"] == "failed":
        stage_label = "生成失败"
    elif info["status"] == "queued":
        stage_label = "等待语音资源"
    if info["status"] == "done":
        message = "配音音轨已生成"
    elif info["status"] == "failed":
        message = str((project.get("pipeline") or {}).get("message") or project.get("last_message") or "生成失败")
    elif stage == "queue":
        message = "排队中：等待语音资源"
    elif current_index and stage in {"generation", "postprocess"}:
        action = "生成" if stage == "generation" else "处理"
        message = f"正在{action}第 {current_index}/{total} 句"
    else:
        message = stage_label

    return {
        "status": info["status"],
        "stage": stage,
        "stage_label": stage_label,
        "progress_mode": progress_mode if progress_mode != "estimated" else stage_mode,
        "percent": max(0, min(100, int(percent))),
        "stage_current": int(stage_current),
        "stage_total": int(total),
        "current_index": int(current_index),
        "current_cue_id": current_cue_id,
        "current_role": current_role,
        "current_text": current_text,
        "current_state": current_state,
        "total": int(total),
        "generated": len(raw_ready),
        "reused": len(reused),
        "processed": len(processed),
        "failed": len(failed),
        "needs_review": len(needs_review),
        "candidate_executions": candidate_executions,
        "asr_executions": asr_executions,
        "elapsed_seconds": round(elapsed),
        "estimated_remaining_seconds": remaining,
        "message": message,
        "last_event": last_event,
    }



def cue_content_signature(cue):
    payload = {
        "text": str(cue.get("text") or "").strip(),
        "role": str(cue.get("role") or ""),
        "emotion": str(cue.get("emotion") or "自然").strip(),
        "speed": float(cue.get("speed") or 1.0),
        "gap": float(cue.get("gap") or 0.25),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def generate_tts(pid, reuse=True):
    observer = TTS_JOB_OBSERVER.get()
    if observer:
        observer("running")
    p=load_project(pid); root=project_dir(pid)
    TTS_PROGRESS.pop(pid, None)
    p = clear_partial_voice_output(pid, p)
    p.setdefault("stages", {})["tts"] = "pending"
    p["stages"]["render"] = "pending"
    p.pop("qa", None)
    save_project(p)
    for stale_path in (root / "status.json", root / "summary.json"):
        stale_path.unlink(missing_ok=True)
    migrate_clone_transcripts(p)
    p=normalize_project(p)
    cues=enabled_cues(p)
    if not cues or any(not c["text"] for c in cues): raise RuntimeError("存在空白台词")
    if not p.get("roles"): raise RuntimeError("至少需要一个角色")
    engine=p["settings"].get("tts_engine","local")
    used_role_ids={c.get("role") for c in cues}
    used_clone_roles=[r for r in p.get("roles",[]) if r.get("id") in used_role_ids and r.get("mode")=="clone"]
    natural_voice = p.get("project_type") == "voice" and p["settings"].get("timeline_mode") == "natural"
    duration=0.0 if natural_voice else project_duration(p)
    if engine=="gemini":
        if used_clone_roles:
            names="、".join(str(role.get("name") or role.get("id") or "未命名") for role in used_clone_roles)
            raise RuntimeError(f"Gemini TTS 不支持授权克隆角色：{names}。请切换到本地 Qwen3-TTS。")
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
        start_tts_progress(pid, len(cues))
        validate_clone_roles(p)
        pronunciation_rules = p.get("settings", {}).get("pronunciation_lexicon", [])
        generation_cues = []
        for cue in cues:
            generated_cue = dict(cue)
            generated_cue["expected_text"] = str(cue.get("text") or "")
            generated_cue["text"] = apply_pronunciation_lexicon(generated_cue["expected_text"], pronunciation_rules)
            generation_cues.append(generated_cue)
        write_json(root/"manifest.json", generation_cues)
        # A worker killed during a previous attempt can leave a valid-looking
        # timeline and status for only part of the current manifest. Remove
        # those aggregate artifacts before starting a new build; per-cue raw
        # caches remain available for a fast, verified resume.
        for stale_path in (root / "voice_timeline.wav", root / "summary.json", root / "status.json"):
            stale_path.unlink(missing_ok=True)
        voices = {
            r["id"]: {**r, "project_direction": p.get("direction_brief") or ""}
            for r in p["roles"]
        }
        write_json(root/"voices.json", voices)
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
        log_path = root / "tts.log"
        with log_path.open("w", encoding="utf-8") as log_handle:
            proc=subprocess.run(
                cmd,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
        output = log_path.read_text(errors="replace")
        if proc.returncode:
            clear_partial_voice_output(pid, load_project(pid))
            raise RuntimeError("Qwen3-TTS 失败，详见 tts.log\n" + output[-2000:])
    # A render is complete only when every enabled cue has verified raw, clean,
    # and fit audio. Missing/failed cues must remain a failed project, never a
    # falsely successful partial timeline.
    # The worker owns the per-cue status list and persists it before reporting
    # success. Read that durable result rather than relying on a local variable
    # that only exists inside the worker process.
    status = read_json(root / "status.json", [])
    if not isinstance(status, list):
        status = []
    enabled_count = len(cues)
    status_ids = set()
    for row in status:
        if not isinstance(row, dict):
            continue
        try:
            status_ids.add(int(row.get("id")))
        except (TypeError, ValueError):
            continue
    generated_count = len(status_ids)
    missing_ids = [
        int(c["id"])
        for c in cues
        if int(c["id"]) not in status_ids
        or not (root / "fit" / f"{int(c['id']):03d}.wav").is_file()
    ]
    if generated_count != enabled_count or missing_ids:
        p=clear_partial_voice_output(pid, load_project(pid))
        p["stages"]["tts"]="failed"; p["stages"]["render"]="pending"
        p["last_message"] = f"配音未完成：{generated_count}/{enabled_count}，这些台词没生成出音频 {missing_ids}，请重试"
        save_project(p); add_log(pid,p["last_message"])
        raise RuntimeError(p["last_message"])
    p=load_project(pid); p["stages"]["tts"]="done"; p["stages"]["render"]="done" if p.get("project_type")=="voice" else p["stages"].get("render","pending")
    p["output_audio"]=str(root/"voice_timeline.wav")
    summary=read_json(root/"summary.json",{}) or {}
    warning_count=int(summary.get("fit_warnings",0) or 0)
    warning_text=f"；{warning_count}句因时间槽较短已加速，请复听" if warning_count else ""
    review_count=int(summary.get("needs_review",0) or 0)
    review_text=f"；{review_count}句自动校验未通过，已用最佳候选发布，建议复听" if review_count else ""
    cache_text=""
    if engine != "gemini":
        raw_hits=int(summary.get("raw_cache_hits",0) or 0)
        post_hits=int(summary.get("postprocess_stage_hits",summary.get("postprocess_cache_hits",0)) or 0)
        reused=max(raw_hits,post_hits)
        if reused:
            cache_text=f"；已复用 {reused}/{len(cues)} 句缓存"
    p["last_message"]=f"配音音轨已生成（{'Gemini云端' if engine=='gemini' else '本地 Qwen3-TTS'}）{cache_text}{warning_text}{review_text}"
    p["pipeline"]={"status":"done" if p.get("project_type")=="voice" else "running","stage":"tts","percent":100 if p.get("project_type")=="voice" else 76,"message":p["last_message"]}
    save_project(p); add_log(pid,p["last_message"]); return p


def preview_voice(pid, role_id, text, emotion="自然", personality_directed=False, cue_id=None, preview_nonce=None):
    p=load_project(pid); root=project_dir(pid)
    migrate_clone_transcripts(p, {role_id})
    p=normalize_project(p)
    role=next((r for r in p.get("roles",[]) if r.get("id")==role_id),None)
    if not role: raise RuntimeError("角色不存在")
    if role.get("mode")=="clone" and p.get("settings",{}).get("tts_engine","local")!="local":
        raise RuntimeError("授权克隆仅支持本地 Qwen3-TTS，请先切换项目引擎")
    if not str(text or "").strip(): raise RuntimeError("请输入试听台词")
    original_text = str(text or "").strip()
    spoken_text = apply_pronunciation_lexicon(original_text, p.get("settings", {}).get("pronunciation_lexicon", []))
    cue = None
    if cue_id is not None:
        try:
            wanted_id = int(cue_id)
        except (TypeError, ValueError):
            raise RuntimeError("试听台词编号无效")
        cue = next((item for item in p.get("cues", []) if int(item.get("id") or 0) == wanted_id), None)
        if not cue:
            raise RuntimeError("试听台词不存在")
        if cue.get("role") != role_id:
            raise RuntimeError("试听台词与角色不匹配")
        if str(cue.get("text") or "").strip() != str(text or "").strip():
            raise RuntimeError("试听台词已变化，请保存最新内容后重试")
        if str(cue.get("emotion") or "自然").strip() != str(emotion or "自然").strip():
            raise RuntimeError("试听表演要求已变化，请保存最新内容后重试")
    validate_clone_roles(p,{role_id})
    preview_dir=root/"previews"; preview_dir.mkdir(parents=True,exist_ok=True)
    preview_role={**role, "project_direction": p.get("direction_brief") or ""}
    ref_value=str(preview_role.get("ref_audio") or "").strip(); ref_path=Path(ref_value).expanduser() if ref_value else None
    ref_signature={"size":ref_path.stat().st_size,"mtime_ns":ref_path.stat().st_mtime_ns} if ref_path and ref_path.is_file() else None
    generation_signature = cue_take_signature(p, cue, role, emotion) if cue else hashlib.sha256(json.dumps({"role":preview_role,"reference":ref_signature,"text":spoken_text,"source_text":original_text,"emotion":emotion,"lang":p["settings"].get("target_lang","zh"),"clone_generation_policy":CLONE_GENERATION_POLICY if role.get("mode")=="clone" else ""},ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    nonce = str(preview_nonce or (f"cue-{cue['id']}-{role_id}" if cue else uuid.uuid4().hex))
    token=hashlib.sha256(f"{generation_signature}:{nonce}".encode()).hexdigest()[:16]
    # A preview always returns audio to listen to. When the automatic check refuses
    # every take, the worker keeps the best one and records the verdict, so the user
    # can sign off by ear when the recognizer cannot transcribe a name.
    output=preview_dir/f"{role_id}-{token}.wav"
    request_path=preview_dir/f"{role_id}-{token}.json"
    seed = 970000 + int(hashlib.sha256(nonce.encode()).hexdigest()[:8], 16) % 1000000
    write_json(request_path,{"voice":preview_role,"text":spoken_text,"expected_text":original_text,"emotion":emotion,"language":p["settings"].get("target_lang","zh"),
                             "design_model":p["settings"].get("voice_design_model",DEFAULT_TTS_MODEL),
                             "clone_model":p["settings"].get("voice_clone_model",DEFAULT_CLONE_MODEL),
                             "seed":seed,"cue_id":int(cue["id"]) if cue else None,"signature":generation_signature,
                             "personality_directed":bool(personality_directed),"created_at":now()})
    env=os.environ.copy(); env["HF_HUB_OFFLINE"]="1"; env["FFMPEG"]=ffmpeg_full(); env["FFPROBE"]=media_tool("ffprobe")
    cmd=[str(DEFAULT_TTS_PYTHON),str(TTS_WORKER),"--preview-request",str(request_path),"--preview-output",str(output),
         "--design-model",p["settings"].get("voice_design_model",DEFAULT_TTS_MODEL),
         "--clone-model",p["settings"].get("voice_clone_model",DEFAULT_CLONE_MODEL)]
    proc=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
    (preview_dir/f"{role_id}-{token}.log").write_text(proc.stdout)
    if proc.returncode:
        output.unlink(missing_ok=True)
        request_path.unlink(missing_ok=True)
        raise RuntimeError("试听生成失败\n"+proc.stdout[-1600:])
    return output


def preview_gate_report(pid, preview_file):
    """Gate verdict recorded by the worker for a specific preview take."""
    filename = Path(str(preview_file or "")).name
    if not filename or filename != str(preview_file or "") or not filename.endswith(".wav"):
        return {}
    report = read_json(project_dir(pid) / "previews" / f"{Path(filename).stem}.review.json", {}) or {}
    return report if isinstance(report, dict) else {}


def adopt_preview_take(pid, cue_id, preview_file, confirm_gate_rejection=False, review_note=""):
    p=normalize_project(load_project(pid)); root=project_dir(pid)
    try:
        cue_id = int(cue_id)
    except (TypeError, ValueError):
        raise RuntimeError("台词编号无效")
    cue = next((item for item in p.get("cues", []) if int(item.get("id") or 0) == cue_id), None)
    if not cue:
        raise RuntimeError("台词不存在")
    role = next((item for item in p.get("roles", []) if item.get("id") == cue.get("role")), None)
    if not role:
        raise RuntimeError("台词角色不存在")
    validate_clone_roles(p, {role.get("id")})
    filename = Path(str(preview_file or "")).name
    if not filename or filename != str(preview_file or "") or not filename.startswith(f"{role['id']}-") or not filename.endswith(".wav"):
        raise RuntimeError("试听文件无效")
    preview_dir=(root/"previews").resolve(); preview=(preview_dir/filename).resolve()
    if preview.parent != preview_dir or not preview.is_file():
        raise RuntimeError("试听文件不存在，请重新生成")
    request_data=read_json(preview.with_suffix(".json"),{}) or {}
    if not request_data:
        raise RuntimeError("试听元数据不存在，请重新生成")
    if int(request_data.get("cue_id") or 0) != cue_id:
        raise RuntimeError("该试听不属于这句台词")
    signature = cue_take_signature(p, cue, role, request_data.get("emotion"))
    if str(request_data.get("signature") or "") != signature:
        raise RuntimeError("台词、角色或表演已变化，请重新试听后再采用")
    gate = preview_gate_report(pid, filename)
    reviewed = bool(gate) and not gate.get("gate_passed", True)
    if reviewed and not confirm_gate_rejection:
        raise RuntimeError("这句试听没有通过自动校验，请先完整听完，再确认“人工确认采用”")
    takes_dir=root/"takes"; takes_dir.mkdir(parents=True,exist_ok=True)
    take_id=uuid.uuid4().hex
    take_file=f"cue-{cue_id:03d}-{take_id}.wav"
    destination=takes_dir/take_file
    cleanup_cue_adopted_take(pid,cue,keep_path=destination)
    cleanup_cue_generation_files(p, cue)
    cleanup_project_generation_files(pid,p)
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(preview,destination)
    take={
        "id":take_id,"file":take_file,"signature":signature,"created_at":now(),"adopted_at":now(),
        "sha256":file_sha256(destination),"seed":int(request_data.get("seed") or 0),
    }
    if reviewed:
        take.update({
            "reviewed": True,
            "reviewed_at": now(),
            "review_transcript": str(gate.get("gate_transcript") or ""),
            "review_expected_text": str(gate.get("expected_text") or cue.get("text") or ""),
            "review_gate_reason": str(gate.get("gate_label") or ""),
            "review_note": str(review_note or "")[:200],
        })
    cue["adopted_take"]=take
    message = (
        f"已人工复听确认采用台词{cue_id}，整轨将直接复用（该句自动校验未通过：{take.get('review_gate_reason') or '识别差异'}）"
        if reviewed else f"已采用台词{cue_id}的试听版本，整轨将直接复用"
    )
    invalidate_tts_outputs(p, message)
    p["last_message"]=message
    save_project(p); add_log(pid,p["last_message"])
    return p


def discard_adopted_take(pid, cue_id):
    p=normalize_project(load_project(pid))
    try:
        cue_id=int(cue_id)
    except (TypeError,ValueError):
        raise RuntimeError("台词编号无效")
    cue=next((item for item in p.get("cues",[]) if int(item.get("id") or 0)==cue_id),None)
    if not cue: raise RuntimeError("台词不存在")
    if not cleanup_cue_adopted_take(pid,cue):
        return p
    cleanup_cue_generation_files(p,cue)
    cleanup_project_generation_files(pid,p)
    invalidate_tts_outputs(p,f"已取消台词{cue_id}的采用版本，需要重新生成")
    p["last_message"]=f"已取消台词{cue_id}的采用版本，需要重新生成"
    save_project(p); add_log(pid,p["last_message"])
    return p


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
