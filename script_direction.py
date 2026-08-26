#!/usr/bin/env python3
import json
import re
import shutil
import urllib.request

VOICE_KEYS = ("male_adult", "male_young", "female_warm", "female_brave", "girl_soft", "girl_lively")
VOICE_LABELS = {
    "male_adult": "沉稳成年男声",
    "male_young": "明亮年轻男声",
    "female_warm": "温暖年轻女声",
    "female_brave": "有力量的英气女声",
    "girl_soft": "柔软清甜女孩声",
    "girl_lively": "明亮活泼女孩声",
}
NARRATOR_NAMES = {"旁白", "解说", "画外音", "narrator", "Narrator"}
FEMALE_NAME_MARKERS = ("姐", "妹", "姨", "奶", "妈", "母", "小姐", "姑娘", "女王", "公主", "夫人", "女士", "阿姨")
MALE_NAME_MARKERS = ("哥", "弟", "叔", "伯", "爷", "爸", "父", "先生", "公子", "少侠", "少爷")
GIRL_NAME_MARKERS = ("小女孩", "女童", "女孩", "萝莉")
BOY_NAME_MARKERS = ("小男孩", "男童", "男孩")
FEMALE_CHARS = set("婷芳娜玲燕娟丽雪梅红霞敏慧静柔晚晴雯淑佳欣")
MALE_CHARS = set("伟强军辉浩宇峰杰勇斌涛刚鹏深凯磊博")
MALE_CYCLE = ("male_adult", "male_young")
FEMALE_CYCLE = ("female_warm", "female_brave")
GIRL_CYCLE = ("girl_soft", "girl_lively")
FALLBACK_EMOTIONS = (
    "自然、清楚",
    "略带思考、语气平稳",
    "认真、把重点说清楚",
    "稍加快、气息更近",
    "放轻声音、像对身边人说",
    "压低音量、更有分量",
    "松一口气、语气缓和",
    "收住情绪、话里有话",
)
EMOTION_RULES = (
    (re.compile(r"(滚出去|闭嘴|混蛋|该死|混账|你敢|岂有此理|可恶)"), ("愤怒、咬牙切齿", "怒意更重、字字用力", "冷怒、从牙缝里挤出来")),
    (re.compile(r"(哈哈|呵呵|嘿嘿|真有趣|太好了|太棒了|太妙了)"), ("轻快、含着笑", "得意、笑意更明显", "轻松打趣、语尾上扬")),
    (re.compile(r"(救命|好怕|不敢|逃啊|别过来|好吓人|好可怕)"), ("害怕、声音发紧", "惊恐、气都喘不匀", "发怯、往后退着说")),
    (re.compile(r"(对不起|抱歉|原谅我|都是我的错)"), ("愧疚、放轻音量", "自责、声音发闷", "恳求、带着悔意")),
    (re.compile(r"(爱你|想你|亲爱的|喜欢你)"), ("温柔、含情", "柔软、气息贴近", "克制的深情")),
    (re.compile(r"(杀了|去死|血|报仇)"), ("阴冷、压抑", "狠厉、压低声音", "决绝、不带温度")),
    (re.compile(r"(必须|立刻|马上|跟上|出发|快点)"), ("坚定、催促", "急迫、不容商量", "干脆、带着命令感")),
    (re.compile(r"[？?]{2,}|什么[？?]|怎么会|不会吧"), ("惊讶、提高音调", "错愕、反应过来", "难以置信、追问")),
    (re.compile(r"[！!]{2,}"), ("激动、气息上涌", "情绪更冲、加快", "用力、把句子甩出去")),
    (re.compile(r"(……|\.\.\.|不说了|算了|罢了)"), ("低落、欲言又止", "疲惫、把话咽回去", "无奈、声音沉下去")),
)
QUESTION = re.compile(r"[？?]$")
OLLAMA_BATCH = 48


def locked_roles(roles, extra=None):
    locked = {role.get("id") for role in roles or [] if role.get("mode") == "clone"}
    locked.update(extra or ())
    locked.discard(None)
    return locked


def direct_script(roles, cues, voice_templates=None, locked_role_ids=None, cue_ids=None):
    templates = voice_templates or {}
    locked = locked_roles(roles, locked_role_ids)
    target_ids = set(cue_ids) if cue_ids is not None else None
    directed_roles = assign_role_voices(roles, cues, templates, locked)
    directed_cues = assign_cue_emotions(cues, target_ids)
    return directed_roles, directed_cues, {
        "source": "rules",
        "refined": False,
        "role_count": len(directed_roles),
        "cue_count": len(directed_cues),
        "cue_ids": [cue["id"] for cue in directed_cues if target_ids is None or cue.get("id") in target_ids],
        "locked_role_ids": sorted(str(item) for item in locked),
    }


def assign_role_voices(roles, cues, templates, locked_role_ids):
    texts_by_role = {}
    for cue in cues:
        texts_by_role.setdefault(cue.get("role"), []).append(str(cue.get("text") or ""))
    used = [role.get("voice") for role in roles if role.get("id") in locked_role_ids]
    directed = []
    for role in roles:
        copy = dict(role)
        if copy.get("id") in locked_role_ids:
            directed.append(copy)
            continue
        key = infer_voice_key(copy.get("name") or "", texts_by_role.get(copy.get("id"), []), used)
        used.append(key)
        description = customize_description(templates.get(key) or VOICE_LABELS[key], copy.get("name") or "角色")
        copy["voice"] = key
        copy["description"] = description
        copy["style"] = description
        directed.append(copy)
    return directed


def infer_voice_key(name, texts, used):
    blob = f"{name}{' '.join(texts[:8])}"
    if any(marker in name for marker in GIRL_NAME_MARKERS) or "小女孩" in blob:
        return next_unused(GIRL_CYCLE, used, "girl_soft")
    if any(marker in name for marker in BOY_NAME_MARKERS):
        return next_unused(("male_young",), used, "male_young")
    if name in NARRATOR_NAMES:
        return next_unused(MALE_CYCLE, used, "male_adult")
    if any(marker in name for marker in FEMALE_NAME_MARKERS) or name_chars(name, FEMALE_CHARS):
        return next_unused(FEMALE_CYCLE, used, "female_warm")
    if any(marker in name for marker in MALE_NAME_MARKERS) or name_chars(name, MALE_CHARS):
        return next_unused(MALE_CYCLE, used, "male_adult")
    if re.search(r"(姐姐|妹妹|女孩|女人|女士|公主)", blob):
        return next_unused(FEMALE_CYCLE, used, "female_warm")
    if re.search(r"(哥哥|弟弟|男孩|男人|先生|少侠)", blob):
        return next_unused(MALE_CYCLE, used, "male_adult")
    if used.count("male_adult") and "female_warm" not in used:
        return next_unused(FEMALE_CYCLE, used, "female_warm")
    return next_unused(MALE_CYCLE + FEMALE_CYCLE, used, "male_adult")


def name_chars(name, charset):
    return any(char in charset for char in name)


def next_unused(cycle, used, fallback):
    for key in cycle:
        if key not in used:
            return key
    if not cycle:
        return fallback
    return cycle[len([item for item in used if item in cycle]) % len(cycle)]


def customize_description(template, name):
    return f"{template} 这是角色「{name}」的专属声线，要和其他角色明显区分，不要播音腔。"


def assign_cue_emotions(cues, target_ids=None):
    last_by_role = {}
    used_by_role = {}
    variant_index = {}
    directed = []
    for index, cue in enumerate(cues):
        copy = dict(cue)
        if target_ids is not None and copy.get("id") not in target_ids:
            directed.append(copy)
            emotion = str(copy.get("emotion") or "")
            last_by_role[copy.get("role")] = emotion
            used_by_role.setdefault(copy.get("role"), set()).add(emotion)
            continue
        previous = cues[index - 1] if index else None
        nxt = cues[index + 1] if index + 1 < len(cues) else None
        emotion = infer_emotion(copy, previous, nxt, index)
        role_id = copy.get("role")
        emotion = vary_emotion(role_id, emotion, last_by_role, used_by_role, variant_index)
        copy["emotion"] = emotion
        last_by_role[role_id] = emotion
        used_by_role.setdefault(role_id, set()).add(emotion)
        directed.append(copy)
    return directed


def infer_emotion(cue, previous, nxt, index):
    text = str(cue.get("text") or "").strip()
    for pattern, variants in EMOTION_RULES:
        if pattern.search(text):
            return variants[0]
    if QUESTION.search(text):
        return "疑问、带着探询"
    prev_text = str((previous or {}).get("text") or "")
    if previous and QUESTION.search(prev_text):
        return "认真回答、语气清楚"
    if nxt and QUESTION.search(str(nxt.get("text") or "")):
        return "抛出问题前的停顿、引起注意"
    if len(text) <= 8:
        return "短促、点到为止"
    if previous and previous.get("role") != cue.get("role"):
        return "接话、针对上一句作出反应"
    return FALLBACK_EMOTIONS[index % len(FALLBACK_EMOTIONS)]


def vary_emotion(role_id, emotion, last_by_role, used_by_role, variant_index):
    used = used_by_role.get(role_id) or set()
    if emotion != last_by_role.get(role_id) and emotion not in used:
        return emotion
    candidates = []
    for _pattern, variants in EMOTION_RULES:
        if emotion in variants:
            candidates.extend(variants)
            break
    candidates.extend(FALLBACK_EMOTIONS)
    slot = variant_index.get(role_id, 0)
    for _ in range(len(candidates) + 1):
        slot += 1
        candidate = candidates[slot % len(candidates)]
        if candidate != last_by_role.get(role_id) and candidate not in used:
            variant_index[role_id] = slot
            return candidate
    variant_index[role_id] = slot
    return candidates[slot % len(candidates)]


def apply_direction(roles, cues, directed_roles, directed_cues, locked_role_ids=None):
    locked = locked_roles(roles, locked_role_ids)
    role_updates = {item.get("id"): item for item in directed_roles or [] if item.get("id")}
    cue_updates = {}
    for item in directed_cues or []:
        if "id" not in item:
            continue
        try:
            cue_updates[int(item["id"])] = item
        except (TypeError, ValueError):
            continue
    next_roles = []
    for role in roles:
        update = role_updates.get(role.get("id"))
        if not update or role.get("id") in locked:
            next_roles.append(role)
            continue
        voice = update.get("voice") if update.get("voice") in VOICE_KEYS else role.get("voice")
        description = str(update.get("description") or update.get("style") or role.get("description") or "").strip()
        next_roles.append({
            **role,
            "voice": voice or role.get("voice"),
            "description": description or role.get("description"),
            "style": description or role.get("style") or role.get("description"),
        })
    next_cues = []
    for cue in cues:
        try:
            update = cue_updates.get(int(cue["id"]))
        except (TypeError, ValueError, KeyError):
            update = None
        if not update:
            next_cues.append(cue)
            continue
        emotion = str(update.get("emotion") or cue.get("emotion") or "自然").strip() or "自然"
        next_cues.append({**cue, "emotion": emotion})
    return next_roles, next_cues


def ollama_direct_script(roles, cues, model, locked_role_ids=None, cue_ids=None):
    if not shutil.which("ollama"):
        raise RuntimeError("未安装 Ollama")
    locked = locked_roles(roles, locked_role_ids)
    target_ids = [cue["id"] for cue in cues if cue_ids is None or cue.get("id") in set(cue_ids)]
    if not target_ids:
        return roles, cues
    directed_roles = []
    directed_cues = []
    for start in range(0, len(target_ids), OLLAMA_BATCH):
        batch_ids = target_ids[start:start + OLLAMA_BATCH]
        parsed = _ollama_direction_batch(roles, cues, model, locked, batch_ids, start == 0)
        directed_roles.extend(parsed.get("roles") or [])
        directed_cues.extend(parsed.get("cues") or [])
    if not directed_cues:
        raise RuntimeError("本地模型没有返回逐句情绪")
    return apply_direction(roles, cues, directed_roles, directed_cues, locked)


def _ollama_direction_batch(roles, cues, model, locked, batch_ids, include_roles):
    wanted = set(batch_ids)
    payload = {
        "roles": [
            {
                "id": role["id"],
                "name": role.get("name") or role["id"],
                "locked": role["id"] in locked,
            }
            for role in roles
        ],
        "cues": [
            {
                "id": cue["id"],
                "speaker": next((role.get("name") for role in roles if role.get("id") == cue.get("role")), cue.get("role")),
                "role": cue.get("role"),
                "text": cue.get("text") or "",
                "previous": str(cues[index - 1].get("text") or "") if index else "",
                "next": str(cues[index + 1].get("text") or "") if index + 1 < len(cues) else "",
            }
            for index, cue in enumerate(cues)
            if cue.get("id") in wanted
        ],
    }
    voice_help = "、".join(f"{key}（{label}）" for key, label in VOICE_LABELS.items())
    role_rule = (
        "为未锁定角色选择 voice，并写一段中文 description（年龄、性别、音高、气质，40-80字）。"
        "同一角色全程共用一个音色；不同角色必须能听出差别。voice 只能是："
        f"{voice_help}。锁定角色不要输出。roles 数组只包含未锁定角色。"
        if include_roles else
        "这一批不要修改角色音色，roles 请输出空数组。"
    )
    prompt = (
        "你是广播剧配音导演。不要改写台词，不要改角色名，不要合并或删减条目。\n"
        f"1. {role_rule}\n"
        "2. 为每句台词写 emotion：中文表演要求，8-18字，必须结合 previous/next。"
        "同一角色相邻句禁止重复同一个情绪词；不要所有句子都写“自然”。\n"
        "只输出 JSON：{\"roles\":[{\"id\",\"voice\",\"description\"}],\"cues\":[{\"id\",\"emotion\"}]}\n"
        "输入：\n" + json.dumps(payload, ensure_ascii=False)
    )
    schema = {
        "type": "object",
        "properties": {
            "roles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "voice": {"type": "string", "enum": list(VOICE_KEYS)},
                        "description": {"type": "string"},
                    },
                    "required": ["id", "voice", "description"],
                },
            },
            "cues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "emotion": {"type": "string"},
                    },
                    "required": ["id", "emotion"],
                },
            },
        },
        "required": ["cues"],
    }
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": 0,
        "format": schema,
        "options": {"temperature": 0.35, "num_predict": min(8192, max(1200, len(batch_ids) * 80))},
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=min(240, 60 + len(batch_ids) * 4)) as response:
        raw = (json.loads(response.read().decode()).get("response") or "").strip()
    parsed = extract_direction_json(raw)
    if not parsed:
        raise RuntimeError("本地模型未返回可解析的导演结果")
    return parsed


def extract_direction_json(raw):
    cleaned = re.sub(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|.)", "", raw or "")
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.S).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.M).strip()
    decoder = json.JSONDecoder()
    try:
        outer = json.loads(cleaned)
        if isinstance(outer, dict) and ("cues" in outer or "roles" in outer):
            return outer
        if isinstance(outer, list):
            return {"cues": outer, "roles": []}
    except (json.JSONDecodeError, TypeError):
        pass
    for pos, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[pos:])
            if isinstance(value, dict) and ("cues" in value or "roles" in value):
                return value
        except json.JSONDecodeError:
            continue
    for pos, char in enumerate(cleaned):
        if char != "[":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[pos:])
            if isinstance(value, list) and value and isinstance(value[0], dict) and "id" in value[0]:
                return {"cues": value, "roles": []}
        except json.JSONDecodeError:
            continue
    return None
