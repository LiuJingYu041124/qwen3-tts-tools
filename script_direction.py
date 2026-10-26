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
FEMALE_IDENTITY_SUFFIXES = ("姐姐", "妹妹", "阿姨", "奶奶", "妈妈", "母亲", "小姐", "姑娘", "女王", "公主", "夫人", "女士")
MALE_IDENTITY_SUFFIXES = ("哥哥", "弟弟", "叔叔", "伯伯", "爷爷", "爸爸", "父亲", "先生", "公子", "少侠", "少爷")
GIRL_NAME_MARKERS = ("小女孩", "女童", "女孩", "萝莉")
BOY_NAME_MARKERS = ("小男孩", "男童", "男孩")
FEMALE_GIVEN_NAMES = {
    "小雪", "雪儿", "小雨", "小雅", "小美", "小丽", "小芳", "小娜", "小慧", "小静", "小梅",
    "晚晴", "晓雪", "晓雯", "雨桐", "雨晴", "若曦", "紫萱", "梦瑶", "欣怡", "诗涵", "婉儿",
    "奇奇", "爱丽丝", "艾米", "安娜", "露西", "莉莉", "玛丽", "苏菲", "莎拉",
}
MALE_GIVEN_NAMES = {
    "小宇", "浩宇", "小海", "小明", "小强", "小刚", "小杰", "小伟", "小凯", "阿乐", "林深",
    "子轩", "浩然", "宇航", "俊杰", "志强", "建国", "大卫", "杰克", "汤姆", "迈克",
}
FEMALE_CHARS = set("婷芳娜玲燕娟丽雪梅红霞敏慧静柔晴雯淑佳欣倩琳蕾薇莉莎雅妍萱瑶怡")
MALE_CHARS = set("伟强军辉浩宇峰杰勇斌涛刚鹏深凯磊博轩然航俊志建国明海乐")
STRONG_FEMALE_CHARS = set("婷芳娜玲燕娟丽雪梅红霞敏慧静柔晴雯淑佳欣倩琳蕾薇莉莎雅妍萱瑶怡")
STRONG_MALE_CHARS = set("伟强军辉浩峰杰勇斌涛刚鹏凯磊博轩宇航俊志建国")
COMMON_SURNAMES = set(
    "赵钱孙李周吴郑王冯陈蒋沈韩杨朱秦许何吕张孔曹严华金魏陶姜谢邹苏潘葛范彭鲁韦马苗方俞任袁柳史唐"
    "薛雷贺倪汤罗郝安常傅齐康伍余顾孟黄萧尹姚邵汪毛戴宋熊纪舒项祝董梁杜阮蓝季贾江童颜郭梅林钟"
    "徐邱高夏蔡田樊胡霍万卢莫房裘解应宗丁宣邓洪包左石崔吉龚程邢裴陆荣翁甄家封靳段巫乌焦侯全"
    "秋仲宫宁栾甘祖武符刘詹龙叶黎乔闻党翟谭申冉牛尚温庄柴阎连容向易廖辛简饶曾关查游权"
)
COMPOUND_SURNAMES = ("欧阳", "司马", "上官", "诸葛", "东方", "皇甫", "尉迟", "公孙", "慕容", "令狐")
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
    (re.compile(r"(救命|好怕|不敢|逃啊|别过来|好吓人|好可怕|着火了|起火了)"), ("惊慌、声音发紧", "惊恐、气都喘不匀", "发怯、往后退着说")),
    (re.compile(r"(对不起|抱歉|原谅我|都是我的错)"), ("愧疚、放轻音量", "自责、声音发闷", "恳求、带着悔意")),
    (re.compile(r"(爱你|想你|亲爱的|喜欢你)"), ("温柔、含情", "柔软、气息贴近", "克制的深情")),
    (re.compile(r"(杀了|去死|血|报仇)"), ("阴冷、压抑", "狠厉、压低声音", "决绝、不带温度")),
    (re.compile(r"(必须|立刻|马上|跟上|出发|快点|赶快|快走)"), ("坚定、催促", "急迫、不容商量", "干脆、带着命令感")),
    (re.compile(r"[？?]{2,}|什么[？?]|怎么会|不会吧"), ("惊讶、提高音调", "错愕、反应过来", "难以置信、追问")),
    (re.compile(r"[！!]{2,}"), ("激动、气息上涌", "情绪更冲、加快", "用力、把句子甩出去")),
    (re.compile(r"((?:……|\.\.\.)\s*$|不说了|算了|罢了)"), ("低落、欲言又止", "疲惫、把话咽回去", "无奈、声音沉下去")),
)
TEXT_RESULT_EMOTIONS = (
    (re.compile(r"(^|[，。！？!?；;、\s])(?:呼(?:……+|\.\.\.+|[，。！？!?；;、\s]|$)|还好(?!奇)|总算(?:是|熬|等|赶|完成|结束|到了|到达|成功|过关|脱险|安全|没事)|终于没事|有惊无险)"), "松一口气、语气缓和"),
    (re.compile(r"(成功获救|终于获救|已经安全|安全了|扑灭了|熄灭了|赢了|胜利了)"), "兴奋、释然、带着笑意"),
)
SCENE_EMOTION_RULES = (
    (re.compile(r"(扑灭|浇灭|熄灭|成功获救|终于获救|已经安全|安全了|胜利|欢呼|庆祝|惊喜|宝箱|成功)"), "兴奋、带着惊喜和笑意"),
    (re.compile(r"(着火|起火|火灾|浓烟|爆炸|坍塌|追来|扑来|袭来|救命|危险|吓得|惊恐|慌乱)"), "惊慌、急促、声音发紧"),
    (re.compile(r"(怪物|脚步.*近|徘徊|潜伏|躲藏|藏起来|不能出声|屏住呼吸|悄悄靠近|黑暗中)"), "紧张、压低声音、保持警惕"),
    (re.compile(r"(哭|泪|抽泣|失去|离别|受伤|悲伤|难过|绝望)"), "悲伤、声音发闷、强忍情绪"),
    (re.compile(r"(拥抱|安慰|重逢|温暖|轻轻|柔和)"), "温柔、放轻声音、真诚关切"),
    (re.compile(r"(战斗|对峙|冲锋|追逐|命令|决战)"), "坚定、紧迫、富有行动感"),
)
QUESTION = re.compile(r"[？?]$")
OLLAMA_BATCH = 48
DIRECTION_BRIEF_MAX_CHARS = 160
ROLE_PERSONALITY_MAX_CHARS = 500
EMOTION_MAX_CHARS = 72


def normalize_direction_brief(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()[:DIRECTION_BRIEF_MAX_CHARS]


def normalize_role_personality(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()[:ROLE_PERSONALITY_MAX_CHARS]


def normalize_emotion(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()[:EMOTION_MAX_CHARS]


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
    directed_cues = assign_cue_emotions(cues, target_ids, directed_roles)
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
    gender = infer_name_gender(name, texts)
    if any(marker in name for marker in GIRL_NAME_MARKERS) or "小女孩" in blob:
        return next_unused(GIRL_CYCLE, used, "girl_soft")
    if any(marker in name for marker in BOY_NAME_MARKERS):
        return next_unused(("male_young",), used, "male_young")
    if name in NARRATOR_NAMES:
        return next_unused(MALE_CYCLE, used, "male_adult")
    if gender == "female":
        return next_unused(FEMALE_CYCLE, used, "female_warm")
    if gender == "male":
        return next_unused(MALE_CYCLE, used, "male_adult")
    return next_unused(MALE_CYCLE, used, "male_adult")


def infer_name_gender(name, texts=None):
    name = str(name or "").strip()
    blob = f"{name}{' '.join(str(item or '') for item in (texts or [])[:8])}"
    if name in FEMALE_GIVEN_NAMES or name.endswith(FEMALE_IDENTITY_SUFFIXES):
        return "female"
    if name in MALE_GIVEN_NAMES or name.endswith(MALE_IDENTITY_SUFFIXES):
        return "male"
    if any(marker in name for marker in GIRL_NAME_MARKERS) or re.search(r"(女童|女孩)", blob):
        return "female"
    if any(marker in name for marker in BOY_NAME_MARKERS) or re.search(r"(男童|男孩)", blob):
        return "male"
    if any(marker in name for marker in FEMALE_NAME_MARKERS + MALE_NAME_MARKERS):
        return None
    given_name = given_name_part(name)
    strong_female = name_char_score(given_name, STRONG_FEMALE_CHARS)
    strong_male = name_char_score(given_name, STRONG_MALE_CHARS)
    if strong_female and not strong_male:
        return "female"
    if strong_male and not strong_female:
        return "male"
    if name.startswith("小") and len(name) <= 3:
        return None
    if re.search(r"(姐姐|妹妹|女孩|女人|女士|公主|女儿|妻子|妈妈|母亲)", blob):
        return "female"
    if re.search(r"(哥哥|弟弟|男孩|男人|先生|少侠|儿子|丈夫|爸爸|父亲)", blob):
        return "male"
    female_score = name_char_score(given_name, FEMALE_CHARS)
    male_score = name_char_score(given_name, MALE_CHARS)
    if female_score >= 2 and male_score == 0:
        return "female"
    if male_score >= 2 and female_score == 0:
        return "male"
    return None


def given_name_part(name):
    name = str(name or "").strip()
    for surname in COMPOUND_SURNAMES:
        if name.startswith(surname) and len(name) > len(surname):
            return name[len(surname):]
    if len(name) in {2, 3, 4} and name[:1] in COMMON_SURNAMES:
        return name[1:]
    return name


def name_char_score(name, charset):
    return sum(char in charset for char in str(name or ""))


def next_unused(cycle, used, fallback):
    for key in cycle:
        if key not in used:
            return key
    if not cycle:
        return fallback
    return cycle[len([item for item in used if item in cycle]) % len(cycle)]


def customize_description(template, name):
    return f"{template} 这是角色「{name}」的专属声线，要和其他角色明显区分，不要播音腔。"


def assign_cue_emotions(cues, target_ids=None, roles=None):
    last_by_role = {}
    variant_index = {}
    directed = []
    role_map = {role.get("id"): role for role in roles or [] if role.get("id")}
    for index, cue in enumerate(cues):
        copy = dict(cue)
        if target_ids is not None and copy.get("id") not in target_ids:
            directed.append(copy)
            last_by_role[copy.get("role")] = str(copy.get("emotion") or "")
            continue
        previous = cues[index - 1] if index else None
        nxt = cues[index + 1] if index + 1 < len(cues) else None
        emotion = infer_emotion(copy, previous, nxt, index)
        role_id = copy.get("role")
        if not (copy.get("scene_before") or copy.get("scene_after")):
            emotion = vary_emotion(role_id, emotion, last_by_role, variant_index)
        copy["emotion"] = normalize_emotion(emotion)
        last_by_role[role_id] = copy["emotion"]
        directed.append(copy)
    return apply_personality_rules(role_map, directed, target_ids)


def apply_personality_rules(role_map, cues, target_ids=None):
    if not role_map or not any(normalize_role_personality(role.get("personality")) for role in role_map.values()):
        return cues
    result = []
    for cue in cues:
        copy = dict(cue)
        if target_ids is not None and copy.get("id") not in target_ids:
            result.append(copy)
            continue
        role = role_map.get(copy.get("role")) or {}
        personality = normalize_role_personality(role.get("personality"))
        if personality:
            copy["emotion"] = personality_emotion_instruction(personality, copy.get("emotion"))
        result.append(copy)
    return result


def personality_reaction_direction(personality):
    value = normalize_role_personality(personality)
    rules = (
        (("克制", "冷静", "沉稳", "内敛", "不吼", "不喊", "压低", "压住", "低声", "平稳", "慢热", "寡言", "外冷"),
         "收住外露，音量偏低，停顿后再落重音"),
        (("外向", "活泼", "爽朗", "开朗", "直接", "冲动", "热情", "爱笑", "情绪鲜明"),
         "反应更快更外放，语速稍快，语调与重音更鲜明"),
        (("温柔", "柔和", "体贴", "善解人意", "心软", "关切"),
         "放轻音量，语速稍慢，气息柔和并留出停顿"),
        (("害羞", "胆小", "自卑", "社恐", "犹豫", "谨慎"),
         "先留短停顿，语气含蓄，情绪不过度外放"),
        (("果断", "强势", "责任感", "理性", "干练", "坚定", "领导", "判断"),
         "先判断后表达，句子干脆，重点清楚"),
    )
    for markers, direction in rules:
        if any(marker in value for marker in markers):
            return direction
    return "保持一贯的情绪外露程度、节奏和重音"


def personality_emotion_instruction(personality, emotion):
    personality = normalize_role_personality(personality)
    emotion = normalize_emotion(emotion) or "自然"
    if not personality or ("按人物性格" in emotion and "不脱离一贯说话方式" in emotion):
        return emotion
    suffix = f"；按人物性格{personality_reaction_direction(personality)}，不脱离一贯说话方式"
    available = max(1, EMOTION_MAX_CHARS - len(suffix))
    emotion = emotion[:available].rstrip("，、；。 ") or "自然"
    return normalize_emotion(f"{emotion}{suffix}")


def infer_emotion(cue, previous, nxt, index):
    text = str(cue.get("text") or "").strip()
    scene = " ".join(
        part for part in (
            str(cue.get("scene_before") or "").strip(),
            str(cue.get("scene_after") or "").strip(),
        )
        if part
    )
    combined = f"{scene} {text}".strip()
    for pattern, emotion in TEXT_RESULT_EMOTIONS:
        if pattern.search(text):
            return emotion
    for pattern, variants in EMOTION_RULES:
        if pattern.search(text):
            return variants[0]
    for pattern, emotion in SCENE_EMOTION_RULES:
        if pattern.search(combined):
            return emotion
    if QUESTION.search(text):
        if re.search(r"(没事|还好|受伤|怎么了|怎么啦|为什么|怎么办|真的)", text):
            return "担心、关切地探询"
        return "疑问、带着探询"
    if re.search(r"[！!]$", text):
        return "激动、语气上扬、重点用力"
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


def vary_emotion(role_id, emotion, last_by_role, variant_index):
    if emotion != last_by_role.get(role_id):
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
        if candidate != last_by_role.get(role_id):
            variant_index[role_id] = slot
            return candidate
    variant_index[role_id] = slot
    return emotion


def protect_obvious_gender(role, voice, description):
    gender = infer_name_gender(role.get("name") or "")
    if gender == "female" and voice not in FEMALE_CYCLE + GIRL_CYCLE:
        voice = role.get("voice") if role.get("voice") in FEMALE_CYCLE + GIRL_CYCLE else "female_warm"
    elif gender == "male" and voice not in MALE_CYCLE:
        voice = role.get("voice") if role.get("voice") in MALE_CYCLE else "male_young"
    if not gender:
        return voice, description
    expected = "女" if gender == "female" else "男"
    opposite = "男" if gender == "female" else "女"
    if expected not in description or opposite in description:
        label = "年轻女声" if gender == "female" else "年轻男声"
        template = VOICE_LABELS.get(voice) or label
        description = f"{template}，明确为{label}。这是角色「{role.get('name') or '角色'}」的专属声线。"
    return voice, description


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
        voice, description = protect_obvious_gender(role, voice, description)
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
        emotion = normalize_emotion(update.get("emotion") or cue.get("emotion") or "自然") or "自然"
        next_cues.append({**cue, "emotion": emotion})
    return next_roles, next_cues


def ollama_direct_script(roles, cues, model, locked_role_ids=None, cue_ids=None, direction_brief=""):
    if not shutil.which("ollama"):
        raise RuntimeError("未安装 Ollama")
    locked = locked_roles(roles, locked_role_ids)
    target_ids = [cue["id"] for cue in cues if cue_ids is None or cue.get("id") in set(cue_ids)]
    if not target_ids:
        return roles, cues
    directed_roles = []
    directed_cues = []
    direction_brief = normalize_direction_brief(direction_brief)
    for start in range(0, len(target_ids), OLLAMA_BATCH):
        batch_ids = target_ids[start:start + OLLAMA_BATCH]
        parsed = _ollama_direction_batch(
            roles,
            cues,
            model,
            locked,
            batch_ids,
            start == 0,
            direction_brief,
        )
        directed_roles.extend(parsed.get("roles") or [])
        directed_cues.extend(parsed.get("cues") or [])
    if not directed_cues:
        raise RuntimeError("本地模型没有返回逐句情绪")
    return apply_direction(roles, cues, directed_roles, directed_cues, locked)


def _ollama_direction_batch(roles, cues, model, locked, batch_ids, include_roles, direction_brief=""):
    wanted = set(batch_ids)
    direction_brief = normalize_direction_brief(direction_brief)
    payload = {
        "direction_brief": direction_brief,
        "roles": [
            {
                "id": role["id"],
                "name": role.get("name") or role["id"],
                "locked": role["id"] in locked,
                "personality": normalize_role_personality(role.get("personality")),
            }
            for role in roles
        ],
        "cues": [
            {
                "id": cue["id"],
                "speaker": next((role.get("name") for role in roles if role.get("id") == cue.get("role")), cue.get("role")),
                "role": cue.get("role"),
                "text": cue.get("text") or "",
                "scene_before": str(cue.get("scene_before") or ""),
                "scene_after": str(cue.get("scene_after") or ""),
                "previous": str(cues[index - 1].get("text") or "") if index else "",
                "next": str(cues[index + 1].get("text") or "") if index + 1 < len(cues) else "",
            }
            for index, cue in enumerate(cues)
            if cue.get("id") in wanted
        ],
    }
    voice_help = "、".join(f"{key}（{label}）" for key, label in VOICE_LABELS.items())
    gender_hints = [
        f"{role.get('name')}={infer_name_gender(role.get('name') or '')}"
        for role in roles
        if infer_name_gender(role.get("name") or "")
    ]
    role_rule = (
        "为未锁定角色选择 voice，并写一段中文 description（年龄、性别、音高、气质，40-80字）。"
        "明显女性姓名只能选 female_warm、female_brave、girl_soft、girl_lively；明显男性姓名只能选 male_adult、male_young。"
        f"已识别的姓名性别约束：{('、'.join(gender_hints) or '无')}。"
        "同一角色全程共用一个音色；不同角色必须能听出差别。voice 只能是："
        f"{voice_help}。锁定角色不要输出。roles 数组只包含未锁定角色。"
        if include_roles else
        "这一批不要修改角色音色，roles 请输出空数组。"
    )
    direction_rule = (
        f"用户指定整部作品的配音方向是“{direction_brief}”。这是一条全局硬约束：角色年龄感、音色、表演强度、节奏和措辞都要符合它；"
        "例如少儿内容应清楚、亲切、生动但不要幼稚尖叫。不要在返回文本中机械重复该方向。\n"
        if direction_brief else
        "用户未指定整部作品的额外配音方向，请只依据正文、角色和场景判断。\n"
    )
    prompt = (
        "你是广播剧配音导演。不要改写台词，不要改角色名，不要合并或删减条目。\n"
        + direction_rule
        + f"1. {role_rule}\n"
        "2. 为每句台词写 emotion：中文、可直接执行的表演要求，12-30字，必须结合标点、scene_before、scene_after、previous 和 next。"
        "角色若提供 personality，先判断此人遇到当前事件会怎样反应，再把稳定性格翻译为音量、语速、停顿、气息、重音或情绪外露程度；"
        "不要把性格原文复制进 emotion，不要让沉稳者突然吼叫、内向者无故外放，也不要让所有台词都维持同一种情绪。"
        "火灾、追逐、怪物逼近、哭泣等场景仍要明确落到惊慌、紧张、悲伤等当前情绪，但表达方式必须符合人物一贯反应。"
        "同一角色相邻句可以在场景连续时保持同类情绪，但不要机械地全部写“自然”。\n"
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
