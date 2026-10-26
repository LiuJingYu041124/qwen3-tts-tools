#!/usr/bin/env python3
import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path


CLONE_GENERATION_POLICY = "clone-length-guard-v3-personality-direction"
CLONE_TOKENS_PER_SECOND = 12.5
CLONE_MIN_MAX_SECONDS = 4.5
CLONE_ABSOLUTE_MAX_SECONDS = 28.0
CLONE_MIN_MAX_TOKENS = 72
CLONE_ABSOLUTE_MAX_TOKENS = 420


def voice_mode(voice):
    mode = (voice or {}).get("mode") or (voice or {}).get("voice_mode") or "design"
    return "clone" if mode == "clone" else "design"


def cue_generation_plan(cues, voices):
    return [
        {
            "cue": cue,
            "voice": deepcopy(voices.get(cue.get("role")) or {}),
            "mode": voice_mode(voices.get(cue.get("role")) or {}),
        }
        for cue in cues or []
    ]


def estimate_spoken_seconds(text, language="auto"):
    value = str(text or "").strip()
    if not value:
        return 0.0
    cjk_count = len(re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", value))
    words = re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", value)
    other_spoken = len(re.findall(r"[^\s\W_]", value, flags=re.UNICODE)) - cjk_count - sum(len(word) for word in words)
    comma_pauses = len(re.findall(r"[,，、;；:]", value))
    sentence_pauses = len(re.findall(r"[.!?。！？…]", value))
    estimated = cjk_count / 3.5 + len(words) / 2.35 + max(0, other_spoken) / 3.5
    estimated += comma_pauses * 0.18 + sentence_pauses * 0.32
    return max(0.45, estimated)


def clone_generation_limits(text, language="auto", slot_seconds=None):
    estimated = estimate_spoken_seconds(text, language)
    allowed_seconds = max(CLONE_MIN_MAX_SECONDS, estimated * 2.35 + 1.25)
    if slot_seconds is not None:
        try:
            slot = max(0.0, float(slot_seconds))
        except (TypeError, ValueError):
            slot = 0.0
        if slot:
            allowed_seconds = min(allowed_seconds, max(CLONE_MIN_MAX_SECONDS, slot * 4.0 + 1.0))
    allowed_seconds = min(CLONE_ABSOLUTE_MAX_SECONDS, allowed_seconds)
    max_tokens = math.ceil((allowed_seconds + 0.75) * CLONE_TOKENS_PER_SECOND)
    max_tokens = max(CLONE_MIN_MAX_TOKENS, min(CLONE_ABSOLUTE_MAX_TOKENS, max_tokens))
    return {
        "policy": CLONE_GENERATION_POLICY,
        "estimated_seconds": round(estimated, 3),
        "max_seconds": round(allowed_seconds, 3),
        "max_tokens": int(max_tokens),
    }


def clone_generation_options(text, language="auto", slot_seconds=None, base_options=None):
    limits = clone_generation_limits(text, language, slot_seconds)
    return {**(base_options or {}), "max_tokens": limits["max_tokens"]}, limits


def clone_performance_profile(voice, cue):
    personality = str((voice or {}).get("personality") or "")
    emotion = str((cue or {}).get("emotion") or "")
    restrained = ("克制", "冷静", "沉稳", "内敛", "不吼", "压低", "低声", "平稳", "慢热", "谨慎")
    expressive = ("外向", "活泼", "爽朗", "直接", "冲动", "热情", "爱笑", "情绪鲜明", "轻快", "含着笑", "语气上扬")
    gentle = ("温柔", "柔和", "体贴", "安慰", "悲伤", "低落", "疲惫", "放轻", "关切")
    intense = ("兴奋", "激动", "惊慌", "愤怒", "急迫", "催促", "命令", "坚定", "惊讶")
    if any(marker in personality for marker in restrained):
        return "restrained"
    if any(marker in personality for marker in expressive):
        return "expressive"
    if any(marker in emotion for marker in restrained):
        return "restrained"
    if any(marker in emotion for marker in expressive):
        return "expressive"
    if any(marker in emotion for marker in gentle):
        return "gentle"
    if any(marker in emotion for marker in intense):
        return "expressive"
    return "natural"


def clone_performance_options(voice, cue, base_options=None):
    options = dict(base_options or {})
    profile = clone_performance_profile(voice, cue)
    if profile == "restrained":
        options["temperature"] = min(float(options.get("temperature", 0.72)), 0.66)
        options["top_k"] = min(int(options.get("top_k", 36)), 30)
        options["top_p"] = min(float(options.get("top_p", 0.9)), 0.86)
    elif profile == "gentle":
        options["temperature"] = min(float(options.get("temperature", 0.72)), 0.69)
        options["top_k"] = min(int(options.get("top_k", 36)), 34)
    elif profile == "expressive":
        options["temperature"] = max(float(options.get("temperature", 0.72)), 0.78)
        options["top_k"] = max(int(options.get("top_k", 36)), 42)
        options["top_p"] = max(float(options.get("top_p", 0.9)), 0.92)
    return options


def clone_performance_text(voice, cue):
    text = str((cue or {}).get("text") or "").strip()
    if not text:
        return text
    if (cue or {}).get("_punctuation_review"):
        return re.sub(r"[！!？?]+", "，", text).rstrip("，。.") + "。"
    emotion = str((cue or {}).get("emotion") or "")
    personality = str((voice or {}).get("personality") or "")
    profile = clone_performance_profile(voice, cue)
    stem = re.sub(r"[。！？!?…\.]+$", "", text).rstrip()
    if not stem:
        return text
    if re.search(r"(疑问|追问|探询|难以置信|错愕)", emotion) or re.search(r"[？?]\s*$", text):
        ending = "？"
    elif re.search(r"(欲言又止|犹豫|哽咽|疲惫|低落|无奈|停顿)", emotion):
        ending = "……"
    elif re.search(r"(惊慌|急迫|激动|兴奋|愤怒|催促|命令|坚定|用力)", emotion):
        ending = "！"
    elif profile in {"restrained", "gentle"}:
        ending = "。"
    else:
        return text
    if profile == "restrained" or re.search(r"(不吼|不喊|压住|克制)", personality):
        ending = "！" if ending == "！" and re.search(r"(催促|命令|坚定|急迫)", emotion) else ("？" if ending == "？" else "。")
    return stem + ending


def cue_generation_options(mode, cue, language="auto", timeline_mode=None, clone_options=None, design_options=None, voice=None):
    if mode != "clone":
        return dict(design_options or {}), None
    slot_seconds = None
    if timeline_mode == "fixed":
        try:
            slot_seconds = max(0.0, float(cue.get("end")) - float(cue.get("start")))
        except (TypeError, ValueError):
            slot_seconds = None
    performance_text = clone_performance_text(voice, cue)
    performance_options = clone_performance_options(voice, cue, clone_options)
    return clone_generation_options(performance_text, language, slot_seconds, performance_options)


def cue_generation_cache_key(cue, voice, language, model, generate_options, reference=None, limits=None):
    cache_options = {**generate_options, "_length_guard": limits} if limits else generate_options
    return generation_cache_key(cue, voice, language, model, cache_options, reference)


def validate_clone_output_duration(text, actual_seconds, language="auto", slot_seconds=None, limits=None):
    policy = limits or clone_generation_limits(text, language, slot_seconds)
    actual = max(0.0, float(actual_seconds or 0.0))
    if actual <= 0.08:
        raise RuntimeError("克隆模型没有生成有效音频")
    if actual > float(policy["max_seconds"]):
        raise RuntimeError(
            f"克隆结果时长异常：生成 {actual:.2f} 秒，当前台词最多允许 {float(policy['max_seconds']):.2f} 秒。"
            "可能是参考逐字稿与音频不一致，请重新播放参考音频、校正并确认逐字稿。"
        )
    return policy


def generation_cache_key(cue, voice, language, model, generate_options, reference=None):
    mode = voice_mode(voice)
    payload = {
        "text": str(cue.get("text") or ""),
        "emotion": str(cue.get("emotion") or ""),
        "role": cue.get("role"),
        "language": language,
        "mode": mode,
        "description": voice.get("description") or voice.get("style") or "",
        "personality": voice.get("personality") or "",
        "project_direction": voice.get("project_direction") or "",
        "ref_text": voice.get("ref_text") or "",
        "reference_revision": voice.get("reference_revision") or voice.get("reference_sha256") or "",
        "reference": reference,
        "model": model,
        "generate": generate_options,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def voice_design_instruction(voice, cue):
    description = str(voice.get("description") or voice.get("style") or "自然真实的声音").strip()
    personality = " ".join(str(voice.get("personality") or "").split()).strip()
    emotion = str(cue.get("emotion") or "自然").strip()
    direction = " ".join(str(voice.get("project_direction") or "").split()).strip()
    direction_instruction = f"整部作品的配音方向：{direction}。" if direction else ""
    personality_instruction = (
        f"人物稳定性格：{personality}。把性格落实为情绪外露程度、节奏、停顿、气息和重音，不要机械朗读性格标签。"
        if personality else ""
    )
    return (
        f"{direction_instruction}{description} {personality_instruction}当前台词的表演要求：{emotion}。"
        "保持角色音色稳定，严格只说给定台词，不增加解释或其他内容。疑问句末尾的语气助词必须清晰、完整地发音，尤其是句末疑问助词，必须按普通话轻声清晰完整地读完，韵母略有支撑但不能拖成长音，不能读成“猫”或吞字。"
    )


def generate_cue_results(model, mode, voice, cue, language, clone_options, design_options):
    text = str(cue.get("text") or "").strip()
    if not text:
        raise RuntimeError("台词为空")
    if mode == "clone":
        if not voice.get("clone_consent"):
            raise RuntimeError("声音克隆尚未确认已获授权")
        ref_value = str(voice.get("ref_audio") or "").strip()
        ref_audio = Path(ref_value).expanduser() if ref_value else None
        if not ref_audio or not ref_audio.is_file():
            raise RuntimeError("克隆参考音频不存在")
        ref_text = str(voice.get("ref_text") or "").strip()
        if not ref_text:
            raise RuntimeError("克隆参考逐字稿为空")
        current_revision = str(voice.get("reference_revision") or "")
        if not current_revision or not voice.get("ref_text_confirmed") or str(voice.get("ref_text_confirmed_revision") or "") != current_revision:
            raise RuntimeError("克隆参考逐字稿尚未核听确认")
        return model.generate(
            text=clone_performance_text(voice, cue),
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            lang_code=language,
            **clone_options,
            verbose=False,
        )
    instruction = voice_design_instruction(voice, cue)
    # Narrow, waveform-verified phrase candidate: preserve every spoken character.
    # Do not generalize across voices/phrases from a single successful seed.
    if re.sub(r"[。！？!?….,，\s]+$", "", text) == "环境勉强有几分相似":
        instruction += "句末最后一个音节的韵母自然支撑后再收声。"
    return model.generate_voice_design(
        text=text,
        instruct=instruction,
        language=language,
        **design_options,
        verbose=False,
    )
