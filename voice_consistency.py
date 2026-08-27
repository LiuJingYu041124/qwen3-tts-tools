#!/usr/bin/env python3
import hashlib
import json
from copy import deepcopy
from pathlib import Path


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


def generation_cache_key(cue, voice, language, model, generate_options, reference=None):
    mode = voice_mode(voice)
    payload = {
        "text": str(cue.get("text") or ""),
        "emotion": str(cue.get("emotion") or ""),
        "role": cue.get("role"),
        "language": language,
        "mode": mode,
        "description": voice.get("description") or voice.get("style") or "",
        "ref_text": voice.get("ref_text") or "",
        "reference": reference,
        "model": model,
        "generate": generate_options,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def voice_design_instruction(voice, cue):
    description = str(voice.get("description") or voice.get("style") or "自然真实的声音").strip()
    emotion = str(cue.get("emotion") or "自然").strip()
    return (
        f"{description} 当前台词的表演要求：{emotion}。"
        "保持角色音色稳定，严格只说给定台词，不增加解释或其他内容。"
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
        ref_text = str(voice.get("ref_text") or "").strip() or None
        return model.generate(
            text=text,
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            lang_code=language,
            **clone_options,
            verbose=False,
        )
    return model.generate_voice_design(
        text=text,
        instruct=voice_design_instruction(voice, cue),
        language=language,
        **design_options,
        verbose=False,
    )
