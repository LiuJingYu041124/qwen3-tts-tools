#!/usr/bin/env python3
from copy import deepcopy
from pathlib import Path


def voice_mode(voice):
    mode = (voice or {}).get("mode") or (voice or {}).get("voice_mode") or "design"
    return "clone" if mode == "clone" else "design"


def first_enabled_cue_by_role(cues):
    first = {}
    for cue in cues or []:
        if cue.get("enabled") is False:
            continue
        role = cue.get("role")
        if role and role not in first:
            first[role] = cue
    return first


def anchor_path(raw_dir, cue):
    if raw_dir is None or not cue:
        return None
    return Path(raw_dir) / f"{int(cue['id']):03d}-{cue['role']}.wav"


def anchored_voice(voice, first_cue, raw_dir):
    """Keep the designed identity, but clone later lines from the first generated take."""
    cloned = deepcopy(voice or {})
    cloned["mode"] = "clone"
    cloned["clone_consent"] = True
    cloned["ref_audio"] = str(anchor_path(raw_dir, first_cue) or "")
    cloned["ref_text"] = str((first_cue or {}).get("text") or "").strip()
    cloned["anchor_cue_id"] = int(first_cue["id"]) if first_cue and first_cue.get("id") is not None else None
    return cloned


def should_anchor_cue(cue, voice, first_cue):
    if voice_mode(voice) != "design" or not first_cue:
        return False
    return int(cue.get("id") or 0) != int(first_cue.get("id") or 0)


def anchored_generation_plan(cues, voices, raw_dir=None):
    first_by_role = first_enabled_cue_by_role(cues)
    plan = []
    for cue in cues or []:
        voice = deepcopy(voices.get(cue.get("role")) or {})
        first = first_by_role.get(cue.get("role"))
        anchored = should_anchor_cue(cue, voice, first)
        if anchored:
            voice = anchored_voice(voice, first, raw_dir)
        plan.append({
            "cue": cue,
            "voice": voice,
            "mode": voice_mode(voice),
            "anchored": anchored,
            "anchor_cue_id": int(first["id"]) if anchored else None,
        })
    return plan
