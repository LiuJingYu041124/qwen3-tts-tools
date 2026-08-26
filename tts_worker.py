#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model
from scipy.io import wavfile

from voice_consistency import anchored_generation_plan

VOICE_DESIGN_MODEL = os.environ.get(
    "QWEN3_TTS_VOICE_DESIGN_MODEL",
    os.environ.get("QWEN3_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"),
)
VOICE_CLONE_MODEL = os.environ.get(
    "QWEN3_TTS_VOICE_CLONE_MODEL",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
)
FFMPEG = os.environ.get("FFMPEG") or (
    "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"
    if Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg").exists()
    else shutil.which("ffmpeg")
) or "ffmpeg"
FFPROBE = os.environ.get("FFPROBE") or (
    "/opt/homebrew/opt/ffmpeg-full/bin/ffprobe"
    if Path("/opt/homebrew/opt/ffmpeg-full/bin/ffprobe").exists()
    else shutil.which("ffprobe")
) or "ffprobe"
SAMPLE_RATE = 24000
LANGUAGES = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean"}
MAX_FIXED_TEMPO = 4.0


def comfortable_fixed_tempo(slot):
    return 1.45 if slot >= 0.85 else (1.85 if slot >= 0.60 else 2.5)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def duration(path):
    return float(
        subprocess.check_output(
            [
                FFPROBE,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            text=True,
        ).strip()
    )


def atempo_chain(value):
    filters = []
    while value > 2:
        filters.append("atempo=2.0")
        value /= 2
    while value < 0.5:
        filters.append("atempo=0.5")
        value /= 0.5
    filters.append(f"atempo={value:.8f}")
    return ",".join(filters)


def voice_mode(voice):
    mode = voice.get("mode") or voice.get("voice_mode") or "design"
    return "clone" if mode == "clone" else "design"


def reference_signature(voice):
    ref_value = str(voice.get("ref_audio") or "").strip()
    ref = Path(ref_value).expanduser() if ref_value else None
    if not ref or not ref.is_file():
        return None
    stat = ref.stat()
    return {"path": str(ref.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


CLONE_GENERATE = {
    "temperature": 0.72,
    "top_k": 36,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "max_tokens": 1400,
}
DESIGN_GENERATE = {
    "temperature": 0.7,
    "top_k": 36,
    "top_p": 0.9,
    "repetition_penalty": 1.18,
    "max_tokens": 1400,
}


def cache_key(cue, voice, language, models):
    mode = voice_mode(voice)
    anchored = voice.get("anchor_cue_id") is not None
    payload = {
        "text": str(cue.get("text") or ""),
        "emotion": str(cue.get("emotion") or ""),
        "role": cue.get("role"),
        "language": language,
        "mode": mode,
        "description": voice.get("description") or voice.get("style") or "",
        "ref_text": voice.get("ref_text") or "",
        "reference": None if anchored else reference_signature(voice),
        "model": models[mode],
        "generate": CLONE_GENERATE if mode == "clone" else DESIGN_GENERATE,
        "anchor_cue_id": voice.get("anchor_cue_id"),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def collect_audio(results):
    parts = []
    sample_rate = None
    for result in results:
        parts.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
        sample_rate = result.sample_rate
    if not parts or not sample_rate:
        raise RuntimeError("模型没有返回音频")
    return np.clip(np.concatenate(parts), -1, 1), sample_rate


def generate_audio(model, mode, voice, cue, language, seed):
    mx.random.seed(seed)
    text = str(cue.get("text") or "").strip()
    emotion = str(cue.get("emotion") or "自然").strip()
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
        return collect_audio(
            model.generate(
                text=text,
                ref_audio=str(ref_audio),
                ref_text=ref_text,
                lang_code=language,
                **CLONE_GENERATE,
                verbose=False,
            )
        )
    description = str(voice.get("description") or voice.get("style") or "自然真实的声音").strip()
    instruction = (
        f"{description} 当前台词的表演要求：{emotion}。"
        "保持角色音色稳定，严格只说给定台词，不增加解释或其他内容。"
    )
    return collect_audio(
        model.generate_voice_design(
            text=text,
            instruct=instruction,
            language=language,
            **DESIGN_GENERATE,
            verbose=False,
        )
    )


def clean_audio(source, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    filters = (
        "silenceremove=start_periods=1:start_duration=0.02:start_threshold=-46dB:"
        "stop_periods=-1:stop_duration=0.06:stop_threshold=-46dB,"
        "areverse,silenceremove=start_periods=1:start_duration=0.02:"
        "start_threshold=-46dB,areverse"
    )
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-af",
            filters,
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-c:a",
            "pcm_f32le",
            str(output),
        ],
        check=True,
    )


def load_and_generate(model_name, mode, pending, language, models):
    if not pending:
        return
    print(f"LOAD {mode} {model_name}", flush=True)
    model = load_model(model_name)
    for cue, voice, raw_path, meta_path, fingerprint, item in pending:
        role = cue["role"]
        if item.get("anchored"):
            ref = Path(str(voice.get("ref_audio") or "")).expanduser()
            if not ref.is_file():
                raise RuntimeError(f"角色“{role}”还没有首句定音样本，无法保持后续台词音色一致")
        audio, sample_rate = generate_audio(
            model,
            mode,
            voice,
            cue,
            language,
            930000 + int(cue["id"]),
        )
        wavfile.write(raw_path, sample_rate, audio)
        meta_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "model": models[mode],
                    "mode": mode,
                    "role": role,
                    "anchored": bool(item.get("anchored")),
                    "anchor_cue_id": item.get("anchor_cue_id"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"GENERATED {cue['id']} {role} {mode}{' anchored' if item.get('anchored') else ''}", flush=True)
    del model
    gc.collect()
    mx.clear_cache()


def build_project(args):
    root = Path(args.root)
    raw_dir = root / "raw"
    clean_dir = root / "clean"
    fit_dir = root / "fit"
    for directory in (raw_dir, clean_dir, fit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cues = read_json(root / "manifest.json", [])
    voices = read_json(root / "voices.json", {})
    models = {"design": args.design_model, "clone": args.clone_model}
    language = LANGUAGES.get(args.language, args.language)
    pending = {"design": [], "clone": []}
    cache_hits = set()
    generation_plan = {}
    dirty_roles = set()

    for item in anchored_generation_plan(cues, voices, raw_dir):
        cue = item["cue"]
        role = cue.get("role")
        if role not in voices:
            raise RuntimeError(f"台词 {cue.get('id')} 使用了不存在的角色：{role}")
        cue_id = int(cue["id"])
        generation_plan[cue_id] = item
        raw_path = raw_dir / f"{cue_id:03d}-{role}.wav"
        meta_path = raw_path.with_suffix(".json")
        fingerprint = cache_key(cue, item["voice"], language, models)
        cached = read_json(meta_path, {})
        reuse_hit = False
        if args.reuse and raw_path.exists() and cached.get("fingerprint") == fingerprint:
            try:
                reuse_hit = duration(raw_path) > 0.1
            except Exception:
                reuse_hit = False
        if item.get("anchored") and role in dirty_roles:
            reuse_hit = False
        if reuse_hit:
            cache_hits.add(cue_id)
            continue
        if not item.get("anchored") and item.get("mode") == "design":
            dirty_roles.add(role)
        pending[item["mode"]].append((cue, item["voice"], raw_path, meta_path, fingerprint, item))

    load_and_generate(models["design"], "design", pending["design"], language, models)
    load_and_generate(models["clone"], "clone", pending["clone"], language, models)

    status = []
    cursor = 0.0
    for cue in cues:
        cue_id = int(cue["id"])
        role = cue["role"]
        item = generation_plan.get(cue_id) or {}
        mode = item.get("mode") or voice_mode(voices[role])
        raw_path = raw_dir / f"{cue_id:03d}-{role}.wav"
        clean_path = clean_dir / f"{cue_id:03d}.wav"
        fit_path = fit_dir / f"{cue_id:03d}.wav"
        clean_audio(raw_path, clean_path)
        clean_duration = duration(clean_path)

        if args.timeline_mode == "natural":
            start = cursor
            applied_tempo = max(0.5, min(2.0, float(cue.get("speed") or 1.0)))
            expected_duration = clean_duration / applied_tempo
            fade_out_start = max(0.0, expected_duration - 0.018)
            filters = [atempo_chain(applied_tempo), "afade=t=in:st=0:d=0.006", f"afade=t=out:st={fade_out_start:.6f}:d=0.018"]
            subprocess.run(
                [
                    FFMPEG,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(clean_path),
                    "-af",
                    ",".join(filters),
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_f32le",
                    str(fit_path),
                ],
                check=True,
            )
            fitted_duration = duration(fit_path)
            end = start + fitted_duration
            gap_after = max(0.0, float(cue.get("gap_after", args.gap)))
            cursor = end + gap_after
        else:
            start = float(cue["start"])
            end = float(cue["end"])
            slot = max(0.18, end - start - 0.035)
            required = clean_duration / slot
            comfortable_tempo = comfortable_fixed_tempo(slot)
            requested_tempo = max(0.86, required)
            if requested_tempo > MAX_FIXED_TEMPO:
                raise RuntimeError(
                    f"台词{cue_id}无法装入时间槽：清理后{clean_duration:.2f}s，槽位{slot:.2f}s，"
                    f"需要{required:.2f}倍，安全上限{MAX_FIXED_TEMPO:.2f}倍。请缩短台词或调宽时间。"
                )
            applied_tempo = min(MAX_FIXED_TEMPO, requested_tempo)
            fit_warning = required > comfortable_tempo
            if fit_warning:
                print(
                    f"WARNING cue={cue_id} compressed={applied_tempo:.2f}x "
                    f"comfortable={comfortable_tempo:.2f}x slot={slot:.2f}s text={cue['text']}",
                    flush=True,
                )
            filters = [
                atempo_chain(applied_tempo),
                f"apad=whole_dur={slot:.6f}",
                f"atrim=duration={slot:.6f}",
                "afade=t=in:st=0:d=0.006",
            ]
            subprocess.run(
                [
                    FFMPEG,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(clean_path),
                    "-af",
                    ",".join(filters),
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_f32le",
                    str(fit_path),
                ],
                check=True,
            )
            fitted_duration = duration(fit_path)

        row = {
            **cue,
            "start": round(start, 3),
            "end": round(end, 3),
            "raw_duration": clean_duration,
            "tempo": applied_tempo,
            "duration": fitted_duration,
            "end_error": (start + fitted_duration) - end,
            "voice_mode": mode,
            "anchored": bool(item.get("anchored")),
            "anchor_cue_id": item.get("anchor_cue_id"),
            "cache_hit": cue_id in cache_hits,
            "fit_warning": fit_warning if args.timeline_mode == "fixed" else False,
            "comfortable_tempo": comfortable_tempo if args.timeline_mode == "fixed" else None,
        }
        status.append(row)
        (root / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2))
        print(cue_id, role, mode, cue["text"], f"tempo={applied_tempo:.3f}", flush=True)

    final_end = max((float(row["start"]) + float(row["duration"]) for row in status), default=0.0)
    timeline_duration = max(float(args.duration or 0), final_end)
    timeline = np.zeros(max(1, round(timeline_duration * SAMPLE_RATE)), np.float32)
    for row in status:
        _, audio = wavfile.read(fit_dir / f"{int(row['id']):03d}.wav")
        audio = np.asarray(audio, np.float32)
        begin = round(float(row["start"]) * SAMPLE_RATE)
        finish = min(len(timeline), begin + len(audio))
        timeline[begin:finish] += audio[: finish - begin]
    peak = float(np.max(np.abs(timeline))) if len(timeline) else 0.0
    if peak > 0.92:
        timeline *= 0.92 / peak
    wavfile.write(root / "voice_timeline.wav", SAMPLE_RATE, timeline)
    summary = {
        "cues": len(status),
        "roles": len({row["role"] for row in status}),
        "voice_modes": sorted({row["voice_mode"] for row in status}),
        "max_tempo": max((row["tempo"] for row in status), default=1),
        "fit_warnings": sum(bool(row.get("fit_warning")) for row in status),
        "peak": float(np.max(np.abs(timeline))) if len(timeline) else 0.0,
        "duration": timeline_duration,
        "engine": "qwen3-tts-mlx",
        "models": models,
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("DONE", json.dumps(summary, ensure_ascii=False), flush=True)


def build_preview(args):
    request = read_json(args.preview_request, {})
    voice = request.get("voice") or {}
    cue = {
        "id": 1,
        "role": voice.get("id") or "preview",
        "text": request.get("text") or "",
        "emotion": request.get("emotion") or "自然",
    }
    mode = voice_mode(voice)
    models = {
        "design": request.get("design_model") or args.design_model,
        "clone": request.get("clone_model") or args.clone_model,
    }
    model = load_model(models[mode])
    language = LANGUAGES.get(request.get("language", "zh"), request.get("language", "zh"))
    audio, sample_rate = generate_audio(model, mode, voice, cue, language, 970001)
    output = Path(args.preview_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.with_suffix(".raw.wav")
    wavfile.write(raw, sample_rate, audio)
    clean_audio(raw, output)
    raw.unlink(missing_ok=True)
    del model
    gc.collect()
    mx.clear_cache()
    print("PREVIEW_DONE", output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--timeline-mode", choices=("fixed", "natural"), default="fixed")
    parser.add_argument("--gap", type=float, default=0.25)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--design-model", default=VOICE_DESIGN_MODEL)
    parser.add_argument("--clone-model", default=VOICE_CLONE_MODEL)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--preview-request")
    parser.add_argument("--preview-output")
    args = parser.parse_args()
    if args.preview_request or args.preview_output:
        if not args.preview_request or not args.preview_output:
            parser.error("预听模式必须同时提供 --preview-request 与 --preview-output")
        build_preview(args)
        return
    if not args.root:
        parser.error("项目生成模式需要 --root")
    build_project(args)


if __name__ == "__main__":
    main()
