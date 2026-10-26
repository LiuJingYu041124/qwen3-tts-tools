#!/usr/bin/env python3
import argparse
import datetime
import fcntl
import gc
import hashlib
import json
import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

# mlx-audio loads Qwen3-TTS with its own MLX config class. Transformers 5.x emits
# a harmless model_type warning because that local class has an empty HF model_type.
# Keep actual exceptions visible while hiding this misleading warning.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import re
from opencc import OpenCC

from pypinyin_dict.phrase_pinyin_data import large_pinyin
large_pinyin.load()

_ASR_SIMPLIFIER = OpenCC("t2s")
import shutil
import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model
from scipy.io import wavfile

from voice_consistency import (
    cue_generation_cache_key,
    cue_generation_options,
    cue_generation_plan,
    generate_cue_results,
    validate_clone_output_duration,
)

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
COMMA_BREATH_SECONDS = 0.34
COMMA_BREATH_POLICY = "sentence-breath-v4-short-context"
CLONE_CONTEXT_FIRST_POLICY = "clone-short-context-first-v1"
CLONE_CONTEXT_FIRST_MAX_CHARS = 32
SENTENCE_TAIL_PAD_SECONDS = 0.14
SENTENCE_COMPLETION_POLICY = "sentence-completion-v1"
TERMINAL_PARTICLE_PAUSE_SECONDS = 0.045
WHISPER_CLI = os.environ.get("WHISPER_CLI", "/opt/homebrew/bin/whisper-cli")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", str(Path(__file__).parent / "models" / "ggml-small.bin"))
LANGUAGES = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean"}
MAX_FIXED_TEMPO = 4.0
TRUSTED_TAKE_POLICY = "trusted-take-v1"
RAW_CACHE_POLICY = "raw-generation-v23-design-pronunciation-strategy"
POSTPROCESS_CACHE_POLICY = "postprocess-unity-preserve-v3"
DESIGN_FALLBACK_POLICY = "design-fallback-v6-name-companion-minecraft-boundary-clear-cut"
CLONE_FALLBACK_POLICY = "clone-fallback-v5-performance-emotion-profile"
FILE_ASR_CACHE_POLICY = "file-full-cue-asr-v1-strict-tone-boundary-medium"
FILE_ASR_CACHE_SCHEMA = 1
MLX_LOCK_PATH = Path(os.environ.get("QWEN3_TTS_MLX_LOCK", Path(tempfile.gettempdir()) / "qwen3-tts-mlx.lock"))
EXPRESSIVE_REPEAT_PATTERN = re.compile(r"([哈呵嘿呜乌哇嗯啊呀哼])\1{1,}")
# Ordinary reduplicated words. A generation-only breath boundary inside these
# would distort a normal word, so the doubled-syllable repair skips them.
CLONE_REDUPLICATED_WORDS = frozenset({
    "妈妈", "爸爸", "姐姐", "妹妹", "哥哥", "弟弟", "爷爷", "奶奶", "娃娃",
    "谢谢", "慢慢", "刚刚", "好好", "看看", "说说", "想想", "试试", "等等",
    "常常", "天天", "人人", "家家", "处处", "时时", "大大", "小小", "多多",
    "早早", "悄悄", "轻轻", "渐渐", "隐隐", "明明", "白白", "偏偏", "徐徐",
    "匆匆", "纷纷", "微微", "淡淡", "静静", "缓缓", "深深", "远远", "紧紧",
    "狠狠", "层层", "点点", "步步", "种种", "个个", "件件", "条条", "面面",
})
# A take the length guard refused is still audio. Anything past this multiple of the
# allowed length is runaway generation rather than a slow read, and is never shipped.
BEST_EFFORT_MAX_OVERSHOOT = 2.5
BEST_EFFORT_MAX_SECONDS = 30.0


class CandidateRejected(RuntimeError):
    """A generated take the automatic checks refused, carrying the audio with it."""

    def __init__(self, message, take=None, degraded=False):
        super().__init__(message)
        self.take = take
        self.degraded = degraded


def best_effort_overshoot_ok(limits):
    if not limits or not limits.get("max_seconds"):
        return False
    return True


def offerable_take(exc):
    """Only a bounded overshoot is worth publishing; runaway audio never is."""
    take = getattr(exc, "take", None)
    if take is None or take[0] is None:
        return None
    if not getattr(exc, "degraded", False):
        return take
    limits = take[3] or {}
    allowed = float(limits.get("max_seconds") or 0.0)
    actual = len(take[0]) / float(take[1] or SAMPLE_RATE)
    if not allowed or actual > allowed * BEST_EFFORT_MAX_OVERSHOOT or actual > BEST_EFFORT_MAX_SECONDS:
        return None
    return take


@contextmanager
def exclusive_mlx_job(label):
    MLX_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MLX_LOCK_PATH.open("a+", encoding="utf-8") as handle:
        waiting_started = time.monotonic()
        print(f"MLX_QUEUE_WAIT label={label} lock={MLX_LOCK_PATH}", flush=True)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        waited = time.monotonic() - waiting_started
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({
            "pid": os.getpid(),
            "label": label,
            "acquired_at": datetime.datetime.now().isoformat(),
        }, ensure_ascii=False))
        handle.flush()
        print(f"MLX_QUEUE_ACQUIRED label={label} wait_seconds={waited:.3f}", flush=True)
        try:
            yield waited
        finally:
            held = time.monotonic() - waiting_started - waited
            print(f"MLX_QUEUE_RELEASED label={label} held_seconds={held:.3f}", flush=True)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def comfortable_fixed_tempo(slot):
    return 1.45 if slot >= 0.85 else (1.85 if slot >= 0.60 else 2.5)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def write_json_atomic(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def progress_stage(stage, state, total=None):
    fields = [f"PROGRESS_STAGE stage={stage}", f"state={state}"]
    if total is not None:
        fields.append(f"total={int(total)}")
    print(" ".join(fields), flush=True)


def progress_cue(cue, phase, state):
    print(
        f"PROGRESS_CUE cue={cue.get('id')} phase={phase} state={state}",
        flush=True,
    )


def duration(path):
    source = Path(path)
    if source.suffix.lower() == ".wav":
        try:
            sample_rate, audio = wavfile.read(source, mmap=True)
            if sample_rate > 0:
                return len(audio) / float(sample_rate)
        except Exception:
            pass
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
                str(source),
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


def fit_natural_audio(clean_path, fit_path, tempo):
    # atempo=1 still removes samples; an identity speed must be an identity
    # waveform, with no redundant fade that changes ASR or initial phonemes.
    if tempo == 1.0:
        shutil.copy2(clean_path, fit_path)
        return
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(clean_path),
                    "-af", atempo_chain(tempo) + ",afade=t=in:st=0:d=0.006",
                    "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_f32le",
                    str(fit_path)], check=True)


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
}
DESIGN_GENERATE = {
    "temperature": 0.7,
    "top_k": 36,
    "top_p": 0.9,
    "repetition_penalty": 1.18,
    "max_tokens": 1400,
}


def cue_slot_seconds(cue, timeline_mode=None):
    if timeline_mode != "fixed":
        return None
    try:
        return max(0.0, float(cue.get("end")) - float(cue.get("start")))
    except (TypeError, ValueError):
        return None


def generation_options(mode, cue, language, timeline_mode=None, voice=None):
    return cue_generation_options(
        mode,
        cue,
        language,
        timeline_mode,
        clone_options=CLONE_GENERATE,
        design_options=DESIGN_GENERATE,
        voice=voice,
    )


def raw_cache_key(cue, voice, language, models, timeline_mode=None):
    mode = voice_mode(voice)
    generate_options, limits = generation_options(mode, cue, language, timeline_mode, voice)
    # Changing candidate order does not invalidate a previously approved take;
    # regeneration is only needed when its actual inputs or safety policy change.
    payload = {
        "generation": cue_generation_cache_key(
            cue,
            voice,
            language,
            models[mode],
            generate_options,
            reference_signature(voice),
            limits,
        ),
        "raw_cache_policy": RAW_CACHE_POLICY,
    }
    if mode == "design" and design_pronunciation_variants(cue.get("text"), language):
        payload["design_fallback_policy"] = DESIGN_FALLBACK_POLICY
    if mode == "clone" and clone_pronunciation_variants(cue.get("text"), language):
        payload["clone_fallback_policy"] = CLONE_FALLBACK_POLICY
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def adopted_raw_cache_key(generation_fingerprint, take, source):
    payload = {
        "generation_fingerprint": generation_fingerprint,
        "source": "adopted_take",
        "take_id": str(take.get("id") or ""),
        "take_sha256": file_sha256(source),
        "trusted_take_policy": TRUSTED_TAKE_POLICY,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def postprocess_cache_key(raw_fingerprint, cue, timeline_mode=None):
    payload = {
        "raw_fingerprint": raw_fingerprint,
        "speed": round(float(cue.get("speed") or 1.0), 6),
        "timeline_mode": timeline_mode or "natural",
        "slot_seconds": round(float(cue_slot_seconds(cue, timeline_mode) or 0.0), 6),
        "postprocess_policy": POSTPROCESS_CACHE_POLICY,
        "trusted_take_policy": TRUSTED_TAKE_POLICY,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def cache_key(cue, voice, language, models, timeline_mode=None):
    return raw_cache_key(cue, voice, language, models, timeline_mode)


def collect_audio(results):
    parts = []
    sample_rate = None
    token_count = 0
    for result in results:
        parts.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
        sample_rate = result.sample_rate
        token_count += max(0, int(getattr(result, "token_count", 0) or 0))
    if not parts or not sample_rate:
        raise RuntimeError("模型没有返回音频")
    return np.clip(np.concatenate(parts), -1, 1), sample_rate, token_count


def ensure_sentence_completion(text, language):
    value = str(text or "").strip()
    if not value or re.search(r"[。！？!?…；;：:]$", value):
        return value
    # A terminal punctuation cue helps Qwen3-TTS finish the final syllable naturally.
    return value + ("。" if str(language).lower() in {"chinese", "zh", "zh-cn"} else ".")


def comma_breath_segments(text, language):
    value = ensure_sentence_completion(text, language)
    # Short utterances need their original context; split additional sentence
    # boundaries only for longer text. Existing comma behavior is unchanged.
    boundary = r"[，,]" if len(re.findall(r"[^\W_]", value, re.UNICODE)) <= 12 else r"[，,。！？!?；;]"
    if not re.search(boundary, value):
        return [value]
    # Each phrase is synthesized independently, so a continuation comma can
    # shorten its final syllable. Give the model a completed utterance; the
    # original comma pause is still represented by COMMA_BREATH_SECONDS.
    # Only punctuation changes here: no hidden repeated/substituted words.
    # Ignore punctuation-only fragments (e.g. the synthetic period after a
    # trailing comma); retain every spoken word, including one-character particles.
    return [ensure_sentence_completion(part.strip().rstrip("，,"), language)
            for part in re.split(r"(?<=" + boundary + r")\s*", value)
            if re.search(r"[^\W_]", part, re.UNICODE)]


def clone_context_first(cue, language, timeline_mode=None):
    if os.environ.get("QWEN3_TTS_CLONE_CONTEXT_FIRST", "1") == "0":
        return False
    if timeline_mode == "fixed" or str(language).lower() not in {"zh", "zh-cn", "chinese"}:
        return False
    if any(cue.get(key) for key in ("_context_review", "_punctuation_review", "_generation_strategy")):
        return False
    text = str(cue.get("text") or "")
    count = len(re.findall(r"[^\W_]", text, re.UNICODE))
    return 0 < count <= CLONE_CONTEXT_FIRST_MAX_CHARS and len(comma_breath_segments(text, language)) > 1


def design_pronunciation_variants(text, language):
    """Return generation-only spellings for unstable Chinese design-model tokens."""
    value = str(text or "").strip()
    if not value or str(language).lower() not in {"chinese", "zh", "zh-cn"}:
        return []

    variants = []
    seen = {value}

    def add(label, candidate):
        candidate = str(candidate or "").strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            variants.append((label, candidate))

    if re.match(r"^\s*杰杰[，,]我来帮你[！!]$", value):
        for spelling in ("截", "结", "洁"):
            add(
                f"support-name-{spelling}-clear",
                re.sub(
                    r"^\s*杰杰[，,]我来帮你[！!]$",
                    f"{spelling}，{spelling}。我来帮你！",
                    value,
                ),
            )

    name_match = re.match(r"^(\s*)杰杰(?P<tail>.*)$", value, re.DOTALL)
    if name_match:
        prefix = name_match.group(1)
        tail = name_match.group("tail")
        has_clause_separator = bool(tail and tail[0] in "，,、")
        remainder = tail[1:] if has_clause_separator else tail
        for spelling, separator in (("杰", "，"), ("洁", "。"), ("杰", "。"), ("洁", "，")):
            pair = f"{spelling}，{spelling}"
            label = "comma" if separator == "，" else "period"
            add(
                f"leading-name-{spelling}-{label}",
                f"{prefix}{pair}{separator}{remainder}",
            )

    embedded_companion = re.search(r"杰杰(还有)杰妹", value)
    if embedded_companion:
        for spelling in ("截", "洁", "杰"):
            candidate = value.replace(
                embedded_companion.group(0),
                f"{spelling}，{spelling}{embedded_companion.group(1)}{spelling}妹",
                1,
            )
            add(f"embedded-name-{spelling}-companion", candidate)

    # The same generation-only spelling repair is needed when a name follows
    # an address or appears inside a longer sentence.
    if "杰杰" in value:
        for spelling in ("截", "洁"):
            add(f"embedded-name-pair-{spelling}", value.replace("杰杰", f"{spelling}，{spelling}", 1))
    if "杰妹" in value:
        for spelling in ("截", "洁"):
            add(f"embedded-name-companion-{spelling}", value.replace("杰妹", f"{spelling}妹", 1))

    if re.search(r"呜{2,}", value):
        spelled = re.sub(r"呜{2,}", lambda match: "乌" * len(match.group(0)), value)
        expressive = re.sub(r"(?:\.{2,}|…+)", "！", spelled)
        expressive = re.sub(
            r"[，,、；;:：。！？!?]+\s*(乌{2,})\s*[，,、；;:：。！？!?]+",
            r"！\1！",
            expressive,
        )
        add("onomatopoeia-expressive", expressive)
        add("onomatopoeia-spelling", spelled)

    if re.search(r"末影人卡片的高机动性", value):
        add(
            "minecraft-term-boundary",
            re.sub(
                r"末影人卡片的高机动性",
                "末影人卡片的高机动性",
                value,
            ).replace("你了！", "你了。"),
        )

    if EXPRESSIVE_REPEAT_PATTERN.search(value):
        comma_spelled = EXPRESSIVE_REPEAT_PATTERN.sub(
            lambda match: "，".join(("乌" if match.group(1) == "呜" else match.group(1))
                                   for _ in match.group(0)),
            value,
        )
        comma_spelled = re.sub(r"(?:\.{2,}|…+)", "！", comma_spelled)
        add("expressive-repeat-commas", comma_spelled)

    return variants


def design_pronunciation_performance_variants(strategy, generation_text):
    """Return narrow performance fallbacks after the original direction fails."""
    spoken_length = len(re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", str(generation_text or "")))
    variants = []
    if strategy == "onomatopoeia-expressive" and re.search(r"乌{2,}", generation_text):
        variants.append(("controlled-fear", "惊慌、声音发紧"))
    if strategy == "leading-name-洁-period" and spoken_length <= 8:
        variants.append(("clear-name-emphasis", "激动、语气上扬、重点用力"))
    if strategy == "support-name-截-clear":
        variants.append(("clear-support-name", "自然、清晰、平稳，不要拖长音"))
    return variants


def clone_pronunciation_performance_variants(strategy, generation_text):
    if strategy == "clone-interjection-comma":
        return [("clear-leading-laugh", "轻快、含着笑，句首两个“嘿”清晰发音，不要吞掉")]
    if strategy == "clone-interjection-period":
        return [("clear-leading-interjection", "激动、语气上扬，句首“啊”清晰发音，不要吞掉")]
    if strategy.startswith("clone-name-pair-"):
        return [("clear-name-pair", "自然、清晰、平稳，名字两个音节分开完整发音，不要读成姐姐，不要添加语气词")]
    if strategy.startswith("clone-name-companion-"):
        return [("clear-name-companion", "自然、清晰、平稳，名字和“妹”完整发音，不要读成姐妹，不要添加语气词")]
    if strategy == "clone-name-tone-boundary":
        return [("clear-name-tone", "自然、清晰、平稳，专名完整发音，保持原本声调，不要改成同音字")]
    if strategy == "clone-placement-tone-boundary":
        return [("clear-placement-tone", "自然、清晰、平稳，‘身处’两个字分开但连贯发音，‘处’读第三声，不要读成深处")]
    if strategy.startswith("clone-latin-letter-"):
        return [("clear-latin-spelling", "自然、清晰、平稳，英文字符逐个完整读出，不要合并、吞字或改读成中文词")]
    if strategy.startswith("clone-tilde-") or strategy == "clone-terminal-period":
        return [("restrained-boundary", "自然、清晰、克制，句首语气词只发一次，不要拖长、重复或添加语气词")]
    if strategy == "clone-terminal-particle-boundary":
        return [("clear-terminal-particle", "自然、清晰、句末语气词必须单独、完整发音，不要吞掉或并入前一个字，不要添加其他语气词")]
    if strategy.startswith("clone-repeat-boundary-"):
        return [("clear-repeat-syllable", "自然、清晰、平稳，重复的两个音节都完整发音，中间有极短的停顿，不要合并成一个音或改成别的音")]
    if strategy.startswith("clone-hard-word-"):
        return [("clear-compound", "自然、清晰、平稳，这个词的两个字都完整发音，不要读成别的词，不要添加语气词")]
    return []


def clone_pronunciation_variants(text, language):
    """Keep unstable clone boundaries verbal with generation-only changes."""
    value = str(text or "").strip()
    if str(language).lower() not in {"chinese", "zh", "zh-cn"}:
        return []

    variants = []
    seen = {value}

    def add(label, candidate):
        candidate = str(candidate or "").strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            variants.append((label, candidate))

    match = re.fullmatch(r"(嘿{2,3}|啊)[！!]+\s*(.+)", value, re.DOTALL)
    if match:
        interjection, remainder = match.groups()
        boundaries = (("comma", "，"), ("period", "。"))
        if interjection == "啊":
            boundaries = tuple(reversed(boundaries))
        for label, boundary in boundaries:
            add(f"clone-interjection-{label}", f"{interjection}{boundary}{remainder}")

    # A short pause after a proper name prevents the clone model from
    # neutralizing its final tone when it immediately starts a narration verb.
    if re.match(r"^阿远打算联系管理员反馈Bug[，,]", value):
        add("clone-name-boundary", re.sub(r"^阿远打算", "阿远，打算", value, count=1))
    if re.match(r"^阿远打了", value):
        add("clone-name-segmented", re.sub(r"^阿远", "阿远，", value, count=1))
    elif re.match(r"^阿远有事", value):
        add("clone-name-boundary", re.sub(r"^阿远", "阿远，", value, count=1))
    elif "阿远" in value:
        add("clone-embedded-name-boundary", re.sub(r"阿远(?![，,。！？!?])", "阿远，", value, count=1))

    # The reference transcript for some clone voices makes two-syllable names
    # collapse into “姐姐/姐妹”. Use generation-only spelling boundaries; the
    # expected cue stays untouched and must still pass the strict ASR gate.
    # Keeping the true character and adding only a breath boundary is the
    # weakest possible edit, so it is tried before any homophone respelling.
    if "杰杰" in value:
        add("clone-name-pair-杰", value.replace("杰杰", "杰，杰", 1))
        for spelling in ("截", "洁"):
            add("clone-name-pair-" + spelling, value.replace("杰杰", f"{spelling}，{spelling}", 1))
    if "杰妹" in value:
        add("clone-name-companion-杰", value.replace("杰妹", "杰，妹", 1))
        for spelling in ("截", "洁"):
            add("clone-name-companion-" + spelling, value.replace("杰妹", f"{spelling}妹", 1))

    # Doubled syllables (人名 金金, 晶晶) are read as a single long syllable or
    # shifted to a neighbouring nasal by ICL. Split the pair for generation only.
    repeat = re.search(r"([\u4e00-\u9fff])\1", value)
    if (repeat and repeat.group(0) not in CLONE_REDUPLICATED_WORDS
            and not EXPRESSIVE_REPEAT_PATTERN.fullmatch(repeat.group(0))):
        add("clone-repeat-boundary-" + repeat.group(1),
            value[:repeat.start()] + f"{repeat.group(1)}，{repeat.group(1)}" + value[repeat.end():])

    # Two-syllable noun compounds with a difficult onset are merged into an
    # unrelated homophone (馅料 read back as 限量/下流). Keep both words and add
    # a generation-time breath boundary only.
    for compound in ("馅料",):
        if compound in value:
            add("clone-hard-word-" + compound,
                value.replace(compound, "，".join(compound), 1))

    # A pause after a name helps ICL preserve the intended tone without
    # changing the expected transcript used by the strict gate.
    if re.search(r"小宇", value):
        add("clone-name-tone-boundary", re.sub(r"小宇", "小，宇", value, count=1))

    # Similar-looking characters in a placement phrase are frequently confused
    # by clone ASR. Keep the phrase intact for validation and expose the tone
    # boundary only to generation.
    if "身处" in value:
        add("clone-placement-tone-boundary", value.replace("身处", "身，处", 1))

    # Latin-heavy identifiers are more stable when the model is told to spell
    # their letters rather than interpret them as an English word.
    if re.search(r"[A-Za-z]{2,}", value):
        spelled = re.sub(r"[A-Za-z]+", lambda match: "，".join(match.group(0)), value)
        spelled = re.sub(r"([A-Za-z])(?=[\u3400-\u9fff])", r"\1，", spelled)
        spelled = re.sub(r"(?:…{2,}|\.{3,})$", "。", spelled)
        add("clone-latin-letter-boundary", spelled)

    # Technical noun compounds are often merged into a homophone by ICL.
    # Preserve the words and add only a generation-time breath boundary.
    if re.match(r"^这串字符看上去像乱码[，,]", value):
        add("clone-technical-boundary", re.sub(r"^这串字符", "这串，字符", value, count=1))

    # A tilde is removed by the comparison normalizer but can make the clone
    # model sustain an interjection. Replace it with ordinary punctuation only
    # for generation, then validate against the untouched expected text.
    if "~" in value:
        add("clone-tilde-comma", re.sub(r"~+\s*[，,]?", "，", value))
        add("clone-tilde-period", re.sub(r"~+\s*[，,]?", "。", value))

    # Ellipses in short clone cues can trigger an extra filler syllable or an
    # unbounded tail. A completed sentence is safer while preserving all words.
    if re.search(r"(?:…{2,}|\.{3,})$", value):
        add("clone-terminal-period", re.sub(r"(?:…{2,}|\.{3,})$", "。", value))

    # A sentence-final neutral particle is easy for ICL to merge into the
    # preceding syllable (especially short, defensive lines such as “想玩一下啊”).
    # Add a generation-only breath boundary while keeping the original cue as the
    # strict ASR expectation. This repairs the omission without weakening the gate.
    terminal = re.search(r"(?P<stem>[\u3400-\u9fff])(?P<particle>[吗呢吧啊呀嘛])[！!？?。.]$", value)
    if terminal:
        add(
            "clone-terminal-particle-boundary",
            value[:terminal.start("particle")] + "，" + value[terminal.start("particle"):],
        )

    return variants


class CueWorkCache:
    """Bounded, process-local work reuse; never a substitute for the quality gate."""

    def __init__(self, max_entries=256, max_audio_bytes=32 * 1024 * 1024):
        self.max_entries = max_entries
        self.max_audio_bytes = max_audio_bytes
        self.transcripts = OrderedDict()
        self.reviews = OrderedDict()
        self.candidates = OrderedDict()
        self.audio_bytes = 0
        self.asr_review_failures = 0
        self.stats = {key: 0 for key in (
            "asr_executions", "asr_cache_hits", "candidate_executions", "candidate_cache_hits",
        )}

    def remember_transcript(self, key, transcript):
        self.transcripts[key] = transcript
        self.transcripts.move_to_end(key)
        while len(self.transcripts) > self.max_entries:
            self.transcripts.popitem(last=False)

    def remember_candidate(self, key, take):
        audio, rate, tokens = take
        if audio.nbytes > self.max_audio_bytes:
            return
        cached_audio = audio.copy()
        cached_audio.setflags(write=False)
        previous = self.candidates.pop(key, None)
        if previous is not None:
            self.audio_bytes -= previous[0].nbytes
        self.candidates[key] = (cached_audio, rate, tokens)
        self.audio_bytes += cached_audio.nbytes
        while self.audio_bytes > self.max_audio_bytes or len(self.candidates) > self.max_entries:
            _, previous = self.candidates.popitem(last=False)
            self.audio_bytes -= previous[0].nbytes


_CUE_WORK_CACHE = ContextVar("tts_cue_work_cache", default=None)
_ASR_RUNTIME = ContextVar("tts_asr_runtime", default=None)


@contextmanager
def asr_job_runtime():
    try:
        from whisper_runtime import WhisperRuntime
    except ImportError as exc:
        print(f"ASR_RUNTIME_IMPORT_FALLBACK reason={exc}", flush=True)
        yield None
        return

    runtime = WhisperRuntime(WHISPER_CLI, enabled=os.environ.get("QWEN3_TTS_ASR_BACKEND", "server") != "cli")
    try:
        with runtime:
            token = _ASR_RUNTIME.set(runtime)
            try:
                yield runtime
            finally:
                _ASR_RUNTIME.reset(token)
    finally:
        print("ASR_RUNTIME " + json.dumps(runtime.stats, sort_keys=True), flush=True)


def asr_backend_identity():
    runtime = _ASR_RUNTIME.get()
    return runtime.identity() if runtime is not None else {"backend": "cli-legacy"}


@contextmanager
def cue_work_cache():
    # A new cue owns its cache even when a caller has another context active.
    cache = CueWorkCache()
    token = _CUE_WORK_CACHE.set(cache)
    try:
        yield cache
    finally:
        _CUE_WORK_CACHE.reset(token)
        # Retry tracebacks can retain the owner until cyclic GC runs.
        cache.transcripts.clear()
        cache.reviews.clear()
        cache.candidates.clear()
        cache.audio_bytes = 0


def audio_array_identity(audio, sample_rate):
    samples = np.ascontiguousarray(audio)
    return (
        int(sample_rate), samples.dtype.str, tuple(samples.shape),
        hashlib.sha256(samples.tobytes()).hexdigest(),
    )


def asr_runtime_identity(model):
    return json.dumps({
        "cli": _artifact_signature(WHISPER_CLI),
        "model": _artifact_signature(model),
        "decode": ["-l", "zh", "-nt", "-np"],
        "backend": asr_backend_identity(),
    }, sort_keys=True)


def transcribe_audio_array(audio, sample_rate, model=None):
    model = model or WHISPER_MODEL
    if not Path(WHISPER_CLI).is_file() or not Path(model).is_file():
        return None
    cache = _CUE_WORK_CACHE.get()
    key = None
    if cache is not None:
        key = (audio_array_identity(audio, sample_rate), asr_runtime_identity(model))
        if key in cache.transcripts:
            cache.stats["asr_cache_hits"] += 1
            cache.transcripts.move_to_end(key)
            return cache.transcripts[key]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        wavfile.write(temp_path, sample_rate, audio)
        if cache is not None:
            cache.stats["asr_executions"] += 1
        runtime = _ASR_RUNTIME.get()
        if runtime is not None:
            output = runtime.transcribe(temp_path, model)
            if output is None:
                return None
        else:
            process = subprocess.run(
                [WHISPER_CLI, "-m", str(model), "-f", str(temp_path), "-l", "zh", "-nt", "-np"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=45,
            )
            if process.returncode != 0:
                return None
            output = process.stdout
        transcript = re.sub(r"\s+", "", output or "").strip()
        # Empty output and failed processes must remain retryable.
        if cache is not None and transcript:
            cache.remember_transcript(key, transcript)
        return transcript
    except Exception as exc:
        print(f"CUE_ASR_ERROR reason={exc}", flush=True)
        return None
    finally:
        temp_path.unlink(missing_ok=True)


def _asr_number_words(match):
    # Comparison-only cardinal rendering. Preserve leading-zero digit sequences;
    # never erase decimal points or discard a digit to manufacture a match.
    digits = "零一二三四五六七八九"
    token = match.group(0)
    integer, dot, fraction = token.partition(".")
    def cardinal(n):
        if n < 10:
            return digits[n]
        for unit, label in ((100000000, "亿"), (10000, "万"), (1000, "千"), (100, "百"), (10, "十")):
            if n >= unit:
                quotient, rest = divmod(n, unit)
                prefix = "" if unit == 10 and quotient == 1 else cardinal(quotient)
                return prefix + label + (("零" if rest < unit // 10 else "") + cardinal(rest) if rest else "")
    if len(integer) > 12 or (len(integer) > 1 and integer.startswith("0")):
        spoken = "".join(digits[int(c)] for c in integer)
    else:
        spoken = cardinal(int(integer))
    return spoken + ("点" + "".join(digits[int(c)] for c in fraction) if dot else "")


def normalized_spoken_text(value):
    import unicodedata
    text = _ASR_SIMPLIFIER.convert(unicodedata.normalize("NFKC", str(value or ""))).lower()
    text = re.sub(r"[0-9]+(?:\.[0-9]+)?", _asr_number_words, text)
    # Whisper uses ASCII tilde as a prosodic mark (Unicode Sm, not P).
    text = text.replace("~", "")
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


def contextual_spoken_phones(value):
    """Resolve clause-final neutral particles before discarding punctuation.

    Zdic 哪: neutral na is 啊 after an /n/ coda. Wiktionary 吶,
    etymology 4 documents the sentence-final variant spelling. This is
    context disambiguation, not permission to match na4 against na5.
    Lexical dictionary phrases and non-final/standalone 呐 remain untouched.

    Zdic 啦: 啦 is the exclamatory fusion of 了 with 啊, and both surface as the
    same neutral [l]-initial syllable at the end of a clause. Measured on the
    clone voice, the identical audio is read back as 了 at one boundary and 啦
    at another (cue logs show 太棒啦/太棒了 and ...百了/...百啦 crossing over in
    both directions), so a clause-final 了/啦 swap is an ASR view difference,
    not a wrong word. Only that neutral clause-final pair is normalized.
    Non-final 啦 (哗啦一声), lexical 了 (了解, 为了 mid-clause) and every other
    syllable or tone still fail the gate, so this is not a general relaxation.
    """
    import unicodedata
    from pypinyin import lazy_pinyin, Style
    from pypinyin.phrases_dict import phrases_dict
    text = _ASR_SIMPLIFIER.convert(unicodedata.normalize("NFKC", str(value or ""))).lower()
    text = re.sub(r"[0-9]+(?:\.[0-9]+)?", _asr_number_words, text)
    # Whisper uses ASCII tilde as a prosodic mark (Unicode Sm, not P).
    text = text.replace("~", "")
    spoken = normalized_spoken_text(value)
    result = lazy_pinyin(spoken, style=Style.TONE3, neutral_tone_with_five=True,
                         errors=lambda chars: list(chars))
    offsets = [i for i,c in enumerate(text) if not c.isspace() and not unicodedata.category(c).startswith("P")]
    if len(result) != len(offsets):
        return result
    for position, offset in enumerate(offsets):
        if text[offset] not in "呐哪" or position == 0:
            continue
        after = text[offset+1:].lstrip()
        if after and after[0] not in "，,。.!！?？;；:：…":
            continue
        before = text[:offset]
        if not before or before[-1].isspace() or unicodedata.category(before[-1]).startswith("P"):
            continue
        preceding = re.sub(r"[1-5]$", "", result[position-1])
        if not preceding.endswith("n") or preceding.endswith("ng"):
            continue
        # Never turn a dictionary lexical word into a sentence particle.
        if any(before.endswith(word[:-1]) for word in phrases_dict if len(word)>1 and word.endswith(text[offset])):
            continue
        result[position] = "na5"
    for position, offset in enumerate(offsets):
        if text[offset] not in "了啦" or result[position] not in {"le5", "la5"}:
            continue
        after = text[offset+1:].lstrip()
        if after and after[0] not in "，,。.!！?？;；:：…":
            continue
        result[position] = "le5"
    return result


def cue_asr_quality(expected, transcript):
    from difflib import SequenceMatcher
    wanted=normalized_spoken_text(expected); actual=normalized_spoken_text(transcript)
    ratio=SequenceMatcher(None,wanted,actual).ratio() if wanted and actual else 0.0
    length_ok=bool(wanted and actual and len(wanted)==len(actual))
    final_ok=bool(wanted and actual and wanted[-1]==actual[-1])
    # Equal-length, position-aligned syllables WITH tones: never fall back to
    # a fuzzy character ratio, which can hide omissions and wrong-tone words.
    wanted_phones=contextual_spoken_phones(expected); actual_phones=contextual_spoken_phones(transcript)
    phonetic_ok=bool(length_ok and len(wanted_phones)==len(wanted)
                     and len(actual_phones)==len(actual)
                     and wanted_phones==actual_phones)
    passed=bool(length_ok and (wanted==actual or phonetic_ok))
    return {"passed":passed,"similarity":ratio,"final_ok":final_ok,
            "length_ok":length_ok,"phonetic_ok":phonetic_ok,
            "match_mode":"exact" if passed and wanted==actual else
                         "tone_homophone" if passed else "rejected"}


def transcribe_full_cue(audio, sample_rate, expected):
    """Unprompted boundary review, never alter phonetic acceptance criteria.

    Real cue2's identical samples yield omissions under Whisper at one boundary
    and complete transcripts at others. On a short-cue failure require BOTH
    fixed silent-boundary views to independently satisfy the original strict
    length/position/tone check. No expected text is given to the recognizer.
    """
    def transcribe_view(view, **kwargs):
        transcript = None
        try:
            transcript = transcribe_audio_array(view, sample_rate, **kwargs)
            return transcript
        finally:
            # A fallback transcript can hide a failed review view from the caller.
            if not transcript:
                cache = _CUE_WORK_CACHE.get()
                if cache is not None:
                    cache.asr_review_failures += 1

    original = transcribe_view(audio)
    if original is None or cue_asr_quality(expected, original)["passed"]:
        return original
    if len(audio) / sample_rate > 12.0:
        return original
    reviewed = []
    for seconds in (0.25, 1.0):
        frames = round(seconds * sample_rate)
        padding = [(frames, frames)] + [(0, 0)] * (np.asarray(audio).ndim - 1)
        transcript = transcribe_view(np.pad(audio, padding))
        passed = cue_asr_quality(expected, transcript)["passed"]
        print(f"ASR_BOUNDARY_REVIEW pad={seconds} passed={passed} transcript={transcript}", flush=True)
        if transcript is None or not passed:
            break
        reviewed.append(transcript)
    if len(reviewed) == 2:
        return reviewed[-1]
    # Independent local model review: all three fixed views must pass the same
    # strict gate. No reference text/prompt, lexicon exception or tone relaxation.
    model = Path(__file__).parent / "models" / "ggml-medium.bin"
    if not model.is_file() or model.stat().st_size != 1533763059:
        return original
    for seconds in (0.0, 0.25, 1.0):
        frames = round(seconds * sample_rate)
        padding = [(frames, frames)] + [(0, 0)] * (np.asarray(audio).ndim - 1)
        transcript = transcribe_view(np.pad(audio, padding), model=model)
        passed = cue_asr_quality(expected, transcript)["passed"]
        print(f"ASR_SECOND_MODEL_REVIEW pad={seconds} passed={passed} transcript={transcript}", flush=True)
        if transcript is None or not passed:
            return original
    return transcript


def terminal_particle_present(audio, sample_rate, particle):
    transcript = transcribe_audio_array(audio, sample_rate)
    if transcript is None:
        return None, ""
    # Use the full-cue normalization here too: Whisper can emit 嗎, and
    # trailing punctuation is not a spoken syllable or evidence of truncation.
    spoken = normalized_spoken_text(transcript)
    expected = normalized_spoken_text(particle)
    endings = {expected, "嘛"} if expected == "吗" else {expected}
    accepted = bool(spoken and expected and spoken[-1] in endings)
    return accepted, transcript


def review_candidate_rank(quality, degraded=False):
    """Rank rejected candidates so the worst offenders are not what gets published."""
    return (
        not degraded,
        bool(quality.get("length_ok")),
        bool(quality.get("phonetic_ok")),
        bool(quality.get("final_ok")),
        float(quality.get("similarity") or 0.0),
    )


def keep_review_candidate(review_sink, take, transcript, quality, label, degraded=False):
    if review_sink is None:
        return
    best = review_sink.get("best")
    if best is not None and review_candidate_rank(best["quality"], best.get("degraded", False)) >= review_candidate_rank(quality, degraded):
        return
    review_sink["best"] = {
        "take": take,
        "transcript": transcript,
        "quality": quality,
        "label": label,
        "degraded": degraded,
    }


def generation_qa_identity(audio, sample_rate, expected):
    return (
        audio_array_identity(audio, sample_rate), str(expected), FILE_ASR_CACHE_POLICY,
        file_asr_validator_fingerprint(), asr_runtime_identity(WHISPER_MODEL),
        asr_runtime_identity(Path(__file__).parent / "models" / "ggml-medium.bin"),
    )


def selected_candidate_transcript(selected, expected):
    metadata = selected[5] if len(selected) > 5 and isinstance(selected[5], dict) else {}
    record = metadata.get("_generation_qa")
    if isinstance(record, dict) and record.get("transcript"):
        identity = generation_qa_identity(selected[0], selected[1], expected)
        if record.get("identity") == identity:
            return record["transcript"], True
    return transcribe_full_cue(selected[0], selected[1], expected), False


def generate_audio(model, mode, voice, cue, language, seed, timeline_mode=None,
                   review_sink=None, fast_review=False):
    expected = str(cue.get("expected_text") or cue.get("text") or "")
    with cue_work_cache() as cache:
        try:
            selected = _generate_audio_with_retries(
                model, mode, voice, cue, language, seed, timeline_mode,
                review_sink=review_sink, fast_review=fast_review,
            )
            identity = generation_qa_identity(selected[0], selected[1], expected)
            record = cache.reviews.get(identity)
            metadata = dict(selected[5]) if len(selected) > 5 and isinstance(selected[5], dict) else {}
            if record is not None:
                metadata["_generation_qa"] = {"identity": identity, "transcript": record}
            metadata["work_cache"] = dict(cache.stats)
            return (*selected[:5], metadata)
        finally:
            print(f"CUE_WORK_CACHE cue={cue.get('id')} " +
                  " ".join(f"{key}={value}" for key, value in cache.stats.items()), flush=True)


def _generate_audio_with_retries(model, mode, voice, cue, language, seed, timeline_mode=None,
                                 review_sink=None, fast_review=False):
    # Every cue must come out of here with audio: the track ships first and the
    # automatic verdict is reported as a quality grade. review_sink collects the best
    # rejected take, ranked, so the published audio is the least bad one available.
    expected = str(cue.get("expected_text") or cue.get("text") or "")
    transcript = None
    last_candidate_error = None
    attempt_seeds = [seed, seed + 7919, seed + 15838]
    # An interactive preview is not hunting for a pass; it needs one listenable take.
    # Keep the strategies that repair wording, drop the prosody-only sweeps, and
    # spend one seed per strategy instead of three.
    review_mode = fast_review

    def evaluate(take, label, degraded=False):
        nonlocal transcript
        cache = _CUE_WORK_CACHE.get()
        if cache is not None:
            identity = generation_qa_identity(take[0], take[1], expected)
            cache.reviews.pop(identity, None)
            failures_before = cache.asr_review_failures
        transcript = transcribe_full_cue(take[0], take[1], expected)
        if transcript is None:
            raise RuntimeError("完整性校验不可用：ASR未返回结果，未保存未经验证的音频")
        quality = cue_asr_quality(expected, transcript)
        if cache is not None and transcript and cache.asr_review_failures == failures_before:
            cache.reviews[identity] = transcript
            cache.reviews.move_to_end(identity)
            while len(cache.reviews) > cache.max_entries:
                cache.reviews.popitem(last=False)
        if not quality["passed"]:
            keep_review_candidate(review_sink, take, transcript, quality, label, degraded)
        return quality

    def offer_rejected(exc, label):
        """A candidate the length guard refused is still audio; keep it as last resort."""
        if review_sink is None:
            return
        take = offerable_take(exc)
        if take is None:
            return
        try:
            evaluate(take, label, degraded=True)
        except RuntimeError:
            pass

    # Short/repeated utterances are especially sensitive to sampling variance.
    # Add deterministic role-stable retries without changing any gate.
    if mode == "design" and len(re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", str(cue.get("text") or ""))) <= 6:
        attempt_seeds.extend([934261, 942180, 950099])
        attempt_seeds = list(dict.fromkeys(attempt_seeds))
    base_cue = dict(cue)
    # Short design lines are generated in one context-preserving pass. Keep this
    # internal decision off the manifest cue so retries and status stay immutable.
    ordinary_cue = base_cue
    if mode == "design" and len(re.sub(r"[^\u4e00-\u9fff]", "", str(cue.get("text") or ""))) <= 12:
        ordinary_cue = dict(base_cue, _context_review=True)
    # Random preview variants may all mispronounce a short name. Before failing,
    # try the same role-stable baseline used by project generation, unchanged
    # reference/text and the same full-cue gate. Never special-case a word.
    role = str(cue.get("role") or voice.get("id") or "")
    if mode == "clone" and role:
        baseline = 930000 + int(hashlib.sha256(role.encode()).hexdigest()[:8], 16) % 10000
        if baseline not in attempt_seeds:
            attempt_seeds.append(baseline)
    if review_mode:
        attempt_seeds = list(dict.fromkeys(attempt_seeds))[:2]
    context_seeds_tried = set()
    # Move one existing full-context candidate ahead of fragment retries. The
    # remaining repair strategies and gates stay available if it is rejected.
    if mode == "clone" and clone_context_first(base_cue, language, timeline_mode):
        context_seed = attempt_seeds[0]
        context_seeds_tried.add(context_seed)
        context_cue = dict(base_cue, _context_review=True)
        try:
            result = _generate_audio_candidate(model, mode, voice, context_cue, language, context_seed, timeline_mode)
            quality = evaluate(result, f"clone-context:{context_seed}")
        except RuntimeError as exc:
            last_candidate_error = exc
            offer_rejected(exc, f"clone-context-degraded:{context_seed}")
            print(f"CLONE_CONTEXT_FIRST cue={cue.get('id')} seed={context_seed} error={exc}", flush=True)
        else:
            print(f"CLONE_CONTEXT_FIRST cue={cue.get('id')} seed={context_seed} "
                  f"passed={quality['passed']} transcript={transcript}", flush=True)
            if quality["passed"]:
                metadata = dict(result[5]) if len(result) > 5 and isinstance(result[5], dict) else {}
                return (*result[:5], {
                    **metadata, "generation_strategy": "clone-context-first", "generation_seed": context_seed,
                    "generation_strategy_policy": CLONE_CONTEXT_FIRST_POLICY,
                })
    for attempt, attempt_seed in enumerate(attempt_seeds, 1):
        try:
            result = _generate_audio_candidate(model, mode, voice, ordinary_cue, language, attempt_seed, timeline_mode)
        except RuntimeError as exc:
            last_candidate_error = exc
            offer_rejected(exc, f"ordinary-degraded:{attempt_seed}")
            print(
                f"CANDIDATE_REJECTED cue={cue.get('id')} attempt={attempt} seed={attempt_seed} reason={exc}",
                flush=True,
            )
            continue
        quality = evaluate(result, f"ordinary:{attempt_seed}")
        print(f"FULL_CUE_ASR cue={cue.get('id')} attempt={attempt} seed={attempt_seed} "
              f"passed={quality['passed']} transcript={transcript}", flush=True)
        if quality["passed"]:
            return result
    # Clone ICL is sensitive to phrase boundaries: independently synthesized
    # comma fragments can turn names and short words into homophones. Retry the
    # complete cue before changing punctuation, preserving the same strict gate.
    if (mode == "clone" and not ordinary_cue.get("_context_review")
            and len(comma_breath_segments(cue.get("text"), language)) > 1):
        review_cue = dict(cue, _context_review=True)
        for review_seed in attempt_seeds[:3]:
            if review_seed in context_seeds_tried:
                continue
            try:
                result = _generate_audio_candidate(
                    model, mode, voice, review_cue, language, review_seed, timeline_mode,
                )
                quality = evaluate(result, f"clone-context:{review_seed}")
            except RuntimeError as exc:
                last_candidate_error = exc
                offer_rejected(exc, f"clone-context-degraded:{review_seed}")
                print(
                    f"CLONE_CONTEXT_REVIEW cue={cue.get('id')} seed={review_seed} error={exc}",
                    flush=True,
                )
                continue
            print(
                f"CLONE_CONTEXT_REVIEW cue={cue.get('id')} seed={review_seed} "
                f"passed={quality['passed']} transcript={transcript}",
                flush=True,
            )
            if quality["passed"]:
                return result
    # Only after ordinary and full-context attempts fail: punctuation-only prosody review.
    # Preserve words, role, reference, seed family and all strict ASR checks.
    if mode == "clone" and not review_mode and re.search(r"[！!？?]", str(cue.get("text") or "")):
        review_cue = dict(cue, _punctuation_review=True)
        for review_seed in attempt_seeds[:3]:
            try:
                result = _generate_audio_candidate(model, mode, voice, review_cue, language, review_seed, timeline_mode)
            except RuntimeError as exc:
                last_candidate_error = exc
                offer_rejected(exc, f"punctuation-degraded:{review_seed}")
                print(
                    f"PUNCTUATION_REVIEW cue={cue.get('id')} seed={review_seed} error={exc}",
                    flush=True,
                )
                continue
            try:
                quality = evaluate(result, f"punctuation:{review_seed}")
            except RuntimeError as exc:
                last_candidate_error = exc
                print(
                    f"PUNCTUATION_REVIEW cue={cue.get('id')} seed={review_seed} error={exc}",
                    flush=True,
                )
                continue
            print(f"PUNCTUATION_REVIEW cue={cue.get('id')} seed={review_seed} passed={quality['passed']} transcript={transcript}", flush=True)
            if quality["passed"]:
                return result
    # Failed independently synthesized fragments can lose expressive context.
    # Review the unmodified full text, preserving voice, seed family and gates.
    if (mode == "design" and not ordinary_cue.get("_context_review")
            and len(comma_breath_segments(cue.get("text"), language)) > 1):
        review_cue = dict(cue, _context_review=True)
        for review_seed in attempt_seeds[:3]:
            try:
                result = _generate_audio_candidate(model, mode, voice, review_cue, language, review_seed, timeline_mode)
                quality = evaluate(result, f"context:{review_seed}")
            except RuntimeError as exc:
                last_candidate_error = exc
                offer_rejected(exc, f"context-degraded:{review_seed}")
                print(f"CONTEXT_REVIEW cue={cue.get('id')} seed={review_seed} error={exc}", flush=True)
                continue
            print(f"CONTEXT_REVIEW cue={cue.get('id')} seed={review_seed} passed={quality['passed']} transcript={transcript}", flush=True)
            if quality["passed"]:
                return result
    if mode == "clone":
        # Keep fallback work bounded: every strategy gets the cue's three
        # deterministic seeds, while the extra role-stable seeds are reserved
        # for the ordinary pass above. This prevents one bad cue from spending
        # minutes repeating equivalent clone candidates.
        variant_seeds = list(dict.fromkeys(attempt_seeds[:3]))
        if review_mode:
            variant_seeds = variant_seeds[:1]
        for strategy, generation_text in clone_pronunciation_variants(cue.get("text"), language):
            performance_plans = [("original", None)] + clone_pronunciation_performance_variants(
                strategy, generation_text,
            )
            for performance_strategy, performance_emotion in performance_plans:
                for variant_seed in variant_seeds:
                    review_cue = dict(
                        cue,
                        text=generation_text,
                        _context_review=strategy != "clone-name-segmented",
                        _generation_strategy=strategy,
                        _generation_text=generation_text,
                    )
                    if performance_emotion is not None:
                        review_cue["emotion"] = performance_emotion
                        review_cue["_performance_strategy"] = performance_strategy
                    try:
                        result = _generate_audio_candidate(
                            model, mode, voice, review_cue, language, variant_seed, timeline_mode,
                        )
                    except RuntimeError as exc:
                        last_candidate_error = exc
                        offer_rejected(
                            exc,
                            f"clone-pronunciation-degraded:{strategy}:{performance_strategy}:{variant_seed}",
                        )
                        print(
                            f"CLONE_PRONUNCIATION_REVIEW cue={cue.get('id')} strategy={strategy} "
                            f"performance={performance_strategy} seed={variant_seed} error={exc}",
                            flush=True,
                        )
                        continue
                    try:
                        quality = evaluate(
                            result, f"clone-pronunciation:{strategy}:{performance_strategy}:{variant_seed}",
                        )
                    except RuntimeError as exc:
                        last_candidate_error = exc
                        print(
                            f"CLONE_PRONUNCIATION_REVIEW cue={cue.get('id')} strategy={strategy} "
                            f"performance={performance_strategy} seed={variant_seed} error={exc}",
                            flush=True,
                        )
                        continue
                    print(
                        f"CLONE_PRONUNCIATION_REVIEW cue={cue.get('id')} strategy={strategy} "
                        f"performance={performance_strategy} seed={variant_seed} "
                        f"passed={quality['passed']} input={generation_text} transcript={transcript}",
                        flush=True,
                    )
                    if quality["passed"]:
                        metadata = {
                            "generation_strategy": strategy,
                            "generation_text": generation_text,
                            "generation_seed": variant_seed,
                            "fallback_policy": CLONE_FALLBACK_POLICY,
                        }
                        if performance_emotion is not None:
                            metadata["original_emotion"] = cue.get("emotion")
                        if performance_emotion is not None:
                            metadata.update({
                                "performance_strategy": performance_strategy,
                                "generation_emotion": performance_emotion,
                            })
                        return (*result, metadata)
    if mode == "design":
        variant_seeds = list(dict.fromkeys(attempt_seeds[:3]))
        if review_mode:
            variant_seeds = variant_seeds[:1]
        for strategy, generation_text in design_pronunciation_variants(cue.get("text"), language):
            performance_plans = [("original", None)] + design_pronunciation_performance_variants(
                strategy, generation_text,
            )
            for performance_strategy, performance_emotion in performance_plans:
                for variant_seed in variant_seeds:
                    review_cue = dict(
                        cue,
                        text=generation_text,
                        _context_review=True,
                        _generation_strategy=strategy,
                        _generation_text=generation_text,
                    )
                    if performance_emotion is not None:
                        review_cue["emotion"] = performance_emotion
                        review_cue["_performance_strategy"] = performance_strategy
                    try:
                        result = _generate_audio_candidate(
                            model, mode, voice, review_cue, language, variant_seed, timeline_mode,
                        )
                    except RuntimeError as exc:
                        last_candidate_error = exc
                        offer_rejected(
                            exc,
                            f"pronunciation-degraded:{strategy}:{performance_strategy}:{variant_seed}",
                        )
                        print(
                            f"PRONUNCIATION_REVIEW cue={cue.get('id')} strategy={strategy} "
                            f"performance={performance_strategy} seed={variant_seed} error={exc}",
                            flush=True,
                        )
                        continue
                    try:
                        quality = evaluate(
                            result, f"pronunciation:{strategy}:{performance_strategy}:{variant_seed}",
                        )
                    except RuntimeError as exc:
                        last_candidate_error = exc
                        print(
                            f"PRONUNCIATION_REVIEW cue={cue.get('id')} strategy={strategy} "
                            f"performance={performance_strategy} seed={variant_seed} error={exc}",
                            flush=True,
                        )
                        continue
                    print(
                        f"PRONUNCIATION_REVIEW cue={cue.get('id')} strategy={strategy} "
                        f"performance={performance_strategy} seed={variant_seed} "
                        f"passed={quality['passed']} input={generation_text} transcript={transcript}",
                        flush=True,
                    )
                    if quality["passed"]:
                        metadata = {
                            "generation_strategy": strategy,
                            "generation_text": generation_text,
                            "generation_seed": variant_seed,
                            "fallback_policy": DESIGN_FALLBACK_POLICY,
                        }
                        if performance_emotion is not None:
                            metadata.update({
                                "performance_strategy": performance_strategy,
                                "generation_emotion": performance_emotion,
                            })
                        return (*result, metadata)
        for review_seed in ([] if review_mode else attempt_seeds[:3]):
            review_cue = dict(cue, _temperature_review=True)
            try:
                result = _generate_audio_candidate(model, mode, voice, review_cue, language, review_seed, timeline_mode)
                quality = evaluate(result, f"temperature:{review_seed}")
            except RuntimeError as exc:
                last_candidate_error = exc
                offer_rejected(exc, f"temperature-degraded:{review_seed}")
                print(f"TEMPERATURE_REVIEW cue={cue.get('id')} seed={review_seed} error={exc}", flush=True)
                continue
            print(f"TEMPERATURE_REVIEW cue={cue.get('id')} seed={review_seed} passed={quality['passed']} transcript={transcript}", flush=True)
            if quality["passed"]:
                return result
    if review_sink is not None and review_sink.get("best"):
        best = review_sink["best"]
        print(
            f"BEST_EFFORT_TAKE cue={cue.get('id')} label={best['label']} "
            f"degraded={str(best.get('degraded', False)).lower()} transcript={best['transcript']}",
            flush=True,
        )
        selected = best["take"]
        base_meta = selected[5] if len(selected) > 5 and isinstance(selected[5], dict) else {}
        return (*selected[:5], {
            **base_meta,
            "best_effort": True,
            "degraded": bool(best.get("degraded")),
            "gate_label": best["label"],
            "gate_transcript": best["transcript"],
            "gate_quality": best["quality"],
        })
    if last_candidate_error is not None:
        raise RuntimeError(
            f"台词{cue.get('id')}生成失败：{len(attempt_seeds)}次候选都没有产出可用音频，"
            f"最后候选被拒绝：{last_candidate_error}"
        ) from last_candidate_error
    raise RuntimeError(f"台词{cue.get('id')}生成失败：{len(attempt_seeds)}次候选都没有产出可用音频")


def generate_seeded_audio(model, mode, voice, cue, language, clone_options, design_options, seed):
    cache = _CUE_WORK_CACHE.get()
    key = None
    if cache is not None:
        try:
            reference = reference_signature(voice) if mode == "clone" else None
            if reference is not None:
                reference = {**reference, "sha256": file_sha256(reference["path"])}
            key = json.dumps({
                "model_instance": id(model), "mode": mode, "voice": voice, "cue": cue,
                "language": language, "clone_options": clone_options,
                "design_options": design_options, "seed": int(seed), "reference": reference,
            }, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError, OSError):
            # Unsupported inputs still use the original validation/generation path.
            key = None
    # Every leaf invocation reseeds; no caller consumes the preceding leaf's RNG state.
    mx.random.seed(seed)
    if key is not None and key in cache.candidates:
        cache.stats["candidate_cache_hits"] += 1
        cache.candidates.move_to_end(key)
        audio, rate, tokens = cache.candidates[key]
        return audio.copy(), rate, tokens
    if cache is not None:
        cache.stats["candidate_executions"] += 1
    take = collect_audio(generate_cue_results(
        model, mode, voice, cue, language, clone_options, design_options,
    ))
    if key is not None:
        cache.remember_candidate(key, take)
    return take


def _generate_audio_candidate(model, mode, voice, cue, language, seed, timeline_mode=None):
    design_options = dict(DESIGN_GENERATE)
    if cue.get("_temperature_review"):
        design_options["temperature"] = 0.3
    original_text = str((cue or {}).get("text") or "").strip()
    segments = ([ensure_sentence_completion(original_text, language)] if cue.get("_context_review")
                else comma_breath_segments(original_text, language))
    pauses = [COMMA_BREATH_SECONDS] * max(0, len(segments) - 1)
    # Keep the exact original word and validate the generated final particle by ASR.
    terminal_match = re.search(r"([吗呢吧啊呀嘛])[？?!！。.]$", segments[-1]) if segments else None
    terminal_particle = bool(terminal_match)
    clone_options, limits = generation_options(mode, cue, language, timeline_mode, voice)

    if len(segments) == 1:
        generation_cue = dict(cue)
        generation_cue["text"] = segments[0]
        single_clone_options, _ = generation_options(mode, generation_cue, language, timeline_mode, voice)
        if terminal_particle:
            particle = terminal_match.group(1)
            fallback = None
            for attempt, attempt_seed in enumerate((seed, seed + 7919, seed + 15838, seed + 23757), 1):
                candidate_audio, candidate_rate, candidate_tokens = generate_seeded_audio(
                    model, mode, voice, generation_cue, language,
                    single_clone_options, design_options, attempt_seed,
                )
                accepted, transcript = terminal_particle_present(candidate_audio, candidate_rate, particle)
                fallback = (candidate_audio, candidate_rate, single_clone_options, limits, candidate_tokens)
                print(
                    f"TERMINAL_ASR cue={cue.get('id')} attempt={attempt} seed={attempt_seed} "
                    f"accepted={accepted} transcript={transcript}", flush=True,
                )
                if accepted is True:
                    audio, sample_rate, token_count = fallback[0], fallback[1], fallback[4]
                    break
            else:
                message = f"台词{cue.get('id')}句尾完整性校验失败（4次），未采用缺字音频：{transcript}"
                # Keep a bounded take for the normal project-level review sink. The
                # strict full-cue ASR gate still runs before publication; this only
                # prevents one stable terminal-particle omission from deleting the
                # entire track. ASR-unavailable failures remain fail-closed.
                if mode == "clone" and transcript:
                    raise CandidateRejected(message, take=fallback, degraded=True)
                raise RuntimeError(message)
        else:
            audio, sample_rate, token_count = generate_seeded_audio(
                model, mode, voice, generation_cue, language,
                single_clone_options, design_options, seed,
            )
    else:
        audio_parts = []
        sample_rate = None
        token_count = 0
        for index, segment in enumerate(segments):
            segment_cue = dict(cue)
            segment_cue["text"] = segment
            segment_clone_options, _ = generation_options(mode, segment_cue, language, None, voice)
            # Validate each phrase: a whole-cue ratio can hide an internal missing tail.
            for segment_attempt, segment_seed in enumerate((seed, seed + 7919, seed + 15838), 1):
                segment_audio, segment_rate, segment_tokens = generate_seeded_audio(
                    model, mode, voice, segment_cue, language,
                    segment_clone_options, design_options, segment_seed,
                )
                segment_transcript = transcribe_audio_array(segment_audio, segment_rate)
                segment_quality = cue_asr_quality(segment, segment_transcript)
                print(f"SEGMENT_ASR cue={cue.get('id')} segment={index + 1} attempt={segment_attempt} "
                      f"passed={segment_quality['passed']} transcript={segment_transcript}", flush=True)
                if segment_quality["passed"]:
                    break
            else:
                # Short comma phrases (especially names/interjections) are not a
                # reliable ASR unit: Whisper may map “巫妖” to “无忧”. Do not
                # discard or accept on this local guess. Keep the candidate and
                # validate the complete reconstructed cue below.
                print(f"SEGMENT_ASR_DEFER cue={cue.get('id')} segment={index + 1} "
                      f"reason=short-fragment transcript={segment_transcript}", flush=True)
            if sample_rate is None:
                sample_rate = segment_rate
            elif sample_rate != segment_rate:
                raise RuntimeError("逗号分段生成的采样率不一致")
            audio_parts.append(segment_audio)
            token_count += segment_tokens
            if index < len(segments) - 1:
                pause_seconds = pauses[index]
                audio_parts.append(np.zeros(round(pause_seconds * sample_rate), dtype=np.float32))
        audio = np.clip(np.concatenate(audio_parts), -1, 1)
        pause_text = ",".join(f"{value:.3f}" for value in pauses)
        print(
            f"PHRASE_BOUNDARIES cue={cue.get('id')} segments={len(segments)} "
            f"pauses={pause_text} comma_policy={COMMA_BREATH_POLICY}",
            flush=True,
        )

    if mode == "clone":
        actual_seconds = len(audio) / sample_rate
        try:
            validate_clone_output_duration(
                original_text,
                actual_seconds,
                language,
                cue_slot_seconds(cue, timeline_mode),
                limits,
            )
        except RuntimeError as exc:
            # Keep the audio: a bounded overshoot is still publishable when it is the
            # only take this sentence produced, and a person can rule on it.
            raise CandidateRejected(
                str(exc),
                take=(audio, sample_rate, clone_options, limits, token_count),
                degraded=True,
            ) from exc
        hit_token_limit = bool(token_count) and token_count >= int(limits["max_tokens"])
        print(
            f"CLONE_GUARD cue={cue.get('id')} chars={len(original_text)} "
            f"tokens={token_count or 'unknown'} max_tokens={limits['max_tokens']} "
            f"hit_token_limit={str(hit_token_limit).lower()} duration={actual_seconds:.3f}s "
            f"allowed={limits['max_seconds']:.3f}s policy={limits['policy']}",
            flush=True,
        )
    return audio, sample_rate, clone_options, limits, token_count


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_signature(path):
    source = Path(path)
    try:
        stat = source.stat()
        return {"path": str(source.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        return {"path": str(source), "missing": True}


def file_asr_validator_fingerprint():
    medium_model = Path(__file__).parent / "models" / "ggml-medium.bin"
    packages = {}
    for package in ("opencc", "pypinyin", "pypinyin-dict"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "missing"
    payload = {
        "schema": FILE_ASR_CACHE_SCHEMA,
        "policy": FILE_ASR_CACHE_POLICY,
        "whisper_cli": _artifact_signature(WHISPER_CLI),
        "primary_model": _artifact_signature(WHISPER_MODEL),
        "secondary_model": _artifact_signature(medium_model),
        "packages": packages,
        "quality_contract": "exact-length-or-equal-phonetic-tone-sequence-v2-clause-final-neutral-particle",
        "transcription_contract": "unprompted-small-boundary-medium-review-v1",
        "runtime": asr_backend_identity(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def file_asr_cache_key(path, cue):
    expected = str(cue.get("expected_text") or cue.get("text") or "")
    return {
        "policy": FILE_ASR_CACHE_POLICY,
        "validator_fingerprint": file_asr_validator_fingerprint(),
        "asr_backend": asr_backend_identity(),
        "audio_sha256": file_sha256(path),
        "expected_sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
    }


def cached_file_asr(metadata, path, cue, identity=None):
    cached = (metadata or {}).get("file_asr") or {}
    identity = identity or file_asr_cache_key(path, cue)
    transcript = str(cached.get("transcript") or "")
    quality = cached.get("quality") or {}
    graded = cached.get("passed") is False and cached.get("best_effort") is True
    if cached.get("schema") != FILE_ASR_CACHE_SCHEMA or not (cached.get("passed") is True or graded):
        return None
    if not transcript or not all(cached.get(key) == value for key, value in identity.items()):
        return None
    if quality.get("passed") is True:
        if cue_asr_quality(
            str(cue.get("expected_text") or cue.get("text") or ""), transcript,
        ).get("passed") is True:
            return cached
        return None
    # A graded file was already judged against this exact audio, text and validator.
    # Re-running Whisper would only reproduce the same verdict, so trust the record.
    return cached


def _validate_audio_file_record(path, cue, identity=None, grade_only=False):
    rate, audio = wavfile.read(path)
    expected = str(cue.get("expected_text") or cue.get("text") or "")
    transcript = transcribe_full_cue(audio, rate, expected)
    quality = cue_asr_quality(expected, transcript)
    print(f"FILE_FULL_CUE_ASR cue={cue.get('id')} file={path} passed={quality['passed']} transcript={transcript}", flush=True)
    if transcript is None:
        raise RuntimeError(f"台词{cue.get('id')}成品校验不可用：ASR 未返回结果")
    if not quality["passed"] and not grade_only:
        raise RuntimeError(f"台词{cue.get('id')}成品完整性校验失败，未发布：{transcript}")
    return {
        "schema": FILE_ASR_CACHE_SCHEMA,
        **(identity or file_asr_cache_key(path, cue)),
        "passed": bool(quality["passed"]),
        "best_effort": not quality["passed"],
        "quality": quality,
        "transcript": transcript,
    }


def validate_audio_file(path, cue):
    return _validate_audio_file_record(path, cue)["quality"]


def validate_or_reuse_audio_file(path, cue, metadata, shared_results=None, asr_stats=None, grade_only=False):
    identity = file_asr_cache_key(path, cue)
    cached = cached_file_asr(metadata, path, cue, identity)
    shared_key = json.dumps(identity, sort_keys=True)
    if cached:
        if shared_results is not None:
            shared_results[shared_key] = cached
        print(f"FILE_FULL_CUE_ASR_CACHE cue={cue.get('id')} file={path}", flush=True)
        return cached, "cache"
    if shared_results is not None and shared_key in shared_results:
        print(f"FILE_FULL_CUE_ASR_DEDUP cue={cue.get('id')} file={path}", flush=True)
        return shared_results[shared_key], "dedup"
    if asr_stats is not None:
        asr_stats["executions"] = int(asr_stats.get("executions", 0)) + 1
    result = _validate_audio_file_record(path, cue, identity, grade_only=grade_only)
    if shared_results is not None:
        shared_results[shared_key] = result
    return result, "checked"


def clean_audio(source, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve BOTH boundaries. silenceremove's duration window clipped 3487
    # samples (145ms, peak .102) from the real cue2 onset. Energy thresholds
    # cannot distinguish quiet onset phonemes from silence safely.
    filters = f"apad=pad_dur={SENTENCE_TAIL_PAD_SECONDS:.3f}"
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


def raw_cache_approved(metadata):
    """Both a verified take and a graded best-effort take are reusable.

    Reusing the graded take is the whole point: a rebuild must not re-roll the same
    sentence and land on different audio after the user already listened to it.
    """
    quality = metadata.get("asr_quality") or {}
    if quality.get("passed") is True:
        return True
    return metadata.get("best_effort") is True


def clear_cue_failure_markers(raw_dir, cue_id):
    for failure_path in Path(raw_dir).glob(f"{int(cue_id):03d}-*.failed.json"):
        failure_path.unlink(missing_ok=True)


def valid_cached_audio(path):
    try:
        return path.is_file() and duration(path) > 0.1
    except Exception:
        return False


def load_and_generate(model_name, mode, pending, language, models, timeline_mode, timings=None, progress_total=None):
    if not pending:
        return
    load_started = time.monotonic()
    progress_stage("model_load", "loading", total=progress_total if progress_total is not None else len(pending))
    print(f"LOAD {mode} {model_name}", flush=True)
    model = load_model(model_name)
    load_seconds = time.monotonic() - load_started
    if timings is not None:
        timings["model_load_seconds"] += load_seconds
    print(f"TIMING stage=model_load mode={mode} seconds={load_seconds:.3f}", flush=True)
    progress_stage("generation", "running", total=progress_total if progress_total is not None else len(pending))
    try:
        for cue, voice, raw_path, meta_path, fingerprint, item in pending:
            role = cue["role"]
            progress_cue(cue, "generation", "running")
            raw_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            try:
                base_seed=930000+int(hashlib.sha256(str(cue["role"]).encode()).hexdigest()[:8],16)%10000
                expected_text=str(cue.get("expected_text") or cue.get("text") or "")
                generation_started = time.monotonic()
                try:
                    selected=generate_audio(
                        model,mode,voice,cue,language,base_seed,timeline_mode,review_sink={},
                    )
                finally:
                    generation_seconds = time.monotonic() - generation_started
                    if timings is not None:
                        timings["generation_seconds"] += generation_seconds
                    print(f"TIMING stage=generate cue={cue.get('id')} mode={mode} seconds={generation_seconds:.3f}", flush=True)
                audio,sample_rate,generate_options,limits,token_count,*selection_meta=selected
                selection_meta=dict(selection_meta[0]) if selection_meta else {}
                qa_started = time.monotonic()
                try:
                    transcript, qa_reused = selected_candidate_transcript(selected, expected_text)
                    selection_meta.pop("_generation_qa", None)
                    if qa_reused:
                        print(f"GENERATION_QA_REUSE cue={cue.get('id')}", flush=True)
                finally:
                    qa_seconds = time.monotonic() - qa_started
                    if timings is not None:
                        timings["generation_qa_seconds"] += qa_seconds
                    print(f"TIMING stage=generation_qa cue={cue.get('id')} seconds={qa_seconds:.3f}", flush=True)
                if transcript is None:
                    raise RuntimeError("完整性校验不可用：ASR未返回结果，未保存未经验证的音频")
                qa_result=cue_asr_quality(expected_text,transcript)
                best_effort = bool(selection_meta.get("best_effort"))
                print(f"CUE_ASR cue={cue.get('id')} seed={selection_meta.get('generation_seed', base_seed)} passed={str(qa_result['passed']).lower()} best_effort={str(best_effort).lower()} similarity={qa_result['similarity']:.3f} final_ok={str(qa_result['final_ok']).lower()} transcript={transcript}",flush=True)
                if not qa_result.get("passed") and not best_effort:
                    raise RuntimeError(f"台词{cue.get('id')}没有产出可发布音频：{transcript}")
                # The track ships with the best take available; the automatic verdict is
                # recorded as a quality grade so the user can review the flagged lines.
                wavfile.write(raw_path, sample_rate, audio)
                write_json_atomic(
                    meta_path,
                    {
                        "fingerprint": fingerprint,
                        "model": models[mode],
                        "mode": mode,
                        "role": role,
                        "generate": generate_options,
                        "clone_limits": limits,
                        "token_count": token_count,
                        "asr_quality": qa_result,
                        "asr_transcript": transcript,
                        "generation_qa_reused": qa_reused,
                        "expected_text": expected_text,
                        "best_effort": best_effort,
                        "needs_review": bool(best_effort and not qa_result.get("passed")),
                        **selection_meta,
                    },
                )
                clear_cue_failure_markers(raw_path.parent, cue["id"])
                print(f"GENERATED {cue['id']} {role} {mode} best_effort={str(best_effort).lower()}", flush=True)
                progress_cue(cue, "generation", "done")
            except Exception as exc:
                raw_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                failure = {
                    "cue_id": cue.get("id"), "role": role, "mode": mode,
                    "expected_text": expected_text if "expected_text" in locals() else str(cue.get("text") or ""),
                    "error": str(exc), "timestamp": datetime.datetime.now().isoformat(),
                }
                failure_path = raw_path.parent / f"{int(cue.get('id', 0)):03d}-{role}.failed.json"
                write_json_atomic(failure_path, failure)
                print(f"GENERATION_FAILED_CONTINUE cue={cue.get('id')} role={role} mode={mode} reason={exc}", flush=True)
                progress_cue(cue, "generation", "failed")
                continue
    finally:
        del model
        gc.collect()
        mx.clear_cache()


def reviewed_take_audit(take):
    """Human sign-off recorded when the user listens to a take the ASR gate rejected.

    The gate is never relaxed: the audio is only trusted because a person listened to
    it and took responsibility for the line. The audit fields are what makes that
    decision reproducible after the fact.
    """
    if not isinstance(take, dict) or not take.get("reviewed"):
        return None
    transcript = str(take.get("review_transcript") or "").strip()
    expected = str(take.get("review_expected_text") or "").strip()
    if not transcript or not expected or not str(take.get("reviewed_at") or "").strip():
        return None
    return {
        "policy": TRUSTED_TAKE_POLICY,
        "reviewed_at": str(take.get("reviewed_at") or ""),
        "review_transcript": transcript,
        "review_expected_text": expected,
        "review_gate_reason": str(take.get("review_gate_reason") or ""),
        "review_note": str(take.get("review_note") or ""),
    }


def adopted_take_source(root, cue, include_validation=False, validate_audio=True, require_gate=True):
    take = cue.get("adopted_take")
    if not isinstance(take, dict):
        return (None, None, None) if include_validation else None
    filename = Path(str(take.get("file") or "")).name
    if not filename or filename != str(take.get("file") or "") or not filename.endswith(".wav"):
        return (None, None, None) if include_validation else None
    takes_dir = (root.resolve() / "takes").resolve()
    source = (takes_dir / filename).resolve()
    if source.parent != takes_dir or not source.is_file():
        raise RuntimeError(f"台词{cue.get('id')}采用的试听版本不存在，请重新试听并采用")
    expected_hash = str(take.get("sha256") or "")
    if expected_hash and file_sha256(source) != expected_hash:
        raise RuntimeError(f"台词{cue.get('id')}采用的试听版本已损坏，请重新试听并采用")
    audit = reviewed_take_audit(take)
    if not validate_audio:
        return (source, None, audit) if include_validation else source
    expected = str(cue.get("expected_text") or cue.get("text") or "")
    if audit:
        print(
            f"ADOPTED_TAKE_REVIEWED cue={cue.get('id')} policy={TRUSTED_TAKE_POLICY} "
            f"reviewed_at={audit['reviewed_at']} transcript={audit['review_transcript']}",
            flush=True,
        )
        return (source, audit, audit) if include_validation else source
    if not require_gate:
        return (source, None, None) if include_validation else source
    rate, audio = wavfile.read(source)
    transcript = transcribe_full_cue(audio, rate, expected)
    quality = cue_asr_quality(expected, transcript)
    print(f"ADOPTED_FULL_CUE_ASR cue={cue.get('id')} passed={quality['passed']} transcript={transcript}", flush=True)
    if transcript is None or not quality["passed"]:
        raise RuntimeError(f"台词{cue.get('id')}采用的试听版本完整性校验失败，未采用：{transcript}")
    if include_validation:
        return source, {"quality": quality, "transcript": transcript}, None
    return source


def inspected_take_source(root, cue):
    """Resolve a take for the review UI without enforcing the ASR gate."""
    return adopted_take_source(root, cue, validate_audio=False)


def build_project(args, queue_wait_seconds=0.0):
    build_started = time.monotonic()
    root = Path(args.root)
    raw_dir = root / "raw"
    clean_dir = root / "clean"
    fit_dir = root / "fit"
    for directory in (raw_dir, clean_dir, fit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for legacy_meta in (*clean_dir.glob("*.wav.json"), *fit_dir.glob("*.wav.json")):
        legacy_meta.unlink(missing_ok=True)

    cues = read_json(root / "manifest.json", [])
    voices = read_json(root / "voices.json", {})
    models = {"design": args.design_model, "clone": args.clone_model}
    language = LANGUAGES.get(args.language, args.language)
    pending = {"design": [], "clone": []}
    cache_hits = set()
    postprocess_cache_hits = set()
    generation_plan = {}
    raw_fingerprints = {}
    timings = {
        "queue_wait_seconds": float(queue_wait_seconds),
        "model_load_seconds": 0.0,
        "generation_seconds": 0.0,
        "generation_qa_seconds": 0.0,
        "postprocess_seconds": 0.0,
        "file_asr_seconds": 0.0,
        "timeline_seconds": 0.0,
    }
    file_asr_cache_hits = 0
    file_asr_dedup_hits = 0
    file_asr_checks = 0
    file_asr_requests = 0
    file_asr_results = {}
    file_asr_stats = {"executions": 0}

    for item in cue_generation_plan(cues, voices):
        cue = item["cue"]
        role = cue.get("role")
        if role not in voices:
            raise RuntimeError(f"台词 {cue.get('id')} 使用了不存在的角色：{role}")
        cue_id = int(cue["id"])
        generation_plan[cue_id] = item
        raw_path = raw_dir / f"{cue_id:03d}-{role}.wav"
        meta_path = raw_path.with_suffix(".json")
        fingerprint = raw_cache_key(cue, item["voice"], language, models, args.timeline_mode)
        take_source, _, take_audit = adopted_take_source(root, cue, include_validation=True, validate_audio=False)
        take_sha256 = ""
        if take_source:
            take = cue.get("adopted_take") or {}
            take_sha256 = file_sha256(take_source)
            fingerprint = adopted_raw_cache_key(fingerprint, take, take_source)
        raw_fingerprints[cue_id] = fingerprint
        if take_source:
            cached = read_json(meta_path, {})
            take_id = str(take.get("id") or "")
            reuse_take = (
                args.reuse
                and raw_path.is_file()
                and cached.get("fingerprint") == fingerprint
                and cached.get("source") == "adopted_take"
                and cached.get("take_id") == take_id
                and cached.get("take_sha256") == take_sha256
                and raw_cache_approved(cached)
                and valid_cached_audio(raw_path)
            )
            if not reuse_take:
                take_source, take_asr, take_audit = adopted_take_source(root, cue, include_validation=True)
                raw_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                shutil.copy2(take_source, raw_path)
                write_json_atomic(
                    meta_path,
                    {
                        "fingerprint": fingerprint,
                        "model": models[item["mode"]],
                        "mode": item["mode"],
                        "role": role,
                        "source": "adopted_take",
                        "take_id": take_id,
                        "take_file": take_source.name,
                        "take_sha256": take_sha256,
                        "seed": int(take.get("seed") or 0),
                        "asr_quality": (take_asr or {}).get("quality"),
                        "asr_transcript": (take_asr or {}).get("transcript"),
                        "reviewed_take_audit": take_audit,
                        "expected_text": str(cue.get("expected_text") or cue.get("text") or ""),
                    },
                )
            cache_hits.add(cue_id)
            clear_cue_failure_markers(raw_dir, cue_id)
            print(f"ADOPTED_TAKE {cue_id} {role} {take_source.name} cache_hit={str(reuse_take).lower()}", flush=True)
            progress_cue(cue, "generation", "reused")
            continue
        cached = read_json(meta_path, {})
        reuse_hit = False
        if args.reuse and raw_path.exists() and cached.get("fingerprint") == fingerprint and raw_cache_approved(cached):
            try:
                cached_seconds = duration(raw_path)
                if item["mode"] == "clone":
                    _, limits = generation_options("clone", cue, language, args.timeline_mode, item["voice"])
                    validate_clone_output_duration(
                        cue.get("text"), cached_seconds, language,
                        cue_slot_seconds(cue, args.timeline_mode), limits,
                    )
                else:
                    reuse_hit = cached_seconds > 0.1
                if item["mode"] == "clone":
                    reuse_hit = True
            except Exception as exc:
                reuse_hit = False
                raw_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                print(f"CACHE_REJECTED cue={cue.get('id')} role={role} mode={item['mode']} reason={exc}", flush=True)
        if reuse_hit:
            cache_hits.add(cue_id)
            clear_cue_failure_markers(raw_dir, cue_id)
            progress_cue(cue, "generation", "reused")
            continue
        for stale_path in (
            clean_dir / f"{cue_id:03d}.wav",
            clean_dir / f"{cue_id:03d}.json",
            fit_dir / f"{cue_id:03d}.wav",
            fit_dir / f"{cue_id:03d}.json",
        ):
            stale_path.unlink(missing_ok=True)
        pending[item["mode"]].append((cue, item["voice"], raw_path, meta_path, fingerprint, item))

    generated_this_run = [
        (raw_path, meta_path)
        for mode_pending in pending.values()
        for _, _, raw_path, meta_path, _, _ in mode_pending
    ]
    try:
        load_and_generate(models["design"], "design", pending["design"], language, models, args.timeline_mode, timings, len(cues))
        load_and_generate(models["clone"], "clone", pending["clone"], language, models, args.timeline_mode, timings, len(cues))
    except Exception:
        # Preserve validated raw/meta pairs already produced. A later cue
        # failure must not erase earlier successful work; retry resumes caches.
        for raw_path, meta_path in generated_this_run:
            if not raw_path.is_file() or not raw_cache_approved(read_json(meta_path, {})):
                raw_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
        raise

    status = []
    progress_stage("postprocess", "running", total=len(cues))
    cursor = 0.0
    adopted_ids = {
        int(cue["id"])
        for cue in cues
        if isinstance(cue.get("adopted_take"), dict)
    }
    for index, cue in enumerate(cues, 1):
        print(f"CUE_START index={index}/{len(cues)} cue={cue.get('id')}", flush=True)
        cue_id = int(cue["id"])
        role = cue["role"]
        item = generation_plan.get(cue_id) or {}
        mode = item.get("mode") or voice_mode(voices[role])
        raw_path = raw_dir / f"{cue_id:03d}-{role}.wav"
        clean_path = clean_dir / f"{cue_id:03d}.wav"
        clean_meta_path = clean_dir / f"{cue_id:03d}.json"
        fit_path = fit_dir / f"{cue_id:03d}.wav"
        fit_meta_path = fit_dir / f"{cue_id:03d}.json"
        raw_fingerprint = raw_fingerprints[cue_id]
        if not raw_path.is_file():
            for stale_path in (
                clean_path,
                clean_meta_path,
                fit_path,
                fit_meta_path,
            ):
                stale_path.unlink(missing_ok=True)
            failure_path = raw_dir / f"{cue_id:03d}-{role}.failed.json"
            if not failure_path.exists():
                write_json_atomic(
                    failure_path,
                    {"cue_id": cue_id, "role": role, "error": "raw missing after generation"},
                )
            print(f"POSTPROCESS_SKIP_MISSING_RAW cue={cue_id} role={role}", flush=True)
            continue
        raw_audio_sha256 = file_sha256(raw_path)
        raw_meta = read_json(raw_path.with_suffix(".json"), {})
        clean_meta = read_json(clean_meta_path, {})
        clean_cache_hit = (
            args.reuse
            and clean_meta.get("schema_version") == 2
            and clean_meta.get("stage") == "clean"
            and clean_meta.get("raw_fingerprint") == raw_fingerprint
            and clean_meta.get("source_audio_sha256") == raw_audio_sha256
            and clean_meta.get("policy") == POSTPROCESS_CACHE_POLICY
            and valid_cached_audio(clean_path)
        )
        if not clean_cache_hit:
            postprocess_started = time.monotonic()
            try:
                clean_path.unlink(missing_ok=True)
                clean_meta_path.unlink(missing_ok=True)
                clean_audio(raw_path, clean_path)
            finally:
                timings["postprocess_seconds"] += time.monotonic() - postprocess_started
            clean_meta = {
                "schema_version": 2,
                "stage": "clean",
                "raw_fingerprint": raw_fingerprint,
                "source_audio_sha256": raw_audio_sha256,
                "policy": POSTPROCESS_CACHE_POLICY,
            }
            write_json_atomic(clean_meta_path, clean_meta)
        clean_duration = duration(clean_path)
        clean_audio_sha256 = file_sha256(clean_path)
        postprocess_fingerprint = postprocess_cache_key(raw_fingerprint, cue, args.timeline_mode)
        fit_meta = read_json(fit_meta_path, {})
        fit_cache_hit = (
            args.reuse
            and fit_meta.get("schema_version") == 2
            and fit_meta.get("stage") == "fit"
            and fit_meta.get("fingerprint") == postprocess_fingerprint
            and fit_meta.get("source_audio_sha256") == clean_audio_sha256
            and fit_meta.get("policy") == POSTPROCESS_CACHE_POLICY
            and valid_cached_audio(fit_path)
        )

        if args.timeline_mode == "natural":
            start = cursor
            applied_tempo = max(0.5, min(2.0, float(cue.get("speed") or 1.0)))
            expected_duration = clean_duration / applied_tempo
            # Do not fade out the sentence tail: it can suppress the last phoneme.
            filters = [atempo_chain(applied_tempo), "afade=t=in:st=0:d=0.006"]
            if not fit_cache_hit:
                postprocess_started = time.monotonic()
                try:
                    fit_path.unlink(missing_ok=True)
                    fit_meta_path.unlink(missing_ok=True)
                    fit_natural_audio(clean_path, fit_path, applied_tempo)
                finally:
                    timings["postprocess_seconds"] += time.monotonic() - postprocess_started
                fit_meta = {
                    "schema_version": 2,
                    "stage": "fit",
                    "fingerprint": postprocess_fingerprint,
                    "source_audio_sha256": clean_audio_sha256,
                    "policy": POSTPROCESS_CACHE_POLICY,
                }
                write_json_atomic(fit_meta_path, fit_meta)
            else:
                postprocess_cache_hits.add(cue_id)
            fitted_duration = duration(fit_path)
            end = start + fitted_duration
            gap_after = max(0.0, float(cue.get("gap_after", args.gap)))
            cursor = end + gap_after
        else:
            # Fixed timestamps are anchors, never a license to alter phonemes.
            # Preserve the validated waveform; extend and shift on overrun.
            original_start = float(cue["start"])
            original_end = float(cue["end"])
            start = max(original_start, cursor)
            applied_tempo = 1.0
            comfortable_tempo = None
            if not fit_cache_hit:
                postprocess_started = time.monotonic()
                try:
                    fit_path.unlink(missing_ok=True)
                    fit_meta_path.unlink(missing_ok=True)
                    shutil.copy2(clean_path, fit_path)
                finally:
                    timings["postprocess_seconds"] += time.monotonic() - postprocess_started
                fit_meta = {
                    "schema_version": 2,
                    "stage": "fit",
                    "fingerprint": postprocess_fingerprint,
                    "source_audio_sha256": clean_audio_sha256,
                    "policy": POSTPROCESS_CACHE_POLICY,
                }
                write_json_atomic(fit_meta_path, fit_meta)
            else:
                postprocess_cache_hits.add(cue_id)
            fitted_duration = duration(fit_path)
            end = start + fitted_duration
            cursor = end
            over_slot = end > original_end + 0.000001
            fit_warning = over_slot or abs(float(cue.get("speed") or 1.0) - 1.0) > 0.000001
            if fit_warning:
                print(f"WARNING cue={cue_id} over_slot={over_slot} "
                      f"original_end={original_end:.3f} actual_end={end:.3f} "
                      "固定模式保留完整原速音频，不执行槽位强压或截断；超槽后续顺延。"
                      "自定义语速仅自然模式应用；请复核视频同步。", flush=True)

        try:
            validation_started = time.monotonic()
            file_asr_requests += 2
            try:
                clean_asr, clean_status = validate_or_reuse_audio_file(
                    clean_path, cue, clean_meta, file_asr_results, file_asr_stats, grade_only=True,
                )
                fit_asr, fit_status = validate_or_reuse_audio_file(
                    fit_path, cue, fit_meta, file_asr_results, file_asr_stats, grade_only=True,
                )
            finally:
                timings["file_asr_seconds"] += time.monotonic() - validation_started
            file_asr_checks = file_asr_stats["executions"]
            for result_status in (clean_status, fit_status):
                if result_status == "cache":
                    file_asr_cache_hits += 1
                elif result_status == "dedup":
                    file_asr_dedup_hits += 1
            clean_meta["file_asr"] = clean_asr
            fit_meta["file_asr"] = fit_asr
            write_json_atomic(clean_meta_path, clean_meta)
            write_json_atomic(fit_meta_path, fit_meta)
        except Exception:
            for failed_path in (clean_path, fit_path, clean_meta_path, fit_meta_path):
                failed_path.unlink(missing_ok=True)
            raise
        row = {
            **cue,
            "start": start if args.timeline_mode == "fixed" else round(start, 3),
            "end": end if args.timeline_mode == "fixed" else round(end, 3),
            "raw_duration": clean_duration,
            "tempo": applied_tempo,
            "duration": fitted_duration,
            "end_error": (start + fitted_duration) - end,
            "voice_mode": mode,
            "cache_hit": cue_id in cache_hits,
            "generation_qa_reused": bool(cue_id not in cache_hits and raw_meta.get("generation_qa_reused")),
            "generation_work_cache": raw_meta.get("work_cache", {}) if cue_id not in cache_hits else {},
            "postprocess_cache_hit": cue_id in postprocess_cache_hits,
            "clean_asr_cache_hit": clean_status == "cache",
            "fit_asr_cache_hit": fit_status == "cache",
            "fit_asr_reused_from_clean": fit_status == "dedup" and clean_asr.get("audio_sha256") == fit_asr.get("audio_sha256"),
            "file_asr_cache_hits": int(clean_status == "cache") + int(fit_status == "cache"),
            "file_asr_dedup_hits": int(clean_status == "dedup") + int(fit_status == "dedup"),
            "adopted_take": cue_id in adopted_ids,
            "over_slot": over_slot if args.timeline_mode == "fixed" else False,
            "original_start": cue.get("start"),
            "original_end": cue.get("end"),
            "timing_shift": round(start - float(cue.get("start") or 0), 6),
            "fit_notice": ("固定模式保留原速完整音频；超槽顺延，自定义语速未应用，请复核视频同步。" if args.timeline_mode == "fixed" and fit_warning else ""),
            "fit_warning": fit_warning if args.timeline_mode == "fixed" else False,
            "comfortable_tempo": comfortable_tempo if args.timeline_mode == "fixed" else None,
            "verified": bool((raw_meta.get("asr_quality") or {}).get("passed")),
            "best_effort": bool(raw_meta.get("best_effort")),
            "needs_review": bool(raw_meta.get("needs_review")),
            "degraded": bool(raw_meta.get("degraded")),
            "gate_transcript": str(raw_meta.get("gate_transcript") or raw_meta.get("asr_transcript") or ""),
            "gate_reason": str(raw_meta.get("gate_label") or ""),
        }
        status.append(row)
        if row["needs_review"]:
            print(
                f"NEEDS_REVIEW cue={cue_id} role={role} degraded={str(row['degraded']).lower()} "
                f"transcript={row['gate_transcript']}",
                flush=True,
            )
        print(cue_id, role, mode, cue["text"], f"tempo={applied_tempo:.3f}", flush=True)
        progress_cue(cue, "postprocess", "done")

    expected_ids = {int(cue["id"]) for cue in cues}
    generated_ids = {int(row["id"]) for row in status}
    missing_ids = sorted(expected_ids - generated_ids)
    if missing_ids:
        # No audio at all means the model never produced a take, which a rebuild can
        # fix. Shipping a silent hole in the dub would not be noticed in time.
        write_json_atomic(root / "status.json", status)
        for stale_path in (root / "voice_timeline.wav", root / "summary.json"):
            stale_path.unlink(missing_ok=True)
        print(
            f"INCOMPLETE missing_ids={missing_ids} generated={len(generated_ids)}/{len(expected_ids)}",
            flush=True,
        )
        raise RuntimeError(
            f"整轨构建未完成：{len(generated_ids)}/{len(expected_ids)}，这些台词没生成出音频 {missing_ids}，请重试"
        )
    write_json_atomic(root / "status.json", status)
    progress_stage("timeline", "running", total=len(cues))
    timeline_started = time.monotonic()
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
    timeline_path = root / "voice_timeline.wav"
    temporary_timeline = root / f".{timeline_path.name}.{os.getpid()}.tmp"
    try:
        wavfile.write(temporary_timeline, SAMPLE_RATE, timeline)
        os.replace(temporary_timeline, timeline_path)
    finally:
        temporary_timeline.unlink(missing_ok=True)
    timings["timeline_seconds"] = time.monotonic() - timeline_started
    timings["total_seconds"] = time.monotonic() - build_started
    summary = {
        "cues": len(status),
        "roles": len({row["role"] for row in status}),
        "voice_modes": sorted({row["voice_mode"] for row in status}),
        "adopted_takes": sum(bool(row.get("adopted_take")) for row in status),
        "verified_cues": sum(bool(row.get("verified")) for row in status),
        "best_effort_cues": sum(bool(row.get("best_effort")) for row in status),
        "needs_review": sum(bool(row.get("needs_review")) for row in status),
        "needs_review_ids": [int(row["id"]) for row in status if row.get("needs_review")],
        "degraded_cues": sum(bool(row.get("degraded")) for row in status),
        "raw_cache_hits": len(cache_hits),
        "generation_qa_reuses": sum(bool(row.get("generation_qa_reused")) for row in status),
        "generation_work_cache": {
            key: sum(int(row.get("generation_work_cache", {}).get(key, 0)) for row in status)
            for key in ("asr_executions", "asr_cache_hits", "candidate_executions", "candidate_cache_hits")
        },
        "postprocess_cache_hits": len(postprocess_cache_hits),
        "postprocess_stage_hits": sum(bool(row.get("postprocess_cache_hit")) for row in status),
        "file_asr_requests": file_asr_requests,
        "file_asr_checks": file_asr_checks,
        "file_asr_executions": file_asr_stats["executions"],
        "file_asr_cache_hits": file_asr_cache_hits,
        "file_asr_dedup_hits": file_asr_dedup_hits,
        "identical_stage_reuses": file_asr_dedup_hits,
        "max_tempo": max((row["tempo"] for row in status), default=1),
        "fit_warnings": sum(bool(row.get("fit_warning")) for row in status),
        "peak": float(np.max(np.abs(timeline))) if len(timeline) else 0.0,
        "duration": timeline_duration,
        "engine": "qwen3-tts-mlx",
        "models": models,
        "asr_runtime": dict(_ASR_RUNTIME.get().stats) if _ASR_RUNTIME.get() is not None else {},
        "clone_context_first_policy": CLONE_CONTEXT_FIRST_POLICY,
        "timings": {key: round(value, 3) for key, value in timings.items()},
    }
    write_json_atomic(root / "summary.json", summary)
    progress_stage("complete", "done", total=len(cues))
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
    output = Path(args.preview_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    review_report = output.with_suffix(".review.json")
    review_report.unlink(missing_ok=True)
    # A preview must always come back with audio to listen to. When the automatic
    # checks refuse every take, keep the best one and say so in the report; adopting it
    # into the track still needs an explicit human sign-off.
    raw = output.with_suffix(".raw.wav")
    output.unlink(missing_ok=True)
    raw.unlink(missing_ok=True)
    model = None
    try:
        model = load_model(models[mode])
        language = LANGUAGES.get(request.get("language", "zh"), request.get("language", "zh"))
        seed = int(request.get("seed") or 970001)
        selected = generate_audio(
            model, mode, voice, cue, language, seed, review_sink={}, fast_review=True,
        )
        audio, sample_rate = selected[0], selected[1]
        review_meta = selected[5] if len(selected) > 5 and isinstance(selected[5], dict) else {}
        awaiting_sign_off = bool(review_meta.get("best_effort"))
        wavfile.write(raw, sample_rate, audio)
        clean_audio(raw, output)
        if not awaiting_sign_off:
            _validate_audio_file_record(output, cue)
        if mode == "clone":
            # Log the guard for the record, but a preview exists to be listened to: the
            # overshoot ceiling was already enforced when this take was accepted.
            cleaned_seconds = duration(output)
            allowed = float((selected[3] or {}).get("max_seconds") or 0.0)
            print(
                f"CLONE_PREVIEW_GUARD cue=preview tokens={selected[4] or 'unknown'} duration={cleaned_seconds:.3f}s "
                f"allowed={allowed:.3f}s over={'true' if allowed and cleaned_seconds > allowed else 'false'}",
                flush=True,
            )
        if awaiting_sign_off:
            write_json_atomic(review_report, {
                "gate_passed": False,
                "gate_label": review_meta.get("gate_label") or "",
                "gate_transcript": review_meta.get("gate_transcript") or "",
                "gate_quality": review_meta.get("gate_quality") or {},
                "degraded": bool(review_meta.get("degraded")),
                "expected_text": cue["text"],
                "created_at": datetime.datetime.now().isoformat(),
            })
            print(f"PREVIEW_BEST_EFFORT transcript={review_meta.get('gate_transcript')}", flush=True)
        print("PREVIEW_DONE", output, flush=True)
    except Exception as exc:
        output.unlink(missing_ok=True)
        raw.unlink(missing_ok=True)
        review_report.unlink(missing_ok=True)
        print(f"PREVIEW_FAILED mode={mode} reason={exc}", flush=True)
        raise
    finally:
        raw.unlink(missing_ok=True)
        if model is not None:
            del model
        gc.collect()
        mx.clear_cache()


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
        with exclusive_mlx_job(f"preview:{Path(args.preview_output).parent.parent.name}"), asr_job_runtime():
            build_preview(args)
        return
    if not args.root:
        parser.error("项目生成模式需要 --root")
    with exclusive_mlx_job(f"project:{Path(args.root).name}") as wait_seconds, asr_job_runtime():
        build_project(args, queue_wait_seconds=wait_seconds)


if __name__ == "__main__":
    main()
