#!/usr/bin/env python3
"""Gemini TTS worker：与 tts_worker.py 接口一致（--root --duration [--reuse]）。
读取 manifest.json（台词）与 voices.json（角色 -> {"voice": Gemini音色, "style": 表演风格}），
逐条调用 Google Gemini TTS（单说话人模式，音色数不限），
沿用 silenceremove + atempo 拟合 + apad 对齐时间轴，输出 fit/*.wav、status.json、voice_timeline.wav。
"""
import argparse, base64, json, re, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path

# 按优先级回退：某模型日配额（每模型每天10次）用完自动切下一个 Gemini TTS 模型
MODELS = ['gemini-3.1-flash-tts-preview', 'gemini-2.5-flash-preview-tts', 'gemini-2.5-pro-preview-tts']
API = 'https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent'
SR = 24000
FF = '/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg'
if not Path(FF).exists():
    import shutil
    FF = shutil.which('ffmpeg') or 'ffmpeg'


def dur(p):
    return float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                          '-of', 'default=nw=1:nk=1', str(p)], text=True).strip())


def atempo_chain(x):
    out = []
    while x > 2: out.append('atempo=2.0'); x /= 2
    while x < .5: out.append('atempo=0.5'); x /= .5
    out.append(f'atempo={x:.8f}')
    return ','.join(out)


def api_key(root):
    import os
    for k in (root / 'gemini_key.txt', Path(__file__).parent / 'gemini_key.txt'):
        if k.exists():
            v = k.read_text().strip()
            if v: return v
    k = os.environ.get('GEMINI_API_KEY', '').strip()
    if not k:
        raise RuntimeError('缺少 Gemini API Key（gemini_key.txt 或环境变量 GEMINI_API_KEY）')
    return k


# 显式直连 Google，绕过 macOS 系统代理（系统代理若指向受限地区出口，Google 会按 IP 拒 400）
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def synthesize(key, text, voice, style, max_retries=5):
    prompt = f'{style}，请只用给定台词朗读，不要添加任何解释或额外内容：{text}'
    payload = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {
            'responseModalities': ['AUDIO'],
            'speechConfig': {'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': voice}}},
        },
    }
    data_bytes = json.dumps(payload, ensure_ascii=False).encode()
    quota_dead = []
    last_err = None
    for model in MODELS:
        if model in quota_dead:
            continue
        req = urllib.request.Request(API.format(m=model) + f'?key={key}',
                                     data=data_bytes,
                                     headers={'Content-Type': 'application/json'})
        for attempt in range(1, max_retries + 1):
            try:
                with _OPENER.open(req, timeout=180) as r:
                    data = json.loads(r.read().decode())
                for part in data.get('candidates', [{}])[0].get('content', {}).get('parts', []):
                    inline = part.get('inlineData') or part.get('inline_data')
                    if inline and inline.get('data'):
                        if model != MODELS[0]:
                            print(f'  （已回退到模型 {model}）', flush=True)
                        return base64.b64decode(inline['data'])
                raise RuntimeError('响应中没有音频数据: ' + json.dumps(data, ensure_ascii=False)[:300])
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors='replace')[:2000]
                last_err = f'{model} HTTP {e.code}: {body[:400]}'
                if e.code == 429 and 'PerDay' in body:
                    quota_dead.append(model)
                    print(f'  {model} 今日配额已用完，切换下一模型…', flush=True)
                    break
                if e.code in (400, 401, 403):
                    raise RuntimeError(f'Gemini TTS 请求被拒（不重试）：{last_err}')
                if e.code == 429:
                    # 分钟级限流：按 RetryInfo 的 retryDelay 等待后重试
                    m = re.search(r'"retryDelay"\s*:\s*"(\d+)s"', body)
                    wait = min(90, (int(m.group(1)) if m else 30) + 5)
                    if attempt < max_retries:
                        print(f'  {model} 分钟限流，{wait}s 后重试（第{attempt}次）…', flush=True)
                        time.sleep(wait)
                    continue
            except Exception as e:
                last_err = f'{model} {e}'
            if attempt < max_retries:
                wait = 5 * attempt
                print(f'  第{attempt}次失败（{str(last_err)[:200]}），{wait}s 后重试…', flush=True)
                time.sleep(wait)
        # 内层结束（重试耗尽或配额 break）：换下一模型
    if len(quota_dead) == len(MODELS):
        raise RuntimeError('所有 Gemini TTS 模型今日免费配额均已用完（每模型每天10次）。'
                           '配额在太平洋时间零点重置，或在 Google AI Studio 开通计费后重试。')
    raise RuntimeError(f'Gemini TTS 连续失败 {max_retries} 次：{str(last_err)[:400]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--duration', type=float, required=True)
    ap.add_argument('--reuse', action='store_true')
    a = ap.parse_args()
    root = Path(a.root)
    raw = root / 'raw'; clean = root / 'clean'; fit = root / 'fit'
    for p in (raw, clean, fit):
        p.mkdir(parents=True, exist_ok=True)
    rec = json.loads((root / 'manifest.json').read_text())
    voices = json.loads((root / 'voices.json').read_text())
    key = api_key(root)
    status = []
    for r in rec:
        i = int(r['id']); role = r['role']
        v = voices.get(role) or {}
        if isinstance(v, str):  # 兼容旧格式：纯描述文本 -> 用默认音色
            voice, style = 'Puck', v
        else:
            voice, style = v.get('voice', 'Puck'), v.get('style', '自然、符合情境地朗读')
        rp = raw / f'{i:03d}-{role}.wav'
        if not (a.reuse and rp.exists() and dur(rp) > .1):
            pcm = synthesize(key, r['text'], voice, style)
            (root / f'_tmp_{i:03d}.raw').write_bytes(pcm)
            subprocess.run([FF, '-y', '-v', 'error', '-f', 's16le', '-ar', str(SR), '-ac', '1',
                            '-i', str(root / f'_tmp_{i:03d}.raw'), str(rp)], check=True)
            (root / f'_tmp_{i:03d}.raw').unlink(missing_ok=True)
        cp = clean / f'{i:03d}.wav'
        subprocess.run([FF, '-y', '-v', 'error', '-i', str(rp), '-af',
                        'silenceremove=start_periods=1:start_duration=0.02:start_threshold=-46dB:stop_periods=-1:stop_duration=0.06:stop_threshold=-46dB,areverse,silenceremove=start_periods=1:start_duration=0.02:start_threshold=-46dB,areverse',
                        '-ar', str(SR), '-ac', '1', '-c:a', 'pcm_f32le', str(cp)], check=True)
        slot = max(.18, r['end'] - r['start'] - .035)
        cd = dur(cp); required = cd / slot
        max_tempo = 1.65 if slot >= .85 else (1.95 if slot >= .60 else 2.5)
        applied = max(.86, min(max_tempo, required))
        if required > max_tempo:
            print(f'  ⚠ 台词{i}偏长：清理后{cd:.2f}s，槽位{slot:.2f}s，需要{required:.2f}倍，已按上限{max_tempo:.2f}倍压缩（可能略赶）', flush=True)
        fp = fit / f'{i:03d}.wav'
        filters = [atempo_chain(applied), f'apad=whole_dur={slot:.6f}', f'atrim=duration={slot:.6f}', 'afade=t=in:st=0:d=0.006']
        subprocess.run([FF, '-y', '-v', 'error', '-i', str(cp), '-af', ','.join(filters),
                        '-ar', str(SR), '-ac', '1', '-c:a', 'pcm_f32le', str(fp)], check=True)
        x = {**r, 'raw_duration': cd, 'tempo': applied, 'duration': dur(fp),
             'end_error': (r['start'] + dur(fp)) - r['end']}
        status.append(x)
        (root / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
        print(i, role, voice, r['text'], f'tempo={applied:.3f}', flush=True)
    # 时间轴混音（numpy 无依赖版本：用 ffmpeg amix 逐条叠加太慢，改用 scipy 若可用，否则纯 Python）
    try:
        import numpy as np
        from scipy.io import wavfile
        timeline = np.zeros(round(a.duration * SR), np.float32)
        for r in status:
            _, x = wavfile.read(fit / f"{r['id']:03d}.wav")
            x = np.asarray(x, np.float32)
            b = round(r['start'] * SR); e = min(len(timeline), b + len(x))
            timeline[b:e] += x[:e - b]
        peak = float(np.max(np.abs(timeline))) if len(timeline) else 0
        if peak > .92:
            timeline *= .92 / peak
        wavfile.write(root / 'voice_timeline.wav', SR, timeline)
        peak_out = float(np.max(np.abs(timeline)))
    except ImportError:
        # 无 numpy：用 ffmpeg 逐条 adelay + amix
        inputs = []
        fc_parts = []
        for idx, r in enumerate(status):
            inputs += ['-i', str(fit / f"{r['id']:03d}.wav")]
            ms = round(r['start'] * 1000)
            fc_parts.append(f'[{idx}:a]adelay={ms}|{ms}[d{idx}]')
        mix_in = ''.join(f'[d{i}]' for i in range(len(status)))
        fc_parts.append(f'{mix_in}amix=inputs={len(status)}:duration=longest:normalize=0,alimiter=limit=0.92[out]')
        subprocess.run([FF, '-y', '-v', 'error', *inputs, '-filter_complex', ';'.join(fc_parts),
                        '-map', '[out]', '-ar', str(SR), '-ac', '1', '-c:a', 'pcm_f32le',
                        str(root / 'voice_timeline.wav')], check=True)
        peak_out = 0.92
    summary = {'cues': len(status), 'roles': len(set(x['role'] for x in status)),
               'max_tempo': max(x['tempo'] for x in status), 'peak': peak_out, 'engine': 'gemini-tts'}
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('DONE', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
