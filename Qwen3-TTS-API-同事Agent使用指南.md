# 本地 Qwen3-TTS API 使用指南（交给 Agent 执行）

## 1. 给执行 Agent 的任务

请通过本文的 HTTP API 调用同事 Mac 上的 Qwen3-TTS，根据用户提供的文本和音色要求生成语音，并把 WAV 下载到当前电脑。

- 服务地址：`http://192.168.5.77:8765`。
- 当前无需 API Key，不要编造认证信息。
- 模型和推理都在服务提供者的 Mac 上，调用端无需安装 Qwen3-TTS、MLX、Whisper 或下载模型。
- 这是项目/角色式的异步 API：提交文本 → 查询任务 → 下载 WAV。当前没有 `/v1/audio/speech`，不能直接套用 OpenAI 语音 SDK。
- 只使用已有的、获准使用的角色。不要主动创建、修改、删除项目，不要替换参考音频，不要伪造声音授权或逐字稿确认。
- 不要调用 `/tts`、`/auto`、`/retry` 来完成本任务：这些是整项目操作，不是这里需要的单段文本合成。
- 不要重启、修改或部署服务。没有网络访问权限时，报告阻塞原因，不要声称生成成功。

服务端于 **2026-10-13** 从本机使用上述局域网地址访问 `/api/projects` 返回 HTTP 200；这不代表调用端网络一定可达。IP 以后可能变化，以服务提供者确认为准。

### 执行顺序

1. 检查用户是否给出了待合成文本；没有则询问，不要用测试台词替代用户最终内容。
2. 在用户电脑上运行下面脚本的 `--list`，查询可用项目和角色。
3. 用户已指定项目、角色时直接匹配；否则只列出名称、ID、模式和配置状态，请用户选择获准使用的音色，不要默认使用列表第一个克隆声音。
4. 确认选中的项目/角色在本次调用期间没有其他客户端或网页提交试听。当前没有独立任务 ID，必须独占该项目/角色，并串行调用。
5. 使用下面脚本提交一次请求，轮询到结束，下载明确的结果文件，保存自动校验报告。
6. 返回本地 WAV 路径、校验报告路径、所用角色，以及是否需要人工复听。

## 2. 网络与使用范围

调用程序必须运行在能访问 `192.168.5.77` 的电脑上：通常是同一公司局域网，或经管理员配置的 VPN。

- 云端 Agent 的远程执行环境通常访问不到这个私网地址；请让 Agent 在用户本机终端执行。
- 不要把地址换成 `localhost`：那指向调用者自己的电脑，而不是模型所在的 Mac。
- 不要自行给路由器做端口转发，或把服务直接暴露到公网。目前没有登录和项目权限隔离。
- 服务提供者的 Mac 需要开机、保持服务运行且不休眠。
- 如果连接超时，先检查局域网、VPN、访客网络隔离和 macOS 防火墙，联系服务提供者确认地址；不要反复提交生成任务。

## 3. 接口约定

### 查询项目与角色

- `GET /api/projects`：返回项目对象数组；使用 `id`、`name` 和 `roles` 发现可用角色。不要转发整个响应，里面可能包含脚本和参考音频信息。
- `GET /api/projects/{project_id}`：返回指定项目，包括 `roles` 和 `settings`。
- 角色用 `roles[].id` 标识，不是显示名称。`mode=design` 为描述式造声，`mode=clone` 为已配置参考音频的声音克隆。
- 克隆角色必须具备声音授权、服务端参考音频、逐字稿，以及与当前参考音频版本匹配的确认记录。配置不完整时联系服务提供者处理，不要绕过检查。
- `ref_audio` 是服务端的本地路径，不要尝试在调用端打开它。

### 提交文本

`POST /api/projects/{project_id}/preview-voice`，请求头 `Content-Type: application/json`。

```json
{
  "role_id": "role1",
  "text": "你好，这是通过接口生成的一段语音。",
  "emotion": "自然、温暖",
  "preview_nonce": "为每次新请求生成一个UUID"
}
```

- `role_id`：该项目中已有角色的真实 ID。
- `text`：本次要朗读的文本，不要求它已经写在项目台词中。
- `emotion`：表演要求，默认 `自然`；不是选择另一种音色。
- `preview_nonce`：每次新请求使用新 UUID，用于区分候选文件和随机种子；**不是任务 ID，也不是幂等键**。
- 不传 `cue_id`：该参数会绑定项目中已有台词，并检查文本是否一致，不适用于任意文本调用。
- 语言、音色描述、参考音频和模型使用项目/角色的既有配置；本接口没有独立的 `model`、`voice`、`response_format` 参数。
- HTTP `202` 表示请求已接受，不表示音频已完成。响应中的 `status` 可能已经变为 `running`、`done` 或 `failed`，需继续检查。

### 查询任务

`GET /api/projects/{project_id}/jobs`，读取键 `preview-{role_id}`。

完成时的示例（文件名仅为示意，必须使用真实响应）：

```json
{
  "preview-role1": {
    "status": "done",
    "message": "完成",
    "result": "role1-0123456789abcdef.wav"
  }
}
```

状态为 `queued`、`running`、`done` 或 `failed`。只有 `done` 且 `result` 有效时才下载；失败信息在 `message`。

### 下载与校验

- `GET /api/projects/{project_id}/preview-voice/{role_id}?file={result}`：返回 `audio/wav`。
- `GET /api/projects/{project_id}/preview-voice/{role_id}/gate?file={result}`：返回自动校验报告。

始终带上本次任务的 `result`，不要请求“最近一次音频”。当前实现即使指定文件不存在，也可能回退到最近文件，因此下面脚本还会核对响应的 `Content-Disposition` 文件名，不匹配就拒绝保存。

`gate_passed=false` 表示返回的是需要人工复听的候选，不应自动标记为合格。当前接口在没有 review 报告时会返回 `gate_passed=true`，这也不等于人工验收或逐字绝对正确。

## 4. 完整调用脚本

请在调用者电脑上把下面代码保存为 `qwen_tts_client.py`。只使用 Python 3 标准库，无需 `pip install`。

```python
#!/usr/bin/env python3
import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import ProxyHandler, Request, build_opener


class Client:
    def __init__(self, base):
        self.base = base.rstrip("/")
        # 仅此客户端直连局域网，不修改系统代理设置。
        self.http = build_opener(ProxyHandler({}))

    def request(self, path, payload=None, timeout=30):
        body = None
        headers = {}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(self.base + path, data=body, headers=headers)
        try:
            with self.http.open(req, timeout=timeout) as response:
                return response.status, response.headers, response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                info = json.loads(detail)
                if isinstance(info, dict):
                    detail = info.get("error") or info.get("message") or detail
            except ValueError:
                pass
            note = ""
            if exc.code == 409:
                note = "；已有任务，本次未提交成功，不要取走已有任务的结果"
            elif payload is not None and exc.code >= 500:
                note = "；提交结果可能不确定，请先查任务，不要自动重提"
            raise RuntimeError(f"HTTP {exc.code}: {str(detail)[:600]}{note}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            note = (
                "提交状态未知；任务可能已被接受。先查 jobs，不要自动重提 POST。"
                if payload is not None else
                "GET 失败；可重试查询，但不要因此重新提交生成。"
            )
            raise RuntimeError(f"网络错误：{exc}。{note}") from exc

    def json(self, path, payload=None, timeout=30):
        status, _, body = self.request(path, payload, timeout)
        try:
            return status, json.loads(body)
        except ValueError as exc:
            raise RuntimeError("响应不是 JSON；若已提交生成，请先检查任务，不要重提") from exc


def clone_ready(role):
    revision = role.get("reference_revision")
    return bool(
        role.get("clone_consent")
        and role.get("ref_audio")
        and str(role.get("ref_text") or "").strip()
        and role.get("ref_text_confirmed")
        and revision
        and role.get("ref_text_confirmed_revision") == revision
    )


def list_voices(api):
    _, projects = api.json("/api/projects")
    if not isinstance(projects, list):
        raise RuntimeError("项目列表格式异常")
    if not projects:
        print("没有项目，请联系服务提供者准备专用项目和获准使用的角色。")
    for project in projects:
        print(f"项目：{project.get('name', '')}  ID={project.get('id', '')}")
        for role in project.get("roles", []):
            mode = role.get("mode", "design")
            state = "可提交，服务端仍会校验"
            if mode == "clone" and not clone_ready(role):
                state = "克隆配置待确认，请联系服务提供者"
            print(
                f"  角色：{role.get('name', '')}  ID={role.get('id', '')}"
                f"  模式={mode}  {state}"
            )


def wait_for_result(api, project_path, key, first_job, wait_seconds):
    deadline = time.monotonic() + wait_seconds
    job = first_job
    previous = None
    while True:
        state = job.get("status") if isinstance(job, dict) else None
        if state != previous:
            print(f"任务状态：{state}", flush=True)
            previous = state
        if state == "done":
            return job.get("result")
        if state == "failed":
            raise RuntimeError(job.get("message") or "生成失败")
        if state not in {"queued", "running"}:
            raise RuntimeError("任务状态缺失或异常，可能服务已重启；请联系提供者，勿自动重提")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("等待超时，服务端任务未被取消；继续查 jobs，不要重跑提交命令")
        time.sleep(min(2, remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            continue
        _, jobs = api.json(project_path + "/jobs", timeout=min(30, remaining))
        job = jobs.get(key) if isinstance(jobs, dict) else None


def main():
    parser = argparse.ArgumentParser(description="调用局域网 Qwen3-TTS，下载单段 WAV")
    parser.add_argument("--base", default="http://192.168.5.77:8765")
    parser.add_argument("--list", action="store_true", help="仅列出项目和角色，不生成")
    parser.add_argument("--project", help="项目 ID")
    parser.add_argument("--role", help="角色 ID")
    text_args = parser.add_mutually_exclusive_group()
    text_args.add_argument("--text", help="要朗读的文本")
    text_args.add_argument("--text-file", help="UTF-8 文本文件")
    parser.add_argument("--emotion", default="自然")
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--wait", type=float, default=1800, help="排队和生成的等待秒数")
    args = parser.parse_args()
    api = Client(args.base)
    if args.list:
        list_voices(api)
        return
    if not args.project or not args.role:
        parser.error("生成需要 --project 和 --role；先用 --list 查询真实 ID")
    if not 0 < args.wait <= 86400:
        parser.error("--wait 必须大于 0 且不超过 86400 秒")
    text = (
        Path(args.text_file).read_text(encoding="utf-8-sig")
        if args.text_file else args.text or ""
    ).strip()
    if not text:
        parser.error("请提供非空的 --text 或 --text-file")

    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(".gate.json")
    if output.suffix.lower() != ".wav":
        parser.error("输出文件必须使用 .wav 扩展名")
    if output.exists() or report_path.exists():
        raise RuntimeError("输出或报告已存在；请换一个 --output，避免覆盖原文件")
    output.parent.mkdir(parents=True, exist_ok=True)

    project_path = "/api/projects/" + quote(args.project, safe="")
    _, project = api.json(project_path)
    role = next((r for r in project.get("roles", []) if r.get("id") == args.role), None)
    if role is None:
        raise RuntimeError("该项目没有指定角色，请重新 --list 查询")
    if role.get("mode") == "clone" and not clone_ready(role):
        raise RuntimeError("克隆授权、参考音频或逐字稿确认未就绪；请服务提供者处理")
    key = "preview-" + args.role
    _, jobs = api.json(project_path + "/jobs")
    if jobs.get(key, {}).get("status") in {"queued", "running"}:
        raise RuntimeError("该角色已有任务，请等原调用者完成；本次没有提交")

    nonce = uuid.uuid4().hex
    status, first_job = api.json(project_path + "/preview-voice", {
        "role_id": args.role,
        "text": text,
        "emotion": args.emotion,
        "preview_nonce": nonce,
    })
    if status != 202:
        raise RuntimeError(f"非预期提交状态 HTTP {status}，请检查任务后再操作")
    print(f"已接受：project={args.project} role={args.role} nonce={nonce}", flush=True)
    filename = wait_for_result(api, project_path, key, first_job, args.wait)
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.startswith(args.role + "-")
        or not filename.endswith(".wav")
    ):
        raise RuntimeError("任务结果文件名异常，停止下载")

    file_path = project_path + "/preview-voice/" + quote(args.role, safe="")
    query = "?" + urlencode({"file": filename})
    _, gate = api.json(file_path + "/gate" + query)
    if not isinstance(gate, dict) or not isinstance(gate.get("gate_passed"), bool):
        raise RuntimeError("自动校验报告格式异常；不要重提任务，请联系服务提供者")
    _, headers, audio = api.request(file_path + query, timeout=180)
    if headers.get_filename() != filename:
        raise RuntimeError("下载文件名不匹配，服务可能回退到了旧文件；停止保存")
    if (
        headers.get_content_type() != "audio/wav"
        or len(audio) < 44
        or audio[:4] != b"RIFF"
        or audio[8:12] != b"WAVE"
    ):
        raise RuntimeError("下载内容不是预期 WAV，停止保存")
    with output.open("xb") as handle:
        handle.write(audio)
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump({
            "project_id": args.project,
            "role_id": args.role,
            "file": filename,
            "preview_nonce": nonce,
            "gate": gate,
        }, handle, ensure_ascii=False, indent=2)
    print(f"WAV：{output}")
    print(f"自动校验报告：{report_path}")
    if gate["gate_passed"] is False:
        print("注意：自动校验未通过，这是候选音频，必须人工复听后决定是否采用。")
    else:
        print("自动接口未报告需复听候选；正式使用前仍请试听，不等于人工验收。")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"失败：{exc}", file=sys.stderr)
        sys.exit(1)
```

## 5. 执行命令

macOS / Linux 使用 `python3`；Windows 根据安装情况把命令中的 `python3` 换成 `py -3` 或 `python`。

先发现项目和角色，不触发生成：

```bash
python3 qwen_tts_client.py --list
```

把下面的 `实际项目ID`、`实际角色ID` 替换为获准使用的真实 ID，把文本替换为用户要朗读的内容：

```bash
python3 qwen_tts_client.py --project "实际项目ID" --role "实际角色ID" --text "你好，这是需要合成的文本。" --emotion "自然、温暖" --output "result.wav"
```

也可以读取本机的 UTF-8 文本文件，避免命令行引号问题：

```bash
python3 qwen_tts_client.py --project "实际项目ID" --role "实际角色ID" --text-file "input.txt" --output "result.wav" --wait 1800
```

服务地址变更时，在命令中添加 `--base "http://新的地址:8765"`。不要因此修改服务端。

第一次联调建议用用户同意的一句短文本，确认音色和连接。长文按完整句子拆成多次请求，等待前一段下载完成后再提交下一段，每段使用不同的输出文件名；这是单段试听路径，不保证适合一次发送整篇文档。

## 6. 并发、超时与失败处理

- **必须独占项目/角色。** 任务状态按 `project_id + role_id` 存储，没有调用者隔离、独立 `job_id` 或请求归属校验。不同用户复用同一角色时，后续请求可能覆盖先前的任务状态。脚本的“提交前查询”不是分布式锁，不能解决客户端之间的竞态；请服务提供者分配专用项目/角色，或在所有调用端统一串行调度。
- **HTTP 409：** 同一角色有排队/执行中的请求，本次未被接受。不要把已有任务结果当成本次结果，不要自动连续重试提交。
- **HTTP 400：** 参数、声音授权或逐字稿确认不满足要求。依据错误消息修正；不得自动把未确认角色改成已确认。
- **HTTP 404 / 500：** 可能是项目/角色/文件无效，或服务内部异常。旧接口部分错误可能返回 HTML 而非 JSON；保留状态码和必要错误说明，联系服务提供者，不要靠循环提交试错。
- **POST 超时或断连：** 服务端可能已接受任务。先查询 `/jobs` 并与提供者确认，不要自动重提；重复 `preview_nonce` 也不能保证幂等。
- **GET 超时或断连：** 只重试查询/下载，不要重新提交文本。客户端退出不会取消服务器上的任务。
- **等待 30 分钟超时：** 是脚本默认的等待上限，不是承诺的生成时长。任务可能仍在排队或运行。
- **任务突然消失：** 当前任务表保存在服务内存中，重启可能使状态丢失。不要无限轮询或猜测结果文件。
- **生成失败：** 报告 `message`；确需重试时先取得用户同意，避免无限重试消耗服务资源。
- **自动校验未通过：** 下载结果仍可能成功，但必须明确标记“候选、待复听”，不要自动发布为合格配音。
- **多人使用：** 计算资源共享，语音任务会排队；并发提交不会使这台 Mac 同时无限量推理。

## 7. Agent 完成标准

只有以下条件全部满足，才能报告“已生成并下载”：

1. 提交返回 HTTP 202，且观察到对应角色任务为 `done`。
2. 使用该任务 `result` 指定文件下载，并核对响应文件名和 WAV 头。
3. 本机成功保存 `.wav` 和 `.gate.json`，未覆盖原文件。
4. 向用户提供本地文件位置和质量提示；如未人工试听，不声称已完成听感验收。

本文不要求修改模型服务器、重启服务、创建新 API 或安装语音模型。完成的是：调用现有服务生成单段音频，并把结果带回调用者电脑。
