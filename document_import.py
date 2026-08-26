#!/usr/bin/env python3
import json
import re
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ALLOWED_DOCUMENTS = {".txt", ".md", ".docx", ".doc", ".rtf", ".odt", ".pdf"}
DOCUMENT_MAX_BYTES = 12 * 1024 * 1024
DOCUMENT_MAX_CHARS = 80_000
DOCUMENT_MAX_PAGES = 80
DOCUMENT_MAX_CUES = 400
DOCUMENT_MAX_CUE_CHARS = 280
DOCUMENT_TARGET_CUE_CHARS = 220
TEXTUTIL = Path("/usr/bin/textutil")
OSASCRIPT = Path("/usr/bin/osascript")
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
ODT_TEXT_NS = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
INVALID_ROLE_NAMES = {"注意", "提示", "备注", "说明", "例如", "比如", "注", "http", "https"}
ROLE_LINE = re.compile(
    r"^(?:【(?P<bracket>[^】]{1,12})】|(?P<name>[^:：\n]{1,12}))\s*[:：]\s*(?P<line>\S.*)$"
)


class DocumentError(ValueError):
    """User-facing document import failure."""


def extract_document(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_DOCUMENTS:
        raise DocumentError("不支持该文档格式。请使用 TXT、MD、DOCX、DOC、RTF、ODT 或 PDF。")
    if not path.is_file():
        raise DocumentError("找不到要导入的文档。")
    size = path.stat().st_size
    if size <= 0:
        raise DocumentError("文档是空文件，请换一份带文字的脚本。")
    if size > DOCUMENT_MAX_BYTES:
        raise DocumentError("文档超过 12 MB，请先删减后再导入。")

    warnings = []
    pages = None
    if suffix in {".txt", ".md"}:
        text = read_text_file(path)
        if suffix == ".md":
            text = simplify_markdown(text)
        fmt = suffix[1:]
    elif suffix == ".pdf":
        text, pages, warnings = extract_pdf(path)
        fmt = "pdf"
    else:
        text = extract_office(path, suffix)
        fmt = suffix[1:]

    text = normalize_extracted_text(text)
    if not has_meaningful_text(text):
        if suffix == ".pdf":
            raise DocumentError("这是扫描版 PDF 或没有可提取文字。请改用 Word、TXT，或先做文字识别。")
        raise DocumentError("没有从文档中读到可配音的文字。请确认文件不是空白稿，或另存为 TXT / DOCX。")
    if len(text) > DOCUMENT_MAX_CHARS:
        raise DocumentError(f"提取出的文字超过 {DOCUMENT_MAX_CHARS:,} 字，请先删减后再导入。")
    return {"text": text, "format": fmt, "pages": pages, "warnings": warnings, "size": size}


def parse_script(text):
    items = []
    buffer_role = None
    buffer_parts = []

    def flush():
        nonlocal buffer_role, buffer_parts
        body = join_parts(buffer_parts)
        if body:
            items.append({"role_name": buffer_role or "旁白", "text": body})
        buffer_role = None
        buffer_parts = []

    for raw_line in normalize_extracted_text(text).split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            flush()
            continue
        labeled = labeled_line(stripped)
        matched = match_role_line(stripped)
        if matched:
            flush()
            buffer_role = matched["name"]
            buffer_parts = [matched["line"]]
            continue
        if labeled:
            flush()
            buffer_role = None
            buffer_parts = [stripped]
            continue
        buffer_parts.append(stripped)
    flush()
    return items


def materialize_script(items, default_gap=0.25, voice_templates=None):
    if not items:
        raise DocumentError("没有从文档中读到可配音的台词。请检查正文，或改用 TXT / DOCX。")
    templates = voice_templates or {}
    default_description = templates.get("male_adult") or "自然真实的中文声音，普通话标准，情绪清楚。"
    role_ids = {}
    roles = []
    cues = []
    cursor = 0.0
    for item in items:
        name = str(item.get("role_name") or "旁白").strip() or "旁白"
        if name not in role_ids:
            role_id = f"role{len(roles) + 1}"
            role_ids[name] = role_id
            roles.append({
                "id": role_id,
                "name": name,
                "voice": "male_adult",
                "mode": "design",
                "description": default_description,
                "style": default_description,
                "clone_consent": False,
                "ref_audio": "",
                "ref_text": "",
            })
        for piece in split_cue_text(item.get("text") or ""):
            duration = max(1.6, min(12.0, len(piece) / 4.2))
            cues.append({
                "id": len(cues) + 1,
                "start": round(cursor, 3),
                "end": round(cursor + duration, 3),
                "source_text": piece,
                "text": piece,
                "role": role_ids[name],
                "emotion": "自然",
                "speed": 1.0,
                "gap_after": float(default_gap),
                "enabled": True,
            })
            cursor += duration + float(default_gap)
    if not cues:
        raise DocumentError("没有从文档中读到可配音的台词。请检查正文，或改用 TXT / DOCX。")
    if len(cues) > DOCUMENT_MAX_CUES:
        raise DocumentError(f"台词超过 {DOCUMENT_MAX_CUES} 条，请先删减文档后再导入。")
    return roles, cues


def read_text_file(path):
    data = Path(path).read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    for encoding in ("utf-16-le", "utf-16-be"):
        try:
            decoded = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\x00" not in decoded:
            return decoded
    return data.decode("utf-8", errors="replace")


def simplify_markdown(text):
    text = re.sub(r"```.*?```", lambda match: re.sub(r"```\w*", "", match.group(0)).strip("`"), text, flags=re.S)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"!\[(.*?)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[(.*?)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_~]{1,3}(.+?)[*_~]{1,3}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def extract_office(path, suffix):
    try:
        return extract_with_textutil(path)
    except Exception as exc:
        if suffix == ".docx":
            return extract_docx_xml(path)
        if suffix == ".odt":
            return extract_odt_xml(path)
        if suffix == ".rtf":
            raise DocumentError("无法读取 RTF。请另存为 TXT 或 DOCX 后再导入。") from exc
        if suffix == ".doc":
            raise DocumentError("无法读取旧版 DOC。请用 Word 另存为 DOCX 或 TXT 后再导入。") from exc
        raise DocumentError("无法读取该文档。请另存为 TXT 或 DOCX 后再导入。") from exc


def extract_with_textutil(path):
    if not TEXTUTIL.exists():
        raise RuntimeError("缺少 /usr/bin/textutil")
    proc = subprocess.run(
        [str(TEXTUTIL), "-convert", "txt", "-stdout", str(path)],
        capture_output=True,
        timeout=45,
    )
    if proc.returncode:
        detail = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or "textutil 转换失败")
    return (proc.stdout or b"").decode("utf-8", errors="replace")


def extract_docx_xml(path):
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
    except KeyError as exc:
        raise DocumentError("DOCX 文件不完整，缺少正文。请重新保存后再导入。") from exc
    except zipfile.BadZipFile as exc:
        raise DocumentError("DOCX 无法打开。请确认文件未损坏，或另存为 TXT。") from exc
    root = ET.fromstring(xml)
    paragraphs = []
    for paragraph in root.iter(f"{WORD_NS}p"):
        pieces = [node.text or "" for node in paragraph.iter(f"{WORD_NS}t")]
        paragraphs.append("".join(pieces))
    return "\n".join(paragraphs)


def extract_odt_xml(path):
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("content.xml")
    except KeyError as exc:
        raise DocumentError("ODT 文件不完整，缺少正文。请重新保存后再导入。") from exc
    except zipfile.BadZipFile as exc:
        raise DocumentError("ODT 无法打开。请确认文件未损坏，或另存为 TXT。") from exc
    root = ET.fromstring(xml)
    lines = []
    for node in root.iter():
        if node.tag in {f"{ODT_TEXT_NS}p", f"{ODT_TEXT_NS}h"}:
            lines.append("".join(node.itertext()))
    return "\n".join(lines)


def extract_pdf(path):
    warnings = []
    text = ""
    page_count = 0
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception as exc:
                raise DocumentError("这是加密 PDF，请先解除保护或另存为 Word / TXT。") from exc
        page_count = len(reader.pages)
        if page_count > DOCUMENT_MAX_PAGES:
            raise DocumentError(f"PDF 超过 {DOCUMENT_MAX_PAGES} 页，请先删减后再导入。")
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        if has_meaningful_text(text):
            return text, page_count, warnings
    except DocumentError:
        raise
    except ImportError:
        pass
    except Exception:
        text = ""
    try:
        kit_text, kit_pages = extract_pdf_pdfkit(path)
    except DocumentError as exc:
        if has_meaningful_text(text):
            return text, page_count, warnings
        if "超过" in str(exc):
            raise
        kit_text, kit_pages = "", page_count
    except Exception:
        kit_text, kit_pages = "", page_count
    if has_meaningful_text(kit_text):
        if not has_meaningful_text(text):
            warnings.append("已使用系统 PDF 引擎重新提取文字")
        return kit_text, kit_pages or page_count, warnings
    raise DocumentError("这是扫描版 PDF 或没有可提取文字。请改用 Word、TXT，或先做文字识别。")


def extract_pdf_pdfkit(path):
    if not OSASCRIPT.exists():
        raise DocumentError("无法提取 PDF 文字。请改用 Word 或 TXT。")
    path_json = json.dumps(str(Path(path).resolve()))
    script = f"""
    ObjC.import("PDFKit");
    ObjC.import("Foundation");
    var p = $.NSString.stringWithString({path_json}).stringByStandardizingPath;
    var u = $.NSURL.fileURLWithPath(p);
    var d = $.PDFDocument.alloc.initWithURL(u);
    if (!d) {{ throw new Error("无法打开 PDF"); }}
    var n = Number(d.pageCount);
    if (n > {DOCUMENT_MAX_PAGES}) {{ throw new Error("PDF_TOO_LONG"); }}
    var out = [];
    for (var i = 0; i < n; i++) {{
      var page = d.pageAtIndex(i);
      out.push(ObjC.unwrap((page && page.string) ? page.string : $("")));
    }}
    JSON.stringify({{pages: n, text: out.join("\\n\\n")}});
    """
    proc = subprocess.run(
        [str(OSASCRIPT), "-l", "JavaScript", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode:
        detail = (proc.stderr or proc.stdout or "").strip()
        if "PDF_TOO_LONG" in detail:
            raise DocumentError(f"PDF 超过 {DOCUMENT_MAX_PAGES} 页，请先删减后再导入。")
        raise DocumentError("无法提取 PDF 文字。请改用 Word 或 TXT。")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except json.JSONDecodeError as exc:
        raise DocumentError("无法提取 PDF 文字。请改用 Word 或 TXT。") from exc
    return str(payload.get("text") or ""), int(payload.get("pages") or 0)


def labeled_line(line):
    match = ROLE_LINE.match(line.strip())
    if not match:
        return None
    name = (match.group("bracket") or match.group("name") or "").strip()
    spoken = (match.group("line") or "").strip()
    if not name or not spoken:
        return None
    return {"name": name, "line": spoken}


def match_role_line(line):
    labeled = labeled_line(line)
    if not labeled or not looks_like_role_name(labeled["name"]):
        return None
    return labeled


def looks_like_role_name(name):
    name = (name or "").strip()
    if not 1 <= len(name) <= 12:
        return False
    if name.lower() in {item.lower() for item in INVALID_ROLE_NAMES}:
        return False
    if re.search(r"[。！？、，;；/\\()（）\[\]《》]", name):
        return False
    if re.fullmatch(r"\d+", name):
        return False
    if re.search(r"[章节篇部回]", name) and re.search(r"[\d一二三四五六七八九十]", name):
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", name))


def split_cue_text(text, limit=DOCUMENT_TARGET_CUE_CHARS):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    limit = max(1, min(int(limit), DOCUMENT_MAX_CUE_CHARS))
    if len(text) <= limit:
        return [text]
    sentences = [part for part in re.split(r"(?<=[。！？!?；;])\s*", text) if part]
    chunks = []
    current = ""
    for sentence in sentences or [text]:
        if current and len(current) + len(sentence) > limit:
            chunks.append(current.strip())
            current = sentence
        else:
            current += sentence
        while len(current) > limit:
            chunks.append(current[:limit].strip())
            current = current[limit:]
    if current.strip():
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def join_parts(parts):
    body = " ".join(str(part).strip() for part in parts if str(part).strip())
    return re.sub(r"[ \t]+", " ", body).strip()


def normalize_extracted_text(text):
    text = str(text or "").replace("\ufeff", "").replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def has_meaningful_text(text):
    return len(re.sub(r"\s+", "", text or "")) >= 2
