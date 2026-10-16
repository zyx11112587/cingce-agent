from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path
from typing import Iterable

from .models import Chunk, ImageRef, Question


QUESTION_HEADING_RE = re.compile(
    r"^#{1,6}\s*(?:第\s*)?(?P<number>\d+)\s*题"
    r"(?:\s*(?:[:：、.\-]\s*)?(?P<suffix>.*))?$",
    re.MULTILINE,
)
NUMBERED_RE = re.compile(
    r"^[ \t]*(?P<number>\d{1,4})[.、．]\s+(?P<body>.*)$",
    re.MULTILINE,
)
OPTION_RE = re.compile(
    r"^[ \t]*(?:[-*][ \t]+)?\*{0,2}(?P<letter>[A-H])\*{0,2}"
    r"\s*[.、．:：]\s*(?P<text>.+?)\s*$",
    re.MULTILINE,
)
ANSWER_RE = re.compile(
    r"^[ \t]*(?:\*{0,2})(?:答案|正确答案|参考答案)(?:\*{0,2})"
    r"\s*[:：]\s*(?P<answer>.+?)\s*$",
    re.MULTILINE,
)
EXPLANATION_HEADING_RE = re.compile(
    r"^[ \t]*(?:\*{0,2})(?:官方解析|答案解析|解析|分析)"
    r"(?:\*{0,2})\s*[:：]?\s*$",
    re.MULTILINE,
)
QID_RE = re.compile(r"\bqid\s*[:：]?\s*([A-Za-z0-9_-]+)", re.IGNORECASE)
YEAR_RE = re.compile(r"(?<!\d)(20\d{2}|19\d{2})(?!\d)")
MARKDOWN_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
HTML_IMAGE_RE = re.compile(
    r"<img\b[^>]*?\bsrc=[\"'](?P<src>[^\"']+)[\"'][^>]*>",
    re.IGNORECASE,
)
HTML_TAG_RE = re.compile(r"<[^>]+>")


def stable_id(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def clean_text(value: str) -> str:
    """Convert Markdown/HTML fragments into compact text while keeping image anchors."""
    value = html.unescape(value or "")
    value = MARKDOWN_IMAGE_RE.sub(
        lambda match: f"[图片: {match.group('alt') or match.group('src')}]",
        value,
    )
    value = HTML_IMAGE_RE.sub(
        lambda match: f"[图片: {Path(match.group('src')).name}]",
        value,
    )
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    value = HTML_TAG_RE.sub(" ", value)
    value = re.sub(r"\*\*|__|`", "", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def parse_front_matter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    result: dict[str, str] = {}
    for raw_line in parts[1].splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        result[key.strip()] = value.strip().strip("'\"")
    return result


def _first_value(frontmatter: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = frontmatter.get(key, "").strip()
        if value:
            return value
    return ""


def _infer_subject(source_rel: str, frontmatter: dict[str, str]) -> str:
    value = _first_value(frontmatter, "模块", "科目", "类别")
    if value:
        return value
    parts = Path(source_rel).parts
    for part in parts:
        if re.match(r"^\d{2}-", part):
            return re.sub(r"^\d{2}-", "", part)
    return ""


def _resolve_image_path(source_path: Path, src: str) -> str:
    src = src.strip()
    if src.startswith("<") and src.endswith(">"):
        src = src[1:-1].strip()
    candidate = Path(src)
    if not candidate.is_absolute():
        candidate = source_path.parent / candidate
    return str(candidate.resolve())


def extract_images(text: str, source_path: Path) -> list[ImageRef]:
    images: list[ImageRef] = []
    seen: set[tuple[str, str]] = set()
    matches: list[tuple[str, str, str]] = []
    for match in MARKDOWN_IMAGE_RE.finditer(text):
        matches.append((match.group("src"), match.group("alt"), match.group(0)))
    for match in HTML_IMAGE_RE.finditer(text):
        matches.append((match.group("src"), "", match.group(0)))

    for src, alt, anchor in matches:
        resolved = _resolve_image_path(source_path, src)
        key = (resolved, anchor)
        if key in seen:
            continue
        seen.add(key)
        images.append(
            ImageRef(
                image_id=stable_id(resolved, anchor),
                source_path=resolved,
                alt=alt or Path(src).name,
                anchor=anchor,
            )
        )
    return images


def _heading_title(text: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return clean_text(match.group(1)) if match else ""


def _infer_year(title: str, source_rel: str, frontmatter: dict[str, str]) -> str:
    value = _first_value(frontmatter, "年份", "年度")
    if value:
        match = YEAR_RE.search(value)
        if match:
            return match.group(1)
    match = YEAR_RE.search(title) or YEAR_RE.search(source_rel)
    return match.group(1) if match else ""


def _question_heading_matches(text: str) -> list[re.Match[str]]:
    matches = list(QUESTION_HEADING_RE.finditer(text))
    if matches:
        return matches
    return list(NUMBERED_RE.finditer(text))


def _parse_options(block: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for match in OPTION_RE.finditer(block):
        letter = match.group("letter").upper()
        text = clean_text(match.group("text"))
        if text:
            options[letter] = text
    return options


def _parse_answer(block: str) -> str:
    match = ANSWER_RE.search(block)
    return clean_text(match.group("answer")) if match else ""


def _parse_explanation(block: str) -> str:
    lines = block.splitlines()
    start = None
    for index, line in enumerate(lines):
        if EXPLANATION_HEADING_RE.match(line):
            start = index + 1
            break
    if start is None:
        answer_match = ANSWER_RE.search(block)
        start = block[: answer_match.end()].count("\n") + 1 if answer_match else 0
    explanation = "\n".join(lines[start:])
    explanation = re.split(
        r"\n\s*---+\s*\n|\n\s*##\s+第\s*\d+\s*题",
        explanation,
        maxsplit=1,
    )[0]
    return clean_text(explanation)


def _parse_stem(block: str) -> str:
    lines = block.splitlines()
    if not lines:
        return ""
    heading_match = QUESTION_HEADING_RE.match(lines[0])
    stem_lines: list[str] = []
    body_lines = lines[1:] if heading_match else lines
    for line in body_lines:
        if not heading_match:
            line = NUMBERED_RE.sub(lambda match: match.group("body"), line)
        if OPTION_RE.match(line) or ANSWER_RE.match(line):
            break
        if EXPLANATION_HEADING_RE.match(line):
            break
        stem_lines.append(line)
    return clean_text("\n".join(stem_lines))


def parse_question_block(
    block: str,
    source_path: Path,
    source_rel: str,
    frontmatter: dict[str, str],
    fallback_number: str = "",
) -> Question | None:
    heading_matches = QUESTION_HEADING_RE.finditer(block)
    heading = next(heading_matches, None)
    number = heading.group("number") if heading else fallback_number
    if not number:
        numbered = NUMBERED_RE.search(block)
        number = numbered.group("number") if numbered else ""

    qid_match = QID_RE.search(block)
    qid = qid_match.group(1) if qid_match else ""
    stem = _parse_stem(block)
    options = _parse_options(block)
    answer = _parse_answer(block)
    explanation = _parse_explanation(block)

    if not (stem or options or answer or explanation):
        return None
    if not qid:
        qid = stable_id(source_rel, number, stem, "\x1f".join(options.values()))

    title = _first_value(frontmatter, "试卷", "题名", "标题") or _heading_title(block)
    subject = _infer_subject(source_rel, frontmatter)
    year = _infer_year(title, source_rel, frontmatter)
    region = _first_value(frontmatter, "地区", "省份")
    question_type = _first_value(frontmatter, "题型", "类型")
    if not question_type and heading and heading.group("suffix"):
        suffix = clean_text(heading.group("suffix"))
        if suffix:
            question_type = suffix.split("·")[-1].strip()

    record_id = stable_id(source_rel, qid, number)
    images = extract_images(block, source_path)
    for image in images:
        image.record_id = record_id

    return Question(
        qid=qid,
        record_id=record_id,
        number=number,
        stem=stem,
        options=options,
        answer=answer,
        explanation=explanation,
        title=title,
        year=year,
        region=region,
        subject=subject,
        question_type=question_type,
        source_path=str(source_path.resolve()),
        source_rel=source_rel,
        raw_text=block.strip(),
        images=images,
    )


def parse_questions(
    text: str,
    source_path: Path,
    source_rel: str,
    frontmatter: dict[str, str],
) -> list[Question]:
    heading_matches = list(QUESTION_HEADING_RE.finditer(text))
    matches = heading_matches
    if not heading_matches:
        matches = [
            match
            for match in NUMBERED_RE.finditer(text)
            if _parse_options(
                text[match.start(): match.end() + 2500]
            ) or _parse_answer(text[match.start(): match.end() + 2500])
        ]
    if not matches:
        return []
    questions: list[Question] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start():end]
        question = parse_question_block(
            block,
            source_path,
            source_rel,
            frontmatter,
            match.group("number"),
        )
        if question:
            if not heading_matches and not (
                question.options or question.answer or question.explanation
            ):
                continue
            questions.append(question)
    return questions


def _question_ranges(text: str) -> list[tuple[int, int]]:
    matches = list(QUESTION_HEADING_RE.finditer(text))
    if not matches:
        matches = [
            match
            for match in NUMBERED_RE.finditer(text)
            if _parse_options(
                text[match.start(): match.end() + 2500]
            ) or _parse_answer(text[match.start(): match.end() + 2500])
        ]
    return [
        (match.start(), matches[index + 1].start() if index + 1 < len(matches) else len(text))
        for index, match in enumerate(matches)
    ]


def _remove_question_ranges(text: str) -> str:
    ranges = _question_ranges(text)
    if not ranges:
        return text
    pieces: list[str] = []
    cursor = 0
    for start, end in ranges:
        pieces.append(text[cursor:start])
        pieces.append("\n")
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _chunk_generic_text(
    text: str,
    source_path: Path,
    source_rel: str,
    frontmatter: dict[str, str],
    max_chars: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    text = _remove_question_ranges(text)
    text = re.sub(r"^---[\s\S]*?---", "", text, count=1)
    sections = re.split(r"(?m)^(#{1,6}\s+.+)$", text)
    title = _heading_title(text) or Path(source_rel).stem
    subject = _infer_subject(source_rel, frontmatter)
    year = _infer_year(title, source_rel, frontmatter)
    region = _first_value(frontmatter, "地区", "省份")
    chunks: list[Chunk] = []

    heading = title
    for index in range(1, len(sections), 2):
        heading = clean_text(sections[index])
        body = clean_text(sections[index + 1] if index + 1 < len(sections) else "")
        if len(body) < 80:
            continue
        start = 0
        while start < len(body):
            piece = body[start:start + max_chars].strip()
            if piece:
                chunk_id = stable_id(source_rel, heading, start, piece[:80])
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        qid="",
                        record_id=stable_id(source_rel, chunk_id),
                        chunk_type="knowledge",
                        text=f"{heading}\n{piece}".strip(),
                        title=title,
                        year=year,
                        region=region,
                        subject=subject,
                        question_type="知识点",
                        source_path=str(source_path.resolve()),
                        source_rel=source_rel,
                    )
                )
            if start + max_chars >= len(body):
                break
            start += max_chars - overlap
    return chunks


def parse_source_file(path: Path, input_dir: Path) -> tuple[list[Question], list[Chunk]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    source_rel = str(path.relative_to(input_dir)).replace("\\", "/")
    frontmatter = parse_front_matter(text)
    questions = parse_questions(text, path, source_rel, frontmatter)
    generic_chunks = _chunk_generic_text(
        text,
        path,
        source_rel,
        frontmatter,
    )
    return questions, generic_chunks


def iter_markdown_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*.md")):
        if ".git" not in path.parts and path.is_file():
            yield path


def question_chunk(question: Question) -> Chunk:
    option_text = "\n".join(
        f"{letter}. {text}" for letter, text in sorted(question.options.items())
    )
    text = "\n".join(
        piece
        for piece in [
            f"题目 {question.number}",
            question.stem,
            option_text,
        ]
        if piece
    )
    return Chunk(
        chunk_id=stable_id(question.record_id, "question"),
        qid=question.qid,
        record_id=question.record_id,
        chunk_type="question",
        text=text,
        title=question.title,
        year=question.year,
        region=question.region,
        subject=question.subject,
        question_type=question.question_type,
        source_path=question.source_path,
        source_rel=question.source_rel,
    )


def explanation_chunk(question: Question) -> Chunk:
    text = "\n".join(
        piece
        for piece in [
            f"题目 {question.number}",
            question.stem,
            f"答案：{question.answer}" if question.answer else "",
            question.explanation,
        ]
        if piece
    )
    return Chunk(
        chunk_id=stable_id(question.record_id, "explanation"),
        qid=question.qid,
        record_id=question.record_id,
        chunk_type="explanation",
        text=text,
        title=question.title,
        year=question.year,
        region=question.region,
        subject=question.subject,
        question_type=question.question_type,
        source_path=question.source_path,
        source_rel=question.source_rel,
    )
