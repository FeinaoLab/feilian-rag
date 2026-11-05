"""文档加载：把一个目录里的知识文件读成 LangChain Document。

相比原始 demo 的改进：
1. 递归遍历子目录，而不只是顶层 *.txt；
2. 支持 .txt / .md / .pdf / .docx，PDF 按页保留页码，回答时可以指到具体页；
3. 每个 Document 都带 source（相对路径）+ chunk 之外的位置信息，方便引用；
4. 单个文件读失败不影响整批，问题收集起来一起报。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from .errors import DocumentError

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
PDF_SUFFIXES = {".pdf"}
DOCX_SUFFIXES = {".docx"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES


def iter_files(root: Path, exclude: Iterable[Path] = ()) -> list[Path]:
    """递归列出 root 下所有受支持的文件，按路径排序保证幂等。"""
    root = Path(root)
    if not root.exists():
        raise DocumentError(f"文档目录不存在：{root}")
    if not root.is_dir():
        raise DocumentError(f"文档目录不是一个文件夹：{root}")

    excluded = [p.resolve() for p in exclude]
    found: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if any(str(path.resolve()).startswith(str(ex)) for ex in excluded):
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        found.append(path)
    return sorted(found)


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _load_text(path: Path, meta: dict) -> list[Document]:
    text = _read_text(path).strip()
    if not text:
        return []
    return [Document(page_content=text, metadata=meta)]


def _load_pdf(path: Path, meta: dict) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentError("解析 PDF 需要 pypdf，请执行 pip install pypdf") from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pypdf 的异常类型不稳定，统一收敛
        raise DocumentError(f"PDF 读取失败：{path.name}（{exc}）") from exc

    docs: list[Document] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:  # pragma: no cover - 个别页损坏时跳过
            text = ""
        if not text:
            continue
        docs.append(Document(page_content=text, metadata={**meta, "page": index}))
    return docs


def _load_docx(path: Path, meta: dict) -> list[Document]:
    try:
        import docx
    except ImportError as exc:
        raise DocumentError("解析 docx 需要 python-docx，请执行 pip install python-docx") from exc

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise DocumentError(f"docx 读取失败：{path.name}（{exc}）") from exc

    blocks = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))
    text = "\n".join(blocks).strip()
    if not text:
        return []
    return [Document(page_content=text, metadata=meta)]


def load_file(path: Path, root: Path) -> list[Document]:
    """读单个文件，返回若干 Document（PDF 会一页一个）。"""
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.name

    meta = {
        "source": relative,
        "file_type": suffix.lstrip("."),
        "size_bytes": path.stat().st_size,
    }

    if suffix in TEXT_SUFFIXES:
        return _load_text(path, meta)
    if suffix in PDF_SUFFIXES:
        return _load_pdf(path, meta)
    if suffix in DOCX_SUFFIXES:
        return _load_docx(path, meta)
    return []


def load_documents(root: Path, exclude: Iterable[Path] = ()) -> tuple[list[Document], list[str]]:
    """加载整个目录，返回 (文档列表, 告警列表)。"""
    documents: list[Document] = []
    warnings: list[str] = []
    for path in iter_files(root, exclude=exclude):
        try:
            documents.extend(load_file(path, root))
        except DocumentError as exc:
            warnings.append(str(exc))
    return documents, warnings
