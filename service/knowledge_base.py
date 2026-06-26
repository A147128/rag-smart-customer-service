"""
Knowledge-base ingestion service.

Uploads multiple file types into Chroma with deduplication, richer chunk metadata, and
knowledge-base versioning for cache invalidation.

Supported file types: txt, md, pdf, docx, xlsx, pptx
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config import config_data as config
from service.knowledge_version import bump_kb_version, get_kb_version
from service.parsers import ParsedDocument, parse_file, validate_file


def check_md5(md5_str: str) -> bool:
    """Return True when the content MD5 has already been ingested."""
    path = Path(config.md5_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")
        return False
    return any(line.strip() == md5_str for line in path.read_text(encoding="utf-8").splitlines())


def save_md5(md5_str: str) -> None:
    """Persist an ingested content MD5."""
    path = Path(config.md5_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(md5_str + "\n")


def get_string_md5(input_str: str, encoding: str = "utf-8") -> str:
    """Return MD5 hex digest for a string."""
    return hashlib.md5(input_str.encode(encoding=encoding)).hexdigest()


def get_bytes_md5(input_bytes: bytes) -> str:
    """Return MD5 hex digest for bytes."""
    return hashlib.md5(input_bytes).hexdigest()


def _safe_file_id(filename: str, content_md5: str) -> str:
    stem = Path(filename or "knowledge").stem or "knowledge"
    safe_stem = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", stem).strip("_") or "knowledge"
    return f"{safe_stem}_{content_md5[:8]}"


def _guess_section(chunk: str, fallback: str) -> str:
    for line in chunk.splitlines():
        cleaned = line.strip().lstrip("#").strip()
        if cleaned:
            return cleaned[:80]
    return fallback


class KnowledgeBaseService:
    def __init__(self) -> None:
        os.makedirs(config.persist_directory, exist_ok=True)

        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),
            persist_directory=config.persist_directory,
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def upload_file(self, file_bytes: bytes, filename: str) -> dict:
        """
        Upload a file (supports txt, md, pdf, docx, xlsx, pptx).

        Args:
            file_bytes: File binary content
            filename: File name

        Returns:
            Result dict with status, chunks count, kb_version, and any warnings
        """
        # Validate file
        is_valid, error_msg = validate_file(file_bytes, filename)
        if not is_valid:
            return {"status": "error", "message": error_msg, "filename": filename}

        # Parse file
        try:
            parsed_docs = parse_file(file_bytes, filename)
        except ValueError as e:
            return {"status": "error", "message": str(e), "filename": filename}

        if not parsed_docs:
            return {"status": "skipped", "message": "文件内容为空或无有效内容", "filename": filename}

        # Process each parsed document
        results = []
        total_chunks = 0
        warnings = []

        for parsed_doc in parsed_docs:
            result = self._process_parsed_document(parsed_doc, file_bytes)
            if result["status"] == "success":
                total_chunks += result["chunks"]
            elif result["status"] == "skipped":
                warnings.append(result["message"])
            elif result["status"] == "error":
                warnings.append(result["message"])

        # Bump KB version once for the whole file
        new_version = bump_kb_version()

        if total_chunks > 0:
            return {
                "status": "success",
                "message": f"文件已成功载入向量库，共 {total_chunks} 个 chunks",
                "filename": filename,
                "chunks": total_chunks,
                "kb_version": new_version,
                "warnings": warnings,
            }
        else:
            return {
                "status": "skipped",
                "message": "所有内容已存在或为空",
                "filename": filename,
                "warnings": warnings,
            }

    def _process_parsed_document(
        self, parsed_doc: ParsedDocument, original_file_bytes: bytes
    ) -> dict:
        """Process a single parsed document and add to vector store."""
        normalized = parsed_doc.content.strip()
        if not normalized:
            return {"status": "skipped", "message": "内容为空"}

        content_md5 = get_string_md5(normalized)
        if check_md5(content_md5):
            return {"status": "skipped", "message": "内容已存在知识库中"}

        chunks = self._split_text(normalized)
        file_md5 = get_bytes_md5(original_file_bytes)
        file_id = _safe_file_id(parsed_doc.source, file_md5)
        next_version = get_kb_version() + 1
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        metadatas = []
        ids = []
        for idx, chunk in enumerate(chunks):
            chunk_md5 = get_string_md5(chunk)
            chunk_id = f"{file_id}_{idx:04d}"

            # Build metadata with extended fields
            metadata = {
                "source": parsed_doc.source,
                "file_id": file_id,
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "chunk_count": len(chunks),
                "content_md5": chunk_md5,
                "file_md5": file_md5,
                "kb_version": next_version,
                "section": _guess_section(chunk, parsed_doc.source),
                "doc_type": parsed_doc.doc_type,
                "create_time": create_time,
                "operator": "system",
            }

            # Add extended metadata fields
            if parsed_doc.page_number is not None:
                metadata["page_number"] = parsed_doc.page_number
            if parsed_doc.sheet_name is not None:
                metadata["sheet_name"] = parsed_doc.sheet_name
            if parsed_doc.sheet_index is not None:
                metadata["sheet_index"] = parsed_doc.sheet_index
            if parsed_doc.extra_metadata:
                metadata.update(parsed_doc.extra_metadata)

            ids.append(chunk_id)
            metadatas.append(metadata)

        self.chroma.add_texts(chunks, metadatas=metadatas, ids=ids)
        save_md5(content_md5)

        return {"status": "success", "chunks": len(chunks)}

    def upload_by_str(self, data: str, filename: str) -> str:
        """Vectorize text content and store it in Chroma. (Legacy method for txt only)"""
        normalized = data.strip()
        if not normalized:
            return "[跳过]上传内容为空"

        content_md5 = get_string_md5(normalized)
        if check_md5(content_md5):
            return "[跳过]内容已经存在知识库中"

        chunks = self._split_text(normalized)
        file_id = _safe_file_id(filename, content_md5)
        next_version = get_kb_version() + 1
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        metadatas = []
        ids = []
        for idx, chunk in enumerate(chunks):
            chunk_md5 = get_string_md5(chunk)
            chunk_id = f"{file_id}_{idx:04d}"
            ids.append(chunk_id)
            metadatas.append(
                {
                    "source": filename,
                    "file_id": file_id,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "chunk_count": len(chunks),
                    "content_md5": chunk_md5,
                    "file_md5": content_md5,
                    "kb_version": next_version,
                    "section": _guess_section(chunk, filename),
                    "doc_type": "txt",
                    "create_time": create_time,
                    "operator": "system",
                }
            )

        self.chroma.add_texts(chunks, metadatas=metadatas, ids=ids)
        save_md5(content_md5)
        version = bump_kb_version()

        return f"[成功]内容已经成功载入向量库，chunks={len(chunks)}，kb_version={version}"

    def _split_text(self, data: str) -> list[str]:
        if len(data) > config.max_split_char_number:
            return [chunk for chunk in self.spliter.split_text(data) if chunk.strip()]
        return [data]


if __name__ == "__main__":
    service = KnowledgeBaseService()
    r = service.upload_by_str("周杰轮222", "testfile")
    print(r)