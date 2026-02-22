import logging
import uuid
from pathlib import Path

from llama_index.core import SimpleDirectoryReader

from rag.core.interfaces import DocumentLoader
from rag.core.models import LoadedDocument

logger = logging.getLogger(__name__)

DEFAULT_SUPPORTED_EXTENSIONS = frozenset({".md", ".txt"})


class LlamaIndexDocumentLoader(DocumentLoader):
    def __init__(self, supported_extensions: frozenset[str] | None = None) -> None:
        self._supported = (
            supported_extensions
            if supported_extensions is not None
            else DEFAULT_SUPPORTED_EXTENSIONS
        )

    def load(self, path: Path) -> list[LoadedDocument]:
        """Load one or more documents from path."""
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            return [self._load_file(path)]

        # Directory: collect and load all supported files, sorted for determinism
        files = sorted(
            f
            for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in self._supported
        )
        return [self._load_file(f) for f in files]

    def _load_file(self, file_path: Path) -> LoadedDocument:
        """Load a single file and return a LoadedDocument."""
        if file_path.suffix.lower() not in self._supported:
            raise ValueError(
                f"Unsupported file extension '{file_path.suffix}'. "
                f"Supported: {sorted(self._supported)}"
            )

        document_id = self._create_document_id(file_path)

        reader = SimpleDirectoryReader(input_files=[file_path], encoding="utf-8")
        llama_docs = reader.load_data()

        text = "\n\n".join(doc.text for doc in llama_docs if doc.text.strip())
        if not text.strip():
            raise ValueError(f"File is empty or whitespace-only: {file_path}")

        llama_meta = llama_docs[0].metadata if llama_docs else {}

        logger.info(
            "Loaded '%s' (%d chars, id=%s...)",
            file_path.name,
            len(text),
            document_id[:8],
        )

        return LoadedDocument(
            text=text,
            document_id=document_id,
            source_name=file_path.name,
            file_path=str(file_path.resolve()),
            file_type=llama_meta.get("file_type", "text/plain"),
            metadata={
                "creation_date": llama_meta.get("creation_date"),
                "last_modified_date": llama_meta.get("last_modified_date"),
            },
        )

    def _create_document_id(self, file_path: Path) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, file_path.name))
