import logging
from pathlib import Path

import pytest

from rag.core.models import LoadedDocument
from rag.services.loader import DEFAULT_SUPPORTED_EXTENSIONS, LlamaIndexDocumentLoader


def make_loader(
    supported_extensions: frozenset[str] | None = None,
) -> LlamaIndexDocumentLoader:
    return LlamaIndexDocumentLoader(supported_extensions=supported_extensions)


@pytest.mark.unit
class TestInputValidation:
    """Loader rejects bad inputs before any I/O."""

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        """Non-existent path raises FileNotFoundError."""
        loader = make_loader()
        missing = tmp_path / "nonexistent.md"

        with pytest.raises(FileNotFoundError, match=r"nonexistent\.md"):
            loader.load(missing)

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """Unsupported file extension raises ValueError before any reading."""
        loader = make_loader()
        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        with pytest.raises(ValueError, match=r"[Uu]nsupported"):
            loader.load(pdf_file)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        """An existing file with no content raises ValueError."""
        loader = make_loader()
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match=r"[Ee]mpty|[Ww]hitespace"):
            loader.load(empty_file)

    def test_whitespace_only_raises(self, tmp_path: Path) -> None:
        """A file containing only whitespace is rejected."""
        loader = make_loader()
        ws_file = tmp_path / "ws.txt"
        ws_file.write_text("   \n\n\t  ", encoding="utf-8")

        with pytest.raises(ValueError, match=r"[Ee]mpty|[Ww]hitespace"):
            loader.load(ws_file)


@pytest.mark.unit
class TestLoadedDocumentShape:
    """load() returns correctly populated LoadedDocuments."""

    def test_document_metadata_fields(self, tmp_path: Path) -> None:
        """Returned document contains correct content and metadata shape."""
        ext = next(iter(DEFAULT_SUPPORTED_EXTENSIONS))
        filename = f"test_doc.{ext}"
        content = "Standard test content."
        file_path = tmp_path / filename
        file_path.write_text(content, encoding="utf-8")

        result = make_loader().load(file_path)
        doc = result[0]

        # Structure and Content
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(doc, LoadedDocument)
        assert doc.text.strip() == content

        # Path-based Metadata
        assert Path(doc.file_path).is_absolute()
        assert doc.file_path == str(file_path.resolve())
        assert doc.source_name == filename

        # Type Metadata
        assert doc.file_type
        assert "text" in doc.file_type.lower()

    def test_document_id_logic(self, tmp_path: Path) -> None:
        """document_id is deterministic."""
        ext = next(iter(DEFAULT_SUPPORTED_EXTENSIONS))
        path_a = tmp_path / f"a{ext}"
        path_b = tmp_path / f"b{ext}"
        path_a.write_text("content", encoding="utf-8")
        path_b.write_text("content", encoding="utf-8")

        loader = make_loader()
        doc_a = loader.load(path_a)[0]
        doc_a_repeat = loader.load(path_a)[0]
        doc_b = loader.load(path_b)[0]

        # Determinism
        assert doc_a.document_id == doc_a_repeat.document_id

        # Uniqueness
        assert doc_a.document_id != doc_b.document_id


@pytest.mark.unit
class TestDirectoryLoading:
    """load() with a directory path returns one document per supported file."""

    def test_directory_returns_multiple_documents(self, tmp_path: Path) -> None:
        """Loading a directory returns as many docs as supported files in it."""
        extensions = list(DEFAULT_SUPPORTED_EXTENSIONS)
        for i, ext in enumerate(extensions):
            (tmp_path / f"test_{i}.{ext}").write_text(f"Content {i}", encoding="utf-8")

        (tmp_path / "ignored.unsupported").write_text("Ignore", encoding="utf-8")

        result = make_loader().load(tmp_path)

        assert len(result) == len(extensions)

    def test_each_file_becomes_one_document(self, tmp_path: Path) -> None:
        """Each file in the directory produces exactly one document."""
        ext = next(iter(DEFAULT_SUPPORTED_EXTENSIONS))
        for i in range(3):
            (tmp_path / f"doc{i}{ext}").write_text(f"Content {i}", encoding="utf-8")

        result = make_loader().load(tmp_path)

        assert len(result) == 3
        source_names = {doc.source_name for doc in result}
        assert source_names == {f"doc0{ext}", f"doc1{ext}", f"doc2{ext}"}

    def test_unsupported_files_skipped_in_directory(self, tmp_path: Path) -> None:
        """Unsupported extensions are silently skipped; no error raised."""
        valid_ext = next(iter(DEFAULT_SUPPORTED_EXTENSIONS))
        invalid_ext = f"{valid_ext}_invalid"

        (tmp_path / f"valid.{valid_ext}").write_text("Valid", encoding="utf-8")
        (tmp_path / f"invalid.{invalid_ext}").write_text("Skip", encoding="utf-8")

        result = make_loader().load(tmp_path)

        assert len(result) == 1
        assert result[0].source_name.endswith(valid_ext)

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """An empty directory returns an empty list without raising."""
        result = make_loader().load(tmp_path)

        assert result == []


@pytest.mark.unit
class TestSupportedFormats:
    """Loader handles all supported file formats correctly."""

    @pytest.mark.parametrize("ext", list(DEFAULT_SUPPORTED_EXTENSIONS))
    def test_loads_markdown_file(self, tmp_path: Path, ext: str) -> None:
        """Default extensions are loaded and produce non-empty text."""
        content = f"Sample content for {ext}"
        test_file = tmp_path / f"test{ext}"
        test_file.write_text(content, encoding="utf-8")

        result = make_loader().load(test_file)

        assert len(result) == 1
        assert len(result[0].text) > 0

    def test_unicode_content_preserved(self, tmp_path: Path) -> None:
        """Unicode content is preserved exactly round-trip."""
        unicode_text = "こんにちは, 🌍, €100, 🏁"
        test_file = tmp_path / "unicode.txt"
        test_file.write_text(unicode_text, encoding="utf-8")

        result = make_loader(supported_extensions={".txt"}).load(test_file)

        assert result[0].text.strip() == unicode_text

    def test_custom_supported_extensions(self, tmp_path: Path) -> None:
        """Constructor accepts a custom extension allow-list."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\nval1,val2", encoding="utf-8")

        # Verify rejection by default
        if ".csv" not in DEFAULT_SUPPORTED_EXTENSIONS:
            with pytest.raises(ValueError, match=r"[Uu]nsupported"):
                make_loader().load(csv_file)

        # Verify acceptance by custom loader
        custom_loader = make_loader(supported_extensions={".csv"})
        result = custom_loader.load(csv_file)

        assert len(result) == 1
        assert "val1" in result[0].text


@pytest.mark.unit
class TestLoggingBehavior:
    """Loader emits expected log messages."""

    def test_successful_load_logs_info(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A successful load emits at least one INFO log containing the filename."""
        ext = next(iter(DEFAULT_SUPPORTED_EXTENSIONS))
        filename = f"test{ext}"
        test_file = tmp_path / filename
        test_file.write_text("Content", encoding="utf-8")

        loader = make_loader()

        with caplog.at_level(logging.INFO, logger="rag.services.loader"):
            loader.load(test_file)

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]

        assert info_records, "Expected at least one INFO log from the loader"
        assert any(filename in r.message for r in info_records), (
            f"Expected filename '{filename}' in INFO logs. Found: {[r.message for r in info_records]}"
        )
