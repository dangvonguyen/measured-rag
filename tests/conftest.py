"""Shared test fixtures used across unit and integration tests."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DOCUMENTS_DIR = FIXTURES_DIR / "documents"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def documents_dir() -> Path:
    return DOCUMENTS_DIR


@pytest.fixture
def sample_tos_text() -> str:
    """Full text of the Terms of Service test document."""
    return (DOCUMENTS_DIR / "terms_of_service.md").read_text(encoding="utf-8")


@pytest.fixture
def sample_privacy_text() -> str:
    return (DOCUMENTS_DIR / "privacy_policy.md").read_text(encoding="utf-8")


@pytest.fixture
def sample_single_sentence() -> str:
    return (DOCUMENTS_DIR / "single_sentence.md").read_text(encoding="utf-8")


@pytest.fixture
def sample_unicode_text() -> str:
    return (DOCUMENTS_DIR / "unicode_text.md").read_text(encoding="utf-8")


@pytest.fixture
def sample_metadata() -> dict[str, str | int | None]:
    """Standard metadata dict for test chunks."""
    return {
        "document_id": "doc-test-001",
        "source_name": "terms_of_service.md",
        "page_number": None,
    }
