"""Unit test fixtures."""

import pytest
import tiktoken


@pytest.fixture
def tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")
