"""Pin-down tests for the shipped system prompt.

These don't test logic — they test the *content* of ``prompts/system.txt`` so
that prompt edits later don't silently delete load-bearing instructions.
"""

from pathlib import Path

import pytest


PROMPT_FILE = Path(__file__).resolve().parents[2] / "prompts" / "system.txt"


@pytest.fixture(scope="module")
def prompt_text() -> str:
    return PROMPT_FILE.read_text()


def test_prompt_file_exists():
    assert PROMPT_FILE.exists(), f"system prompt missing at {PROMPT_FILE}"


def test_prompt_identifies_naila(prompt_text):
    """Without this, the model loses the persona on long turns."""
    assert "NAILA" in prompt_text


def test_prompt_includes_memory_instruction(prompt_text):
    """NAILA must be told she has memory of recent turns and should reference
    them. Llama 3.1's default training pattern is 'AI assistants are stateless'
    — without an explicit override, she'll deny having memory even when we
    feed her the history.

    Pinning this so future prompt edits can't silently regress the behavior.
    """
    text = prompt_text.lower()
    memory_terms = ["memory", "remember", "history", "previous", "recent exchange"]
    assert any(term in text for term in memory_terms), (
        "system prompt must instruct NAILA about her memory; "
        f"none of {memory_terms} found"
    )


def test_prompt_warns_against_stateless_disclaimer(prompt_text):
    """Specifically push back on the trained pattern of 'I have no memory' /
    'each conversation starts fresh' — those phrases keep showing up because
    the prompt didn't explicitly forbid them."""
    text = prompt_text.lower()
    # At least one anti-pattern marker should be present.
    anti_patterns = [
        "no memory",
        "starts fresh",
        "stateless",
        "don't have memory",
        "do not have memory",
    ]
    assert any(p in text for p in anti_patterns), (
        "system prompt should explicitly counter the 'AI is stateless' trained "
        f"pattern; none of {anti_patterns} found"
    )
