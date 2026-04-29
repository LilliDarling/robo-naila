"""Buffer streamed LLM tokens; emit text at sentence boundaries.

Used by the streaming response path so TTS can begin synthesizing the first
sentence as soon as the LLM emits it, instead of waiting for the entire
response to finish generating.
"""

from __future__ import annotations

import re
from typing import List

# Sentence terminators followed by whitespace (or end of buffer at flush).
# Trailing quote/paren/bracket are absorbed so "She said 'hi.'" stays intact.
_SENTENCE_END = re.compile(r'[.!?]["\')\]]?(?=\s|$)')

# If no terminator appears within this many chars, force-flush so a runaway
# LLM (e.g. emitting a giant comma-separated list) doesn't block playback.
DEFAULT_MAX_BUFFER_CHARS = 240

# Don't emit fragments shorter than this — short sub-sentences (e.g. "Yes.")
# in mid-stream look like artifacts; the receiver-side jitter buffer also
# handles tiny chunks poorly.
DEFAULT_MIN_FRAGMENT_CHARS = 2


class SentenceBuffer:
    """Accumulate streamed tokens, emit complete sentences as they form."""

    def __init__(
        self,
        max_buffer_chars: int = DEFAULT_MAX_BUFFER_CHARS,
        min_fragment_chars: int = DEFAULT_MIN_FRAGMENT_CHARS,
    ) -> None:
        self._buf: str = ""
        self._max_buffer_chars = max_buffer_chars
        self._min_fragment_chars = min_fragment_chars

    def feed(self, chunk: str) -> List[str]:
        """Append new tokens; return list of complete sentences ready to send."""
        if not chunk:
            return []
        self._buf += chunk
        return self._extract_sentences()

    def _extract_sentences(self) -> List[str]:
        out: List[str] = []
        while True:
            match = _SENTENCE_END.search(self._buf)
            if match is None:
                # No terminator; force-flush only if buffer is dangerously long.
                if len(self._buf) >= self._max_buffer_chars:
                    text = self._buf.strip()
                    self._buf = ""
                    if len(text) >= self._min_fragment_chars:
                        out.append(text)
                break

            cut = match.end()
            sentence = self._buf[:cut].strip()
            self._buf = self._buf[cut:].lstrip()
            if len(sentence) >= self._min_fragment_chars:
                out.append(sentence)
        return out

    def flush(self) -> str:
        """Return any text remaining in the buffer (call once after stream end)."""
        rest = self._buf.strip()
        self._buf = ""
        return rest if len(rest) >= self._min_fragment_chars else ""
