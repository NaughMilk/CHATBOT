"""
Google Cloud Text-to-Speech with SSML support.
Handles mixed Vietnamese / English / IPA in a single audio stream.
"""

import io
import os
import re
from typing import List, Tuple

from google.cloud import texttospeech
from google.oauth2 import service_account

# ────────────────── credentials ──────────────────
_TTS_CLIENT = None

def _get_client() -> texttospeech.TextToSpeechClient:
    global _TTS_CLIENT
    if _TTS_CLIENT is not None:
        return _TTS_CLIENT

    sa_path = os.getenv("TTS_SERVICE_ACCOUNT", "")
    if sa_path and os.path.isfile(sa_path):
        creds = service_account.Credentials.from_service_account_file(sa_path)
        _TTS_CLIENT = texttospeech.TextToSpeechClient(credentials=creds)
    else:
        # fallback: default credentials (GOOGLE_APPLICATION_CREDENTIALS)
        _TTS_CLIENT = texttospeech.TextToSpeechClient()
    return _TTS_CLIENT


# ────────────────── language detection ──────────────────
_VI_DIACRITIC_RE = re.compile(r"[À-ỹ]")
_IPA_CHARS = re.compile(r"[ˈˌəɛɪɔʊʃʒŋθðæɑɜɐɒʌɹɾɫʔ]")
_IPA_BLOCK_RE = re.compile(r"[/\[]([\u0250-\u02FF\u0300-\u036F\u0041-\u005A\u0061-\u007Aˈˌː.ᵻ\s]+)[/\]]")


def _is_ipa_block(text: str) -> bool:
    """Check if text looks like an IPA transcription /.../ or [...]."""
    t = text.strip()
    if (t.startswith("/") and t.endswith("/")) or (t.startswith("[") and t.endswith("]")):
        inner = t[1:-1]
        return bool(_IPA_CHARS.search(inner))
    return False


def _is_english(text: str) -> bool:
    """Heuristic: text is English if it has ≥1 Latin word and no Vietnamese diacritics."""
    if _VI_DIACRITIC_RE.search(text):
        return False
    words = re.findall(r"[A-Za-z]{2,}", text)
    return len(words) >= 1


# ────────────────── SSML builder ──────────────────
def _escape_xml(text: str) -> str:
    """Escape XML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _segment_text(text: str) -> List[Tuple[str, str]]:
    """
    Split text into (segment, type) tuples.
    type is one of: 'vi', 'en', 'ipa'
    """
    segments: List[Tuple[str, str]] = []

    # First, split out IPA blocks like /ˌɛdʒʊˈkeɪʃən/ or [ˌɛdʒʊˈkeɪʃən]
    ipa_pattern = re.compile(r"([/\[][\u0020-\u007E\u0250-\u02FF\u0300-\u036Fˈˌː.ᵻ]+[/\]])")
    parts = ipa_pattern.split(text)

    for part in parts:
        if not part.strip():
            continue

        if _is_ipa_block(part):
            # Extract the IPA content without delimiters
            inner = part.strip()[1:-1].strip()
            segments.append((inner, "ipa"))
            continue

        # Split remaining text by lines first, then sentences for better language detection
        lines = part.split("\n")
        for line in lines:
            if not line.strip():
                continue
            # Split on sentence boundaries within each line
            sentences = re.split(r"(?<=[.!?])\s+", line)

            for sentence in sentences:
                if not sentence.strip():
                    continue

                # Check for colon-separated patterns like "Nghĩa là: education"
                if ":" in sentence:
                    colon_idx = sentence.index(":")
                    head = sentence[: colon_idx + 1]
                    tail = sentence[colon_idx + 1 :]
                    if head.strip():
                        segments.append((head, "vi"))
                    if tail.strip():
                        if _is_english(tail):
                            segments.append((tail, "en"))
                        else:
                            segments.append((tail, "vi"))
                    continue

                # Check for quoted English: 'word' or "word"
                quote_re = re.compile(r"""(['"])(.*?)\1""")
                last = 0
                found_quotes = False
                for m in quote_re.finditer(sentence):
                    found_quotes = True
                    before = sentence[last : m.start()]
                    if before.strip():
                        lang = "en" if _is_english(before) else "vi"
                        segments.append((before, lang))
                    inner = m.group(2)
                    if inner.strip():
                        segments.append((inner, "en"))
                    last = m.end()

                if found_quotes:
                    tail = sentence[last:]
                    if tail.strip():
                        lang = "en" if _is_english(tail) else "vi"
                        segments.append((tail, lang))
                    continue

                # No special pattern — detect by content
                lang = "en" if _is_english(sentence) else "vi"
                segments.append((sentence, lang))

    return segments


def build_ssml(text: str) -> str:
    """Build SSML from mixed-language text."""
    segments = _segment_text(text)
    if not segments:
        return f"<speak>{_escape_xml(text)}</speak>"

    parts: List[str] = []
    for seg_text, seg_type in segments:
        escaped = _escape_xml(seg_text.strip())
        if not escaped:
            continue

        if seg_type == "ipa":
            # Find a word nearby to use as fallback text
            # Use the IPA itself as display text
            parts.append(
                f'<phoneme alphabet="ipa" ph="{escaped}">{escaped}</phoneme>'
            )
        elif seg_type == "en":
            parts.append(f'<lang xml:lang="en-US">{escaped}</lang>')
        else:
            parts.append(escaped)

    return "<speak>" + " ".join(parts) + "</speak>"


# ────────────────── synthesis ──────────────────
_VOICE = texttospeech.VoiceSelectionParams(
    language_code="vi-VN",
    name="vi-VN-Neural2-A",
)

_AUDIO_CONFIG = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3,
    speaking_rate=1.0,
    pitch=0.0,
)

# Google Cloud TTS has a 5000 byte SSML limit
_MAX_SSML_BYTES = 4800


def _chunk_ssml(ssml: str) -> List[str]:
    """
    If SSML exceeds the byte limit, split into multiple <speak> blocks.
    Splits on sentence boundaries in the original text.
    """
    if len(ssml.encode("utf-8")) <= _MAX_SSML_BYTES:
        return [ssml]

    # Strip <speak> tags and split content
    inner = ssml
    if inner.startswith("<speak>"):
        inner = inner[7:]
    if inner.endswith("</speak>"):
        inner = inner[:-8]

    # Split on sentence-like boundaries (period, !, ?)
    # but be careful with XML tags
    sentences = re.split(r"(?<=[.!?])\s+", inner)

    chunks: List[str] = []
    current = ""
    for s in sentences:
        test = f"<speak>{current} {s}</speak>"
        if len(test.encode("utf-8")) > _MAX_SSML_BYTES and current:
            chunks.append(f"<speak>{current.strip()}</speak>")
            current = s
        else:
            current = f"{current} {s}" if current else s

    if current.strip():
        chunks.append(f"<speak>{current.strip()}</speak>")

    return chunks if chunks else [ssml]


def synthesize_speech(text: str) -> bytes:
    """
    Convert text to speech audio (MP3 bytes).
    Handles mixed Vietnamese / English / IPA via SSML.
    """
    text = (text or "").strip()
    if not text:
        return b""

    client = _get_client()
    ssml = build_ssml(text)
    chunks = _chunk_ssml(ssml)

    audio_parts: List[bytes] = []
    for chunk in chunks:
        synthesis_input = texttospeech.SynthesisInput(ssml=chunk)
        try:
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=_VOICE,
                audio_config=_AUDIO_CONFIG,
            )
            if response.audio_content:
                audio_parts.append(response.audio_content)
        except Exception as exc:
            print(f"[TTS] Error synthesizing chunk: {exc}", flush=True)
            continue

    if not audio_parts:
        raise RuntimeError("TTS synthesis failed for all chunks")

    # If single chunk, return directly
    if len(audio_parts) == 1:
        return audio_parts[0]

    # Multiple chunks: concatenate MP3 bytes (MP3 is concatenable)
    return b"".join(audio_parts)
