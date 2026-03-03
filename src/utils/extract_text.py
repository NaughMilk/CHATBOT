import re


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting that TTS would read aloud."""
    # Remove bold **text** and __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    # Remove italic *text* and _text_ (single)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'\1', text)
    # Remove backticks `code`
    text = re.sub(r'`(.+?)`', r'\1', text)
    # Replace smart/curly quotes with nothing (TTS reads "quote" literally)
    text = text.replace('"', '').replace('"', '').replace(''', "'").replace(''', "'")
    # Remove standard double quotes around words (keep apostrophes)
    text = re.sub(r'"([^"]*)"', r'\1', text)
    # Remove markdown headers (# ## ###)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Remove bullet points (- or *)
    text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
    # Clean up extra whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_clean_text(content) -> str:
    """Extract clean text from Gemini response."""
    if not content:
        return ""
    
    if isinstance(content, str):
        return _strip_markdown(content)
    
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                return _strip_markdown(item['text'])
            elif isinstance(item, str):
                return _strip_markdown(item)
    
    return _strip_markdown(str(content)) if content else ""
