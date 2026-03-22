# tools/memory_tools.py
from typing import Optional, Dict, Any
from langchain_core.tools import tool
import json
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from tools.schema import UpdateThreadFieldsInput
memory_store = None
def _simple_concat_summary(chat_history) -> str:
    """Create summary of recent conversation."""
    summary_lines = []
    
    # Chỉ lấy message thực sự có nội dung
    for m in chat_history[-10:]:  # Giảm xuống 10 message
        # Lấy text an toàn
        text = ""
        
        if hasattr(m, 'content'):
            content = m.content
            
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # Xử lý Gemini format
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        text = item['text']
                        break
                    elif isinstance(item, str):
                        text = item
                        break
        
        if text.strip():
            # Xác định role ngắn gọn
            if m.__class__.__name__ == "HumanMessage":
                summary_lines.append(f"👤 {text[:100]}")
            else:
                summary_lines.append(f"🤖 {text[:100]}")
    
    return "\n".join(summary_lines)


def init_memory_tools(store):
    global memory_store
    memory_store = store

@tool
def db_get_user_profile(user_id: str) -> Dict[str, Any]:
    """Load user profile from DB."""
    return memory_store.load_user_profile(user_id)

@tool
def db_upsert_user_profile(
    user_id: str,
    level: Optional[str] = None,
    focus: Optional[str] = None,
    session_minutes: Optional[int] = None,
    accessibility: Optional[str] = None,
) -> Dict[str, Any]:
    """Upsert user learning profile into DB, return updated profile."""
    memory_store.upsert_user_profile(
        user_id,
        level=level,
        focus=focus,
        session_minutes=session_minutes,
        accessibility=accessibility,
    )
    return memory_store.load_user_profile(user_id)

@tool
def db_get_thread(user_id: str, thread_id: str) -> Dict[str, Any]:
    """Load thread state from DB."""
    return memory_store.load_thread(user_id, thread_id)

def _robust_json_loads(raw: str) -> dict:
    """Parse JSON robustly, handling extra data after the JSON object."""
    s = (raw or "").strip()
    if not s:
        return {}
    # Fast path
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Find first '{' and match its closing '}' with bracket counting
    start = s.find("{")
    if start == -1:
        return {}
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except json.JSONDecodeError:
                    break
    # Last resort: first { to last }
    end = s.rfind("}")
    if end > start:
        try:
            return json.loads(s[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


@tool(args_schema=UpdateThreadFieldsInput)
def db_update_thread_fields(user_id: str, thread_id: str, fields_json: str, fields: Optional[Dict[str, Any]] = None):
    """Update thread document fields in MongoDB using $set. Accepts either fields_json (JSON string) or fields (dict)."""
    # Support both payload styles: "fields_json" (string) or "fields" (dict)
    if fields is None:
        fields = _robust_json_loads(fields_json) if fields_json else {}
    memory_store.update_thread_fields(user_id, thread_id, fields)
    return {"ok": True, "updated_keys": list(fields.keys())}

