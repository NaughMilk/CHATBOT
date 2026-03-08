# agents/score_agent.py
import re
from difflib import SequenceMatcher
from typing import Dict, Any, Optional

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_SENT_RE = re.compile(r"[.!?]+")

def _norm(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(_WORD_RE.findall(str(text).lower()))

def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()

def _best_sentence_similarity(text: str, answer_norm: str) -> float:
    parts = [p.strip() for p in _SENT_RE.split(text or "") if p.strip()]
    if not parts:
        return _similarity(_norm(text or ""), answer_norm)
    best = 0.0
    for p in parts:
        best = max(best, _similarity(_norm(p), answer_norm))
    return best

def _tokens(text: Optional[str]) -> list:
    return _WORD_RE.findall((text or "").lower())

def _has_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in (text or ""))

def _choice_from_text(text: str) -> Optional[str]:
    raw = (text or "").strip().lower()
    if not raw:
        return None
    if raw in {"a", "b", "c", "d"}:
        return raw.upper()

    m = re.search(r"\b(?:option|choice|answer|choose|select|chon|lua chon)\b\s*([abcd])\b", raw)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([abcd])\b\s*(?:option|choice|answer|choose|select|chon|lua chon)\b", raw)
    if m:
        return m.group(1).upper()

    return None


def score_step(phase: str, expected: Dict[str, Any], user_text: Optional[str]) -> Dict[str, Any]:
    answer = _norm(user_text)
    if not answer:
        return {"passed": False, "feedback": "Mình chưa nghe rõ. Bạn nói lại nhé."}

    if phase == "learn_vocab":
        raw_expected = expected.get("word") or ""
        raw_expected = _norm(raw_expected)
        ratio = _best_sentence_similarity(raw_expected, answer)
        passed = ratio >= 0.8
        feedback = "Tốt, bạn đọc rất chính xác luôn đó!" if passed else "Mình thấy chưa đúng lắm, bạn thử lại giúp mình nhé!"
        return {"passed": passed, "feedback": feedback}

    if phase == "learn_grammar":
        target = _norm(expected.get("example_en") or "")
        ratio = _similarity(target, answer)
        passed = ratio >= 0.8
        feedback = "Tốt, bạn đọc rất chính xác luôn đó!" if passed else "Mình thấy chưa đúng lắm, bạn thử lại giúp mình nhé!"
        return {"passed": passed, "feedback": feedback}
    
    if phase == "learn_conversation":
        prompt_en = expected.get("en") or ""
        word_count = len(answer.split())
        # If user gives a short reply, treat it as attempting to repeat the prompt.
        if prompt_en and word_count <= 10:
            tgt_tokens = _tokens(prompt_en)
            ans_tokens = _tokens(user_text or "")
            if not ans_tokens:
                return {"passed": False, "feedback": "Mình chưa nghe rõ. Bạn nói lại nhé."}
            
            # Case-insensitive comparison (ignore capitalization and punctuation)
            if ans_tokens == tgt_tokens:
                return {"passed": True, "feedback": "Tuyệt vời! Bạn đã nói đúng rồi."}

            sm = SequenceMatcher(None, tgt_tokens, ans_tokens)
            wrong_pairs = []
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag in {"replace", "delete", "insert"}:
                    left = " ".join(tgt_tokens[i1:i2]).strip()
                    right = " ".join(ans_tokens[j1:j2]).strip()
                    if left or right:
                        wrong_pairs.append((right, left))
            if wrong_pairs:
                wrong, correct = wrong_pairs[0]
                wrong = (wrong or "").lower()
                correct = (correct or "").lower()
                if correct and wrong:
                    fb = f"Bạn sai từ: '{wrong}' cần '{correct}'. Bạn thử lại nhé."
                elif correct and not wrong:
                    fb = f"Bạn còn thiếu từ: '{correct}'. Bạn thử lại nhé."
                else:
                    fb = f"Bạn thừa từ: '{wrong}'. Bạn thử lại nhé."
                return {"passed": False, "feedback": fb}
            return {"passed": False, "feedback": "Bạn nói chưa đúng. Bạn thử lại nhé."}

        passed = word_count >= 6
        feedback = "Tốt. Bạn nói đầy đủ." if passed else "Bạn nói thêm chút nữa nhé."
        return {"passed": passed, "feedback": feedback}

    if phase == "evaluation_material":
        q_type = expected.get("type")
        if q_type == "multiple_choice":
            choice = _choice_from_text(user_text or "")
            if not choice:
                return {"passed": False, "feedback": "Bạn cần chọn A, B, C hoặc D nhé."}

            key = expected.get("answer_key")
            print(key, choice)
            choices = expected.get("choices") or []
            print(choices)
    # 1) Nếu answer_key là letter A/B/C/D
            if key in {"A", "B", "C", "D"}:
                passed = (choice == key)
                return {"passed": passed, "feedback": "Chính xác." if passed else "Chưa đúng. Bạn thử lại nhé."}

    # 2) Nếu answer_key là TEXT (vd: "Watching movies") -> map letter -> text rồi so
            idx = ord(choice) - ord("A")
            picked = choices[idx] if 0 <= idx < len(choices) else ""
            passed = _norm(picked) == _norm(expected.get("answer_key") or "")
            return {"passed": passed, "feedback": "Chính xác." if passed else "Chưa đúng. Bạn thử lại nhé."}

        target = _norm(expected.get("answer_key") or "")
        ratio = _similarity(target, answer)
        passed = ratio >= 0.7
        feedback = "Tốt." if passed else "Mình nghe chưa đúng. Bạn thử lại nhé."
        return {"passed": passed, "feedback": feedback}

    return {"false": True, "feedback": "Mình chưa có tiêu chí chấm cho bước này."}
