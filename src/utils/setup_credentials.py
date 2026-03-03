"""
Xử lý Google Service Account khi deploy trên cloud (Render).
Trên cloud không có file service_account_1.json -> đọc từ env var GOOGLE_SA_JSON.
"""
import os
import json
import tempfile

def setup_google_credentials():
    """
    Nếu GOOGLE_APPLICATION_CREDENTIALS trỏ tới file tồn tại -> giữ nguyên.
    Ngược lại, đọc GOOGLE_SA_JSON (nội dung JSON) -> ghi ra file tạm -> set env.
    """
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if creds_path and os.path.isfile(creds_path):
        return  # local dev, file exists

    sa_json = os.getenv("GOOGLE_SA_JSON", "")
    if not sa_json:
        print("[WARN] No GOOGLE_SA_JSON env var found. Vertex AI may not work.")
        return

    try:
        creds = json.loads(sa_json)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(creds, tmp)
        tmp.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
        print(f"[INFO] Created temp service account file: {tmp.name}")
    except Exception as e:
        print(f"[ERROR] Failed to parse GOOGLE_SA_JSON: {e}")
