import os
from urllib.parse import urlparse

import boto3
import whisper
from sqlalchemy.orm import Session

from app import crud
from app.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_BUCKET_NAME, AWS_REGION

model = whisper.load_model("base")

def _s3_url_to_key(url: str) -> str:
    # https://{bucket}.s3.{region}.amazonaws.com/<key>  -> <key>
    return urlparse(url).path.lstrip("/")

def _download_s3_to_path(audio_url: str, save_path: str):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
    key = _s3_url_to_key(audio_url)
    s3.download_file(AWS_BUCKET_NAME, key, save_path)

def build_speed_rows_from_segments(result: dict):
    rows = []
    for seg in result.get("segments", []):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = seg.get("text", "") or ""
        num_words = len(seg["words"]) if "words" in seg else len(text.split())
        duration = max(end - start, 1e-6)
        wps = num_words / duration
        rows.append({
            "stn_start": start,
            "stn_end": end,
            "text": text,
            "num_words": num_words,
            "duration": duration,
            "wps": wps,
            "wpm": wps * 60.0,
        })
    return rows

def analyze_and_save_speed(db: Session, audio_id: int, audio_url: str, work_dir: str):
    """
    S3에서 오디오를 work_dir 안에 다운로드 → Whisper STT → speed 저장 → 임시 파일 삭제
    """
    os.makedirs(work_dir, exist_ok=True)
    tmp_path = os.path.join(work_dir, "audio.wav")
    _download_s3_to_path(audio_url, tmp_path)

    try:
        result = model.transcribe(
            tmp_path,
            word_timestamps=True,
            language="ko",
        )
        speed_rows = build_speed_rows_from_segments(result)
        crud.bulk_insert_speed(db, audio_id, speed_rows)
        return {
            "segments": result.get("segments", []),
            "speed_rows": speed_rows,
        }
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
