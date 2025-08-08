import os
from urllib.parse import urlparse

import boto3
import whisper
from sqlalchemy.orm import Session

import numpy as np
from sklearn.neighbors import NearestNeighbors

from app import crud
from app.models import Knn
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
        
        # 필터 조건 추가
        if duration <= 0.3:
            continue
        if wps >= 4:
            continue
        if num_words <= 1:
            continue

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


def get_knn_model_from_db(db: Session, k=3, alpha=0.05):
    # DB에서 mean_wpm 리스트 조회
    wpm_records = db.query(Knn.mean_wpm).all()
    wpm_values = [[r[0]] for r in wpm_records]  # 2차원 배열

    knn = NearestNeighbors(n_neighbors=k).fit(wpm_values)

    import numpy as np
    wpm_array = np.array([r[0] for r in wpm_records])
    scale = np.std(wpm_array) * alpha

    return knn, scale


def calculate_overall_wpm_and_knn_score_db(result, knn, scale):
    all_words = []
    all_start = None
    all_end = None
    for seg in result['segments']:
        if all_start is None:
            all_start = seg['start']
        all_end = seg['end']
        if 'words' in seg:
            all_words.extend(seg['words'])
        else:
            all_words.extend(seg['text'].split())

    total_words = len(all_words)
    total_duration = all_end - all_start if all_start is not None and all_end is not None else 0

    wpm = 0
    score = 0
    if total_duration > 0:
        wps = total_words / total_duration
        wpm = wps * 60

        dist, _ = knn.kneighbors([[wpm]])
        mean_dist = np.mean(dist)
        score = 100 * np.exp(-mean_dist / (scale + 1e-6))

    return wpm, score


def analyze_and_save_speed(db: Session, audio_id: int, audio_url: str, work_dir: str):
    """
    S3에서 오디오 다운로드 → Whisper 처리 → speed 저장  
    → DB 기준 knn으로 전체 WPM 점수 계산 → 결과 반환
    """
    os.makedirs(work_dir, exist_ok=True)
    tmp_path = os.path.join(work_dir, "audio.wav")
    _download_s3_to_path(audio_url, tmp_path)

    knn, scale = get_knn_model_from_db(db)

    try:
        result = model.transcribe(
            tmp_path,
            word_timestamps=True,
            language="ko",
        )
        speed_rows = build_speed_rows_from_segments(result)
        crud.bulk_insert_speed(db, audio_id, speed_rows)

        wpm, knn_score = calculate_overall_wpm_and_knn_score_db(result, knn, scale)

        return {
            "segments": result.get("segments", []),
            "speed_rows": speed_rows,
            "overall_wpm": wpm,
            "knn_score": knn_score,
        }
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass