
import os
import sys
import shutil

import whisper
import numpy as np
from sqlalchemy.orm import Session
from sklearn.neighbors import NearestNeighbors

from app import crud
from app.models import Knn

# speech_pronunciation.py와 동일한 우선순위로 ffmpeg 탐색
try:
    from imageio_ffmpeg import get_ffmpeg_exe  # pip install imageio-ffmpeg
except ImportError:
    get_ffmpeg_exe = None


def _ensure_ffmpeg_on_path():
    """
    ffmpeg 실행파일을 찾고 PATH에 주입.
    우선순위:
      1) venv 루트(=sys.prefix)/ffmpeg(.exe)
      2) venv/Scripts/ffmpeg(.exe) (Windows)
      3) imageio-ffmpeg 번들
      4) 이미 PATH에 있는 ffmpeg
      5) 백업: C:\\ffmpeg\\bin, /usr/bin, /usr/local/bin
    """
    venv_root_ffmpeg = os.path.join(sys.prefix, "ffmpeg.exe")
    venv_root_ffmpeg_nix = os.path.join(sys.prefix, "ffmpeg")
    venv_scripts_ffmpeg = os.path.join(sys.prefix, "Scripts", "ffmpeg.exe")

    candidates = [
        venv_root_ffmpeg,
        venv_root_ffmpeg_nix,
        venv_scripts_ffmpeg,
    ]

    if get_ffmpeg_exe is not None:
        try:
            bundle = get_ffmpeg_exe()
            if bundle and os.path.exists(bundle):
                candidates.append(bundle)
        except Exception:
            pass

    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        candidates.append(path_ffmpeg)

    candidates.extend([
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ])

    ffmpeg_path = None
    for c in candidates:
        if c and os.path.exists(c):
            ffmpeg_path = c
            os.environ["PATH"] = os.path.dirname(c) + os.pathsep + os.environ.get("PATH", "")
            break

    if not ffmpeg_path and not shutil.which("ffmpeg"):
        raise FileNotFoundError(
            "ffmpeg executable not found. "
            "Put ffmpeg in your venv or install 'imageio-ffmpeg', "
            "or ensure it's on PATH (e.g., C:\\ffmpeg\\bin)."
        )


# Whisper 모델 전역 캐시
_MODEL = None
def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = whisper.load_model("base")
    return _MODEL


def build_speed_rows_from_segments(result: dict):
    rows = []
    for seg in result.get("segments", []):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = seg.get("text", "") or ""
        num_words = len(seg["words"]) if "words" in seg else len(text.split())
        duration = max(end - start, 1e-6)
        wps = num_words / duration

        # 필터 조건
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
    wpm_records = db.query(Knn.mean_wpm).all()
    if not wpm_records:
        # 빈 DB 대비
        return None, 0.0

    wpm_values = [[r[0]] for r in wpm_records]  # 2차원 배열
    knn = NearestNeighbors(n_neighbors=min(k, len(wpm_values))).fit(wpm_values)

    wpm_array = np.array([r[0] for r in wpm_records], dtype=float)
    scale = float(np.std(wpm_array)) * alpha

    return knn, scale


def calculate_overall_wpm_and_knn_score_db(result, knn, scale):
    all_words = []
    all_start = None
    all_end = None
    for seg in result.get('segments', []):
        if all_start is None:
            all_start = seg.get('start', None)
        all_end = seg.get('end', all_end)
        if 'words' in seg:
            all_words.extend(seg['words'])
        else:
            all_words.extend((seg.get('text') or '').split())

    total_words = len(all_words)
    total_duration = (all_end - all_start) if (all_start is not None and all_end is not None) else 0.0

    wpm = 0.0
    score = 0.0
    if total_duration > 0:
        wps = total_words / total_duration
        wpm = wps * 60.0

        if knn is not None:
            dist, _ = knn.kneighbors([[wpm]])
            mean_dist = float(np.mean(dist))
            score = float(100 * np.exp(-mean_dist / (scale + 1e-6)))
        else:
            score = 0.0

    return wpm, score


def analyze_and_save_speed(db: Session, audio_id: int, wav_path: str):
    """
    로컬 WAV 경로를 직접 받아 Whisper로 속도를 분석하고 DB에 저장.
    (S3 다운로드 없음, 파일 삭제 없음)
    """
    # ✅ FFmpeg PATH 보정
    _ensure_ffmpeg_on_path()

    if not wav_path or not os.path.exists(wav_path):
        raise FileNotFoundError(f"Local wav not found: {wav_path}")

    knn, scale = get_knn_model_from_db(db)
    model = _get_model()

    # Whisper에 로컬 파일 경로 전달
    result = model.transcribe(
        wav_path,
        word_timestamps=True,
        language="ko",
    )

    speed_rows = build_speed_rows_from_segments(result)
    if speed_rows:
        crud.bulk_insert_speed(db, audio_id, speed_rows)

    wpm, knn_score = calculate_overall_wpm_and_knn_score_db(result, knn, scale)

    return {
        "segments": result.get("segments", []),
        "speed_rows": speed_rows,
        "overall_wpm": wpm,
        "knn_score": knn_score,
    }
