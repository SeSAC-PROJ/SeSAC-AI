
import os
import sys
import shutil
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import whisper
from sqlalchemy.orm import Session
from sklearn.neighbors import NearestNeighbors

from app import crud
from app.models import Knn  # mean_wpm 컬럼을 갖는 테이블(벤치마크 WPM 저장)

# -------------------------------
# 설정값
# -------------------------------
WPM_GOOD_MIN = 100.0
WPM_GOOD_MAX = 150.0
MAX_PENALTY_RATIO = 0.40   # bad 비율 100%일 때 KNN 점수의 최대 40% 감점 
K_FOR_KNN = 3
ALPHA_FOR_SCALE = 0.05     # KNN 거리 스케일 민감도

# speech_pronunciation.py와 동일한 우선순위로 ffmpeg 탐색
try:
    from imageio_ffmpeg import get_ffmpeg_exe  # pip install imageio-ffmpeg
except ImportError:
    get_ffmpeg_exe = None


def _ensure_ffmpeg_on_path() -> None:
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


def build_speed_rows_from_segments(result: dict) -> List[Dict[str, Any]]:
    """
    Whisper result에서 구간별 속도 지표 생성 + 필터링 + wpm_band 라벨링
    필터:
      - duration <= 0.3s 제외
      - wps >= 4 제외
      - num_words <= 1 제외
    라벨:
      - 100 <= wpm <= 150 -> 'good' else 'bad'
    """
    rows: List[Dict[str, Any]] = []
    for seg in result.get("segments", []):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
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

        wpm = wps * 60.0
        row = {
            "stn_start": start,
            "stn_end": end,
            "text": text,
            "num_words": num_words,
            "duration": duration,
            "wps": wps,
            "wpm": wpm,
            "wpm_band": "good" if (WPM_GOOD_MIN <= wpm <= WPM_GOOD_MAX) else "bad",
        }
        rows.append(row)
    return rows


def get_knn_model_from_db(db: Session, k: int = K_FOR_KNN, alpha: float = ALPHA_FOR_SCALE
                          ) -> Tuple[Optional[NearestNeighbors], float]:
    """
    DB의 Knn.mean_wpm 샘플을 불러 KNN 모델 및 거리 스케일을 구성
    반환:
      - knn: NearestNeighbors or None(데이터 없을 때)
      - scale: float (std * alpha)
    """
    wpm_records = db.query(Knn.mean_wpm).all()
    if not wpm_records:
        return None, 0.0

    wpm_values = [[float(r[0])] for r in wpm_records]
    knn = NearestNeighbors(n_neighbors=min(k, len(wpm_values))).fit(wpm_values)

    wpm_array = np.array([float(r[0]) for r in wpm_records], dtype=float)
    scale = float(np.std(wpm_array)) * float(alpha)

    return knn, scale


def calculate_overall_wpm_and_knn_score_db(
    result: dict,
    knn: Optional[NearestNeighbors],
    scale: float
) -> Tuple[float, float]:
    """
    전체 발화 기준 WPM + (선택)KNN 기반 점수 계산
    - knn이 None이면 knn_score=0
    """
    all_words: List[str] = []
    all_start: Optional[float] = None
    all_end: Optional[float] = None

    for seg in result.get("segments", []):
        if all_start is None:
            all_start = seg.get("start", None)
        all_end = seg.get("end", all_end)
        if "words" in seg:
            all_words.extend(seg["words"])
        else:
            all_words.extend((seg.get("text") or "").split())

    total_words = len(all_words)
    total_duration = (all_end - all_start) if (all_start is not None and all_end is not None) else 0.0

    wpm = 0.0
    knn_score = 0.0
    if total_duration > 0:
        wps = total_words / max(total_duration, 1e-6)
        wpm = wps * 60.0

        if knn is not None:
            dist, _ = knn.kneighbors([[wpm]])
            mean_dist = float(np.mean(dist))
            # 거리가 작을수록 점수↑, scale로 민감도 조절
            knn_score = float(100 * np.exp(-mean_dist / (scale + 1e-6)))
        else:
            knn_score = 0.0

    return wpm, knn_score


def apply_bad_ratio_penalty(knn_score: float, speed_rows: List[Dict[str, Any]],
                            max_penalty_ratio: float = MAX_PENALTY_RATIO
                            ) -> Tuple[float, float, float]:
    """
    good/bad 비율 기반 감점
    반환: (final_score, bad_ratio, penalty_ratio)
    """
    total = len(speed_rows)
    if total == 0:
        return knn_score, 0.0, 0.0

    bad_cnt = sum(1 for r in speed_rows if r.get("wpm_band") == "bad")
    bad_ratio = bad_cnt / total
    penalty_ratio = bad_ratio * max_penalty_ratio
    final_score = max(0.0, knn_score * (1.0 - penalty_ratio))
    return final_score, bad_ratio, penalty_ratio


def analyze_and_save_speed(db: Session, audio_id: int, wav_path: str) -> Dict[str, Any]:
    """
    로컬 WAV 경로를 받아 Whisper로 속도 분석 후:
      1) segment speed rows 생성 및 저장(구간별 wpm_band 포함)
      2) 전체 wpm 및 KNN 점수 계산
      3) good/bad 비율 기반 감점 적용 → final_score 도출
    반환 dict은 프론트 디버깅/로그용. 실제 점수 저장은 기존 점수 테이블 로직에 연결.
    """
    _ensure_ffmpeg_on_path()

    if not wav_path or not os.path.exists(wav_path):
        raise FileNotFoundError(f"Local wav not found: {wav_path}")

    # KNN 벤치마크 구성
    knn, scale = get_knn_model_from_db(db)

    # Whisper
    model = _get_model()
    result = model.transcribe(
        wav_path,
        word_timestamps=True,
        language="ko",
    )

    # 세그먼트 속도 계산 + 라벨링
    speed_rows = build_speed_rows_from_segments(result)
    if speed_rows:
        # 구간 저장(wpm_band 포함)
        crud.bulk_insert_speed(db, audio_id, speed_rows)

    # 전체 WPM + KNN 점수
    overall_wpm, knn_score = calculate_overall_wpm_and_knn_score_db(result, knn, scale)

    # good/bad 비율 기반 감점
    final_score, bad_ratio, penalty_ratio = apply_bad_ratio_penalty(knn_score, speed_rows)


    return {
        "segments": result.get("segments", []),
        "speed_rows": speed_rows,           # 구간별 wpm_band 포함
        "overall_wpm": overall_wpm,
        "knn_score": knn_score,
        "final_score": final_score,
        "bad_ratio": bad_ratio,
        "penalty_ratio": penalty_ratio,     # = bad_ratio * MAX_PENALTY_RATIO
        "wpm_range": (WPM_GOOD_MIN, WPM_GOOD_MAX),
    }
