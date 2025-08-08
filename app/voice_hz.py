import numpy as np
import librosa
from sklearn.neighbors import NearestNeighbors
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.models import Pitch, Knn

def load_knn_model():
    db: Session = SessionLocal()
    pitch_std_list = [row[0] for row in db.query(Knn.pitch_std).all()]
    db.close()
    if len(pitch_std_list) == 0:
        # 기준 데이터 없으면 기본 점수 0으로
        return None, None
    pitch_std_array = np.array(pitch_std_list, dtype=float).reshape(-1, 1)
    knn_model = NearestNeighbors(n_neighbors=min(3, len(pitch_std_array))).fit(pitch_std_array)
    return knn_model, pitch_std_array

def _aggregate_f0_to_halfsec(f0, sr, hop_length, agg_sec=0.5):
    # f0: 20ms 등 짧은 간격으로 계산된 배열 (NaN 포함)
    # 0.5초 버킷으로 median 집계
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    duration = times[-1] if len(times) else 0.0

    if duration == 0 or len(f0) == 0:
        return np.array([])

    # 버킷 경계
    edges = np.arange(0, duration + agg_sec, agg_sec)
    agg_vals = []

    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i+1]
        mask = (times >= s) & (times < e)
        if not np.any(mask):
            agg_vals.append(np.nan)
            continue
        vals = f0[mask]
        # 전부 NaN이면 NaN 유지
        if np.all(np.isnan(vals)):
            agg_vals.append(np.nan)
        else:
            agg_vals.append(np.nanmedian(vals))

    return np.array(agg_vals)

def analyze_pitch(wav_path: str, knn_model, pitch_std_array):
    y, sr = librosa.load(wav_path, sr=None)

    # 1) 짧은 hop(20ms)로 안정적으로 f0 추출
    base_hop = int(sr * 0.02)  # 20ms
    if base_hop < 1:
        base_hop = 1

    try:
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr, hop_length=base_hop)
    except Exception:
        # 그래도 문제가 나면 Viterbi 끄고 시도 (가끔 transition 오류 회피)
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr, hop_length=base_hop, viterbi=False)

    f0 = np.asarray(f0, dtype=float)

    # 2) 0.5초 간격으로 다운샘플(집계)
    f0_half = _aggregate_f0_to_halfsec(f0, sr, base_hop, agg_sec=0.5)

    # 전체 NaN 제거
    hz_values = f0_half[~np.isnan(f0_half)]
    hz_values_list = hz_values.tolist()

    hz_std = float(np.nanstd(hz_values)) if len(hz_values) > 0 else 0.0

    # 3) KNN 점수
    if knn_model is None or pitch_std_array is None or len(pitch_std_array) == 0:
        pitch_score = 0.0
    else:
        alpha = 2.0
        scale_pitch = float(np.std(pitch_std_array)) * alpha
        # scale이 0이면(모두 같은 값) 0으로 나눔 방지
        if scale_pitch == 0:
            pitch_score = 100.0 if hz_std == float(pitch_std_array[0][0]) else 0.0
        else:
            mean_dist_pitch, _ = knn_model.kneighbors([[hz_std]])
            mean_dist_pitch = float(np.mean(mean_dist_pitch))
            pitch_score = float(100 * np.exp(-mean_dist_pitch / (scale_pitch + 1e-9)))

    return hz_values_list, hz_std, pitch_score

def save_pitch_to_db(audio_id: int, wav_path: str):
    knn_model, pitch_std_array = load_knn_model()
    hz_array, hz_std, pitch_score = analyze_pitch(wav_path, knn_model, pitch_std_array)

    db: Session = SessionLocal()
    try:
        pitch_obj = Pitch(
            audio_id=audio_id,
            # ⚠️ DB 스키마가 Float이면 리스트 저장 안 됩니다.
            # JSON/Text라면 json.dumps(hz_array)로 직렬화해서 넣으세요.
            hz=0.0,                # 스키마가 Float라면 대표값(예: median)만 저장
            time=float(len(hz_array) * 0.5),  # 0.5초 간격 갯수 → 총 길이(초)
            hz_std=hz_std,
            proper_csv=0.0,
            pitch_score=pitch_score
        )
        db.add(pitch_obj)
        db.commit()
        print(f"[INFO] Pitch 저장 완료: audio_id={audio_id}, hz_std={hz_std:.3f}, score={pitch_score:.1f}")
    finally:
        db.close()
