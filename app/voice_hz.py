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
        return None, None
    pitch_std_array = np.array(pitch_std_list, dtype=float).reshape(-1, 1)
    knn_model = NearestNeighbors(n_neighbors=min(3, len(pitch_std_array))).fit(pitch_std_array)
    return knn_model, pitch_std_array


def _aggregate_f0_to_halfsec(f0, sr, hop_length, agg_sec=0.5):
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    duration = times[-1] if len(times) else 0.0
    if duration == 0 or len(f0) == 0:
        return np.array([])

    edges = np.arange(0, duration + agg_sec, agg_sec)
    agg_vals = []
    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i + 1]
        mask = (times >= s) & (times < e)
        if not np.any(mask):
            agg_vals.append(np.nan)
            continue
        vals = f0[mask]
        if np.all(np.isnan(vals)):
            agg_vals.append(np.nan)
        else:
            agg_vals.append(np.nanmedian(vals))
    return np.array(agg_vals)


def analyze_pitch(wav_path: str, knn_model, pitch_std_array):
    y, sr = librosa.load(wav_path, sr=None)
    base_hop = int(sr * 0.02)  # 20ms
    if base_hop < 1:
        base_hop = 1

    try:
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr, hop_length=base_hop)
    except Exception:
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr, hop_length=base_hop, viterbi=False)

    f0 = np.asarray(f0, dtype=float)
    f0_half = _aggregate_f0_to_halfsec(f0, sr, base_hop, agg_sec=0.5)

    hz_values = f0_half[~np.isnan(f0_half)]
    hz_values_list = hz_values.tolist()
    hz_std = float(np.nanstd(hz_values)) if len(hz_values) > 0 else 0.0

    if knn_model is None or pitch_std_array is None or len(pitch_std_array) == 0:
        pitch_score = 0.0
    else:
        alpha = 1.0
        scale_pitch = float(np.std(pitch_std_array)) * alpha
        if scale_pitch == 0:
            pitch_score = 100.0 if hz_std == float(pitch_std_array[0][0]) else 0.0
        else:
            mean_dist_pitch, _ = knn_model.kneighbors([[hz_std]])
            mean_dist_pitch = float(np.mean(mean_dist_pitch))
            pitch_score = float(100 * np.exp(-mean_dist_pitch / (scale_pitch + 1e-9)))

    return f0_half, hz_std, pitch_score


def save_pitch_to_db(audio_id: int, wav_path: str):
    knn_model, pitch_std_array = load_knn_model()
    hz_array, hz_std, pitch_score = analyze_pitch(wav_path, knn_model, pitch_std_array)

    db: Session = SessionLocal()
    try:
        # 0.5초 간격으로 하나씩 INSERT
        for idx, hz_val in enumerate(hz_array):
            t_sec = (idx + 0.5) * 0.5  # 중앙 시간 (0.5, 1.0, 1.5 ...)

            # NaN이면 None → DB에서 NULL .. db에 not null 제약조건이 있어서 0.0으로 작성함.
            hz_clean = 0.0 if np.isnan(hz_val) else float(hz_val)

            db.add(Pitch(
                audio_id=audio_id,
                hz=hz_clean,
                time=t_sec,
                hz_std=hz_std,
                proper_csv=0.0,
                pitch_score=pitch_score
            ))

        db.commit()
        print(f"[INFO] Pitch 저장 완료: audio_id={audio_id}, 총 {len(hz_array)}개 구간, "
              f"hz_std={hz_std:.3f}, score={pitch_score:.1f}")
    finally:
        db.close()