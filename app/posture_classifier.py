# === app/posture_classifier.py (전체 교체본) ===
import os
import re
import io
import hashlib
import logging
from typing import Optional, List
from sqlalchemy import or_

import boto3
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

import tensorflow as tf
load_model = tf.keras.models.load_model

from app.models import Frame, Pose, Score

# -----------------------------
# 로깅 설정
# -----------------------------
logger = logging.getLogger("posture")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[POSTURE] %(levelname)s %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.DEBUG)  # 시끄러우면 INFO로

# -----------------------------
# 설정값
# -----------------------------
DEFAULT_THRESHOLD = 0.65
VALID_EXTS = (".jpg", ".jpeg", ".png")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "my_pose_classifier2.keras")

# -----------------------------
# 유틸
# -----------------------------
def _sha256(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return "N/A"

def _s3_client(aws_access_key_id: str, aws_secret_access_key: str, region_name: str):
    return boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )

def _poses_key_to_frame_url(bucket: str, region: str, img_key: str, video_id: int) -> Optional[str]:
    base = os.path.basename(img_key)
    m = re.search(r"pose_(\d+)\.(?:jpg|jpeg|png)$", base, re.IGNORECASE)
    if not m:
        return None
    ms = m.group(1)
    mapped_key = f"frames/{video_id}/frame_{ms}.jpg"
    return f"https://{bucket}.s3.{region}.amazonaws.com/{mapped_key}"

def _load_img_from_s3(s3, bucket: str, key: str, target_size=(128, 128)) -> Optional[np.ndarray]:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        img = Image.open(io.BytesIO(data)).convert("RGB").resize(target_size)
        arr = np.asarray(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr
    except Exception as e:
        logger.error(f"Failed to load image from s3://{bucket}/{key}: {e}")
        return None

def _list_all_pose_keys(s3, bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        contents = resp.get("Contents", [])
        for o in contents:
            k = o.get("Key", "")
            if k and k.lower().endswith(VALID_EXTS):
                keys.append(k)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return sorted(keys)

# -----------------------------
# 메인 함수
# -----------------------------
def classify_poses_and_save_to_db(
    *,
    db: Session,
    video_id: int,
    bucket: str,
    region: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    model_path: str = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    logger.info(f"Start posture: video_id={video_id}, threshold={threshold}")

    # 1) 모델 로드
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pose model not found: {model_path}")

    abs_path = os.path.abspath(model_path)
    size = os.path.getsize(model_path)
    sha = _sha256(model_path)

    logger.info(f"TF version: {tf.__version__}")
    logger.info(f"Model path: {abs_path}")
    logger.info(f"Model size: {size} bytes")
    logger.info(f"Model sha256: {sha}")

    model = load_model(model_path, compile=False)

    try:
        n_in, n_out = len(model.inputs), len(model.outputs)
        logger.info(f"Model IO -> inputs={n_in}, outputs={n_out}")
        model.summary(print_fn=lambda x: logger.debug(x))
    except Exception as e:
        logger.error(f"Model structure read failed: {e}")
        raise

    if n_in != 1 or n_out != 1:
        raise ValueError(f"Expected single-input/single-output model, got inputs={n_in}, outputs={n_out}")

    # 2) S3 파일 목록
    s3 = _s3_client(aws_access_key_id, aws_secret_access_key, region)
    prefix = f"poses/{video_id}/"

    try:
        keys = _list_all_pose_keys(s3, bucket, prefix)
        logger.info(f"Found {len(keys)} pose images under s3://{bucket}/{prefix}")
    except Exception as e:
        logger.error(f"S3 list failed: {e}")
        return {"video_id": video_id, "total": 0, "good": 0, "bad": 0, "pose_score": 0.0}

    if not keys:
        logger.warning("No pose images found.")
        return {"video_id": video_id, "total": 0, "good": 0, "bad": 0, "pose_score": 0.0}

    good_cnt = bad_cnt = total = 0

    for key in keys:
        # --- ms 추출 ---
        base = os.path.basename(key)
        m = re.search(r"pose_(\d+)\.(?:jpg|jpeg|png)$", base, re.IGNORECASE)
        if not m:
            logger.warning(f"Skip (cannot parse ms): {key}")
            continue
        ms = int(m.group(1))
        mapped_key = f"frames/{video_id}/frame_{ms}.jpg"
        frame_url = _poses_key_to_frame_url(bucket, region, key, video_id)

        # --- Frame 매칭: 1) 완전일치 ---
        frame = db.query(Frame).filter(Frame.image_url == frame_url).first()

        # --- 2) 접미사 like (CloudFront/서명URL/path-style 커버) ---
        if not frame:
            frame = (
                db.query(Frame)
                  .filter(Frame.video_id == video_id)
                  .filter(
                      or_(
                          Frame.image_url.like(f"%/{mapped_key}"),
                          Frame.image_url.like(f"%{mapped_key}?%")
                      )
                  )
                  .first()
            )

        # --- 3) 타임스탬프 근사(±30ms) ---
        if not frame:
            ts = ms / 1000.0
            tol = 0.03
            frame = (
                db.query(Frame)
                  .filter(Frame.video_id == video_id)
                  .filter(Frame.frame_timestamp.between(ts - tol, ts + tol))
                  .first()
            )

        # --- 4) 그래도 없으면 생성 ---
        if not frame:
            logger.warning(f"No matching frame, creating one: {mapped_key}")
            ts = ms / 1000.0
            frame = Frame(
                video_id=video_id,
                frame_timestamp=ts,
                image_url=f"https://{bucket}.s3.{region}.amazonaws.com/{mapped_key}",
            )
            db.add(frame)
            db.flush()  # frame.id 확보

        # --- 이미지 로드 & 예측 ---
        arr = _load_img_from_s3(s3, bucket, key)
        if arr is None:
            continue

        if isinstance(arr, (list, tuple)):
            logger.warning(f"Multiple inputs detected for {key}, taking the first one only.")
            arr = arr[0]
        arr = np.asarray(arr, dtype="float32")
        if arr.ndim != 4 or arr.shape[1:] != (128, 128, 3):
            logger.error(f"Unexpected shape for {key}: {arr.shape}, skipping.")
            continue

        try:
            pred = model.predict(arr, verbose=0)
            prob = float(pred[0][0])
        except Exception as e:
            logger.error(f"Predict failed for key={key}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            continue

        label = "GOOD" if prob >= threshold else "BAD"
        db.add(Pose(frame_id=frame.id, image_type=label, estimate_score=prob))

        total += 1
        if label == "GOOD":
            good_cnt += 1
        else:
            bad_cnt += 1

    db.commit()

    pose_score = float((good_cnt / total) * 100) if total > 0 else 0.0
    score_obj = db.query(Score).filter(Score.video_id == video_id).first()
    if not score_obj:
        score_obj = Score(video_id=video_id)
        db.add(score_obj)
    score_obj.pose_score = pose_score
    db.commit()

    result = {
        "video_id": video_id,
        "total": total,
        "good": good_cnt,
        "bad": bad_cnt,
        "pose_score": pose_score,
    }
    logger.info(f"Posture classification done: {result}")
    return result
