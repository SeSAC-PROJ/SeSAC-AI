import os
import re
import boto3
import cv2
import numpy as np
from sqlalchemy.orm import Session
from app import crud
from app.models import Frame
from app.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
from deepface import DeepFace

from sqlalchemy import func
from app.models import Emotion, Frame


# ----------------emotion 분석 부분 ---------------------------
def read_image_from_s3(bucket: str, key: str):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    try:
        res = s3_client.get_object(Bucket=bucket, Key=key)
        img_data = res['Body'].read()
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[ERROR] Failed to decode image: {key}")
            return None
        return img
    except Exception as e:
        print(f"[ERROR] Error reading image from S3: {e}")
        return None

def _faces_key_to_frame_url(bucket: str, region: str, img_key: str, video_id: int) -> str | None:
    """
    faces/{video_id}/face_XXXX.jpg  → frames/{video_id}/frame_XXXX.jpg 로 매핑된 URL 반환
    매칭 실패 시 None
    """
    base = os.path.basename(img_key)  #face_3000.jpg
    m = re.search(r"(?:face|frames)_(\d+)\.(?:jpg|png)$", base, re.IGNORECASE)
    if not m:
        return None
    ms = m.group(1)  # '3000'
    mapped_key = f"frames/{video_id}/frame_{ms}.jpg"
    return f"https://{bucket}.s3.{region}.amazonaws.com/{mapped_key}"

def analyze_emotion_and_save_to_db(bucket: str, prefix: str, db: Session, region: str, video_id: int):
    """
    S3 얼굴 crop → DeepFace 감정분석 → Emotion 테이블 저장
    정밀 매칭: faces 키에서 ms 추출 → frames/{video_id}/frame_{ms}.jpg URL로 정확히 매칭
    """
    print(f"[INFO] Starting DeepFace emotion analysis for bucket: {bucket}, prefix: {prefix}")
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

    try:
        result = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    except Exception as e:
        print(f"[ERROR] Error listing S3 objects: {e}")
        return {}

    if 'Contents' not in result:
        print("[WARNING] No images found in S3 for emotion analysis")
        return {}

    image_keys = sorted(
        [obj['Key'] for obj in result['Contents'] if obj['Key'].lower().endswith(('.jpg', '.png'))]
    )
    print(f"[INFO] Found {len(image_keys)} images for DeepFace emotion analysis")

    for img_key in image_keys:
        print(f"[DEBUG] ========== Processing image: {img_key} ==========")
        try:
            # 1) S3에서 얼굴 crop 읽기
            frame_img = read_image_from_s3(bucket, img_key)
            if frame_img is None:
                print(f"[WARNING] Failed to read image: {img_key}")
                continue

            # 2) DeepFace 감정 분석
            analysis = DeepFace.analyze(img_path=frame_img, actions=['emotion'], enforce_detection=False)
            emotion_scores = analysis[0]['emotion']

            # 3) 원본 프레임 URL로 정확 매핑
            frame_url = _faces_key_to_frame_url(bucket, region, img_key, video_id)
            if not frame_url:
                print(f"[WARNING] Unable to parse ms from key: {img_key}")
                continue

            # 4) Frame 정확 조회 (URL 일치)
            frame = db.query(Frame).filter(Frame.image_url == frame_url).first()
            if not frame:
                print(f"[WARNING] No matching frame found for: {frame_url}")
                continue

            # 5) Emotion 저장
            crud.create_emotion(
                db=db,
                frame_id=frame.id,
                angry=float(emotion_scores.get("angry", 0.0)),
                fear=float(emotion_scores.get("fear", 0.0)),
                surprise=float(emotion_scores.get("surprise", 0.0)),
                happy=float(emotion_scores.get("happy", 0.0)),
                sad=float(emotion_scores.get("sad", 0.0)),
                neutral=float(emotion_scores.get("neutral", 0.0)),
            )
            print(f"[INFO] Emotion saved for frame_id: {frame.id}")

        except Exception as e:
            print(f"[ERROR] Failed to process image {img_key}: {e}")
            import traceback; traceback.print_exc()
            continue

    print(f"[INFO] DeepFace emotion analysis & DB save completed.")


# ----------------emotion 평가 부분 ------------------------
def get_emotion_ratios_corrected(db: Session, video_id: int):
    rows = db.query(Emotion).join(Frame, Emotion.frame_id == Frame.id)\
               .filter(Frame.video_id == video_id).all()

    total_count = len(rows)
    if total_count == 0:
        return {"neutral": 0.0, "happy": 0.0}

    neutral_count = 0
    happy_count = 0

    for r in rows:
        # dominant 결정
        dominant = max(
            [("angry", r.angry), ("fear", r.fear), ("surprise", r.surprise),
             ("happy", r.happy), ("sad", r.sad), ("neutral", r.neutral)],
            key=lambda x: x[1]
        )[0]

        # 보정 적용
        if dominant in ["sad", "fear", "angry"]:
            if r.neutral > 20 or r.happy > 25:
                dominant = "neutral (corrected)"

        # 카운트
        if dominant.startswith("neutral"):
            neutral_count += 1
        elif dominant == "happy":
            happy_count += 1

    return {
        "neutral": neutral_count / total_count,
        "happy": happy_count / total_count
    }

def calculate_l1_score(ref_neutral, ref_happy, user_neutral, user_happy):
    neutral_gap = abs(user_neutral - ref_neutral)
    happy_gap = abs(user_happy - ref_happy)
    total_gap = neutral_gap + happy_gap
    return max(0, 100 - (total_gap * 100))


def evaluate_presentation_emotion_corrected(db: Session, video_id: int):
    '''
    Neutral/Happy 비율을 보정 로직 기준으로 계산하고,
    아나운서 표준과 L1 거리 점수 반환
    '''
    rows = db.query(Emotion).join(Frame, Emotion.frame_id == Frame.id)\
               .filter(Frame.video_id == video_id).all()

    total_count = len(rows)
    if total_count == 0:
        return {
            "ref": {"neutral": 0.0, "happy": 0.0},
            "user": {"neutral": 0.0, "happy": 0.0},
            "score": 0.0
        }

    neutral_count = 0
    happy_count = 0

    for r in rows:
        # dominant 계산
        dominant = max(
            [("angry", r.angry), ("fear", r.fear), ("surprise", r.surprise),
             ("happy", r.happy), ("sad", r.sad), ("neutral", r.neutral)],
            key=lambda x: x[1]
        )[0]

        # 보정 로직
        if dominant in ["sad", "fear", "angry"]:
            if r.neutral > 20 or r.happy > 25:
                dominant = "neutral"

        if dominant == "neutral":
            neutral_count += 1
        elif dominant == "happy":
            happy_count += 1

    user_ratios = {
        "neutral": neutral_count / total_count,
        "happy": happy_count / total_count
    }
    
    #아나운서 표준값 ->"/content/drive/MyDrive/faceproject/output/emotion_results_v5.csv" 저장된 내용을 기반으로 도출
    ref_neutral, ref_happy = 0.6902, 0.2102

    # L1 거리 점수
    neutral_gap = abs(user_ratios["neutral"] - ref_neutral)
    happy_gap = abs(user_ratios["happy"] - ref_happy)
    total_gap = neutral_gap + happy_gap
    score = max(0, 100 - (total_gap * 100))

    return {
        "ref": {"neutral": ref_neutral, "happy": ref_happy},
        "user": user_ratios,
        "score": score
    }


def get_all_emotion_averages_corrected(db: Session, video_id: int):
    """
    Emotion 테이블에서 dominant를 계산하고,
    보정 로직 적용 후 각 감정별 비율을 계산
    """
    rows = db.query(Emotion).join(Frame, Emotion.frame_id == Frame.id)\
               .filter(Frame.video_id == video_id).all()

    total_count = len(rows)
    if total_count == 0:
        return {col: 0.0 for col in ["angry", "fear", "surprise", "happy", "sad", "neutral"]}

    # 감정 카운트 초기화
    emotion_counts = {col: 0 for col in ["angry", "fear", "surprise", "happy", "sad", "neutral"]}

    for r in rows:
        # dominant 결정
        dominant = max(
            [("angry", r.angry), ("fear", r.fear), ("surprise", r.surprise),
             ("happy", r.happy), ("sad", r.sad), ("neutral", r.neutral)],
            key=lambda x: x[1]
        )[0]

        # 보정 로직 적용
        if dominant in ["sad", "fear", "angry"]:
            if r.neutral > 20 or r.happy > 25:
                dominant = "neutral"

        # 카운트 증가
        emotion_counts[dominant] += 1

    # 비율로 변환
    return {k: count / total_count for k, count in emotion_counts.items()}


def get_emotion_averages_corrected_for_front(db: Session, video_id: int):
    """
    Emotion 원시값(DeepFace 퍼센티지)을 읽어
    - 프레임별 dominant 산출
    - 보정 로직 적용
    - 감정별 비율 반환
    프론트로 보내는 보정 분포.
    """
    rows = (
        db.query(Emotion)
          .join(Frame, Emotion.frame_id == Frame.id)
          .filter(Frame.video_id == video_id)
          .all()
    )

    keys = ["angry", "fear", "surprise", "happy", "sad", "neutral"]
    total = len(rows)
    if total == 0:
        return {k: 0.0 for k in keys}

    counts = {k: 0 for k in keys}

    for r in rows:
        dominant = max(
            [("angry", r.angry), ("fear", r.fear), ("surprise", r.surprise),
             ("happy", r.happy), ("sad", r.sad), ("neutral", r.neutral)],
            key=lambda x: x[1]
        )[0]

        # 보정 규칙
        if dominant in ["sad", "fear", "angry"]:
            # None 안전 처리
            neutral_val = float(r.neutral or 0.0)
            happy_val = float(r.happy or 0.0)
            if neutral_val > 20.0 or happy_val > 25.0:
                dominant = "neutral"

        counts[dominant] += 1

    return {k: counts[k] / total for k in keys}

