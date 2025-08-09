# 이미지 추출 및 오디오 추출 및 db 저장
# app/video_processing.py
from moviepy.editor import VideoFileClip
import os
from typing import Tuple, Dict, Any

from PIL import Image
import numpy as np
from sqlalchemy.orm import Session

import cv2
import mediapipe as mp

from app import crud
from app.models import Audio, Pronunciation, Pitch
from app.posture_classifier import classify_poses_and_save_to_db
from app.config import AWS_BUCKET_NAME, AWS_REGION

# ---------- MediaPipe 초기화 ----------
mp_face = mp.solutions.face_detection
FACE_DETECTOR = mp_face.FaceDetection(model_selection=1)

mp_pose = mp.solutions.pose
# static_image_mode=True: 프레임마다 독립적으로 감지(동영상에서도 OK, 속도보다 안정성 우선)
POSE_DETECTOR = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

# ---------- 포즈(사람) 크롭 ----------
def _crop_person_rgb_with_mediapipe(frame_rgb: np.ndarray, out_size=(128, 128)) -> Image.Image:
    """
    MediaPipe Pose로 전신 랜드마크를 찾고, 그 최소/최대 xy로 bbox를 만들어 128x128 크롭.
    실패하면 전체 프레임을 128x128로 리사이즈해서 반환.
    """
    h, w = frame_rgb.shape[:2]
    result = POSE_DETECTOR.process(frame_rgb)

    if result.pose_landmarks and result.pose_landmarks.landmark:
        xs, ys = [], []
        # 가시성이 너무 낮은 점은 버림(노이즈 방지)
        for lm in result.pose_landmarks.landmark:
            if lm.visibility is None or lm.visibility < 0.2:
                continue
            xs.append(lm.x * w)
            ys.append(lm.y * h)

        if xs and ys:
            x1 = max(0, int(min(xs)))
            y1 = max(0, int(min(ys)))
            x2 = min(w, int(max(xs)))
            y2 = min(h, int(max(ys)))

            # 여유 마진 추가 (사람을 조금 더 넉넉히)
            margin_x = int(0.10 * (x2 - x1 + 1))
            margin_y = int(0.10 * (y2 - y1 + 1))
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)

            if x2 > x1 and y2 > y1:
                crop = frame_rgb[y1:y2, x1:x2]
                return Image.fromarray(crop).resize(out_size)

    # 실패 시 전체 리사이즈
    return Image.fromarray(frame_rgb).resize(out_size)

# ---------- 얼굴(감정) 크롭(기존 유지) ----------
def extract_face_from_frame(frame: np.ndarray, save_path: str) -> bool:
    """
    MoviePy 프레임(RGB) 기준: 얼굴 검출 → RGB crop → 저장
    """
    rgb_frame = frame  # 이미 RGB
    results = FACE_DETECTOR.process(rgb_frame)
    if results.detections:
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            h, w, _ = rgb_frame.shape
            x1 = max(0, int(box.xmin * w))
            y1 = max(0, int(box.ymin * h))
            x2 = min(w, x1 + int(box.width * w))
            y2 = min(h, y1 + int(box.height * h))
            face_rgb = rgb_frame[y1:y2, x1:x2]
            if face_rgb.shape[0] > 0 and face_rgb.shape[1] > 0:
                face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, face_bgr)
                return True
    return False

def extract_frames_and_audio(
    video_path, out_dir, db: Session, video_id, s3_utils
) -> str:
    """
    - 1초 간격 프레임 추출 → S3(frames/) 업로드 → Frame 저장
    - 얼굴(감정) 크롭 → S3(faces/) 업로드   [분류는 emotion 모듈에서]
    - 사람(포즈) 크롭(128x128) → S3(poses/) 업로드  [분류는 별도 posture_classifier.py]
    - 오디오 추출(WAV, pcm_s16le) → S3(audios/) 업로드 → Audio 저장
    - 반환: 로컬 wav_path (Whisper용)
    """
    print(f"[INFO] Starting video processing for video_id: {video_id}")
    os.makedirs(out_dir, exist_ok=True)

    clip = None
    wav_local_path = os.path.join(out_dir, "audio.wav")

    try:
        clip = VideoFileClip(video_path)
        duration = float(clip.duration or 0.0)

        t = 0.0
        while t < duration:
            frame = clip.get_frame(t)  # RGB numpy array
            ms = int(t * 1000)

            # 1) 원본 프레임 저장 & 업로드
            frame_path = os.path.join(out_dir, f"frame_{ms}.jpg")
            Image.fromarray(frame).save(frame_path)
            s3_frame_key = f"frames/{video_id}/frame_{ms}.jpg"
            s3_img_url = s3_utils.upload_file_to_s3(frame_path, s3_frame_key)
            crud.create_frame(db, video_id, t, s3_img_url)
            print(f"[INFO] Frame saved: {s3_img_url}")

            # 2) 감정용 얼굴 크롭 (원래 로직)
            face_save_path = os.path.join(out_dir, f"frames_{ms}.jpg")
            if extract_face_from_frame(frame, face_save_path):
                s3_face_key = f"faces/{video_id}/frames_{ms}.jpg"
                s3_utils.upload_file_to_s3(face_save_path, s3_face_key)
                print(f"[INFO_FACE] Face crop saved: s3://{AWS_BUCKET_NAME}/{s3_face_key}")

            # 3) 포즈용 사람 크롭 (128x128) – YOLO 제거, MediaPipe Pose 사용
            pose_img = _crop_person_rgb_with_mediapipe(frame)
            pose_path = os.path.join(out_dir, f"pose_{ms}.jpg")
            pose_img.save(pose_path, quality=95)
            s3_pose_key = f"poses/{video_id}/pose_{ms}.jpg"
            s3_utils.upload_file_to_s3(pose_path, s3_pose_key)
            print(f"[INFO_POSE] Pose crop saved: s3://{AWS_BUCKET_NAME}/{s3_pose_key}")

            # temp 정리(원하면 유지)
            try:
                os.remove(frame_path)
                if os.path.exists(face_save_path):
                    os.remove(face_save_path)
                os.remove(pose_path)
            except Exception:
                pass

            t += 1.0  # 1초 간격

        # 4) 오디오 추출 (Whisper 친화적)
        if clip.audio is None:
            raise RuntimeError("No audio track found in the video.")
        clip.audio.write_audiofile(wav_local_path, codec="pcm_s16le")

        # 오디오 S3 업로드 + DB 저장/업데이트
        s3_audio_key = f"audios/{video_id}/audio.wav"
        s3_audio_url = s3_utils.upload_file_to_s3(wav_local_path, s3_audio_key)
        crud.create_audio(db, video_id, s3_audio_url, duration)
        crud.update_video_audio_url(db, video_id, s3_audio_url)

        print(f"[INFO] Video processing completed for video_id: {video_id}")
        return wav_local_path

    finally:
        # MoviePy 리소스 정리
        try:
            if clip is not None:
                clip.reader.close()
                if clip.audio:
                    clip.audio.reader.close_proc()
        except Exception:
            pass

def analyze_presentation_video(
    video_path: str, script_path: str, out_dir: str, db: Session, video_id: int, s3_utils
) -> Tuple[Dict[str, Any], str]:
    from app import gaze_analysis, emotion_analysis

    # 1) 프레임/오디오/크롭 저장
    wav_path = extract_frames_and_audio(video_path, out_dir, db, video_id, s3_utils)

    # 2) 시선 분석
    print(f"[INFO] Starting gaze analysis for video_id: {video_id}")
    gaze_results = []
    try:
        gaze_results = gaze_analysis.analyze_and_save_gaze(
            bucket=AWS_BUCKET_NAME,
            prefix=f"frames/{video_id}/",
            db=db,
            region=AWS_REGION
        )
        print(f"[INFO] Gaze analysis completed with {len(gaze_results)} results")
    except Exception as e:
        print(f"[ERROR] Gaze analysis failed: {e}")

    # 3) 감정 분석 (faces/ 사용)
    print(f"[INFO] Starting emotion analysis for video_id: {video_id}")
    emotion_score_result = {"user": None, "ref": None, "score": None}
    all_emotion_avg = None
    try:
        _ = emotion_analysis.analyze_emotion_and_save_to_db(
            bucket=AWS_BUCKET_NAME,
            prefix=f"faces/{video_id}/",
            db=db,
            region=AWS_REGION,
            video_id=video_id,
        )
        emotion_score_result = emotion_analysis.evaluate_presentation_emotion_corrected(db, video_id)
        all_emotion_avg = emotion_analysis.get_all_emotion_averages_corrected(db, video_id)
        print("유저 감정 평균", all_emotion_avg)
        print("보정 neutral/happy", emotion_score_result.get("user"))
        print(f"[INFO] Emotion analysis completed")
    except Exception as e:
        print(f"[ERROR] Emotion analysis failed: {e}")

    results: Dict[str, Any] = {
        "gaze": gaze_results,
        "emotion": {
            "avg": (emotion_score_result or {}).get("user"),
            "ref": (emotion_score_result or {}).get("ref"),
            "score": (emotion_score_result or {}).get("score"),
            "all_avg": all_emotion_avg
        },
        "voice": {},
        "posture": {}  # ✅ 분류는 별도 posture_classifier.py에서
    }
    return results, wav_path