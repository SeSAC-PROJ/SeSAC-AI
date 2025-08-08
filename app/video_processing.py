# === app/video_processing.py (전체 교체본) ===
# 이미지 추출 및 오디오 추출 및 db 저장
from moviepy.editor import VideoFileClip
import os
import uuid
from app import crud
from PIL import Image
from app.config import AWS_BUCKET_NAME, AWS_REGION
from sqlalchemy.orm import Session
import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Dict, Any

mp_face = mp.solutions.face_detection
FACE_DETECTOR = mp_face.FaceDetection(model_selection=1)


def extract_frames_and_audio(
    video_path: str, out_dir: str, db: Session, video_id: int, s3_utils
) -> str:
    """
    - 1초 간격 프레임 추출 -> S3 업로드 -> DB 저장
    - 얼굴 크롭 -> S3 업로드 (별도 DB 저장 로직 필요시 crud 추가)
    - 오디오 추출(WAV, pcm_s16le) -> S3 업로드 -> DB 저장
    - 반환: 로컬 wav_path (Whisper용)
    """
    print(f"[INFO] Starting video processing for video_id: {video_id}")

    os.makedirs(out_dir, exist_ok=True)

    clip = None
    wav_local_path = os.path.join(out_dir, "audio.wav")

    try:
        clip = VideoFileClip(video_path)
        duration = float(clip.duration or 0.0)

        face_count = 0  # 얼굴 몇개 추출됐는지 확인

        # === 프레임 추출 ===
        t = 0.0
        while t < duration:
            frame = clip.get_frame(t)  # RGB numpy array
            ms = int(t * 1000)

            # 원본 프레임 파일 경로
            frame_path = os.path.join(out_dir, f"frame_{ms}.jpg")
            Image.fromarray(frame).save(frame_path)

            # 프레임 S3 업로드 및 DB 저장
            s3_key = f"frames/{video_id}/frame_{ms}.jpg"
            s3_img_url = s3_utils.upload_file_to_s3(frame_path, s3_key)
            crud.create_frame(db, video_id, t, s3_img_url)
            print(f"[INFO] Frame saved: {s3_img_url}")

            # 얼굴 crop 저장
            face_save_path = os.path.join(out_dir, f"frames_{ms}.jpg")
            if extract_face_from_frame(frame, face_save_path):
                s3_face_key = f"faces/{video_id}/frames_{ms}.jpg"
                s3_face_url = s3_utils.upload_file_to_s3(face_save_path, s3_face_key)
                # TODO: 필요시 crud.create_face(...) 같은 DB 저장 추가
                print(f"[INFO_FACE] Face crop saved: {s3_face_url}")
                face_count += 1

            t += 1.0  # 1초 간격

        # === 오디오 추출 (Whisper 친화적 wav/pcm_s16le) ===
        if clip.audio is None:
            raise RuntimeError("No audio track found in the video.")
        # codec 지정: pcm_s16le (16-bit WAV)
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
    video_path: str, out_dir: str, db: Session, video_id: int, s3_utils
) -> Tuple[Dict[str, Any], str]:
    """
    1) 프레임/오디오 추출 및 S3/DB 저장
       -> 로컬 wav_path 반환 (Whisper용)
    2) 시선(gaze) 분석 및 DB 저장
    3) 감정(emotion) 분석 및 DB 저장
    4) 기타 분석 결과 묶어서 반환

    Returns:
        (results_dict, wav_path)
    """
    from app import gaze_analysis, emotion_analysis
    from app.config import AWS_BUCKET_NAME, AWS_REGION

    # 1) 프레임/오디오 추출
    wav_path = extract_frames_and_audio(video_path, out_dir, db, video_id, s3_utils)

    # 2) 시선 분석 (S3 frames/)
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
        import traceback
        traceback.print_exc()

    # 3) 감정 분석 (S3 faces/)
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
        import traceback
        traceback.print_exc()

    # 4) 결과 패키징 (voice/posture는 상위 단계에서 채우거나 후속 단계)
    results: Dict[str, Any] = {
        "gaze": gaze_results,
        "emotion": {
            "avg": (emotion_score_result or {}).get("user"),    # neutral/happy 비율
            "ref": (emotion_score_result or {}).get("ref"),     # 아나운서 표준값
            "score": (emotion_score_result or {}).get("score"), # 비교 점수
            "all_avg": all_emotion_avg                          # 전체 감정 평균
        },
        "voice": {},    # BackgroundTask에서 run_pronunciation_score/save_pitch_to_db로 채움
        "posture": {}
    }

    # 호출자(main.py BackgroundTask)에서 (results, wav_path)를 받는다.
    return results, wav_path


def extract_face_from_frame(frame: np.ndarray, save_path: str) -> bool:
    """
    MoviePy 프레임(RGB) 기준: 얼굴 검출 → RGB crop → 저장
    - MediaPipe FaceDetection은 RGB 입력을 기대
    - OpenCV 저장 시 BGR 변환 필요하지만 PIL 저장도 가능
    """
    # frame: MoviePy에서 온 RGB numpy array
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
                # 방법 A) OpenCV로 저장: RGB -> BGR 변환
                face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, face_bgr)

                # 방법 B) PIL로 저장 (선호시):
                # Image.fromarray(face_rgb).save(save_path, quality=95)

                return True
    return False
