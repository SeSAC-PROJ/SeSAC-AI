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
from app.models import Audio


def extract_frames_and_audio(
    video_path, out_dir, db: Session, video_id, s3_utils
):
    print(f"[INFO] Starting video processing for video_id: {video_id}")
    
    clip = VideoFileClip(video_path)
    duration = clip.duration

    face_count = 0 #얼굴 몇개 추출됐는지 확인해줌
    # 1초 간격 이미지 추출
    t = 0
    while t < duration:
        frame = clip.get_frame(t)
        frame_path = os.path.join(out_dir, f"frame_{int(t*1000)}.jpg")
        Image.fromarray(frame).save(frame_path)
        
        #프레임 저장 (기존 코드)
        s3_key = f"frames/{video_id}/frame_{int(t*1000)}.jpg"
        s3_img_url = s3_utils.upload_file_to_s3(frame_path, s3_key)
        crud.create_frame(db, video_id, t, s3_img_url)
        print(f"[INFO] Frame saved: {s3_img_url}")

        # ---얼굴 crop 저장 ---
        face_save_path = os.path.join(out_dir, f"frames_{int(t*1000)}.jpg")
        if extract_face_from_frame(frame, face_save_path):
            s3_face_key = f"faces/{video_id}/frames_{int(t*1000)}.jpg"
            s3_face_url = s3_utils.upload_file_to_s3(face_save_path, s3_face_key)
            # 별도 crud.create_face(DB 저장 함수) 필요 (혹은 frame 테이블에 추가 컬럼)
            print(f"[INFO_FACE] Face crop saved: {s3_face_url}")
            face_count += 1

        
        t += 1.0

    # 오디오 추출
    audio_path = os.path.join(out_dir, "audio.wav")
    clip.audio.write_audiofile(audio_path)
    
    s3_audio_key = f"audios/{video_id}/audio.wav"
    s3_audio_url = s3_utils.upload_file_to_s3(audio_path, s3_audio_key)
    crud.create_audio(db, video_id, s3_audio_url, duration)
    crud.update_video_audio_url(db, video_id, s3_audio_url)
    
    print(f"[INFO] Video processing completed for video_id: {video_id}")
    
    # MoviePy 리소스 정리
    clip.reader.close()
    if clip.audio:
        clip.audio.reader.close_proc()

def analyze_presentation_video(
    video_path, out_dir, db: Session, video_id, s3_utils
):
    """
    1) 프레임/오디오 추출 및 S3/DB 저장
    2) 시선(gaze) 분석 및 DB 저장
    3) 감정(emotion) 분석 및 DB 저장
    4) 발표 속도(voice_speed) 분석 및 DB 저장
    """
    from app import gaze_analysis, emotion_analysis, speed_analysis

    from app.config import AWS_BUCKET_NAME, AWS_REGION

    #프레임/오디오 추출
    extract_frames_and_audio(video_path, out_dir, db, video_id, s3_utils)

    #시선 분석 (프레임 폴더 대상)
    print(f"[INFO] Starting gaze analysis for video_id: {video_id}")
    try:
        gaze_results = gaze_analysis.analyze_and_save_gaze( #
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

    #감정 분석 (얼굴 crop 폴더 대상)
    print(f"[INFO] Starting emotion analysis for video_id: {video_id}")
    try:
        emotion_results = emotion_analysis.analyze_emotion_and_save_to_db(
            bucket=AWS_BUCKET_NAME,
            prefix=f"faces/{video_id}/",
            db=db,
            region=AWS_REGION,
            video_id=video_id,
        )
        emotion_score_result = emotion_analysis.evaluate_presentation_emotion_corrected(db, video_id)
        all_emotion_avg = emotion_analysis.get_all_emotion_averages_corrected(db, video_id)
        print("유저 감정 평균", all_emotion_avg)
        print("보정 neutral/happy", emotion_score_result["user"])
        print(f"[INFO] Emotion analysis completed")
    except Exception as e:
        print(f"[ERROR] Emotion analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    #속도 분석
    audio_obj = db.query(Audio).filter(Audio.video_id == video_id).first()
    if audio_obj:
        voice_speed_result = speed_analysis.analyze_and_save_speed(db, audio_obj.id, audio_obj.audio_url, work_dir=out_dir)
    else:
        voice_speed_result = {"segments": [], "speed_rows": []}

    return {
        "gaze": gaze_results,
        "emotion": {
            "avg": emotion_score_result["user"],    # neutral/happy 비율
            "ref": emotion_score_result["ref"],     # 아나운서 표준값
            "score": emotion_score_result["score"], # 비교 점수
            "all_avg": all_emotion_avg              # 전체 감정 평균
        },
        "voice": voice_speed_result,
        "posture": {...}
    }
    

mp_face = mp.solutions.face_detection
FACE_DETECTOR = mp_face.FaceDetection(model_selection=1)

def extract_face_from_frame(frame, save_path):
    """
    MoviePy 프레임(RGB) 기준: 얼굴 검출 → RGB crop → BGR 변환 후 저장
    """
    # frame: MoviePy에서 온 RGB numpy array
    rgb_frame = frame  # 이미 RGB

    results = FACE_DETECTOR.process(rgb_frame)  # MediaPipe는 RGB 기대
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
                #OpenCV로 저장: RGB -> BGR 변환 필수
                face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, face_bgr)
                return True
    return False    