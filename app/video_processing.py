# 이미지 추출 및 오디오 추출 및 db 저장
from moviepy.editor import VideoFileClip
import os
import uuid
from app import crud
from PIL import Image
from app.config import AWS_BUCKET_NAME, AWS_REGION
from sqlalchemy.orm import Session

def extract_frames_and_audio(
    video_path, out_dir, db: Session, video_id, s3_utils
):
    print(f"[INFO] Starting video processing for video_id: {video_id}")
    
    clip = VideoFileClip(video_path)
    duration = clip.duration

    # 1초 간격 이미지 추출
    t = 0
    while t < duration:
        frame = clip.get_frame(t)
        frame_path = os.path.join(out_dir, f"frame_{int(t*1000)}.jpg")
        Image.fromarray(frame).save(frame_path)
        
        s3_key = f"frames/{video_id}/frame_{int(t*1000)}.jpg"
        s3_img_url = s3_utils.upload_file_to_s3(frame_path, s3_key)
        crud.create_frame(db, video_id, t, s3_img_url)
        print(f"[INFO] Frame saved: {s3_img_url}")
        
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

def extract_frames_and_audio_with_gaze(
    video_path, out_dir, db: Session, video_id, s3_utils
):
    """프레임/오디오 추출 후 gaze 분석까지 포함하는 함수"""
    # gaze_analysis import를 함수 내부로 이동
    from app import gaze_analysis
    
    # 기존 추출 작업 수행
    extract_frames_and_audio(video_path, out_dir, db, video_id, s3_utils)
    
    # 프레임 추출 완료 후 바로 gaze 분석 실행
    print(f"[INFO] Starting gaze analysis for video_id: {video_id}")
    try:
        gaze_results = gaze_analysis.analyze_and_save_gaze(
            bucket=AWS_BUCKET_NAME,
            prefix=f"frames/{video_id}/",
            db=db,
            region=AWS_REGION
        )
        print(f"[INFO] Gaze analysis completed with {len(gaze_results)} results")
        return gaze_results
    except Exception as e:
        print(f"[ERROR] Gaze analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {}