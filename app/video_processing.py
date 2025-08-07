#이미지 추출 및 오디오 추출 및 db 저장
from moviepy.editor import VideoFileClip
import os
import uuid
from app import crud
from PIL import Image

def extract_frames_and_audio(
    video_path, out_dir, db, video_id, s3_utils
):
    clip = VideoFileClip(video_path)
    duration = clip.duration

    # 0.5초 간격 이미지 추출
    t = 0
    while t < duration:
        frame = clip.get_frame(t)
        frame_path = os.path.join(out_dir, f"frame_{int(t*1000)}.jpg")
        Image.fromarray(frame).save(frame_path)
        s3_key = f"frames/{video_id}/frame_{int(t*1000)}.jpg"
        s3_img_url = s3_utils.upload_file_to_s3(frame_path, s3_key)
        crud.create_frame(db, video_id, t, s3_img_url)
        t += 0.5

    # 오디오 추출
    audio_path = os.path.join(out_dir, "audio.wav")
    clip.audio.write_audiofile(audio_path)
    s3_audio_key = f"audios/{video_id}/audio.wav"
    s3_audio_url = s3_utils.upload_file_to_s3(audio_path, s3_audio_key)
    crud.create_audio(db, video_id, s3_audio_url, duration)
    crud.update_video_audio_url(db, video_id, s3_audio_url)