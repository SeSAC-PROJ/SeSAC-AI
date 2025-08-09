
# DB에서 데이터 CRUD 작업을 수행하는 함수들 모음.
from app.models import Video, Frame, Audio, Feedback, Gaze, Emotion, Speed, Score

from sqlalchemy.orm import Session

# 비디오 생성 (db에 관련 정보 저장)
def create_video(db, user_id, title, video_totaltime, video_url):
    db_video = Video(
        user_id=user_id,
        title=title,
        video_totaltime=video_totaltime,
        video_url=video_url
    )
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    return db_video

def update_video_audio_url(db: Session, video_id: int, video_url: str):
    video = db.query(Video).filter(Video.id == video_id).one_or_none()
    if video:
        video.video_url = video_url
        db.commit()

def create_frame(db: Session, video_id: int, frame_timestamp: float, image_url: str):
    db_frame = Frame(video_id=video_id, frame_timestamp=frame_timestamp, image_url=image_url)
    db.add(db_frame)
    db.commit()

def create_audio(db: Session, video_id: int, audio_url: str, duration: float):
    db_audio = Audio(video_id=video_id, audio_url=audio_url, duration=duration)
    db.add(db_audio)
    db.commit()

def create_gaze_record(db: Session, frame_id: int, direction: str):
    gaze_record = Gaze(frame_id=frame_id, direction=direction)
    db.add(gaze_record)
    db.commit()

def create_emotion(db: Session, frame_id: int, angry: float, fear: float, surprise: float, happy: float, sad: float, neutral: float):
    db_emotion = Emotion(
        frame_id=frame_id,
        angry=angry,
        fear=fear,
        surprise=surprise,
        happy=happy,
        sad=sad,
        neutral=neutral
    )
    db.add(db_emotion)
    db.commit()
    db.refresh(db_emotion)
    return db_emotion


def create_audio(db: Session, video_id: int, audio_url: str, duration: float):
    db_audio = Audio(video_id=video_id, audio_url=audio_url, duration=duration)
    db.add(db_audio)
    db.commit()
    db.refresh(db_audio)
    return db_audio  # ← id 반환 시 필요


def bulk_insert_speed(db: Session, audio_id: int, rows: list[dict]) -> None:
    """
    rows: [{'stn_start':..., 'stn_end':..., 'duration':..., 'num_words':..., 'wps':..., 'wpm':..., 'text':...}, ...]
    """
    objs = [Speed(audio_id=audio_id, **r) for r in rows]
    db.bulk_save_objects(objs)
    db.commit()

#점수 저장용 테이블
def create_score_record(db: Session, score: float):
    db_score = Score(score=score)
    db.add(db_score)
    db.commit()
    db.refresh(db_score)
    return db_score

# 피드백 저장용 테이블
def create_feedback_record(db: Session, video_id: int, short_feedback: str, detail_feedback: str) -> Feedback:
    fb = Feedback(
        video_id=video_id,
        short_feedback=short_feedback,
        detail_feedback=detail_feedback
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return fb