
# DB에서 데이터 CRUD 작업을 수행하는 함수들 모음.
from typing import Iterable, Optional
from app.models import Video, Frame, Audio, Gaze, Emotion, Speed, Pose, Pronunciation, Score, Pitch, Feedback

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
# Pose: 단건 생성
def create_pose(db: Session, frame_id: int, image_type: str, estimate_score: float) -> Pose:
    obj = Pose(frame_id=frame_id, image_type=image_type, estimate_score=estimate_score)
    db.add(obj)
    # 커밋은 호출측에서 한 번에!
    return obj

# Pose: 벌크 생성 (성능)
def bulk_insert_poses(db: Session, items: Iterable[dict]) -> None:
    """
    items: [{"frame_id": int, "image_type": "GOOD"/"BAD", "estimate_score": float}, ...]
    """
    objs = [Pose(**it) for it in items]
    db.bulk_save_objects(objs)
    # 커밋은 호출측에서 한 번에!

# Pronunciation: script_text upsert (반환 객체 재사용)
def upsert_pronunciation_script(db: Session, audio_id: int, script_text: str) -> Pronunciation:
    pron = db.query(Pronunciation).filter(Pronunciation.audio_id == audio_id).first()
    if pron:
        pron.script_text = script_text
    else:
        pron = Pronunciation(audio_id=audio_id, script_text=script_text)
        db.add(pron)
    return pron

# Pronunciation: stt/matching_rate 업데이트
def update_pronunciation_result(db: Session, audio_id: int, stt_text: str, matching_rate: float) -> Pronunciation:
    pron = db.query(Pronunciation).filter(Pronunciation.audio_id == audio_id).first()
    if not pron:
        pron = Pronunciation(audio_id=audio_id, script_text="")  # 안전장치
        db.add(pron)
    pron.stt_text = stt_text
    pron.matching_rate = matching_rate
    return pron

# Score: 특정 필드만 부분 업데이트 (upsert)
def upsert_score(
    db: Session,
    video_id: int,
    pose_score: Optional[float] = None,
    emotion_score: Optional[float] = None,
    gaze_score: Optional[float] = None,
    pitch_score: Optional[float] = None,
    speed_score: Optional[float] = None,
    pronunciation_score: Optional[float] = None,
) -> Score:
    sc = db.query(Score).filter(Score.video_id == video_id).first()
    if not sc:
        sc = Score(video_id=video_id)
        db.add(sc)
    if pose_score is not None:
        sc.pose_score = float(pose_score)
    if emotion_score is not None:
        sc.emotion_score = float(emotion_score)
    if gaze_score is not None:
        sc.gaze_score = float(gaze_score)
    if pitch_score is not None:
        sc.pitch_score = float(pitch_score)
    if speed_score is not None:
        sc.speed_score = float(speed_score)
    if pronunciation_score is not None:
        sc.pronunciation_score = float(pronunciation_score)
    db.commit()
    db.refresh(sc)
    return sc

# Pitch 벌크 인서트 (voice_hz.py에서 사용)
def bulk_insert_pitch(db: Session, items: Iterable[dict]) -> None:
    """
    items: [{"audio_id":int,"hz":Optional[float],"time":float,"hz_std":float,"proper_csv":float,"pitch_score":float}, ...]
    """
    objs = [Pitch(**it) for it in items]
    db.bulk_save_objects(objs)

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