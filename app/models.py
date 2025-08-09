
# db 테이블 구조를 SQLAlchemy ORM 클래스로 정의
from sqlalchemy import Enum
from sqlalchemy import text
from sqlalchemy import Column, BigInteger, Float, String, ForeignKey, Text, TIMESTAMP, Integer
from sqlalchemy.ext.declarative import declarative_base
from app.db import Base

class Video(Base):
    __tablename__ = "video"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False)
    upload_time = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    title = Column(String(200))
    video_totaltime = Column(Float)
    video_url = Column(String(500), nullable=False)

class Frame(Base):
    __tablename__ = "frame"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    video_id = Column(BigInteger, ForeignKey("video.id", ondelete="CASCADE"), nullable=False)
    frame_timestamp = Column(Float, nullable=False)
    image_url = Column(String(500), nullable=False)

class Audio(Base):
    __tablename__ = "audio"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    video_id = Column(BigInteger, ForeignKey("video.id", ondelete="CASCADE"), nullable=False)
    audio_url = Column(String(500), nullable=False)
    duration = Column(Float)

class Gaze(Base):
    __tablename__ = "gaze"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    frame_id = Column(BigInteger, ForeignKey("frame.id", ondelete="CASCADE"), nullable=False)
    direction = Column(String(10), nullable=False)  # 'up', 'down', etc.

class Emotion(Base):
    __tablename__ = "emotion"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    frame_id = Column(BigInteger, ForeignKey("frame.id", ondelete="CASCADE"), nullable=False)
    angry = Column(Float, nullable=False)
    fear = Column(Float, nullable=False)
    surprise = Column(Float, nullable=False)
    happy = Column(Float, nullable=False)
    sad = Column(Float, nullable=False)
    neutral = Column(Float, nullable=False)

class Speed(Base):
    __tablename__ = "speed"
    id = Column(BigInteger, primary_key=True, autoincrement=True)  # PK
    audio_id = Column(BigInteger, ForeignKey("audio.id", ondelete="CASCADE"), nullable=False)  # FK -> audio.id
    stn_start = Column(Float, nullable=False)   # 문장 시작 시간
    stn_end = Column(Float, nullable=False)     # 문장 끝 시간
    duration = Column(Float, nullable=False)    # 발화 소요 시간
    num_words = Column(Integer, nullable=False) # 단어 개수
    wps = Column(Float, nullable=False)         # 초당 단어 수
    wpm = Column(Float, nullable=False)         # 분당 단어 수
    text = Column(Text, nullable=True)          # 문장 텍스트    

class Pitch(Base):
    __tablename__ = "pitch"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    audio_id = Column(BigInteger, ForeignKey("audio.id", ondelete="CASCADE"), nullable=False)
    hz = Column(Float, nullable=True)
    time = Column(Float, nullable=False)
    hz_std = Column(Float, nullable=False)
    proper_csv = Column(Float, nullable=False)
    pitch_score = Column(Float, nullable=False)


class Pronunciation(Base):
    __tablename__ = "Pronunciation"  # 기존 테이블 명 유지
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    audio_id = Column(BigInteger, ForeignKey("audio.id", ondelete="CASCADE"), nullable=False)
    script_text = Column(Text, nullable=False)
    stt_text = Column(Text, nullable=True)
    matching_rate = Column(Float, nullable=True)  # %


class Score(Base):
    __tablename__ = "score"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    video_id = Column(BigInteger, ForeignKey("video.id", ondelete="CASCADE"), nullable=False)
    pose_score = Column(Float, nullable=True)
    emotion_score = Column(Float, nullable=True)
    gaze_score = Column(Float, nullable=True)
    pitch_score = Column(Float, nullable=True)
    speed_score = Column(Float, nullable=True)
    pronunciation_score = Column(Float, nullable=True)

class Knn(Base):
    __tablename__ = "knn"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source = Column(Text, nullable=False)  # 영상 이름
    mean_wpm = Column(Float, nullable=False)  # 분당 말하는 단어수 평균
    pitch_std = Column(Float, nullable=False)  # 높낮이 변화의 표준편차

class Pose(Base):
    __tablename__ = "Pose"
    id = Column(BigInteger, primary_key=True, autoincrement=True)                 # 포즈 식별자
    frame_id = Column(BigInteger, ForeignKey("frame.id", ondelete="CASCADE"), nullable=False)  # 프레임 FK
    image_type = Column(Enum("GOOD", "BAD", name="pose_image_type"), nullable=False)           # GOOD/BAD
    estimate_score = Column(Float, nullable=False)  

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    video_id = Column(BigInteger, ForeignKey("video.id", ondelete="CASCADE"), nullable=False)
    short_feedback = Column(String(200), nullable=False)
    detail_feedback = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))