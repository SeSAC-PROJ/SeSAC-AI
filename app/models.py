# db 테이블 구조를 SQLAlchemy ORM 클래스로 정의
from sqlalchemy import Column, BigInteger, Float, String, ForeignKey, Text, TIMESTAMP, Integer
from sqlalchemy.ext.declarative import declarative_base
from app.db import Base

class Video(Base):
    __tablename__ = "video"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False)
    upload_time = Column(TIMESTAMP)
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