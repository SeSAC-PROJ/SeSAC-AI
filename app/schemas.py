# 
from pydantic import BaseModel
from datetime import datetime

class VideoCreate(BaseModel):
    title: str
    video_totaltime: float

class VideoOut(BaseModel):
    id: int
    title: str
    video_totaltime: float
    audio_url: str

    class Config:
        orm_mode = True