from fastapi import FastAPI, File, UploadFile, Form, Header, Depends, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
import shutil, os, uuid
from moviepy.editor import VideoFileClip

from app.db import SessionLocal, engine, Base
from app import crud, s3_utils, video_processing
# from jose import jwt, JWTError  # JWT 관련 주석 처리
from app.config import JWT_SECRET

Base.metadata.create_all(bind=engine)
app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# JWT 토큰 부분 주석 처리!
# def extract_user_id_from_jwt(authorization: str):
#     if not authorization:
#         raise HTTPException(status_code=401, detail="Authorization header missing")
#     try:
#         token = authorization.replace("Bearer ", "")
#         payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
#         user_id = int(payload.get("user_id"))
#         if user_id is None:
#             raise HTTPException(status_code=401, detail="User ID not found in token")
#         return user_id
#     except (JWTError, ValueError) as e:
#         raise HTTPException(status_code=401, detail="Invalid token or decoding error")

@app.get("/", response_class=HTMLResponse)
async def main_sample_page():
    return """
    <html>
        <head>
            <title>Video Upload Example</title>
        </head>
        <body>
            <h2>Video Upload Test (FastAPI + S3 + DB)</h2>
            <form action="/videos/upload" enctype="multipart/form-data" method="post">
                Title: <input type="text" name="title" required/><br/>
                <input type="file" name="file" accept="video/*" required/><br/>
                <input type="submit" value="Upload" />
            </form>
            <p>업로드 후 /docs 에서 API 문서도 확인 가능합니다.</p>
        </body>
    </html>
    """

@app.post("/videos/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    # authorization: str = Header(None),  # JWT 부분 제거
    db: Session = Depends(get_db)
):
    # user_id를 기본 값(예: 1)로 임의 설정
    user_id = 1

    os.makedirs("temp", exist_ok=True)
    temp_file_name = f"{uuid.uuid4()}_{file.filename}"
    temp_file_path = os.path.join("temp", temp_file_name)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 영상 길이 추출 (moviepy)
    try:
        clip = VideoFileClip(temp_file_path)
        video_totaltime = clip.duration
        clip.reader.close()
        clip.audio.reader.close_proc() if clip.audio else None
    except Exception as e:
        video_totaltime = 0

    s3_key = f"videos/{uuid.uuid4()}/{file.filename}"
    s3_video_url = s3_utils.upload_file_to_s3(temp_file_path, s3_key)

    db_video = crud.create_video(
        db,
        user_id=user_id,
        title=title,
        video_totaltime=video_totaltime,
        video_url=s3_video_url
    )

    os.makedirs(f"temp/{db_video.id}", exist_ok=True)
    out_dir = f"temp/{db_video.id}"
    background_tasks.add_task(
        video_processing.extract_frames_and_audio,
        video_path=temp_file_path,
        out_dir=out_dir,
        db=db,
        video_id=db_video.id,
        s3_utils=s3_utils
    )

    return {
        "message": "Upload success, processing started",
        "video_id": db_video.id,
        "s3_url": s3_video_url,
        "video_totaltime": video_totaltime,
    }