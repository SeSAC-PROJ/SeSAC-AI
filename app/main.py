from fastapi import FastAPI, File, UploadFile, Form, Header, Depends, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
import shutil, os, uuid
from moviepy.editor import VideoFileClip
from app.db import SessionLocal, engine, Base
from app import crud, s3_utils, video_processing
from app.config import JWT_SECRET

Base.metadata.create_all(bind=engine)
app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session():
    """Background task용 독립적인 DB 세션"""
    return SessionLocal()

def process_video_background(video_path, out_dir, video_id, temp_file_name):
    """Background task용 비디오 처리 함수 - gaze 분석 포함"""
    db = get_db_session()
    try:
        print(f"[INFO] Background processing started for video_id: {video_id}")
        gaze_results = video_processing.extract_frames_and_audio_with_gaze(
            video_path=video_path,
            out_dir=out_dir,
            db=db,
            video_id=video_id,
            s3_utils=s3_utils
        )
        print(f"[INFO] Background processing completed for video_id: {video_id}")
        print(f"[INFO] Gaze results: {gaze_results}")
    except Exception as e:
        print(f"[ERROR] Background processing failed for video_id {video_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
        # 임시 파일 정리
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            # 디렉토리 내 모든 파일 삭제 후 디렉토리 삭제
            if os.path.exists(out_dir):
                for file in os.listdir(out_dir):
                    os.remove(os.path.join(out_dir, file))
                os.rmdir(out_dir)
            print(f"[INFO] Temporary files cleaned up for video_id: {video_id}")
        except Exception as e:
            print(f"[WARNING] Failed to clean up temporary files: {e}")

@app.get("/", response_class=HTMLResponse)
async def main_sample_page():
    return """
    <html>
        <head><title>AI 발표 피드백 서비스</title></head>
        <body>
            <h1>AI 발표 피드백 서비스</h1>
            <form action="/videos/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="video/*"><br><br>
                <input type="text" name="title" placeholder="제목 입력"><br><br>
                <input type="submit" value="업로드">
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
        if clip.audio:
            clip.audio.reader.close_proc()
    except Exception as e:
        print(f"[ERROR] Failed to extract video duration: {e}")
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

    # Background task 실행 (gaze 분석 포함)
    background_tasks.add_task(
        process_video_background,
        video_path=temp_file_path,
        out_dir=out_dir,
        video_id=db_video.id,
        temp_file_name=temp_file_name
    )

    return {
        "message": "Upload success, processing started",
        "video_id": db_video.id,
        "s3_url": s3_video_url,
        "video_totaltime": video_totaltime,
    }

# @app.get("/videos/{video_id}/gaze")
# async def get_gaze_results(video_id: int, db: Session = Depends(get_db)):
#     """특정 비디오의 gaze 분석 결과 조회"""
#     try:
#         # video_id로 frame들을 조회하고, 각 frame의 gaze 데이터를 가져옴
#         from app.models import Frame, Gaze
#         frames = db.query(Frame).filter(Frame.video_id == video_id).all()
#         if not frames:
#             raise HTTPException(status_code=404, detail="No frames found for this video")
        
#         gaze_data = []
#         for frame in frames:
#             gaze = db.query(Gaze).filter(Gaze.frame_id == frame.id).first()
#             if gaze:
#                 gaze_data.append({
#                     "frame_id": frame.id,
#                     "frame_timestamp": frame.frame_timestamp,
#                     "direction": gaze.direction
#                 })
        
#         return {
#             "video_id": video_id,
#             "total_frames": len(frames),
#             "analyzed_frames": len(gaze_data),
#             "gaze_data": gaze_data
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))