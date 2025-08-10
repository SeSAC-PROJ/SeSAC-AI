
# === main.py (전체 교체본) ===

# ffmpeg PATH를 가장 먼저 등록 (subprocess에서 못 찾는 문제 방지)
import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

from fastapi import FastAPI, File, UploadFile, Form, Depends, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
import shutil, uuid
from moviepy.editor import VideoFileClip

from app.db import SessionLocal, engine, Base
from app import crud, s3_utils, video_processing
from app.config import JWT_SECRET  # 사용 안 해도 유지

# 추가 import
from app.speech_pronunciation import run_pronunciation_score  # (audio_id, wav_path, script_path)
from app.voice_hz import save_pitch_to_db                     # (audio_id, wav_path)
from app.models import Audio, Pronunciation, Pitch, Score, Feedback
from app.feedback_chatbot import process_and_feedback


from app.posture_classifier import BASE_DIR, classify_poses_and_save_to_db
from app.config import (
    AWS_BUCKET_NAME,
    AWS_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
)

Base.metadata.create_all(bind=engine)
app = FastAPI()


# --- DB 세션 유틸 ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session():
    """Background task용 독립적인 DB 세션"""
    return SessionLocal()


# --- Background 처리 ---
def process_video_background(video_path: str, script_path: str, out_dir: str, video_id: int, temp_file_name: str):
    """
    Background task용 비디오 처리 함수 - 전체 분석
    analyze_presentation_video가 (results, wav_path)를 반환해야 함.
    run_pronunciation_score(audio_id, wav_path, script_path)로 wav 로컬 경로 직접 전달.
    """
    db = get_db_session()
    wav_path = None
    try:
        print(f"[INFO] Background processing started for video_id: {video_id}")

        # 1) 시각/표정 분석 + 오디오 추출(wav_path)
        results, wav_path = video_processing.analyze_presentation_video(
            video_path=video_path,
            out_dir=out_dir,
            db=db,
            video_id=video_id,
            s3_utils=s3_utils
        )
        if results is None:
            results = {}

        # 2) audio_id 확보
        audio_obj = db.query(Audio).filter(Audio.video_id == video_id).first()
        if audio_obj:
            audio_id = audio_obj.id

            # posture 분류 (S3 poses/{video_id}/pose_*.jpg 기반)
            try:
                pose_res = classify_poses_and_save_to_db(
                    db=db,
                    video_id=video_id,
                    bucket=AWS_BUCKET_NAME,
                    region=AWS_REGION,
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    model_path=os.path.join(BASE_DIR, "my_pose_classifier2.keras"),
                    threshold=0.65,
                )
                results["posture"] = pose_res
            except Exception as e:
                print(f"[WARN] Posture classification failed: {e}")

            # 로컬 wav_path 필수
            if not wav_path or not os.path.exists(wav_path):
                raise FileNotFoundError(f"Local wav file not found: {wav_path}")

            # 3) 발음 분석
            try:
                run_pronunciation_score(audio_id, wav_path, script_path)
            except Exception as e:
                print(f"[WARN] Pronunciation scoring failed: {e}")

            # 4) 피치 분석 (DB 저장은 voice_hz.py 내부에서 crud 사용)
            try:
                save_pitch_to_db(audio_id, wav_path)
            except Exception as e:
                print(f"[WARN] Pitch analysis failed: {e}")

            # 5) voice 결과 병합 (speed는 video_processing에서 넣음)
            pron_obj = db.query(Pronunciation).filter_by(audio_id=audio_id).first()
            pitch_obj = db.query(Pitch).filter_by(audio_id=audio_id).first()
            score_obj = db.query(Score).filter_by(video_id=video_id).first()

            voice_block = results.get("voice", {})  # ✅ 기존 speed 유지
            voice_block["pronunciation"] = {
                "matching_rate": getattr(pron_obj, "matching_rate", None),         # Pronunciation에서
                "score": getattr(score_obj, "pronunciation_score", None),          # ✅ Score에서
            }
            voice_block["pitch"] = {
                "hz_std": getattr(pitch_obj, "hz_std", None),
                "score": getattr(pitch_obj, "pitch_score", None)
            }
            results["voice"] = voice_block

            # 6) Score(emotion/speed/pitch) 최종 반영
            score_obj = db.query(Score).filter(Score.video_id == video_id).first()
            if not score_obj:
                score_obj = Score(video_id=video_id)
                db.add(score_obj)

            # Emotion 점수
            emo_score = (results.get("emotion") or {}).get("score")
            if emo_score is not None:
                try:
                    score_obj.emotion_score = float(emo_score)
                except Exception:
                    pass

            # Speed 점수 (knn_score 우선)
            speed_block = (results.get("voice") or {}).get("speed") or {}
            sp_score = speed_block.get("final_score", speed_block.get("knn_score", speed_block.get("score")))
            if sp_score is not None:
                try:
                    score_obj.speed_score = float(sp_score)
                except Exception:
                    pass

            # Pitch 점수 (Pitch 테이블의 score는 구간마다 동일값이므로 하나만 읽어도 됨)
            if pitch_obj and getattr(pitch_obj, "pitch_score", None) is not None:
                try:
                    score_obj.pitch_score = float(pitch_obj.pitch_score)
                except Exception:
                    pass

            db.commit()

        else:
            print("[WARN] Audio object not found in DB. Skipping voice analyses merge.")

        print(f"[INFO] Video analysis completed for video_id: {video_id}")
        print(f"[INFO] Analysis results: {results}")

        # 7) 피드백 생성
        try:
            fb = process_and_feedback(results)
            print("[INFO] Generated Feedback:", fb)
            
            # === DB 저장 부분 추가 ===
            saved_fb = crud.create_feedback_record(
                db=db,
                video_id=video_id,
                short_feedback=fb.get("short_feedback", ""),
                detail_feedback=fb.get("detailed_feedback", "")
            )
            print(f"[INFO] Feedback saved! ID={saved_fb.id}")
        
        except Exception as e:
            print(f"[ERROR] Failed to generate chatbot feedback: {e}")

    except Exception as e:
        print(f"[ERROR] Background processing failed for video_id {video_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 세션 종료
        try:
            db.close()
        except Exception:
            pass

        # 임시 파일/폴더 정리
        try:
            for path in [video_path, script_path, wav_path]:
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass

            if os.path.exists(out_dir):
                try:
                    for file in os.listdir(out_dir):
                        fpath = os.path.join(out_dir, file)
                        try:
                            if os.path.isfile(fpath):
                                os.remove(fpath)
                            elif os.path.isdir(fpath):
                                shutil.rmtree(fpath, ignore_errors=True)
                        except Exception:
                            pass
                    os.rmdir(out_dir)
                except Exception:
                    pass

            print(f"[INFO] Temporary files cleaned up for video_id: {video_id}")
        except Exception as e:
            print(f"[WARNING] Failed to clean up temporary files: {e}")


# --- 샘플 페이지 ---
@app.get("/", response_class=HTMLResponse)
async def main_sample_page():
    return """
    <html>
        <head><title>AI 발표 피드백 서비스</title></head>
        <body>
            <h1>AI 발표 피드백 서비스</h1>
            <form action="/videos/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="video/*"><br><br>
                <input type="file" name="script" accept=".txt"><br><br>
                <input type="text" name="title" placeholder="제목 입력"><br><br>
                <input type="submit" value="업로드">
            </form>
            <p>업로드 후 /docs 에서 API 문서도 확인 가능합니다.</p>
        </body>
    </html>
    """


# --- 업로드 엔드포인트 ---
@app.post("/videos/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    script: UploadFile = File(...),
    title: str = Form(...),
    db: Session = Depends(get_db)
):
    user_id = 1

    os.makedirs("temp", exist_ok=True)

    # 1) 비디오 파일 저장 (로컬)
    temp_file_name = f"{uuid.uuid4()}_{file.filename}"
    temp_file_path = os.path.join("temp", temp_file_name)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2) 스크립트 파일 저장 (로컬)
    script_file_name = f"{uuid.uuid4()}_{script.filename}"
    script_file_path = os.path.join("temp", script_file_name)
    with open(script_file_path, "wb") as buffer:
        shutil.copyfileobj(script.file, buffer)

    # 3) 영상 길이 계산
    try:
        clip = VideoFileClip(temp_file_path)
        video_totaltime = clip.duration or 0
        clip.reader.close()
        if clip.audio:
            clip.audio.reader.close_proc()
    except Exception as e:
        print(f"[ERROR] Failed to extract video duration: {e}")
        video_totaltime = 0

    # 4) 비디오 원본 S3 업로드
    s3_key = f"videos/{uuid.uuid4()}/{file.filename}"
    s3_video_url = s3_utils.upload_file_to_s3(temp_file_path, s3_key)

    # 5) DB 저장
    db_video = crud.create_video(
        db,
        user_id=user_id,
        title=title,
        video_totaltime=video_totaltime,
        video_url=s3_video_url
    )

    # 6) 작업 디렉토리 준비
    out_dir = os.path.join("temp", str(db_video.id))
    os.makedirs(out_dir, exist_ok=True)

    # 7) Background Task 실행 (로컬 경로 전달)
    background_tasks.add_task(
        process_video_background,
        video_path=temp_file_path,
        script_path=script_file_path,
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

# --- 비디오 분석 결과 조회 엔드포인트 ---
from fastapi import HTTPException

@app.get("/videos/{video_id}/analysis")
def get_video_analysis(video_id: int, db: Session = Depends(get_db)):
    """
    지정된 video_id에 대한 Score와 Feedback을 하나의 응답으로 반환합니다.
    """
    # Score 조회
    score_obj = db.query(Score).filter(Score.video_id == video_id).first()
    if not score_obj:
        raise HTTPException(status_code=404, detail="Score not found for video")

    # Feedback 조회 (가장 최신)
    feedback_obj = (
        db.query(Feedback)
          .filter(Feedback.video_id == video_id)
          .order_by(Feedback.created_at.desc())
          .first()
    )
    if not feedback_obj:
        raise HTTPException(status_code=404, detail="Feedback not found for video")

    return {
        "video_id": video_id,
        "score": {
            "pose_score": score_obj.pose_score,
            "emotion_score": score_obj.emotion_score,
            "gaze_score": score_obj.gaze_score,
            "pitch_score": score_obj.pitch_score,
            "speed_score": score_obj.speed_score,
            "pronunciation_score": score_obj.pronunciation_score,
        },
        "feedback": {
            "short_feedback": feedback_obj.short_feedback,
            "detail_feedback": feedback_obj.detail_feedback,
            "created_at": feedback_obj.created_at.isoformat(),
        },
    }