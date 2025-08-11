# === main.py (전체 교체본) ===

# ffmpeg PATH를 가장 먼저 등록 (subprocess에서 못 찾는 문제 방지)
import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

from fastapi import FastAPI, File, UploadFile, Form, Depends, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
import shutil, uuid
from moviepy.editor import VideoFileClip

from app.db import SessionLocal, engine, Base
from app import crud, s3_utils, video_processing
from app.config import JWT_SECRET  # 사용 안 해도 유지

# 추가 import
from app.speech_pronunciation import run_pronunciation_score  # (audio_id, wav_path, script_path)
from app.voice_hz import save_pitch_to_db                     # (audio_id, wav_path)
from app.models import (
    Audio, Emotion, Frame, Pose, Pronunciation, Pitch, Score, Feedback, Speed, Video
)
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


# --- 공용 유틸 ---
def _safe_float(x, nd=None):
    try:
        f = float(x)
        return round(f, nd) if (nd is not None) else f
    except Exception:
        return None

def _safe_str(x):
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""


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
                "matching_rate": getattr(pron_obj, "matching_rate", None),   # Pronunciation에서
                "score": getattr(score_obj, "pronunciation_score", None),    # ✅ Score에서
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
            print("[WARN] Audio object not found in DB. Skipping voice analyses merge.]")

        print(f"[INFO] Video analysis completed for video_id: {video_id}")
        print(f"[INFO] Analysis results: {results}")

        # 7) 피드백 생성 + 저장 (키 안전화)
        try:
            fb = process_and_feedback(results)
            print("[INFO] Generated Feedback:", fb)

            detail_text = fb.get("detailed_feedback", fb.get("detail_feedback", "")) or ""
            saved_fb = crud.create_feedback_record(
                db=db,
                video_id=video_id,
                short_feedback=fb.get("short_feedback", "") or "",
                detail_feedback=detail_text
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
    # === 기본 정보 ===
    video_obj = db.query(Video).filter(Video.id == video_id).first()
    if not video_obj:
        raise HTTPException(status_code=404, detail="Video not found")

    score_obj = db.query(Score).filter(Score.video_id == video_id).first()
    if not score_obj:
        raise HTTPException(status_code=404, detail="Score not found for video")

    feedback_obj = (
        db.query(Feedback)
          .filter(Feedback.video_id == video_id)
          .order_by(Feedback.created_at.desc())
          .first()
    )
    if not feedback_obj:
        raise HTTPException(status_code=404, detail="Feedback not found for video")

    # === 감정 평균 (버전/0건 안전) ===
    em_row = (
        db.query(
            func.avg(Emotion.angry).label("angry"),
            func.avg(Emotion.fear).label("fear"),
            func.avg(Emotion.surprise).label("surprise"),
            func.avg(Emotion.happy).label("happy"),
            func.avg(Emotion.sad).label("sad"),
            func.avg(Emotion.neutral).label("neutral"),
        )
        .join(Frame, Emotion.frame_id == Frame.id)
        .filter(Frame.video_id == video_id)
        .first()  # one() 대신 first()
    )

    keys = ["angry", "fear", "surprise", "happy", "sad", "neutral"]
    if em_row is None:
        emotion_avg = {k: None for k in keys}
    else:
        emotion_avg = {k: _safe_float(getattr(em_row, k, None), nd=4) for k in keys}

    # === 다른 테이블 데이터 (직렬화 안전) ===
    frames = db.query(Frame).filter(Frame.video_id == video_id).all()
    audios = db.query(Audio).filter(Audio.video_id == video_id).all()

    frame_data = [
        {
            "id": f.id,
            "frame_timestamp": _safe_float(f.frame_timestamp),
            "image_url": _safe_str(f.image_url),
        }
        for f in frames
    ]

    audio_data = []
    for a in audios:
        # speed
        speeds = db.query(Speed).filter(Speed.audio_id == a.id).all()
        speed_data = [
            {
                "id": s.id,
                "stn_start": _safe_float(s.stn_start),
                "stn_end": _safe_float(s.stn_end),
                "duration": _safe_float(s.duration),
                "num_words": s.num_words,
                "wps": _safe_float(s.wps),
                "wpm": _safe_float(s.wpm),
                "text": s.text or "",
                "wpm_band": _safe_str(s.wpm_band),
            }
            for s in speeds
        ]

        # pitch
        pitches = db.query(Pitch).filter(Pitch.audio_id == a.id).all()
        pitch_data = [
            {
                "id": p.id,
                "hz": _safe_float(p.hz),
                "time": _safe_float(p.time),
                "hz_std": _safe_float(p.hz_std),
                "proper_csv": _safe_float(p.proper_csv),
                "pitch_score": _safe_float(p.pitch_score),
            }
            for p in pitches
        ]

        # pronunciation
        prons = db.query(Pronunciation).filter(Pronunciation.audio_id == a.id).all()
        pron_data = [
            {
                "id": pr.id,
                "script_text": pr.script_text or "",
                "stt_text": pr.stt_text or "",
                "matching_rate": _safe_float(pr.matching_rate),
            }
            for pr in prons
        ]

        audio_data.append({
            "id": a.id,
            "audio_url": _safe_str(a.audio_url),
            "duration": _safe_float(a.duration),
            "speed": speed_data,
            "pitch": pitch_data,
            "pronunciation": pron_data,
        })

    # 포즈 (Frame 조인)
    poses = (
        db.query(Pose)
          .join(Frame, Pose.frame_id == Frame.id)
          .filter(Frame.video_id == video_id)
          .all()
    )
    pose_data = [
        {
            "id": p.id,
            "frame_id": p.frame_id,
            "image_type": _safe_str(p.image_type),
            "estimate_score": _safe_float(p.estimate_score),
        }
        for p in poses
    ]

    # results (문자열이면 JSON 변환, 실패시 None)
    results = getattr(score_obj, "results", None)
    if isinstance(results, str):
        try:
            import json
            results = json.loads(results)
        except Exception:
            results = None

    # === 최종 리턴 ===
    return {
        "video": {
            "id": video_obj.id,
            "user_id": video_obj.user_id,
            "upload_time": video_obj.upload_time.isoformat() if video_obj.upload_time else None,
            "title": video_obj.title,
            "video_totaltime": _safe_float(video_obj.video_totaltime),
            "video_url": _safe_str(video_obj.video_url),
        },
        "score": {
            "pose_score": _safe_float(score_obj.pose_score),
            "gaze_score": _safe_float(score_obj.gaze_score),
            "pitch_score": _safe_float(score_obj.pitch_score),
            "speed_score": _safe_float(score_obj.speed_score),
            "pronunciation_score": _safe_float(score_obj.pronunciation_score),
            "emotion_score": _safe_float(score_obj.emotion_score),
        },
        "emotion_avg": emotion_avg,  # ✅ 비디오 단위 감정 평균
        "feedback": {
            "short_feedback": feedback_obj.short_feedback or "",
            "detail_feedback": feedback_obj.detail_feedback or "",
            "created_at": feedback_obj.created_at.isoformat() if feedback_obj.created_at else None,
        },
        
        "audios": audio_data,
        "poses": pose_data,
       
    }
