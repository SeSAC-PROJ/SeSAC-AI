import boto3
import cv2
import numpy as np
import mediapipe as mp
from sqlalchemy.orm import Session
from app import crud
from app.models import Frame
from app.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION


# MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
_FACE_MESH = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)


# 더 관대한 임계값 설정
HORIZ_LEFT_THRESH   = 0.3
HORIZ_RIGHT_THRESH  = 0.7
VERT_DOWN_THRESH    = 0.6


# 눈 관련 랜드마크 인덱스
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
LEFT_EYE_UPPER = 159
LEFT_EYE_LOWER = 145
RIGHT_EYE_UPPER = 380
RIGHT_EYE_LOWER = 385


# 눈 깜빡임 검출용
LEFT_EYE_LANDMARKS = [33, 159, 158, 133, 153, 145]
RIGHT_EYE_LANDMARKS = [362, 380, 374, 263, 386, 385]


def read_image_from_s3(bucket: str, key: str):
    """S3에서 이미지를 읽고 OpenCV 이미지로 반환"""
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    try:
        res = s3.get_object(Bucket=bucket, Key=key)
        arr = np.frombuffer(res['Body'].read(), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] 읽기 실패: {e}")
        return None


def calculate_eye_aspect_ratio(eye_points):
    """눈 깜빡임 검출을 위한 EAR 계산"""
    try:
        # 수직 거리들
        A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        # 수평 거리
        C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        # EAR 계산
        ear = (A + B) / (2.0 * C) if C > 0 else 0
        return ear
    except:
        return 0.3


def is_eye_closed(eye_points, threshold=0.05): # 방금 여기 수정
    """EAR로 눈 깜빡임 검출"""
    ear = calculate_eye_aspect_ratio(eye_points)
    return ear < threshold


def detect_gaze_direction_with_mediapipe(image: np.ndarray) -> str:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = _FACE_MESH.process(rgb)
   
    if not results.multi_face_landmarks:
        print("[DEBUG] No face detected")
        return "center"

    lm = results.multi_face_landmarks[0].landmark
    h, w, _ = image.shape
    landmarks = [(int(p.x * w), int(p.y * h)) for p in lm]

    try:
        # 눈 깜빡임 검사
        left_eye_points = [landmarks[i] for i in LEFT_EYE_LANDMARKS]
        right_eye_points = [landmarks[i] for i in RIGHT_EYE_LANDMARKS]
        left_closed = is_eye_closed(left_eye_points)
        right_closed = is_eye_closed(right_eye_points)

        print(f"[DEBUG] Left eye closed: {left_closed}, Right eye closed: {right_closed}")
        if left_closed or right_closed:
            print("[DEBUG] Eye blink detected -> down")
            return "down"

        # 홍채와 눈 윤곽 landmark
        left_iris = landmarks[LEFT_IRIS_CENTER]
        right_iris = landmarks[RIGHT_IRIS_CENTER]
        left_inner = landmarks[LEFT_EYE_INNER]
        left_outer = landmarks[LEFT_EYE_OUTER]
        right_inner = landmarks[RIGHT_EYE_INNER]
        right_outer = landmarks[RIGHT_EYE_OUTER]
        left_upper = landmarks[LEFT_EYE_UPPER]
        left_lower = landmarks[LEFT_EYE_LOWER]
        right_upper = landmarks[RIGHT_EYE_UPPER]
        right_lower = landmarks[RIGHT_EYE_LOWER]

        # 왼쪽 눈
        left_min_x = min(left_inner[0], left_outer[0])
        left_max_x = max(left_inner[0], left_outer[0])
        left_eye_width = left_max_x - left_min_x
        left_ratio = (left_iris[0] - left_min_x) / left_eye_width if left_eye_width > 0 else 0.5

        # 오른쪽 눈
        right_min_x = min(right_inner[0], right_outer[0])
        right_max_x = max(right_inner[0], right_outer[0])
        right_eye_width = right_max_x - right_min_x
        right_ratio = (right_iris[0] - right_min_x) / right_eye_width if right_eye_width > 0 else 0.5

        # 수평 비율 (중앙값 사용)
        horiz_ratio = np.median([left_ratio, right_ratio])

        # 수직 비율 계산
        left_eye_height = abs(left_upper[1] - left_lower[1])
        left_v_ratio = (left_iris[1] - left_upper[1]) / left_eye_height if left_eye_height > 0 else 0.5
        right_eye_height = abs(right_upper[1] - right_lower[1])
        right_v_ratio = (right_iris[1] - right_upper[1]) / right_eye_height if right_eye_height > 0 else 0.5
        vert_ratio = np.median([left_v_ratio, right_v_ratio])

        print(f"[DEBUG] left_ratio: {left_ratio:.3f}, right_ratio: {right_ratio:.3f}, horiz_ratio: {horiz_ratio:.3f}")
        print(f"[DEBUG] left_v_ratio: {left_v_ratio:.3f}, right_v_ratio: {right_v_ratio:.3f}, vert_ratio: {vert_ratio:.3f}")

        # 시선 방향 결정
        if vert_ratio > VERT_DOWN_THRESH:
            print("[DEBUG] Looking down")
            return "down"
        elif horiz_ratio < HORIZ_LEFT_THRESH:
            print("[DEBUG] Looking left")
            return "left"
        elif horiz_ratio > HORIZ_RIGHT_THRESH:
            print("[DEBUG] Looking right")  
            return "right"
        else:
            print("[DEBUG] Looking center")
            return "center"

    except Exception as e:
        print(f"[DEBUG] 계산 오류: {e}")
        return "center"


def analyze_and_save_gaze(bucket: str, prefix: str, db: Session, region: str):
    """
    눈동자 기반 gaze 분석 수행 후
    - frame별 gaze 테이블에 저장
    - gaze_results 딕셔너리 반환
    - gaze_score를 Score 테이블에 저장
    """
    print(f"[INFO] Gaze 분석 시작: bucket={bucket}, prefix={prefix}")
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' not in resp:
        print("[WARN] 분석할 이미지 없음")
        return {}


    image_keys = sorted(
        obj['Key'] for obj in resp['Contents']
        if obj['Key'].lower().endswith(('.jpg', '.png'))
    )
    print(f"[INFO] 총 {len(image_keys)}개 이미지 처리 예정")


    gaze_results = {}
    processed_count = 0
   
    for idx, key in enumerate(image_keys):
        print(f"[DEBUG] 처리 중 ({idx+1}/{len(image_keys)}): {key}")
       
        img = read_image_from_s3(bucket, key)
        if img is None:
            continue


        direction = detect_gaze_direction_with_mediapipe(img)
        print(f"[DEBUG] 감지된 방향: {direction}")
       
        image_url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
        frame = db.query(Frame).filter(Frame.image_url == image_url).first()
        if not frame:
            print(f"[WARN] 프레임을 찾을 수 없음: {image_url}")
            continue


        crud.create_gaze_record(db, frame.id, direction)
        gaze_results[frame.id] = direction
        processed_count += 1


    print(f"[INFO] 처리 완료: {processed_count}개 프레임")


    # gaze_score 계산
    stats = {}
    for d in gaze_results.values():
        stats[d] = stats.get(d, 0) + 1
   
    print(f"[INFO] 방향별 분포: {stats}")
   
    total = sum(stats.values())
    center_ct = stats.get("center", 0)
    gaze_score = (center_ct / total) * 100 if total > 0 else 0
    print(f"[INFO] Gaze score (center 비율 %): {gaze_score:.2f}")


    # DB에 gaze_score 저장
    try:
        video_id = int(prefix.split("/")[1])
        crud.upsert_score(db, video_id, gaze_score=gaze_score)
        print(f"[INFO] gaze_score 저장 완료: video_id={video_id}, score={gaze_score:.2f}")
    except Exception as e:
        print(f"[WARN] gaze_score 저장 실패: {e}")


    gaze_results["gaze_score"] = gaze_score
    return gaze_results
