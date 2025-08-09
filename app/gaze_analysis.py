import boto3
import cv2
import numpy as np
import mediapipe as mp
from sqlalchemy.orm import Session
from app import crud
from app.models import Frame
from app.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def read_image_from_s3(bucket: str, key: str):
    """S3에서 이미지를 읽고 올바른 형식으로 변환"""
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    try:
        print(f"[DEBUG] Reading image from S3: {key}")
        res = s3_client.get_object(Bucket=bucket, Key=key)
        img_data = res['Body'].read()
        
        # 바이트 데이터를 numpy 배열로 변환
        img_array = np.frombuffer(img_data, np.uint8)
        
        # OpenCV로 이미지 디코딩
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"[ERROR] Failed to decode image: {key}")
            return None
            
        return img
        
    except Exception as e:
        print(f"[ERROR] Error reading image from S3: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_eye_aspect_ratio(eye_landmarks):
    """Eye Aspect Ratio (EAR) 계산으로 눈 깜빡임 감지"""
    # 수직 거리 계산 (위/아래 눈꺼풀)
    A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks))
    B = np.linalg.norm(np.array(eye_landmarks[14]) - np.array(eye_landmarks))
    
    # 수평 거리 계산 (좌/우 눈 모서리)
    C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[15]))
    
    # EAR 계산
    ear = (A + B) / (2.0 * C) if C > 0 else 0
    return ear

def is_eye_open(eye_landmarks, ear_threshold=0.25):
    """눈이 열려있는지 EAR로 판단"""
    ear = calculate_eye_aspect_ratio(eye_landmarks)
    return ear > ear_threshold

def detect_gaze_direction_with_mediapipe(image):
    """개선된 MediaPipe 시선 방향 감지"""
    # RGB로 변환 (MediaPipe는 RGB를 사용)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,  # 신뢰도 높임
        min_tracking_confidence=0.3
    ) as face_mesh:
        
        results = face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            print("[DEBUG] No face landmarks detected")
            return "unknown"
        
        # 첫 번째 얼굴의 랜드마크 사용
        face_landmarks = results.multi_face_landmarks[0]
        
        # 눈 랜드마크 인덱스 (MediaPipe Face Mesh)
        LEFT_EYE_LANDMARKS = [33, 159, 158, 133, 153, 145]   # 왼쪽 눈 6개 점
        RIGHT_EYE_LANDMARKS = [362, 380, 374, 263, 386, 385]  # 오른쪽 눈 6개 점
        
        # 홍채 중심점
        LEFT_IRIS_CENTER = 468
        RIGHT_IRIS_CENTER = 473
        
        # 눈 모서리
        LEFT_EYE_INNER = 133
        LEFT_EYE_OUTER = 33
        RIGHT_EYE_INNER = 362
        RIGHT_EYE_OUTER = 263
        
        # 이미지 크기
        h, w, _ = image.shape
        
        # 랜드마크 좌표 추출
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((x, y))
        
        try:
            if len(landmarks) <= max(LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER):
                print("[DEBUG] Not enough landmarks for iris detection")
                return "center"
            
            # 눈 깜빡임 검사 (EAR 사용)
            left_eye_points = [landmarks[i] for i in LEFT_EYE_LANDMARKS]
            right_eye_points = [landmarks[i] for i in RIGHT_EYE_LANDMARKS]
            
            left_eye_open = is_eye_open(left_eye_points, ear_threshold=0.25)
            right_eye_open = is_eye_open(right_eye_points, ear_threshold=0.25)
            
            print(f"[DEBUG] Left eye open: {left_eye_open}, Right eye open: {right_eye_open}")
            
            # 눈이 감겨있으면 center로 처리
            if not (left_eye_open and right_eye_open):
                print("[DEBUG] Eyes are closed or blinking, returning center")
                return "center"
            
            # 홍채 중심점과 눈 모서리 좌표
            left_iris = landmarks[LEFT_IRIS_CENTER]
            right_iris = landmarks[RIGHT_IRIS_CENTER]
            left_inner = landmarks[LEFT_EYE_INNER]
            left_outer = landmarks[LEFT_EYE_OUTER]
            right_inner = landmarks[RIGHT_EYE_INNER]
            right_outer = landmarks[RIGHT_EYE_OUTER]
            
            # 왼쪽 눈의 시선 비율 계산
            left_eye_width = abs(left_outer[0] - left_inner)
            left_iris_x = left_iris - left_inner
            left_ratio = left_iris_x / left_eye_width if left_eye_width > 0 else 0.5
            
            # 오른쪽 눈의 시선 비율 계산  
            right_eye_width = abs(right_inner[0] - right_outer[0])
            right_iris_x = right_inner - right_iris 
            right_ratio = right_iris_x / right_eye_width if right_eye_width > 0 else 0.5
            
            # 평균 비율 계산
            avg_ratio = (left_ratio + right_ratio) / 2
            
            print(f"[DEBUG] Left ratio: {left_ratio:.3f}, Right ratio: {right_ratio:.3f}, Avg: {avg_ratio:.3f}")
            
            # 더 보수적인 임계값으로 시선 방향 결정
            if avg_ratio < 0.25:      # 0.35 → 0.25로 더 보수적
                return "left"
            elif avg_ratio > 0.75:    # 0.65 → 0.75로 더 보수적  
                return "right"
            else:
                return "center"
                
        except Exception as e:
            print(f"[DEBUG] Landmark processing error: {e}")
            return "center"

def analyze_and_save_gaze(bucket: str, prefix: str, db: Session, region: str):
    """
    개선된 MediaPipe 기반 gaze 분석 및 DB 저장
    """
    print(f"[INFO] Starting improved MediaPipe-based gaze analysis for bucket: {bucket}, prefix: {prefix}")
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    
    try:
        result = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    except Exception as e:
        print(f"[ERROR] Error listing S3 objects: {e}")
        return {}

    if 'Contents' not in result:
        print("[WARNING] No images found in S3 for gaze analysis")
        return {}

    image_keys = sorted([obj['Key'] for obj in result['Contents'] if obj['Key'].lower().endswith(('.jpg', '.png'))])
    print(f"[INFO] Found {len(image_keys)} images for improved MediaPipe gaze analysis")
    
    # 분석 결과 통계
    gaze_results = {}
    processed_count = 0
    failed_count = 0
    face_detected_count = 0
    blink_count = 0
    
    # 연속 프레임 검증을 위한 버퍼
    recent_directions = []
    buffer_size = 3

    for idx, img_key in enumerate(image_keys):
        print(f"[DEBUG] ========== Processing image {idx+1}/{len(image_keys)}: {img_key} ==========")
        
        try:
            # S3에서 이미지 읽기
            frame_img = read_image_from_s3(bucket, img_key)
            if frame_img is None:
                print(f"[WARNING] Failed to read image: {img_key}")
                failed_count += 1
                continue

            # MediaPipe로 시선 방향 감지
            direction = detect_gaze_direction_with_mediapipe(frame_img)
            
            if direction != "unknown":
                face_detected_count += 1
            
            # 연속 프레임 안정성 검사
            recent_directions.append(direction)
            if len(recent_directions) > buffer_size:
                recent_directions.pop(0)
            
            # 최근 프레임들의 일관성 확인 (안정화)
            if len(recent_directions) >= buffer_size:
                # 최근 3개 프레임 중 2개 이상이 같은 방향이면 채택
                direction_counts = {}
                for d in recent_directions:
                    direction_counts[d] = direction_counts.get(d, 0) + 1
                
                # 가장 빈번한 방향 선택
                stable_direction = max(direction_counts, key=direction_counts.get)
                if direction_counts[stable_direction] >= 2:
                    final_direction = stable_direction
                else:
                    final_direction = "center"  # 불안정하면 center로
            else:
                final_direction = direction
            
            print(f"[DEBUG] Raw direction: {direction}, Stabilized: {final_direction}")

            # 이미지 URL 생성
            image_url = f"https://{bucket}.s3.{region}.amazonaws.com/{img_key}"

            # Frame 조회
            frame = db.query(Frame).filter(Frame.image_url == image_url).first()
            if not frame:
                print(f"[WARNING] No matching frame found for: {image_url}")
                failed_count += 1
                continue

            print(f"[DEBUG] Found frame_id: {frame.id}")

            # DB gaze 테이블에 저장 (안정화된 방향 사용)
            crud.create_gaze_record(db, frame.id, final_direction)
            print(f"[INFO] Gaze saved for frame_id: {frame.id}, direction: {final_direction}")
            gaze_results[frame.id] = final_direction
            processed_count += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to process image {img_key}: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue

    print(f"[INFO] ========== Improved MediaPipe Gaze Analysis Summary ==========")
    print(f"[INFO] Total images: {len(image_keys)}")
    print(f"[INFO] Successfully processed: {processed_count}")
    print(f"[INFO] Failed: {failed_count}")
    print(f"[INFO] Face detected in: {face_detected_count} images")
    print(f"[INFO] Face detection rate: {(face_detected_count/len(image_keys)*100):.1f}%")
    print(f"[INFO] Processing success rate: {(processed_count/len(image_keys)*100):.1f}%")
    
    # 시선 방향 분포 출력
    direction_stats = {}
    for direction in gaze_results.values():
        direction_stats[direction] = direction_stats.get(direction, 0) + 1
    print(f"[INFO] Gaze direction distribution: {direction_stats}")
    
    # dictionary 형태로 결과 반환
    return gaze_results