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

def detect_gaze_direction_with_mediapipe(image):
    """MediaPipe를 사용한 시선 방향 감지"""
    # RGB로 변환 (MediaPipe는 RGB를 사용)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        results = face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            print("[DEBUG] No face landmarks detected")
            return "unknown"
        
        # 첫 번째 얼굴의 랜드마크 사용
        face_landmarks = results.multi_face_landmarks[0]
        
        # 눈과 코의 주요 랜드마크 인덱스
        LEFT_EYE_CENTER = 468  # 왼쪽 눈 중심 (홍채)
        RIGHT_EYE_CENTER = 473  # 오른쪽 눈 중심 (홍채)
        LEFT_EYE_INNER = 133   # 왼쪽 눈 안쪽 모서리
        LEFT_EYE_OUTER = 33    # 왼쪽 눈 바깥쪽 모서리
        RIGHT_EYE_INNER = 362  # 오른쪽 눈 안쪽 모서리
        RIGHT_EYE_OUTER = 263  # 오른쪽 눈 바깥쪽 모서리
        NOSE_TIP = 1           # 코끝
        
        # 이미지 크기
        h, w, _ = image.shape
        
        # 랜드마크 좌표 추출
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((x, y))
        
        # 눈 중심점들
        try:
            if len(landmarks) > max(LEFT_EYE_CENTER, RIGHT_EYE_CENTER, LEFT_EYE_INNER, LEFT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER):
                # 눈의 중심점과 모서리를 이용해 시선 방향 추정
                left_eye_center = landmarks[LEFT_EYE_CENTER] if LEFT_EYE_CENTER < len(landmarks) else landmarks[133]
                right_eye_center = landmarks[RIGHT_EYE_CENTER] if RIGHT_EYE_CENTER < len(landmarks) else landmarks[362]
                
                left_eye_inner = landmarks[LEFT_EYE_INNER]
                left_eye_outer = landmarks[LEFT_EYE_OUTER]
                right_eye_inner = landmarks[RIGHT_EYE_INNER]
                right_eye_outer = landmarks[RIGHT_EYE_OUTER]
                
                # 왼쪽 눈의 수평 위치 비율 계산
                left_eye_width = abs(left_eye_outer[0] - left_eye_inner[0])
                left_pupil_x = left_eye_center[0] - left_eye_inner[0]
                left_ratio = left_pupil_x / left_eye_width if left_eye_width > 0 else 0.5
                
                # 오른쪽 눈의 수평 위치 비율 계산
                right_eye_width = abs(right_eye_outer[0] - right_eye_inner[0])
                right_pupil_x = right_eye_center[0] - right_eye_inner[0]
                right_ratio = right_pupil_x / right_eye_width if right_eye_width > 0 else 0.5
                
                # 평균 비율 계산
                avg_ratio = (left_ratio + right_ratio) / 2
                
                print(f"[DEBUG] Left ratio: {left_ratio:.3f}, Right ratio: {right_ratio:.3f}, Avg: {avg_ratio:.3f}")
                
                # 시선 방향 결정
                if avg_ratio < 0.35:
                    return "left"
                elif avg_ratio > 0.65:
                    return "right"
                else:
                    return "center"
            else:
                print("[DEBUG] Using simplified eye landmark detection")
                # 기본적인 눈 랜드마크만 사용
                left_eye = landmarks[133]  # 왼쪽 눈
                right_eye = landmarks[362]  # 오른쪽 눈
                nose = landmarks[1]  # 코끝
                
                # 눈과 코의 위치를 기반으로 간단한 시선 추정
                eye_center_x = (left_eye[0] + right_eye[0]) / 2
                face_center_x = w / 2
                
                if eye_center_x < face_center_x - 20:
                    return "left"
                elif eye_center_x > face_center_x + 20:
                    return "right"
                else:
                    return "center"
                    
        except Exception as e:
            print(f"[DEBUG] Landmark processing error: {e}")
            return "center"  # 기본값으로 center 반환

def analyze_and_save_gaze(bucket: str, prefix: str, db: Session, region: str):
    """
    MediaPipe를 사용한 gaze 분석 및 DB 저장
    """
    print(f"[INFO] Starting MediaPipe-based gaze analysis for bucket: {bucket}, prefix: {prefix}")
    
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
    print(f"[INFO] Found {len(image_keys)} images for MediaPipe gaze analysis")
    
    # 딕셔너리 변수 초기화
    gaze_results = {}
    processed_count = 0
    failed_count = 0
    face_detected_count = 0

    for img_key in image_keys:
        print(f"[DEBUG] ========== Processing image: {img_key} ==========")
        
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
            
            print(f"[DEBUG] Detected gaze direction: {direction}")

            # 이미지 URL 생성: DB에 저장된 형식과 동일하게
            image_url = f"https://{bucket}.s3.{region}.amazonaws.com/{img_key}"

            # Frame 조회
            frame = db.query(Frame).filter(Frame.image_url == image_url).first()
            if not frame:
                print(f"[WARNING] No matching frame found for: {image_url}")
                failed_count += 1
                continue

            print(f"[DEBUG] Found frame_id: {frame.id}")

            # DB gaze 테이블에 저장
            crud.create_gaze_record(db, frame.id, direction)
            print(f"[INFO] Gaze saved for frame_id: {frame.id}, direction: {direction}")
            gaze_results[frame.id] = direction
            processed_count += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to process image {img_key}: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue

    print(f"[INFO] ========== MediaPipe Gaze Analysis Summary ==========")
    print(f"[INFO] Total images: {len(image_keys)}")
    print(f"[INFO] Successfully processed: {processed_count}")
    print(f"[INFO] Failed: {failed_count}")
    print(f"[INFO] Face detected in: {face_detected_count} images")
    print(f"[INFO] Face detection rate: {(face_detected_count/len(image_keys)*100):.1f}%")
    print(f"[INFO] Processing success rate: {(processed_count/len(image_keys)*100):.1f}%")
    
    # 딕셔너리 변수
    return gaze_results
