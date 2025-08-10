import pandas as pd
from sqlalchemy.orm import Session
from app.models import Knn
from app.db import SessionLocal  # 세션 생성 함수

# CSV 로드
df = pd.read_csv("source_pitch_wpm.csv")

# DB 세션
db: Session = SessionLocal()

for _, row in df.iterrows():
    knn_entry = Knn(
        source=row['source'],
        mean_wpm=row['mean_wpm'],
        pitch_std=row['pitch_std']
    )
    db.add(knn_entry)

db.commit()
db.close()

print("✅ KNN 테이블에 데이터 삽입 완료")
