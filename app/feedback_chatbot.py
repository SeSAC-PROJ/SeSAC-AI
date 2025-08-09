import os
import openai
import json
from typing import Dict, Any
from dotenv import load_dotenv

# env에서 OpenAI API 키 로드함.
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

class PresentationFeedbackBot:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model

    def build_prompt(self, analysis: Dict[str, Any]) -> str:
        # 분석 결과를 JSON 형태로 문자열로 변환
        analysis_str = json.dumps(analysis, ensure_ascii=False)
        return (
            "당신은 숙련된 발표 코치입니다. 아래는 AI가 분석한 발표자의 4가지 모달리티 결과입니다.\n\n"
            f"{analysis_str}\n\n"
            "위 내용을 바탕으로 다음 두 가지 피드백을 생성하세요:\n"
            "1) short_feedback: 발표자가 바로 이해할 수 있는 50~100자 이내의 간결한 요약 피드백\n"
            "2) detailed_feedback: 구체적이고 친절한 개선 제안과 칭찬을 포함한 상세 피드백\n\n"
            "사용자는 응답할 수 없으며, 코치인 당신만 일방적으로 말합니다.\n"
            "출력은 반드시 JSON 형식으로만, 키는 short_feedback, detailed_feedback 두 개만 포함하세요.\n"
            '{ "short_feedback": "...", "detailed_feedback": "..." }\n'
        )

    def get_feedback(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        prompt = self.build_prompt(analysis)
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "당신은 경험 많은 발표 코치입니다. 사용자는 응답할 수 없습니다."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        content = response.choices[0].message.content.strip()
        import json
        try:
            return json.loads(content)
        except Exception:
            return {
                "short_feedback": "피드백 생성에 실패했습니다.",
                "detailed_feedback": "상세 피드백 생성 중 오류가 발생했습니다."
            }

# video_processing 흐름 내 피드백 호출 예시
def process_and_feedback(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    분석 완료 직후 이 함수를 호출해 즉시 피드백을 생성·반환합니다.
    analysis_results: video_processing.analyze_.. 에서 반환된 딕셔너리
    """
    bot = PresentationFeedbackBot()
    fb = bot.get_feedback(analysis_results)
    return {
        # "analysis": analysis_results,
        "short_feedback": fb["short_feedback"],
        "detailed_feedback": fb["detailed_feedback"]
    }

# 예시 사용
if __name__ == "__main__":
    # analysis_results는 비디오 처리 직후 생성된 결과
    analysis_results = {
        "gaze": {45: "center", 46: "center"},
        "emotion": {"all_avg": {"neutral": 1.0}},
        "voice": {"pronunciation": {"score": 90.0}, "pitch": {"score": 95.0}},
        "posture": {"pose_score": 85.0}
    }
    result = process_and_feedback(analysis_results)
    print("Short feedback:", result["short_feedback"])
    print("Detailed feedback:", result["detailed_feedback"])