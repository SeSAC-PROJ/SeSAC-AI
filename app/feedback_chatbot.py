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
            "당신은 숙련된 발표 코치입니다. 아래는 AI가 분석한 발표자의 분석 결과(4가지 모달리티)입니다.\n\n"
            f"{analysis_str}\n\n"
            "표정이 neutral이면 중립적이라고 하지 말고, 표정이 밋밋하다고 해주고, 표정을 다채롭게 지어달라고 해야 합니다.\n"
            "만약, 시선(gaze)이 unknown이면, 눈이 잘 보이도록 다시 찍어달라고 하거나, 눈 깜빡임 횟수를 줄여보라고 해야합니다.\n"
            "분석 결과를 바탕으로 다음 두 가지 피드백을 반드시 마크다운(Markdown) 형식으로 작성하세요:\n"
            "1. **short_feedback** : 100자 내외의 간결한 요약 피드백. (짧은 문장, 마크다운 적용)\n"
            "2. **detailed_feedback** : 아래 예시처럼, 시간 구간(초 단위, 예: 5~10초)을 명시하여, 파트별로 구체적으로 작성합니다. "
            "시선, 자세, 감정, 발음·음성 등에서 칭찬도 포함하세요. 대화 없이 코치 입장에서 일방적으로 피드백하세요.\n\n"
            "예시:\n"
            "- **[시선]** 5초에서 10초 사이에 시선이 자주 바닥을 향하고 있네요. 대본을 보고 있다면 발표 내용을 미리 더 숙지하는게 좋을 거 같아요! 발표할 때 카메라를 더 자주 바라보도록 해봐요.\n"
            "- **[자세]** 12초에서 20초 사이에 몸이 좌우로 살짝 흔들렸으니 자세를 한 번 점검해 보세요.\n"
            "- **[표정]** 11초에서 30초 사이에 조금 더 미소를 지으면 좋겠어요. 1분 30초부터는 조금 더 다채로운 표정을 지어보는건 어떨까요? \n"
            "- **[속도]** 전반적으로 너무 빠릅니다. 조금만 더 천천히 여유를 가지고 해봐요. \n"
            "- **[목소리 크기]** 전반적으로 너무 작아요. 조금 더 키워보는게 좋겠어요. \n"
            "- **[발음]** 40초에서 50초에 발음이 뭉개집니다. 조금 더 또박또박 말해봐요.\n\n"
            "출력은 반드시 JSON 형식으로 하세요. 각 값은 마크다운 형식입니다. 예시:\n"
            '{\n'
            '  "short_feedback": "짧고 핵심적인 피드백 내용 (예: **시선과 자세** 모두 우수합니다!)",\n'
            '  "detailed_feedback": "- **[시선]** 5~10초: ...\\n- **[자세]** 12~20초: ..."\n'
            '}\n'
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