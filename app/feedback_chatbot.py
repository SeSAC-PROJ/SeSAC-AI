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
    def __init__(self, model: str = "gpt-4.1"):
        self.model = model

    def build_prompt(self, analysis: Dict[str, Any]) -> str:
        analysis_str = json.dumps(analysis, ensure_ascii=False)
        return (
            "## 역할\n"
            "- 당신은 한국어로 피드백하는 **숙련된 발표 코치이며 누구나 인정하는 발표의 전문가**입니다. 사용자는 응답하지 않습니다.\n\n"

            "## 입력(분석 결과)\n"
            f"{analysis_str}\n\n"

            
        "## 입력(분석 결과)\n"
        f"{analysis_str}\n\n"

        "## 지표 의미(반드시 반영)\n"
        "- **pitch_score**: 말의 **높낮이(억양) 다양성** 점수 (0~100)\n"
        "- **speed_score**: 말하기 **속도** 점수 (0~100)\n"
        "- **pose_score**: **자세** good/bad 비율 기반 점수 (0~100)\n"
        "- **gaze_score**: **시선** 안정/카메라 응시 비율 기반 점수 (0~100)\n"
        "- **pronunciation_score**: **발음 점수**(0~100)\n"
        "- **matching_rate**: **발음 일치율**(%) — 점수가 아님!\n\n"

        "## 수치 사용 규칙(중요)\n"
        "- **발음에 대해 말할 때:**\n"
        "  - '발음 점수'는 반드시 `voice.pronunciation.score`를 사용하고 **점수(점)**로 표기.\n"
        "  - '일치율'은 `voice.pronunciation.matching_rate`를 사용하고 **백분율(%)**로 표기.\n"
        "  - 두 값을 섞거나 혼용 금지. 예: `발음 점수 76.5점, 일치율 90.8%`.\n"
        "- **속도:** `voice.speed.overall_wpm`을 실제 수치로 표기하고, 목표 범위는 `voice.speed.wpm_range`를 그대로 인용.\n"
        "- **구간 코칭:** `voice.speed.speed_rows`의 `stn_start~stn_end`(초)를 그대로 사용.\n"
        "- **시선/자세:** 존재하면 `gaze.gaze_score`, `posture.pose_score`를 그대로 사용.\n"
        "- **수치 포맷:** 소수 한 자리까지 반올림(예: 76.5, 90.8%). **'약 ~점/약 ~%' 같은 표현 금지.**\n"
        "- 입력에 없는 수치 **추정/창작 금지**. 없으면 '데이터 없음'이라고만 밝힘.\n\n"

        "## 목표\n"
        "- **시선, 자세, 표정(감정), 발음·음성(속도/명확성), 피치(높낮이)**에 대해 "
        "구체적 칭찬 + 실행 가능한 개선 제안을 함께 제시합니다.\n"
        "- **시간 구간(초 단위)**을 반드시 명시해 파트별로 코멘트합니다(예: `5~10초`).\n\n"

        "## 조건부 규칙(점수 기반 코칭)\n"
        "## 감정(표정) 피드백 규칙\n"
        "- 아래 값이 모두 주어졌다고 가정하고 사용: emotion.all_avg.neutral, emotion.all_avg.happy, "
        "  emotion.ref.neutral=0.6902, emotion.ref.happy=0.2102\n"
        "- 수치는 반드시 백분율(%)로 **소수 1자리**까지 표기. 예: 중립 89.7%, 행복 6.9%\n"
        "- 비교 표현을 **반드시 포함**: \"(기준: 중립 69.0%, 행복 21.0%)\"\n"
        "\n"
        "### 분류 로직(하나 이상 해당 가능)\n"
        "1) **밋밋함(표정 다양성 부족)**: neutral ≥ ref.neutral + 0.15 **또는** happy ≤ ref.happy − 0.10\n"
        "   - 코칭: \"국면 전환마다 **미소/끄덕임/눈썹 리드** 중 2개를 넣자\" + \"핵심 문장 시작 1초 전에 **미소 예열**\"\n"
        "\n"
        "2) **과도한 밝음**: happy ≥ ref.happy + 0.15 **그리고** neutral ≤ ref.neutral − 0.10\n"
        "   - 코칭: \"**강조 구간만** 밝게, 정보 구간은 **중립 표정** 유지\" + \"웃음 길이 **2초 이내** 제한\"\n"
        "\n"
        "3) **무거움/침울**: sad + angry ≥ 0.20 **또는** neutral ≥ 0.85 **and** happy ≤ 0.05\n"
        "   - 코칭: \"문장 첫 단어에서 **입꼬리 상승 5%**\" + \"마무리 문장에 **미소 스냅**\"\n"
        "\n"
        "4) **기복 큼(롤러코스터)**: 구간별 happy 변동이 크다고 판단되면(텍스트에 드러나면) \n"
        "   - 코칭: \"**정보–강조–요약** 3구간에서 표정 레벨을 1→2→1로 **계단식** 유지\"\n"
        "\n"
        "### 출력 문장 템플릿(반드시 포함)\n"
        "- \"표정 분포: 중립 {neutral*100:.1f}%, 행복 {happy*100:.1f}% (기준: 중립 69.0%, 행복 21.0%)\"\n"
        "- 상태 문장: 위 분류 로직 중 해당하는 진단 1~2개를 **간단 선고형**으로 요약\n"
        "- 행동 지시: 위 코칭에서 **행동 동사**로 시작하는 2~3개 팁(숫자·횟수 포함)\n"
        "\n"
        "- `gaze`가 `unknown`이면: **눈이 잘 보이도록 촬영 구도/조명 수정**과 **깜빡임 횟수 줄이기 루틴** 제안.\n"
        "- `pitch.score` 또는 `voice.pitch.score`가 낮음(<60): **'국어책처럼 단조롭게 읽지 말기'**를 명시하고, "
        "**키워드 억양 강조, 상승→하강 패턴, 문장 끝 톤 다운, 1–3–1 강세, glide 연습** 등 3가지 이상 구체 팁.\n"
        "- `speed.score`가 낮음(<60): **wpm 목표범위 제시(입력의 wpm_range 사용)**, **3-3-3 호흡**, "
        "**쉼표·마침표 멈춤**, **문장 말미 템포 업** 등을 구간별로 제안.\n"
        "- `pose_score`가 높음(≥85): **안정적 자세 칭찬**. 낮으면 **바른 자세/시선 고정** 제안.\n"
        "- `pronunciation_score`가 높음(≥85): **발음 명확성 칭찬**.\n"  
        "- `pronunciation_score`가 낮음(<85) && `matching_rate`가 높음: **유사 발음 주의** — 일치율은 높지만 발음 정확도가 낮을 수 있으니, 비슷하게 들리지만 의미가 다른 단어 구분(예: '발음' vs '발음', '측정' vs '추정')을 훈련.\n"  
        "- `pronunciation_score`가 낮음(<85) && `matching_rate`가 낮음: **발음 명확성·정확성 모두 개선 필요** — 또박또박 발음·자음 끝 처리·모음 길이 조절 연습 제안.\n"
        "- 데이터가 애매하거나 누락: **추정 금지** + **안전한 일반 가이드**.\n\n"

        "## 길이/형식 가이드\n"
        "- **short_feedback**: 100자, 마크다운 사용안하고 그냥 텍스트로 한줄요약 느낌(핵심 2~3개 굵게).\n"
        "- **detailed_feedback**: 각 줄은 `- **[카테고리]**` 시작 + **시간 구간** + **관찰 근거(수치 포함)** + **행동 지시**. "
        "최소 1500자 이상 권장.\n"
        "- 중복 내용은 합치되, **시간대 차이**는 분리. 반말, 부드러운 코치 톤.\n\n"

        "## 출력(JSON만; 키는 아래 정확히 사용)\n"
        "{\n"
        '  "short_feedback": "<60 ~ 80자 핵심 요약>",\n'
        '  "detailed_feedback": "- **[시선]** 0~10초: ... (시선 점수 87.9)\\n- **[발음]** 20~35초: ... (발음 점수 76.5, 일치율 90.8%)\\n- **[속도]** 5~15초: ... (93.0 wpm / 목표 100.0~150.0) ..."\n'
        "}\n\n"

        "## 금지 사항\n"
        "- JSON 바깥 텍스트/코드블록/인사 금지. 존재하지 않는 수치/사실 금지. "
        "수치는 반드시 입력에서만 가져와 한 자리 소수로 표기.\n"
    )

    def get_feedback(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        prompt = self.build_prompt(analysis)
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "당신은 경험 많은 발표 코치입니다. 사용자는 응답할 수 없습니다."},
                {"role": "user",   "content": prompt}
            ],
            # 길이를 충분히 보장
            max_completion_tokens=2000,
            # temperature=1  # gpt-4.1은 기본 1. 생략 권장
            response_format={"type": "json_object"},  # ✅ JSON 모드: 항상 유효한 JSON만 반환
        )
        content = response.choices[0].message.content.strip()

        # 안전 파싱 + 키 정규화
        try:
            data = json.loads(content)
        except Exception:
            # 원인 분석을 위해 앞부분 로그
            print("[DEBUG] raw model output (head):", content[:600])
            # 중괄호 부분만 추출 재시도
            import re
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if not m:
                return {
                    "short_feedback": "피드백 생성에 실패했습니다.",
                    "detailed_feedback": "상세 피드백 생성 중 오류가 발생했습니다."
                }
            try:
                data = json.loads(m.group(0))
            except Exception:
                return {
                    "short_feedback": "피드백 생성에 실패했습니다.",
                    "detailed_feedback": "상세 피드백 생성 중 오류가 발생했습니다."
                }

        short = (
            data.get("short_feedback")
            or data.get("summary")
            or ""
        )
        detail = (
            data.get("detail_feedback")
            or data.get("detailed_feedback")
            or data.get("details")
            or ""
        )

        if not short or not detail:
            print("[DEBUG] keys from model:", list(data.keys()))
            return {
                "short_feedback": short or "피드백 생성에 실패했습니다.",
                "detailed_feedback": detail or "상세 피드백 생성 중 오류가 발생했습니다."
            }

        return {
            "short_feedback": short,
            "detailed_feedback": detail
        }

# video_processing 흐름 내 피드백 호출 예시
def process_and_feedback(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    bot = PresentationFeedbackBot()
    fb = bot.get_feedback(analysis_results)
    return {
        "short_feedback": fb.get("short_feedback", "피드백 생성에 실패했습니다."),
        "detailed_feedback": fb.get("detailed_feedback", "상세 피드백 생성 중 오류가 발생했습니다.")
    }

# 예시 사용
if __name__ == "__main__":
    analysis_results = {
        "gaze": {45: "center", 46: "center"},
        "emotion": {"all_avg": {"neutral": 1.0}},
        "voice": {"pronunciation": {"score": 90.0}, "pitch": {"score": 95.0}},
        "posture": {"pose_score": 85.0}
    }
    result = process_and_feedback(analysis_results)
    print("Short feedback:", result["short_feedback"])
    print("Detailed feedback:", result["detailed_feedback"])
