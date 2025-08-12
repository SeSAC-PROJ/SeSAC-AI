import os
import re
import json
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
import openai

# env에서 OpenAI API 키 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")


def _parse_intervals(text: str) -> List[Tuple[float, float, Tuple[int, int]]]:
    """
    텍스트에서 시간 구간을 찾는다.
    허용 패턴:
      1) a~b초
      2) a b초  (중간 ~ 빠짐)
      3) a,b초  (콤마 연결)
    반환: (start, end, (span_start, span_end))
    """
    intervals: List[Tuple[float, float, Tuple[int, int]]] = []

    # 1) 정상형 a~b초
    for m in re.finditer(r'(\d+(?:\.\d)?)\s*~\s*(\d+(?:\.\d)?)\s*초', text):
        a, b = float(m.group(1)), float(m.group(2))
        if a <= b:
            intervals.append((round(a, 1), round(b, 1), m.span()))

    # 2) 오류형 a b초  또는 a,b초
    for m in re.finditer(r'(?<!\d)(\d+(?:\.\d)?)\s*[, ]\s*(\d+(?:\.\d)?)\s*초', text):
        a, b = float(m.group(1)), float(m.group(2))
        if a <= b:
            intervals.append((round(a, 1), round(b, 1), m.span()))

    return sorted(intervals, key=lambda x: (x[0], x[1]))


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    merged: List[Tuple[float, float]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        # 0.05초 이내면 이어진 것으로 간주
        if s <= cur_e + 0.05:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def normalize_time_ranges(text: str) -> str:
    """취소선 제거, 잘못된 시간 표기 교정, 소수 한 자리 통일 + '~' 공백 강제."""
    if not text:
        return text

    # 우발적 취소선 패턴 제거: '~~' → ' ~ '
    text = text.replace("~~", " ~ ")

    # 쉼표로 이어지는 '...초, ...초, ...초' 덩어리 교정
    def repl_list(match: re.Match) -> str:
        raw = match.group(0)
        inner = _parse_intervals(raw)
        ranges = _merge_intervals([(s, e) for s, e, _ in inner]) or [(s, e) for s, e, _ in inner]
        # 반드시 ' ~ ' (양쪽 공백)로 출력
        return ", ".join([f"{s:.1f} ~ {e:.1f}초" for s, e in ranges])

    text = re.sub(
        r'((?:\d+(?:\.\d)?\s*(?:~\s*\d+(?:\.\d)?|\s*[, ]\s*\d+(?:\.\d)?)\s*초)'
        r'(?:\s*,\s*(?:\d+(?:\.\d)?\s*(?:~\s*\d+(?:\.\d)?|\s*[, ]\s*\d+(?:\.\d)?)\s*초))*)',
        repl_list,
        text,
    )

    # 개별 오류형: 'a b초' 또는 'a,b초' → 'a ~ b초'
    text = re.sub(r'(\d+(?:\.\d)?)\s*[, ]\s*(\d+(?:\.\d)?)\s*초', r'\1 ~ \2초', text)

    # 정상형도 공백 강제: 'a~b초' → 'a ~ b초'
    text = re.sub(r'(\d+(?:\.\d)?)\s*~\s*(\d+(?:\.\d)?)\s*초', r'\1 ~ \2초', text)

    # 혹시 숫자 사이에 공백 없이 남은 '~'가 있으면 강제 공백 삽입
    text = re.sub(r'(?<=\d)~(?=\d)', ' ~ ', text)

    # 소수 둘 이상 → 한 자리
    text = re.sub(r'(\d+\.\d{2,})', lambda m: f"{float(m.group(1)):.1f}", text)

    return text


def fix_decimals(text: str) -> str:
    """숫자 소수 자릿수 정규화(둘 이상 → 한 자리)."""
    if not text:
        return text
    return re.sub(r'(\d+\.\d{2,})', lambda m: f"{float(m.group(1)):.1f}", text)


class PresentationFeedbackBot:
    def __init__(self, model: str = "gpt-4.1"):
        self.model = model

    def build_prompt(self, analysis: Dict[str, Any]) -> str:
        analysis_str = json.dumps(analysis, ensure_ascii=False)
        return (
            "## 역할\n"
            "- 당신은 한국어로 피드백하는 **숙련된 발표 코치**입니다. 사용자는 응답하지 않습니다.\n\n"

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
            "- 발음: `voice.pronunciation.score`는 **점수(점)**, "
            "`voice.pronunciation.matching_rate`는 **백분율(%)**.\n"
            "- 속도: `voice.speed.overall_wpm` 실제 수치, 목표 범위는 `voice.speed.wpm_range` 그대로 인용.\n"
            "- 구간 코칭: `voice.speed.speed_rows`의 시작~끝(초)을 그대로 사용.\n"
            "- **모든 시간 구간은 반드시 `a~b초` 형식**으로 표기(예: `9.2~13.4초`). 쉼표/공백 연결 금지.\n"
            "- 소수 한 자리까지 반올림. **추정/창작 금지**.\n"
            "- **마크다운 취소선(`~~`) 절대 사용 금지.**\n\n"

            "## 목표\n"
            "- 시선/자세/표정/발음/속도/피치에 대해 **구체적 칭찬 + 실행 가능한 개선 제안**.\n"
            "- **시간 구간(초 단위)**을 꼭 명시.\n\n"

            "## 조건부 규칙(점수 기반 코칭)\n"
            "## 감정(표정) 피드백 규칙\n"
            "- 아래 값이 모두 주어졌다고 가정하고 사용: emotion.all_avg.neutral, emotion.all_avg.happy, "
            "  emotion.ref.neutral=0.6902, emotion.ref.happy=0.2102\n"
            "- 수치는 반드시 백분율(%)로 **소수 1자리**까지 표기. 예: 중립 89.7%, 행복 6.9%\n"
            "- 비교 표현 **반드시 포함**: \"(기준: 중립 69.0%, 행복 21.0%)\"\n"
            "\n"
            "### 분류 로직(하나 이상 해당 가능)\n"
            "1) **밋밋함(표정 다양성 부족)**: neutral ≥ ref.neutral + 0.15 **또는** happy ≤ ref.happy − 0.10\n"
            "   - 코칭: \"국면 전환마다 **미소/끄덕임/눈썹 리드** 중 2개를 넣자\" + \"핵심 문장 시작 1초 전에 **미소 예열**\"\n"
            "2) **과도한 밝음**: happy ≥ ref.happy + 0.15 **그리고** neutral ≤ ref.neutral − 0.10\n"
            "   - 코칭: \"**강조 구간만** 밝게, 정보 구간은 **중립 표정** 유지\" + \"웃음 길이 **2초 이내** 제한\"\n"
            "3) **무거움/침울**: sad + angry ≥ 0.20 **또는** neutral ≥ 0.85 **and** happy ≤ 0.05\n"
            "   - 코칭: \"문장 첫 단어에서 **입꼬리 상승 5%**\" + \"마무리 문장에 **미소 스냅**\"\n"
            "4) **기복 큼(롤러코스터)**: 구간별 happy 변동이 크면 "
            "\"**정보–강조–요약** 3구간에서 표정 레벨을 1→2→1로 **계단식** 유지\"\n"
            "\n"
            "### 출력 문장 템플릿(반드시 포함)\n"
            "- \"표정 분포: 중립 {neutral*100:.1f}%, 행복 {happy*100:.1f}% (기준: 중립 69.0%, 행복 21.0%)\"\n"
            "- 상태 문장: 위 분류 로직 중 해당하는 진단 1~2개 **간단 선고형** 요약\n"
            "- 행동 지시: 위 코칭에서 **행동 동사**로 시작하는 2~3개 팁(숫자·횟수 포함)\n"
            "\n"
            "- `gaze`가 `unknown`이면: **촬영 구도/조명 수정** + **깜빡임 루틴** 제안.\n"
            "- `pitch.score` 또는 `voice.pitch.score` < 60: '국어책처럼 단조롭게 읽지 말기'를 명시하고 "
            "키워드 억양, 상승→하강, 문장 끝 톤 다운, 1–3–1 강세, glide 연습 등 3가지 이상 제시.\n"
            "- `speed.score` < 60: wpm 목표범위(`wpm_range`) 제시, 3-3-3 호흡, 쉼표·마침표 멈춤, "
            "문장 말미 템포 업 등을 구간별로 제안.\n"
            "- `pose_score` ≥ 85: 안정적 자세 칭찬. 낮으면 바른 자세/시선 고정 제안.\n"
            "- `pronunciation_score` ≥ 85: 발음 명확성 칭찬.\n"
            "- `pronunciation_score` < 85 && `matching_rate` 높음: 유사 발음 주의 훈련.\n"
            "- `pronunciation_score` < 85 && `matching_rate` 낮음: 명확성·정확성 모두 개선 팁.\n"
            "- 데이터 누락/애매: 추정 금지 + 일반 가이드.\n\n"

            "## 길이/형식 가이드\n"
            "- short_feedback: 60~80자 한줄요약(마크다운 금지).\n"
            "- detailed_feedback: 각 줄은 `- **[카테고리]**` 시작 + **시간 구간** + **관찰 근거(수치)** + **행동 지시**. "
            "최소 1500자 이상 권장.\n"
            "- 중복 내용은 합치되, **시간대 차이**는 분리. 반말, 부드러운 코치 톤.\n\n"

            "## 출력(JSON만; 키는 아래 정확히 사용)\n"
            "{\n"
            '  "short_feedback": "<60 ~ 80자 핵심 요약>",\n'
            '  "detailed_feedback": "- **[시선]** 0~10초: ... (시선 점수 87.9)\\n- **[발음]** 20~35초: ... (발음 점수 76.5, 일치율 90.8%)\\n- **[속도]** 5~15초: ... (93.0 wpm / 목표 100.0~150.0) ..."\n'
            "}\n\n"

            "## 금지 사항\n"
            "- JSON 바깥 텍스트/코드블록/인사 금지. 존재하지 않는 수치/사실 금지. "
            "수치는 반드시 입력에서만 가져와 한 자리 소수로 표기. **취소선(~~) 금지.**\n"
        )

    def get_feedback(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        prompt = self.build_prompt(analysis)
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "당신은 경험 많은 발표 코치입니다. 사용자는 응답할 수 없습니다."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=2000,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content.strip()

        # 안전 파싱 + 키 정규화
        try:
            data = json.loads(content)
        except Exception:
            print("[DEBUG] raw model output (head):", content[:600])
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if not m:
                return {
                    "short_feedback": "피드백 생성에 실패했습니다.",
                    "detailed_feedback": "상세 피드백 생성 중 오류가 발생했습니다.",
                }
            try:
                data = json.loads(m.group(0))
            except Exception:
                return {
                    "short_feedback": "피드백 생성에 실패했습니다.",
                    "detailed_feedback": "상세 피드백 생성 중 오류가 발생했습니다.",
                }

        short = (data.get("short_feedback") or data.get("summary") or "").strip()
        detail = (
            data.get("detail_feedback")
            or data.get("detailed_feedback")
            or data.get("details")
            or ""
        ).strip()

        # 🔧 후처리: 취소선/시간구간 정규화 + 소수 자리 통일
        short = fix_decimals(normalize_time_ranges(short))
        detail = fix_decimals(normalize_time_ranges(detail))

        if not short or not detail:
            print("[DEBUG] keys from model:", list(data.keys()))
            return {
                "short_feedback": short or "피드백 생성에 실패했습니다.",
                "detailed_feedback": detail or "상세 피드백 생성 중 오류가 발생했습니다.",
            }

        return {"short_feedback": short, "detailed_feedback": detail}


# video_processing 흐름 내 피드백 호출 예시
def process_and_feedback(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    bot = PresentationFeedbackBot()
    fb = bot.get_feedback(analysis_results)
    return {
        "short_feedback": fb.get("short_feedback", "피드백 생성에 실패했습니다."),
        "detailed_feedback": fb.get("detailed_feedback", "상세 피드백 생성 중 오류가 발생했습니다."),
    }


# 예시 사용
if __name__ == "__main__":
    analysis_results = {
        "gaze": {45: "center", 46: "center"},
        "emotion": {"all_avg": {"neutral": 1.0, "happy": 0.0}},
        "voice": {
            "pronunciation": {"score": 90.0, "matching_rate": 95.3},
            "pitch": {"score": 95.0},
            "speed": {
                "overall_wpm": 120.0,
                "wpm_range": [100.0, 150.0],
                "speed_rows": [
                    {"stn_start": 7.5, "stn_end": 8.7, "wpm": 96.8},
                    {"stn_start": 23.4, "stn_end": 29.5, "wpm": 89.1},
                ],
            },
        },
        "posture": {"pose_score": 86.0},
        "gaze_score": 100.0,
    }
    result = process_and_feedback(analysis_results)
    print("Short feedback:", result["short_feedback"])
    print("Detailed feedback:", result["detailed_feedback"])
