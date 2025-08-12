import os
import re
import json
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
import openai

# =============== 환경 설정 ===============
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")


# =============== 유틸: 시선 비율 계산 ===============
def compute_gaze_stats(gaze: Dict[Any, Any]) -> Dict[str, Any]:
    """
    gaze 딕셔너리(프레임->라벨)에서 비율(%)을 계산해서 반환.
    반환 예시:
      {
        "center_ratio": 87.9, "down_ratio": 8.6,
        "left_ratio": 2.1, "right_ratio": 1.4, "unknown_ratio": 0.0,
        "side_ratio": 3.5, "total": 123
      }
    """
    from collections import Counter
    labels = [v for v in gaze.values() if isinstance(v, str)]
    c = Counter(labels)
    total = sum(c.values()) or 1
    getp = lambda k: round(c.get(k, 0) * 100.0 / total, 1)
    left = getp("left")
    right = getp("right")
    return {
        "center_ratio": getp("center"),
        "down_ratio": getp("down"),
        "left_ratio": left,
        "right_ratio": right,
        "unknown_ratio": getp("unknown"),
        "side_ratio": round(left + right, 1),
        "total": total,
    }


# =============== 유틸: 시간 문자열 교정 ===============
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


def _strip_gaze_times(detail: str) -> str:
    """
    시선 라인에서 실수로 들어간 시간 구간(a ~ b초)을 제거해 비율만 남기기.
    """
    if not detail:
        return detail
    lines = detail.splitlines()
    out = []
    for ln in lines:
        if ln.startswith("- **[시선]**"):
            ln = re.sub(r'\d+(?:\.\d+)?\s*~\s*\d+(?:\.\d+)?\s*초', '', ln)
            ln = re.sub(r'\s{2,}', ' ', ln).strip()
        out.append(ln)
    return "\n".join(out)


# =============== 코치 봇 ===============
class PresentationFeedbackBot:
    def __init__(self, model: str = "gpt-4.1", fps: int = 30):
        self.model = model
        self.fps = fps  # 프레임→초 변환 지시를 위해 프롬프트에 명시

    def build_prompt(self, analysis: Dict[str, Any]) -> str:
        """
        프롬프트 핵심:
        - 시선: '시간' 금지, '비율(%)'만 사용 (center/down/side/unknown)
        - 숫자 포맷: 소수 1자리, '약 ~' 금지, 취소선 금지
        - 발음: score=점수(점), matching_rate=퍼센트(%)
        - 속도: 전체 wpm + 구간별 speed_rows 초 사용
        - 표정: ref와 비교, 임계값 로직 + 실행 팁
        - 행동 문장: '처음에는, 다음에는, 마지막에는' 형식
        """
        # 파생: 시선 비율 계산
        gaze = analysis.get("gaze", {}) if isinstance(analysis.get("gaze"), dict) else {}
        gaze_stats = compute_gaze_stats(gaze)

        # 메타
        meta = {
            "notes": {
                "fps": self.fps,
                "gaze_time_unit": "frames",
                "convert_rule": "시선 섹션에서는 시간을 쓰지 말 것(비율만). 다른 섹션에서 초 표기는 a ~ b초 형식."
            }
        }

        payload = {"analysis": analysis, "_derived": {"gaze_stats": gaze_stats}, "_meta": meta}
        analysis_str = json.dumps(payload, ensure_ascii=False)

        return (
            "## 역할\n"
            "- 당신은 한국어로 피드백하는 **숙련된 발표 코치**입니다. 사용자는 응답하지 않습니다.\n\n"

            "## 입력(분석 결과 + 파생값 + 메타)\n"
            f"{analysis_str}\n\n"

            "## 시선 작성 규칙(중요: 시간 금지)\n"
            "- **[시선] 섹션에서는 시간 구간(초)을 절대 쓰지 말고, 오직 비율(%)만 사용**.\n"
            "- 사용할 값: _derived.gaze_stats.center_ratio, down_ratio, left_ratio, right_ratio, side_ratio, unknown_ratio.\n"
            "- 템플릿 예: \"- **[시선]** 전체: center {center_ratio}%, down {down_ratio}%, side {side_ratio}% — ...\"\n"
            "- 행동 제안: '처음에는, 다음에는, 마지막에는' 2~3문장.\n\n"

            "## 지표 의미(반드시 반영)\n"
            "- **pitch_score**: 말의 **높낮이(억양) 다양성** 점수 (0~100)\n"
            "- **speed_score**: 말하기 **속도** 점수 (0~100)\n"
            "- **pose_score**: **자세** good/bad 비율 기반 점수 (0~100)\n"
            "- **gaze_score**: **시선** 안정/카메라 응시 비율 기반 점수 (0~100)\n"
            "- **pronunciation_score**: **발음 점수**(0~100)\n"
            "- **matching_rate**: **발음 일치율**(%) — 점수가 아님!\n\n"

            "## 수치/시간 사용 규칙\n"
            "- 발음: `voice.pronunciation.score`는 **점수(점)**, "
            "`voice.pronunciation.matching_rate`는 **백분율(%)**. 혼용 금지.\n"
            "- 속도: `voice.speed.overall_wpm` 실제 수치, 목표 범위는 `voice.speed.wpm_range` 그대로 인용.\n"
            "- 구간 코칭: `voice.speed.speed_rows`의 시작~끝(초)을 그대로 사용.\n"
            "- **시간 표기**(시선 제외): 반드시 `a ~ b초` 형식(공백 포함)만 사용.\n"
            "- 모든 수치는 **소수 1자리**. **“약 ~%/점” 금지**, **취소선(~~) 금지**.\n"
            "- 입력에 없는 수치 **추정·창작 금지**. 없으면 “데이터 없음”.\n\n"

            "## 감정(표정) 피드백 규칙\n"
            "- 사용 값: emotion.all_avg.neutral/happy/sad/angry, emotion.ref.neutral=0.6902, emotion.ref.happy=0.2102\n"
            "- 표기: **백분율(%) 소수 1자리**, 비교 문구 포함: \"(기준: 중립 69.0%, 행복 21.0%)\"\n"
            "  1) **밋밋함**: neutral ≥ ref.neutral + 0.15 또는 happy ≤ ref.happy − 0.10 → "
            "     '미소/끄덕임/눈썹 리드' 2개 + '핵심 문장 1초 전 미소 예열'\n"
            "  2) **과도한 밝음**: happy ≥ ref.happy + 0.15 & neutral ≤ ref.neutral − 0.10 → "
            "     강조 구간만 밝게/웃음 2초 이내\n"
            "  3) **무거움/침울**: (sad+angry) ≥ 0.20 또는 (neutral ≥ 0.85 & happy ≤ 0.05) → "
            "     첫 단어 입꼬리 5% 상승 + 마무리 미소 스냅\n"
            "  4) **기복 큼**: happy 변동이 크면 → 정보–강조–요약을 1→2→1로 계단식 유지\n\n"

            "## 조건부 규칙(점수 기반)\n"
            "- gaze가 unknown이면: 촬영 구도/조명 수정 + 깜빡임 루틴.\n"
            "- voice.pitch.score < 60: '국어책처럼 단조롭게 읽지 말기' 포함 + "
            "키워드 억양, 상승→하강, 문장 끝 톤 다운, 1–3–1 강세, glide 연습 중 3개 이상.\n"
            "- 속도: overall_wpm < wpm_range 하한 또는 speed.bad_ratio ≥ 0.4 → "
            "3-3-3 호흡, 쉼표·마침표 멈춤, 문장 말미 템포 업; speed_rows로 구간 코칭.\n"
            "- 자세: posture.pose_score ≥ 85 → 안정적 자세 칭찬; 미만이면 바른 자세/시선 고정/양발 균형.\n"
            "- 발음: pronunciation.score ≥ 85 → 발음 명확성 칭찬. "
            "score < 85 & matching_rate ≥ 85% → 유사 발음 주의. "
            "score < 85 & matching_rate < 85% → 명확성·정확성 모두 개선.\n\n"

            "## 행동 제안 문장 스타일\n"
            "- 목록은 **'처음에는, 다음에는, 마지막에는'** 사용. '1) 2) 3)' 금지. 문장은 **동사로 시작**.\n"
            "- 카테고리 내 스타일 일관 유지.\n\n"

            "## 길이·형식\n"
            "- **short_feedback**: 100~150자, 한 줄(마크다운 금지).\n"
            "- **detailed_feedback**: 각 줄 `- **[카테고리]**` 시작 + **시간/비율** + **관찰 수치** + **행동 지시**, 900~1500자 권장.\n\n"

            "## 출력(JSON만; 키 이름 정확히 사용)\n"
            "{\n"
            '  "short_feedback": "<100~150자 한 줄 요약>",\n'
            '  "detailed_feedback": "- **[시선]** 전체: center {center_ratio}%, down {down_ratio}%, side {side_ratio}% — ...\\n- **[속도]** 5.0 ~ 15.0초: ... (93.0 wpm / 목표 100.0 ~ 150.0) ..."\n'
            "}\n\n"

            "## 금지 사항\n"
            "- JSON 바깥 텍스트/코드블록/인사 금지. '약 ~%/점' 금지. 취소선(~~) 금지. "
            "입력에 없는 수치/사실 창작 금지.\n"
        )

    def get_feedback(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        prompt = self.build_prompt(analysis)
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "당신은 경험 많은 발표 코치입니다. 사용자는 응답할 수 없습니다."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=1600,
            response_format={"type": "json_object"},  # JSON 모드
            # temperature=1  # gpt-4.1 기본 1
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

        # 🔧 후처리: 시간 표기/소수 자리 통일 + 시선라인 시간 강제 제거
        short = fix_decimals(normalize_time_ranges(short))
        detail = fix_decimals(normalize_time_ranges(detail))
        detail = _strip_gaze_times(detail)

        if not short or not detail:
            print("[DEBUG] keys from model:", list(data.keys()))
            return {
                "short_feedback": short or "피드백 생성에 실패했습니다.",
                "detailed_feedback": detail or "상세 피드백 생성 중 오류가 발생했습니다.",
            }

        return {"short_feedback": short, "detailed_feedback": detail}


# =============== 파이프라인 진입점 ===============
def process_and_feedback(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    bot = PresentationFeedbackBot()
    fb = bot.get_feedback(analysis_results)
    return {
        "short_feedback": fb.get("short_feedback", "피드백 생성에 실패했습니다."),
        "detailed_feedback": fb.get("detailed_feedback", "상세 피드백 생성 중 오류가 발생했습니다."),
    }


# =============== 예시 실행 ===============
if __name__ == "__main__":
    # 시선 dict에 프레임->라벨 혼재 가능. 우리는 비율만 쓰므로 안전.
    analysis_results = {
        "gaze": {45: "center", 46: "center", 47: "down", 48: "center", "gaze_score": 88.0},
        "emotion": {
            "all_avg": {"neutral": 0.90, "happy": 0.05, "sad": 0.03, "angry": 0.02},
            "ref": {"neutral": 0.6902, "happy": 0.2102},
            "score": 65.2
        },
        "voice": {
            "pronunciation": {"score": 76.5, "matching_rate": 90.8},
            "pitch": {"score": 39.8},
            "speed": {
                "overall_wpm": 93.0,
                "wpm_range": [100.0, 150.0],
                "bad_ratio": 0.5,
                "speed_rows": [
                    {"stn_start": 5.9, "stn_end": 10.4, "wpm": 92.5, "wpm_band": "bad"},
                    {"stn_start": 17.9, "stn_end": 25.3, "wpm": 90.2, "wpm_band": "bad"},
                    {"stn_start": 25.9, "stn_end": 34.8, "wpm": 95.0, "wpm_band": "bad"},
                    {"stn_start": 51.4, "stn_end": 56.1, "wpm": 89.4, "wpm_band": "bad"},
                ],
            },
        },
        "posture": {"pose_score": 100.0},
        "gaze_score": 87.9,
    }

    result = process_and_feedback(analysis_results)
    print("Short feedback:", result["short_feedback"])
    print("Detailed feedback:", result["detailed_feedback"])
