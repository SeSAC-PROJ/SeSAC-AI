# === app/speech_pronunciation.py (전체 교체본) ===
import os
import re
import sys
import shutil
import numpy as np
import whisper
from pydub import AudioSegment
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.models import Audio, Pronunciation, Score

try:
    from imageio_ffmpeg import get_ffmpeg_exe  # pip install imageio-ffmpeg
except ImportError:
    get_ffmpeg_exe = None


def _ensure_ffmpeg_on_path():
    """
    ffmpeg 실행파일을 찾고 PATH에 주입.
    """
    venv_root_ffmpeg = os.path.join(sys.prefix, "ffmpeg.exe")
    venv_root_ffmpeg_nix = os.path.join(sys.prefix, "ffmpeg")
    venv_scripts_ffmpeg = os.path.join(sys.prefix, "Scripts", "ffmpeg.exe")

    candidates = [
        venv_root_ffmpeg,
        venv_root_ffmpeg_nix,
        venv_scripts_ffmpeg,
    ]

    if get_ffmpeg_exe is not None:
        try:
            bundle = get_ffmpeg_exe()
            if bundle and os.path.exists(bundle):
                candidates.append(bundle)
        except Exception:
            pass

    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        candidates.append(path_ffmpeg)

    candidates.extend([
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ])

    ffmpeg_path = None
    for c in candidates:
        if c and os.path.exists(c):
            ffmpeg_path = c
            os.environ["PATH"] = os.path.dirname(c) + os.pathsep + os.environ.get("PATH", "")
            break

    if not ffmpeg_path and not shutil.which("ffmpeg"):
        raise FileNotFoundError(
            "ffmpeg executable not found. "
            "Put ffmpeg in your venv or install 'imageio-ffmpeg', "
            "or ensure it's on PATH (e.g., C:\\ffmpeg\\bin)."
        )

    try:
        if ffmpeg_path:
            AudioSegment.converter = ffmpeg_path
            ffprobe_guess_exe = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe.exe")
            ffprobe_guess = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe")
            if os.path.exists(ffprobe_guess_exe):
                AudioSegment.ffprobe = ffprobe_guess_exe
            elif os.path.exists(ffprobe_guess):
                AudioSegment.ffprobe = ffprobe_guess
    except Exception:
        pass


# Whisper 모델 전역 캐시
_WHISPER_MODEL = None
def _get_whisper_model(model_size: str = "base"):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model(model_size)
    return _WHISPER_MODEL


# ------------------ 한글 처리 보조 함수 ------------------
def hangul_to_syllables(text: str):
    text = re.sub(r"[^가-힣 ]", "", text)
    syllables = []
    for word in text.strip().split():
        syllables.extend(list(word))
    return syllables

def split_jamo(syllable):
    BASE = 0xAC00
    CHOSUNG = 588
    JUNGSUNG = 28
    CHO =  ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
    JUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
    JONG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
    if len(syllable) != 1:
        return (None, None, None)
    code = ord(syllable) - BASE
    if code < 0 or code > 11171:
        return (None, None, None)
    cho = CHO[code // CHOSUNG]
    jung = JUNG[(code % CHOSUNG) // JUNGSUNG]
    jong = JONG[(code % JUNGSUNG)]
    return (cho, jung, jong)

similar_cho = [
    {'ㄱ','ㅋ'}, {'ㄷ','ㅌ'}, {'ㅂ','ㅍ'}, {'ㅈ','ㅊ'}, {'ㅅ','ㅆ'}, {'ㄲ','ㅋ','ㄱ'},
    {'ㅈ','ㅉ'}, {'ㄸ','ㄷ'}, {'ㅃ','ㅂ'}, {'ㄴ','ㄹ'}, {'ㅁ','ㅂ'}
]
similar_jung = [
    {'ㅏ','ㅑ'}, {'ㅓ','ㅕ'}, {'ㅗ','ㅛ'}, {'ㅜ','ㅠ'}, {'ㅡ','ㅢ'}, {'ㅐ','ㅔ','ㅒ','ㅖ'}, {'ㅚ','ㅙ'}
]

def is_similar_syllable(a, b):
    if a == b:
        return 0
    cho1, jung1, jong1 = split_jamo(a)
    cho2, jung2, jong2 = split_jamo(b)
    if None in [cho1, jung1, jong1, cho2, jung2, jong2]:
        return 2
    score = 2
    for simset in similar_cho:
        if cho1 in simset and cho2 in simset and jung1 == jung2 and jong1 == jong2:
            score = 1
    for simset in similar_jung:
        if jung1 in simset and jung2 in simset and cho1 == cho2 and jong1 == jong2:
            score = 1
    if cho1 == cho2 and jung1 == jung2 and jong1 != jong2:
        score = min(score, 1.5)
    return score


# ------------------ Levenshtein alignment ------------------
def align_ops(ref, hyp):
    m, n = len(ref), len(hyp)
    dp = np.zeros((m+1, n+1), dtype=int)
    op = [[None]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
        op[i][0] = 'D'
    for j in range(n+1):
        dp[0][j] = j
        op[0][j] = 'I'
    op[0][0] = ' '
    for i in range(1, m+1):
        for j in range(1, n+1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
                op[i][j] = 'M'
            else:
                vals = [dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+1]
                idx = np.argmin(vals)
                dp[i][j] = vals[idx]
                op[i][j] = ['D','I','S'][idx]
    i, j = m, n
    ops = []
    while i > 0 or j > 0:
        ops.append(op[i][j])
        if op[i][j] in ('M', 'S'):
            i -= 1
            j -= 1
        elif op[i][j] == 'D':
            i -= 1
        elif op[i][j] == 'I':
            j -= 1
    return ops[::-1]


# ------------------ 스크립트 텍스트 저장 ------------------
def get_or_create_script_text_from_file(db: Session, audio_id: int, script_path: str):
    """대본(.txt)에서 script_text 읽어 Pronunciation에 저장/업데이트"""
    pron_obj = db.query(Pronunciation).filter(Pronunciation.audio_id == audio_id).first()

    with open(script_path, encoding="utf-8") as f:
        script_text = f.read().strip().replace("\n", " ")

    if pron_obj:
        pron_obj.script_text = script_text
    else:
        pron_obj = Pronunciation(audio_id=audio_id, script_text=script_text)
        db.add(pron_obj)

    db.commit()
    return script_text, pron_obj


# ------------------ 메인 엔트리: 발음 점수 ------------------
def run_pronunciation_score(audio_id: int, wav_path: str, script_file_path: str, model_size: str = "base"):
    """
    ffmpeg PATH 보정 -> Whisper -> 정렬/점수 -> Pronunciation/Score 저장
    """
    db: Session = SessionLocal()
    try:
        _ensure_ffmpeg_on_path()

        # audio_id -> video_id 역추적
        audio_obj = db.query(Audio).filter(Audio.id == audio_id).first()
        if not audio_obj:
            raise ValueError(f"Audio not found for id={audio_id}")
        video_id = audio_obj.video_id

        # 입력 파일 검증
        if not wav_path or not os.path.exists(wav_path):
            raise FileNotFoundError(f"Local wav not found: {wav_path}")

        abs_audio = os.path.abspath(wav_path)
        print("DEBUG | wav_path:", abs_audio, "exists:", os.path.exists(abs_audio))
        print("DEBUG | ffmpeg which:", shutil.which("ffmpeg"))

        # 스크립트 저장/업데이트 (이 시점에 Pronunciation 레코드 보장)
        script_text, pron_obj = get_or_create_script_text_from_file(db, audio_id, script_file_path)

        # 확장자 보정
        audio_path = abs_audio
        ext = os.path.splitext(audio_path)[1].lower()
        if ext != ".wav":
            audio = AudioSegment.from_file(audio_path)
            wav_out = audio_path.rsplit(".", 1)[0] + ".wav"
            audio.export(wav_out, format="wav")
            audio_path = wav_out
            print("DEBUG | converted to wav:", audio_path, "exists:", os.path.exists(audio_path))

        # Whisper STT
        model = _get_whisper_model(model_size)
        result = model.transcribe(audio_path, language="ko")
        stt_text = result["text"].strip()

        # 비교/정렬
        ref_syll = hangul_to_syllables(script_text)
        hyp_syll = hangul_to_syllables(stt_text)

        ops = align_ops(ref_syll, hyp_syll)
        n_insert = ops.count('I')
        n_delete = ops.count('D')
        n_match = ops.count('M')

        # 치환 분석
        sub_pairs = []
        i_ref = i_hyp = 0
        for op in ops:
            if op == 'M':
                i_ref += 1
                i_hyp += 1
            elif op == 'S':
                sub_pairs.append((ref_syll[i_ref], hyp_syll[i_hyp]))
                i_ref += 1
                i_hyp += 1
            elif op == 'D':
                i_ref += 1
            elif op == 'I':
                i_hyp += 1

        sub_scores = [is_similar_syllable(a, b) for a, b in sub_pairs]
        n_sub_similar = sub_scores.count(1)
        n_sub_severe = sub_scores.count(2)
        n_sub_medium = sub_scores.count(1.5)

        # filler & stutter
        filler_words = ['어', '아', '음', '저', '뭐']
        filler_set = set(filler_words)
        filler_in_ref = set([s for s in ref_syll if s in filler_set])
        filler_inserted = [s for s in hyp_syll if s in filler_set and s not in filler_in_ref]

        def find_stutter_patterns(syll_list, filler_set, min_repeat=2):
            stutter_list = []
            i = 0
            while i < len(syll_list):
                if syll_list[i] in filler_set:
                    repeat = 1
                    j = i + 1
                    while j < len(syll_list) and syll_list[j] == syll_list[i]:
                        repeat += 1
                        j += 1
                    if repeat >= min_repeat:
                        stutter_list.append((syll_list[i], i, repeat))
                    i = j
                else:
                    i += 1
            return stutter_list

        stutter_patterns = find_stutter_patterns(hyp_syll, filler_set, min_repeat=2)

        # 점수 계산
        total_syll = len(ref_syll)
        match_score = n_match / total_syll * 100 if total_syll > 0 else 0
        W_INSERT, W_DELETE, W_SIMILAR, W_MEDIUM, W_SEVERE, W_FILLER, W_STUTTER = 50, 50, 5, 5, 200, 50, 50
        penalty = (
            (n_insert / total_syll) * W_INSERT +
            (n_delete / total_syll) * W_DELETE +
            (n_sub_similar / total_syll) * W_SIMILAR +
            (n_sub_medium / total_syll) * W_MEDIUM +
            (n_sub_severe / total_syll) * W_SEVERE +
            (len(filler_inserted) / total_syll) * W_FILLER +
            (len(stutter_patterns) / total_syll) * W_STUTTER
        )
        final_score = max(0, match_score - penalty)

        # DB 저장 (같은 Pronunciation 객체 재사용)
        pron_obj.stt_text = stt_text
        pron_obj.matching_rate = match_score

        # Score는 video_id 기준으로 저장/업데이트
        score_obj = db.query(Score).filter(Score.video_id == video_id).first()
        if not score_obj:
            score_obj = Score(video_id=video_id)
            db.add(score_obj)
        score_obj.pronunciation_score = final_score

        db.commit()

        print(f"[matching_rate]: {match_score:.1f}")
        print(f"[pronounciation_score]: {final_score:.1f}")
        print("[INFO] Pronunciation scoring saved to DB")

    finally:
        try:
            db.close()
        except Exception:
            pass
