import whisper
import parselmouth  # ⭐️ [추가] Praat 라이브러리 임포트
import os
from dotenv import load_dotenv
from pathlib import Path
import numpy as np

# ❗️ 로컬 모델을 전역 변수로 관리하여 한번만 로드
model = None

def load_local_whisper_model():
    """
    서버 시작 시 로컬 Whisper 모델을 로드합니다.
    """
    global model
    if model:
        return model
    
    print("   > [AI 1/3] ❗️ 로컬 음성인식 AI(Whisper 'base' 모델) 로드 중...")
    try:
        model = whisper.load_model("small") 
        print("   > [AI 1/3] ✅ 로컬 Whisper 모델 로드 완료.")
        return model
    except Exception as e:
        print(f"❌ 로컬 Whisper 모델 로드 중 심각한 오류 발생: {e}")
        raise

def transcribe_audio_with_timestamps(audio_path: str):
    """
    로컬 Whisper 모델을 사용하여 타임스탬프가 찍힌 텍스트(대본)를 반환합니다.
    """
    global model
    if not model:
        try:
            model = load_local_whisper_model()
        except Exception as e:
            return [], f"로컬 Whisper 모델 로드 실패: {e}"

    print(f"   > [4/6] ❗️ 로컬 음성 인식(Whisper) 실행 중... (시간 소요)")
    
    try:
        result = model.transcribe(audio_path, language="ko", fp16=False) 
        print("   > [4/6] ✅ 음성 인식 완료.")
        return result["segments"], None # (데이터, 에러 없음)
        
    except Exception as e:
        print(f"❌ 로컬 Whisper 실행 오류: {e}")
        return [], str(e) # (빈 데이터, 에러 메시지)

# ⭐️ [수정] 음성 운율(목소리 떨림) 분석 함수 로직 수정
def analyze_prosody_for_segments(audio_path: Path, segments: list) -> list:
    """
    Whisper가 나눠놓은 'segments' 시간대별로 Jitter와 Shimmer를 계산합니다.
    (segments 리스트를 직접 수정하여 반환합니다)
    """
    print(f"   > [5/6] ❗️ 음성 운율(목소리 떨림) 분석 중... (Praat)")
    try:
        snd = parselmouth.Sound(str(audio_path))
        
        for segment in segments:
            start_time = segment['start']
            end_time = segment['end']
            
            # 음성 파일에서 해당 세그먼트 구간만 잘라내기
            part = snd.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)
            
            # Praat을 사용하여 Pitch(음높이) 객체 생성 (기본 설정 사용)
            pitch = part.to_pitch()
            
            # ⭐️ [수정] Pitch 객체에서 PointProcess 생성
            point_process = parselmouth.praat.call(pitch, "To PointProcess")
            
            # Jitter (주파수 변동성, %): 목소리 높낮이의 불안정성
            jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
            
            # ⭐️ [수정] Shimmer 계산 시 'part'(Sound) 객체도 함께 전달
            shimmer_local = parselmouth.praat.call([part, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
            
            segment['jitter'] = jitter_local * 100  # 백분율로 변환
            segment['shimmer'] = shimmer_local * 100 # 백분율로 변환

        print(f"   > [5/6] ✅ 음성 운율 분석 완료.")
        return segments # Jitter, Shimmer가 추가된 segments
        
    except Exception as e:
        # 오류 발생 시 (예: 세그먼트가 너무 짧거나 음성이 없는 경우) 0으로 처리
        print(f"   > [5/6] ⚠️  음성 운율 분석 경고: {e}")
        for segment in segments:
            if 'jitter' not in segment:
                segment['jitter'] = 0
            if 'shimmer' not in segment:
                segment['shimmer'] = 0
        return segments