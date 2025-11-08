import whisper # ❗️ openai 라이브러리가 아닌, 설치한 'openai-whisper' 라이브러리 임포트
import os
from dotenv import load_dotenv

# ❗️ API 키가 필요 없는 로컬 모델 방식입니다.

# ❗️ 로컬 모델을 전역 변수로 관리하여 한번만 로드
model = None

def load_local_whisper_model():
    """
    서버 시작 시 로컬 Whisper 모델을 로드합니다.
    "base" 모델은 약 142MB입니다. (더 정확한 "small"은 466MB)
    """
    global model
    if model:
        return model
    
    print("   > [1/5] ❗️ 로컬 음성인식 AI(Whisper 'base' 모델) 로드 중...")
    try:
        # "base" 모델을 다운로드하고 로드합니다. (최초 실행 시 시간이 걸림)
        model = whisper.load_model("base") 
        print("   > [1/5] ✅ 로컬 Whisper 모델 로드 완료.")
        return model
    except Exception as e:
        print(f"❌ 로컬 Whisper 모델 로드 중 심각한 오류 발생: {e}")
        print("-> 'pip install -r requirements.txt'로 openai-whisper가 잘 설치되었는지 확인하세요.")
        raise

def transcribe_audio_with_timestamps(audio_path: str):
    """
    로컬 Whisper 모델을 사용하여 타임스탬프가 찍힌 텍스트(대본)를 반환합니다.
    """
    global model
    if not model:
        try:
            model = load_local_whisper_model() # 모델이 없으면 로드 시도
        except Exception as e:
            return [], f"로컬 Whisper 모델 로드 실패: {e}"

    print(f"   > [4/5] ❗️ 로컬 음성 인식(Whisper) 실행 중... (PC 성능에 따라 매우 오래 걸릴 수 있음)")
    
    try:
        # ❗️ 로컬 모델의 transcribe 함수 호출
        # fp16=False는 CPU에서 실행할 때 안정성을 높여줍니다.
        result = model.transcribe(audio_path, language="ko", fp16=False) 
        
        print("   > [4/5] ✅ 음성 인식 완료.")
        # 'segments' 리스트를 반환합니다.
        return result["segments"], None # (데이터, 에러 없음)
        
    except Exception as e:
        print(f"❌ 로컬 Whisper 실행 오류: {e}")
        return [], str(e) # (빈 데이터, 에러 메시지)