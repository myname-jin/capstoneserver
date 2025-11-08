import google.generativeai as genai
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수(API 키) 로드
load_dotenv()

# Gemini API 키 설정
gemini_key = os.getenv("GEMINI_API_KEY")
gemini_model = None

if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        # ⭐️ 404 오류 방지를 위해 가장 범용적인 1.0 pro 모델 사용
        gemini_model = genai.GenerativeModel('gemini-1.5-flash') 
    except Exception as e:
        print(f"Gemini 클라이언트 초기화 실패: {e}")
else:
    print("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")

def is_gemini_configured():
    """Gemini API 키가 올바르게 설정되었는지 확인합니다."""
    return gemini_model is not None # 모델 객체가 성공적으로 생성되었는지 확인

def get_ai_score(aligned_data: list):
    """
    정렬된 데이터를 Gemini API로 보내 채점 및 피드백을 받습니다.
    """
    if not is_gemini_configured():
        return {"error": "Gemini API 키가 설정되지 않아 AI 채점을 수행할 수 없습니다."}

    if not aligned_data:
        return {"error": "분석 데이터가 없어 AI 채점을 할 수 없습니다."}

    print("   > [6/6] Gemini API 호출 중... (최대 30초 소요될 수 있음)")

    prompt = f"""
    당신은 10년차 전문 발표(프레젠테이션) 코칭 AI입니다.
    학생의 발표 영상에서 추출한 [대본]과, 해당 대본을 말하는 동안의 [시선/표정 평균], [음성 운율] 데이터를 제공합니다.

    [데이터]
    {aligned_data}

    [데이터 항목 설명]
    - text: 음성인식된 대본 (무슨 말을 했는지)
    - vision_avg: 해당 대본 구간의 평균 시선/표정 (어떤 표정이었는지)
        - gaze_h (좌우): 0에 가까울수록 정면. (+: 왼쪽, -: 오른쪽)
        - gaze_v (상하): 0에 가까울수록 정면. (+: 위쪽, -: 아래쪽)
        - smile: 미소 수치 (0.25 이상 유의미)
        - frown: 찡그림 수치 (0.25 이상 유의미)
        - status: "얼굴 미검출" (카메라 이탈)
    - prosody: 음성 운율 (어떻게 말했는지)
        - jitter (%): 목소리 높낮이 떨림. (1.0% 이하면 안정, 2.0% 이상이면 불안정)
        - shimmer (%): 목소리 거칠기/잠김. (3.0% 이하면 안정, 5.0% 이상이면 거침)
    - speech_rate_cps: 발표 속도 (초당 글자 수). (3.0 ~ 4.5가 적절)

    [채점 요청]
    위 데이터를 바탕으로, 학생의 발표 태도를 전문적으로 분석하고 채점해주세요.
    결과는 반드시 Markdown 형식을 사용하여 다음 4가지 항목으로 구분해서 작성해주세요.

    1.  **시선 처리 (25점)**:
        * gaze_h/v가 -0.1~0.1 사이인 '정면 응시' 비율을 평가합니다.
        * '얼굴 미검출'(카메라 이탈) 구간이 있다면 감점합니다.
        * 시선이 불안하게 흔들리는 구간을 지적합니다.

    2.  **표정 관리 (25점)**:
        * smile, frown 수치를 보고 긍정적/부정적 표정을 평가합니다.
        * [text] 내용과 [vision_avg] 표정이 일치하는지(예: 긍정적 단어에 smile), 불일치하는지(예: 웃으며 사과) 평가합니다.

    3.  **발표 태도 및 전달력 (50점)**:
        * ⭐️ (신규) prosody의 jitter/shimmer 값을 기준으로 목소리 안정성(긴장도)을 평가합니다. (가장 중요)
        * ⭐️ (신규) speech_rate_cps를 기준으로 발표 속도가 너무 빠르거나 느린 구간을 지적합니다.
        * '얼굴 미검출'이나 과도한 'frown' 등 부적절한 태도를 감점합니다.

    4.  **종합 점수 및 총평**:
        * 위 3개 항목의 합산 점수 (100점 만점)와,
        * 발표자가 어떤 점을 가장 먼저 개선해야 하는지에 대한 상세한 조언을 2~3문장으로 작성해주세요.
    """
    
    try:
        # Gemini API 호출 방식으로 변경
        generation_config = genai.types.GenerationConfig(temperature=0.5)
        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        content = response.text # 응답 텍스트 추출 방식 변경
        print("   > [6/6] ✅ Gemini 채점 완료.")
        return {"ai_feedback": content}
        
    except Exception as e:
        print(f"❌ Gemini API 오류: {e}")
        return {"error": f"Gemini API 호출 중 오류 발생: {e}"}