from openai import OpenAI
import os
from dotenv import load_dotenv
import json

# .env 파일에서 환경 변수(API 키) 로드
load_dotenv()

client = None
api_key = os.getenv("OPENAI_API_KEY") 

if api_key and api_key.startswith("sk-"):
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"OpenAI 클라이언트 초기화 실패: {e}")
        client = None

def is_openai_configured():
    """OpenAI API 키가 올바르게 설정되었는지 확인합니다."""
    return client is not None

# ⭐️ [기본 기준]
DEFAULT_CRITERIA = [
    {
        "name": "시선 처리",
        "score": 25,
        "description": "gaze_h/v가 -0.1~0.1 사이인 '정면 응시' 비율을 평가합니다."
    },
    {
        "name": "표정 관리",
        "score": 25,
        "description": "smile, frown 수치를 보고 긍정적/부정적 표정을 평가합니다."
    },
    {
        "name": "발표 태도 및 전달력",
        "score": 50,
        "description": "prosody의 jitter/shimmer 값을 기준으로 목소리 안정성을 평가합니다."
    }
]

def format_criteria_for_prompt(criteria: list) -> tuple:
    """
    채점 기준 목록을 프롬프트용 문자열로 변환합니다.
    """
    if not criteria or len(criteria) == 0:
        criteria = DEFAULT_CRITERIA
        
    prompt_lines = []
    total_score = 0
    
    for idx, item in enumerate(criteria, 1):
        name = item.get("name", f"항목 {idx}")
        score = item.get("score", 0)
        description = item.get("description", "항목 평가")
        
        prompt_lines.append(f"- **{name}** (배점: {score}점): {description}")
        total_score += score
        
    return "\n".join(prompt_lines), total_score

def get_ai_score(aligned_data: list, custom_criteria: list = None): 
    """
    정렬된 데이터를 OpenAI API로 보내 JSON 형태의 채점 결과를 받습니다.
    """
    if not is_openai_configured():
        return {"error": "OpenAI API 키가 설정되지 않았습니다."}

    if not aligned_data:
        return {"error": "분석 데이터가 없습니다."}

    print("   > [6/6] OpenAI API 호출 중... (JSON Mode)")

    criteria_prompt, total_score = format_criteria_for_prompt(custom_criteria)

    # ⭐️ 최종 프롬프트 구성
    prompt = f"""
    당신은 10년차 전문 발표(프레젠테이션) 코칭 AI입니다.
    학생의 발표 영상에서 추출한 [대본]과, 해당 대본을 말하는 동안의 [시선/표정 평균], [음성 운율] 데이터를 제공합니다.
    제공된 [데이터]를 분석하여 [채점 기준]에 맞춰 평가하고, 반드시 아래의 **JSON 형식**으로만 응답하세요.
    다른 말은 붙이지 마세요.

     [채점 기준]
    {criteria_prompt}
    (총점 만점: {total_score}점)
    
    [데이터 항목 설명]
    - text: 음성인식된 대본
    - vision_avg: 해당 대본 구간의 평균 시선/표정
        - gaze_h (좌우): 0에 가까울수록 정면. (+: 왼쪽, -: 오른쪽)
        - gaze_v (상하): 0에 가까울수록 정면. (+: 위쪽, -: 아래쪽)
        - smile: 미소 수치 (0.25 이상 유의미)
        - frown: 찡그림 수치 (0.25 이상 유의미)
        - status: "얼굴 미검출" (카메라 이탈)
    - prosody: 음성 운율
        - jitter (%): 목소리 높낮이 떨림. (1.0% 이하면 안정, 2.0% 이상이면 불안정)
        - shimmer (%): 목소리 거칠기/잠김. (3.0% 이하면 안정, 5.0% 이상이면 거침)
    - speech_rate_cps: 발표 속도 (초당 글자 수). (3.0 ~ 4.5가 적절)

    [필수 응답 포맷 (JSON)]
    {{
        "reviews": [
            {{
                "name": "평가 항목 이름 (채점 기준에 있는 이름 그대로)",
                "score": 0, 
                "feedback": "해당 항목에 대한 구체적인 피드백 (2~3문장)"
            }}
        ],
        "overall_summary": "전체적인 총평 및 개선할 점 (3문장 내외)",
        "video_summary": "영상의 핵심 내용 요약 (1~2문장)"
    }}
        
    [데이터]
    {str(aligned_data)[:15000]} 
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            response_format={"type": "json_object"} # ⭐️ 핵심: JSON 모드 활성화
        )
        
        content = response.choices[0].message.content
        result_json = json.loads(content) # 문자열을 파이썬 딕셔너리로 변환
        
        print("   > [6/6] ✅ OpenAI 채점 완료 (JSON).")
        return result_json
        
    except json.JSONDecodeError:
        print("❌ OpenAI 응답이 올바른 JSON이 아닙니다.")
        return {"error": "AI 응답 파싱 실패"}
    except Exception as e:
        print(f"❌ OpenAI API 오류: {e}")
        return {"error": f"API 호출 오류: {e}"}