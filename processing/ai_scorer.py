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

def get_ai_score(aligned_data: list, custom_criteria: list = None): 
    """
    정렬된 데이터를 OpenAI API로 보내 JSON 형식의 채점 결과를 받습니다.
    """
    if not is_openai_configured():
        return {"error": "OpenAI API 키가 설정되지 않아 AI 채점을 수행할 수 없습니다."}

    if not aligned_data:
        return {"error": "분석 데이터가 없어 AI 채점을 할 수 없습니다."}

    print("   > [6/6] OpenAI API 호출 중... (JSON 모드)")

    # 1. 기준이 없으면 기본값 설정
    if not custom_criteria:
        custom_criteria = [
            {"name": "전달력", "score": 100, "description": "발표 태도 및 명확성 평가"}
        ]

    # 2. 기준 목록을 텍스트로 변환
    criteria_text = ""
    for item in custom_criteria:
        name = item.get('name', '평가 항목')
        score = item.get('score', 0)
        desc = item.get('description', '')
        criteria_text += f"- {name} (만점: {score}점): {desc}\n"

    # 3. JSON 강제 출력을 위한 프롬프트 구성
    prompt = f"""
    당신은 10년차 전문 발표 코칭 AI입니다. 
    아래 [채점 기준]에 맞춰 [분석 데이터]를 평가하고, 반드시 아래의 **JSON 형식**으로만 응답하세요.
    Markdown이나 다른 설명은 절대 포함하지 마세요.

    [채점 기준]
    {criteria_text}

    [분석 데이터 요약]
    {str(aligned_data)[:4000]}...

    [필수 응답 JSON 포맷]
    {{
        "reviews": [
            {{
                "name": "기준이름(위 채점 기준과 띄어쓰기까지 정확히 일치해야 함)",
                "score": 획득점수(정수),
                "feedback": "해당 기준에 대한 구체적인 피드백 (한글로 작성)"
            }}
        ],
        "overall_summary": "전체적인 총평 (3문장 내외, 구체적인 조언 포함)",
        "video_summary": "발표 내용 요약 (2문장 내외)"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 또는 gpt-3.5-turbo-1106 (JSON 모드를 지원하는 모델 권장)
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            response_format={"type": "json_object"} # ⭐️ 핵심: JSON 응답 강제
        )
        
        content = response.choices[0].message.content
        print("   > [6/6] ✅ OpenAI 채점 완료 (JSON).")
        
        # JSON 문자열을 파이썬 딕셔너리로 변환하여 반환
        return json.loads(content)
        
    except json.JSONDecodeError:
        print("❌ AI 응답이 올바른 JSON 형식이 아닙니다.")
        return {"error": "AI 응답 파싱 실패"}
    except Exception as e:
        print(f"❌ OpenAI API 오류: {e}")
        return {"error": f"OpenAI API 호출 중 오류 발생: {e}"}