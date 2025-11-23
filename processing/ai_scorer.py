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

# ⭐️ [기본 기준] 사용자 정의 기준이 없을 경우 사용
DEFAULT_CRITERIA = [
    {
        "name": "시선 처리",
        "score": 25,
        "description": "gaze_h/v가 -0.1~0.1 사이인 '정면 응시' 비율을 평가합니다. '얼굴 미검출'(카메라 이탈) 구간이 있다면 감점하며, 시선이 불안하게 흔들리는 구간을 지적합니다."
    },
    {
        "name": "표정 관리",
        "score": 25,
        "description": "smile, frown 수치를 보고 긍정적/부정적 표정을 평가하고, [text] 내용과 [vision_avg] 표정이 일치하는지(예: 긍정적 단어에 smile), 불일치하는지 평가합니다."
    },
    {
        "name": "발표 태도 및 전달력",
        "score": 50,
        "description": "prosody의 jitter/shimmer 값을 기준으로 목소리 안정성(긴장도)을 평가하고, speech_rate_cps를 기준으로 발표 속도가 너무 빠르거나 느린 구간을 지적합니다. 부적절한 태도를 감점합니다."
    }
]

def format_criteria_for_prompt(criteria: list) -> tuple:
    """
    채점 기준 목록을 받아 Markdown 형식의 프롬프트 문자열과 총점을 반환합니다.
    """
    # ⭐️ 사용자 정의 기준이 없거나 빈 리스트인 경우 기본 기준을 사용합니다.
    if not criteria or len(criteria) == 0:
        criteria = DEFAULT_CRITERIA
        
    prompt_lines = []
    total_score = 0
    
    for idx, item in enumerate(criteria, 1):
        name = item.get("name", f"항목 {idx}")
        score = item.get("score", 0)
        description = item.get("description", "제공된 데이터에 기반하여 항목을 평가합니다.")
        
        prompt_lines.append(f"{idx}. **{name} ({score}점)**:")
        prompt_lines.append(f"    * 기준: {description}")
        total_score += score
        
    return "\n".join(prompt_lines), total_score, len(criteria)

def get_ai_score(aligned_data: list, custom_criteria: list = None): 
    """
    정렬된 데이터를 OpenAI API로 보내 채점 및 피드백을 받습니다.
    """
    if not is_openai_configured():
        return {"error": "OpenAI API 키가 설정되지 않아 AI 채점을 수행할 수 없습니다."}

    if not aligned_data:
        return {"error": "분석 데이터가 없어 AI 채점을 할 수 없습니다."}

    print("   > [6/6] OpenAI API 호출 중... (최대 30초 소요될 수 있음)")

    # ⭐️ 채점 기준 동적 생성
    criteria_prompt, total_score, num_criteria = format_criteria_for_prompt(custom_criteria)

    # ⭐️ 최종 프롬프트 구성
    prompt = f"""
    당신은 10년차 전문 발표(프레젠테이션) 코칭 AI입니다.
    학생의 발표 영상에서 추출한 [대본]과, 해당 대본을 말하는 동안의 [시선/표정 평균], [음성 운율] 데이터를 제공합니다.

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

    [채점 요청]
    아래 {num_criteria}가지 항목을 기준으로 학생의 발표 태도를 전문적으로 분석하고 채점해주세요.
    결과는 반드시 Markdown 형식을 사용하여 다음 항목으로 구분해서 작성해주세요.

    {criteria_prompt}

    {num_criteria + 1}. **종합 점수 및 총평**:
        * 위 항목들의 합산 점수 (**{total_score}점 만점**)와,
        * 발표자가 어떤 점을 가장 먼저 개선해야 하는지에 대한 상세한 조언을 2~3문장으로 작성해주세요.

    {num_criteria + 2}. **영상요약**:
        * 영상에서 말하고자 하는 바를 판단하여 정리해 주세요.
        
    [데이터]
    {aligned_data}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        
        content = response.choices[0].message.content
        print("   > [6/6] ✅ OpenAI 채점 완료.")
        return {"ai_feedback": content}
        
    except Exception as e:
        print(f"❌ OpenAI API 오류: {e}")
        return {"error": f"OpenAI API 호출 중 오류 발생: {e}"}