# [신규 파일] processing/json_helpers.py
import json
import os
import re
from pathlib import Path

# ⭐️ JSON 기준 저장 디렉토리
STANDARD_DIR = Path(__file__).resolve().parent.parent / "standard"

def setup_json_dirs():
    """서버 시작 시 JSON 저장 폴더가 없으면 생성합니다."""
    os.makedirs(STANDARD_DIR, exist_ok=True)
    
def save_criteria_json(criteria: list, competition_name: str):
    """
    사용자 정의 채점 기준을 competition_name을 파일명으로 JSON 파일에 저장합니다.
    """
    # 파일명으로 사용할 수 없는 문자 제거 및 공백을 언더바로 변환
    safe_name = re.sub(r'[\\/*?:"<>|]', '', competition_name).replace(" ", "_")
    if not safe_name:
        safe_name = "default_criteria"
        
    file_name = f"{safe_name}.json"
    file_path = STANDARD_DIR / file_name
    
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(criteria, f, ensure_ascii=False, indent=4)
        print(f"   > [JSON Save] ✅ 채점 기준이 '{file_name}'으로 저장되었습니다.")
    except Exception as e:
        print(f"   > [JSON Save] ❌ 채점 기준 JSON 저장 실패: {e}")
        
def load_criteria_json(competition_name: str) -> list:
    """
    competition_name에 해당하는 JSON 파일을 로드하여 기준 목록을 반환합니다.
    """
    safe_name = re.sub(r'[\\/*?:"<>|]', '', competition_name).replace(" ", "_")
    file_name = f"{safe_name}.json"
    file_path = STANDARD_DIR / file_name
    
    if file_path.exists():
        try:
            with file_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"   > [JSON Load] ⚠️ JSON 파일 로드 오류 ({file_name}): {e}")
            return []
    return []