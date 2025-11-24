import os
import json
import uvicorn
import uuid 
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, status, BackgroundTasks, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware 

# 유틸리티 및 모델 로더 임포트
from utils.helpers import setup_temp_dirs, create_session_dirs, save_upload_file, BASE_DIR 
from utils.json_helpers import setup_json_dirs, save_criteria_json 
from processing.face_analyzer import setup_face_landmarker
from processing.audio_analyzer import load_local_whisper_model
from processing.ai_scorer import is_openai_configured 
from processing.task_manager import run_analysis_task, job_status

# ⭐️ 지피티 챗봇 기능용 임포트

from processing.chat_manager import ask_gpt

BASE_DIR = Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.name == 'nt':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    print("="*50)
    print("서버가 시작되었습니다. (http://127.0.0.1:8000)")
    print("="*50)
    setup_temp_dirs()
    setup_json_dirs() # ⭐️ JSON 폴더 설정
    
    try:
        setup_face_landmarker()
        load_local_whisper_model()
        
        if not is_openai_configured(): 
            print("="*50)
            print("⚠️  경고: OPENAI_API_KEY가 없습니다. (AI 채점 기능은 비활성화됩니다)") 
            print("="*50)
    except Exception as e:
        print(f"❌ 치명적 오류: AI 모델 로드 실패! {e}")
    yield
    print("="*50)
    print("서버가 종료됩니다.")
    print("="*50)

app = FastAPI(lifespan=lifespan)

# CORS 미들웨어 (모든 접속 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def read_index():
    html_file_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="index.html 파일을 찾을 수 없습니다.")
    return FileResponse(html_file_path)

@app.get("/chat", include_in_schema=False)
async def read_chat():
    html_file_path = os.path.join(BASE_DIR, "chat.html")
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="chat.html 파일을 찾을 수 없습니다.")
    return FileResponse(html_file_path)

@app.post("/analyze")
def upload_and_analyze_video(
    background_tasks: BackgroundTasks, 
    # 1. 안드로이드 Retrofit의 'file' 파트와 이름 일치
    file: UploadFile = File(...),
    
    # 2. 안드로이드 Retrofit의 'criteria' 파트와 이름 일치
    criteria: str = Form(...),
    
    # 3. 안드로이드에서 보내지 않는 값들은 None으로 처리 (에러 방지)
    competitionName: str = Form(None), 
    teamName: str = Form(None)
):
    # 1. 임시 폴더 생성
    video_dir, frame_dir = create_session_dirs()

    # 2. 파일 경로 설정
    safe_filename = file.filename or "uploaded_video.mp4"
    video_path = Path(os.path.join(video_dir, safe_filename))
    
    try:
        # 파일을 실제로 저장
        save_upload_file(file, video_path)

        custom_criteria = json.loads(criteria if criteria else "[]")
        if custom_criteria and competitionName: 
            save_criteria_json(custom_criteria, competitionName)

        print(f"\n[작업 접수] 파일: {file.filename}")
        print(f"   > 저장 경로: {video_path}") # 경로 확인용 로그
        
        job_id = str(uuid.uuid4())
        job_status[job_id] = {"status": "Pending", "message": "0/6: 작업 대기 중..."} 
        
        background_tasks.add_task(run_analysis_task, job_id, video_path, frame_dir, video_dir, custom_criteria)
        
        print(f"   > Job ID 발급: {job_id}")
        return {"job_id": job_id}

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="잘못된 JSON 형식의 채점 기준이 전달되었습니다."
        )
    except Exception as e:
        print(f"❌❌❌ [업로드 실패] 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"파일 업로드 중 오류 발생: {str(e)}"
        )

@app.get("/status/{job_id}", summary="작업 진행 상태 확인")
def get_status(job_id: str):
    status = job_status.get(job_id)
    
    if not status:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="작업 ID를 찾을 수 없습니다.")
    
    if status["status"] == "Complete" or status["status"] == "Error":
        return job_status.pop(job_id)
        
    return status

# ⭐️ 지피티 챗봇 기능 API
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("message", "")
    gpt_response = ask_gpt(prompt)
    return {"response": gpt_response}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True
    )
   
    # .\venv\Scripts\activate    python main.py  http://127.0.0.1:8000
    # pip install -r requirements.txt (라이브러리 설치)
    # http://127.0.0.1:8000/chat

    # --------------핸드폰으로 실행 방법-----------------
    # .\venv\Scripts\activate
    # uvicorn main:app --host 0.0.0.0 --port 8000 (서버 키기)