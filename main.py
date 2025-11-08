from fastapi import FastAPI, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
import uuid 
import os 
from fastapi.middleware.cors import CORSMiddleware 

# 유틸리티 및 모델 로더 임포트
from utils.helpers import setup_temp_dirs, create_session_dirs, save_upload_file
from processing.face_analyzer import setup_face_landmarker
from processing.audio_analyzer import load_local_whisper_model
from processing.ai_scorer import is_gemini_configured # ⭐️ [수정] Gemini로 변경

# [신규] 분리된 태스크 매니저 임포트
from processing.task_manager import run_analysis_task, job_status

# --- 설정 ---
BASE_DIR = Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("="*50)
    print("서버가 시작되었습니다. (http://127.0.0.1:8000)")
    print("="*50)
    setup_temp_dirs()
    try:
        # AI 1: MediaPipe 얼굴 모델 로드
        setup_face_landmarker()
        # AI 2: 로컬 Whisper 모델 로드
        load_local_whisper_model()
        
        # AI 3: Gemini (키 확인)
        if not is_gemini_configured(): # ⭐️ [수정] Gemini로 변경
            print("="*50)
            print("⚠️  경고: GEMINI_API_KEY가 없습니다. (AI 채점 기능은 비활성화됩니다)") # ⭐️ [수정]
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

@app.post("/upload", summary="비디오 분석 작업 시작")
def upload_and_analyze_video(background_tasks: BackgroundTasks, videoFile: UploadFile = File(...)):
    
    video_dir, frame_dir = create_session_dirs()
    safe_filename = videoFile.filename or "uploaded_video"
    video_path = Path(os.path.join(video_dir, safe_filename))
    
    try:
        print(f"\n[작업 접수] 파일: {videoFile.filename}")
        save_upload_file(videoFile, video_path)
        
        job_id = str(uuid.uuid4())
        job_status[job_id] = {"status": "Pending", "message": "0/6: 작업 대기 중..."} 
        
        background_tasks.add_task(run_analysis_task, job_id, video_path, frame_dir, video_dir)
        
        print(f"   > Job ID 발급: {job_id}")
        return {"job_id": job_id}

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
        return job_status.pop(job_id) # 완료/에러 시 메모리에서 제거
    
    return status

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0", # ⭐️ 외부 접속 허용 (팀원 테스트용)
        port=8000,
        reload=True
    )
    # .\venv\Scripts\activate    python main.py  http://127.0.0.1:8000
    # pip install -r requirements.txt (라이브러리 설치)