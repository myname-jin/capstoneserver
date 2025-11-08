from fastapi import FastAPI, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
import time as timer 
import uuid 
import os 
from fastapi.middleware.cors import CORSMiddleware 

# 우리가 만든 모듈 임포트
from utils.helpers import setup_temp_dirs, create_session_dirs, save_upload_file, cleanup_dirs
# ❗️ [수정] ❗️: audio_extractor 임포트
from processing.video_analyzer import extract_all_frames, extract_audio
from processing.face_analyzer import setup_face_landmarker, analyze_image
# ❗️ [수정] ❗️: audio_analyzer (로컬 Whisper) 임포트
from processing.audio_analyzer import transcribe_audio_with_timestamps, load_local_whisper_model
from processing.ai_scorer import is_openai_configured # (GPT 채점은 여전히 제외)

# --- 설정 ---
FRAME_RATE = 5
BASE_DIR = Path(__file__).resolve().parent

job_status = {}

# 
# ❗️ [새 함수] ❗️: 음성/시선 데이터를 합치는 핵심 로직
def align_data(vision_data: list, audio_segments: list) -> list:
    """
    문장(audio_segments)별로 해당 시간대의 평균 시선/표정(vision_data)을 계산합니다.
    """
    aligned_results = []
    
    valid_vision_data = [frame for frame in vision_data if "error" not in frame]
    if not valid_vision_data:
        return []

    for segment in audio_segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        
        frames_in_segment = [
            frame for frame in valid_vision_data 
            if frame['time'] >= start_time and frame['time'] <= end_time
        ]

        if not frames_in_segment:
            avg_vision = {"error": "얼굴 미검출"}
        else:
            avg_vision = {
                "smile": round(sum(f['smile'] for f in frames_in_segment) / len(frames_in_segment), 3),
                "frown": round(sum(f['frown'] for f in frames_in_segment) / len(frames_in_segment), 3),
                "brow_up": round(sum(f['brow_up'] for f in frames_in_segment) / len(frames_in_segment), 3),
                "brow_down": round(sum(f['brow_down'] for f in frames_in_segment) / len(frames_in_segment), 3),
                "jaw_open": round(sum(f['jaw_open'] for f in frames_in_segment) / len(frames_in_segment), 3),
                "mouth_open": round(sum(f['mouth_open'] for f in frames_in_segment) / len(frames_in_segment), 3),
                "squint": round(sum(f['squint'] for f in frames_in_segment) / len(frames_in_segment), 3),
                "gaze_h": round(sum(f['gaze_h'] for f in frames_in_segment) / len(frames_in_segment), 3),
                "gaze_v": round(sum(f['gaze_v'] for f in frames_in_segment) / len(frames_in_segment), 3),
            }

        aligned_results.append({
            "start": start_time,
            "end": end_time,
            "text": text,
            "vision_avg": avg_vision
        })
        
    return aligned_results


# 
# ❗️ [수정] ❗️: 백그라운드 작업 로직 전체 변경 (5단계로)
def run_analysis_task(job_id: str, video_path: Path, frame_dir: Path, video_dir: Path):
    all_vision_results = []
    audio_path = frame_dir / "audio.wav" 
    
    try:
        # 1. 오디오 추출
        job_status[job_id] = {"status": "Analyzing", "message": "1/5: 오디오 트랙 추출 중..."}
        extract_audio(video_path, audio_path)
        
        # 2. 프레임 추출
        job_status[job_id] = {"status": "Analyzing", "message": "2/5: 비디오 프레임 추출 중..."}
        frame_paths = extract_all_frames(video_path, frame_dir, FRAME_RATE)
        
        if not frame_paths:
            raise Exception("비디오에서 프레임을 추출할 수 없습니다.")
        
        total_frames = len(frame_paths)
        
        # 3. 각 프레임 분석 (MediaPipe)
        job_status[job_id] = {"status": "Analyzing", "message": f"3/5: 얼굴 데이터 분석 중... (0/{total_frames})"}
        print(f"   > [3/5] 모든 프레임 분석 시작 (Job: {job_id})...")
        for i, path in enumerate(frame_paths):
            data = analyze_image(str(path))
            data["time"] = i / FRAME_RATE
            all_vision_results.append(data)
            
            if i % 20 == 0 or i == total_frames - 1:
                job_status[job_id] = {
                    "status": "Analyzing", 
                    "message": f"3/5: 얼굴 데이터 분석 중...",
                    "progress": i + 1,
                    "total": total_frames
                }
                print(f"     ... {i+1}/{total_frames} 프레임 처리 중 (Job: {job_id})")
        
        print(f"   > [3/5] ✅ 프레임 분석 완료 (Job: {job_id}).")
        
        # 4. 음성 인식 (로컬 Whisper)
        job_status[job_id] = {"status": "Analyzing", "message": "4/5: ❗️로컬 음성 인식 실행 중... (시간 소요)❗️"}
        audio_segments, whisper_error = transcribe_audio_with_timestamps(str(audio_path))
        
        if whisper_error:
            print(f"   > [4/5] ❗️ 음성 인식 오류: {whisper_error}")
            audio_segments = []
            ai_report_message = f"## 🤖 로컬 음성인식 오류\n\n**오류:** {whisper_error}\n\n시선/표정 분석 데이터는 정상적으로 추출되었습니다."
        else:
            ai_report_message = "## 🤖 음성 인식 완료 (로컬)\n\nAI 채점 기능은 현재 비활성화되어 있습니다. \n\n음성 및 시선/표정 데이터 추출은 정상적으로 완료되었습니다."
            print(f"   > [4/5] ✅ 음성 인식 완료 (Job: {job_id}).")

        # 5. 데이터 정렬 (Alignment)
        job_status[job_id] = {"status": "Analyzing", "message": "5/5: 음성/시선 데이터 정렬 중..."}
        aligned_data = align_data(all_vision_results, audio_segments)
        print("   > [5/5] ✅ 데이터 정렬 완료.")

        # AI 채점은 여전히 건너뜀
        ai_result = {"ai_feedback": ai_report_message} 
        
        final_result = {
            "ai_assessment": ai_result,
            "analysis_summary": {
                "total_frames_processed": len(all_vision_results),
                "duration_analyzed_sec": len(all_vision_results) / FRAME_RATE,
                "face_detected_frames": len([f for f in all_vision_results if "error" not in f]),
            },
            "raw_data": all_vision_results, # ❗️ UI 호환성을 위해 'raw_data' 키 사용
            "aligned_transcript_data": aligned_data
        }
        
        job_status[job_id] = {"status": "Complete", "result": final_result}
        print(f"\n✅✅✅ [작업 완료] (Job: {job_id})")

    except Exception as e:
        print(f"\n❌❌❌ [작업 실패] (Job: {job_id})")
        print(f"오류 내용: {e}")
        job_status[job_id] = {"status": "Error", "message": str(e)}
    
    finally:
        cleanup_dirs(video_dir, frame_dir)


# 
# ❗️ [수정] ❗️: lifespan 함수를 수정하여 로컬 Whisper 모델을 미리 로드합니다.
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("="*50)
    print("서버가 시작되었습니다.")
    print("API 문서는 http://127.0.0.1:8000/docs 에서 확인하세요.")
    print(f"❗️UI 접속: http://127.0.0.1:8000 ❗️")
    print("="*50)
    setup_temp_dirs()
    try:
        # MediaPipe 얼굴 모델 로드
        setup_face_landmarker()
        # ❗️ [수정] ❗️: 로컬 Whisper 모델 로드
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

@app.get("/health", summary="서버 및 AI 모델 상태 확인")
def health_check():
    try:
        model = setup_face_landmarker()
        model_loaded = (model is not None)
        openai_ready = is_openai_configured()
        if not model_loaded:
            raise Exception("MediaPipe 모델이 로드되지 않았습니다.")
        return {
            "ok": True, 
            "message": "서버 및 AI 모델 정상",
            "modelLoaded": model_loaded,
            "openaiConfigured": openai_ready
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ok": False, "error": f"모델 로드 실패: {e}", "modelLoaded": False}
        )

@app.post("/upload", summary="비디오 분석 작업 시작")
def upload_and_analyze_video(background_tasks: BackgroundTasks, videoFile: UploadFile = File(...)):
    
    video_dir, frame_dir = create_session_dirs()
    safe_filename = videoFile.filename or "uploaded_video"
    video_path = Path(os.path.join(video_dir, safe_filename))
    
    try:
        print(f"\n[작업 접수] 파일: {videoFile.filename} (Type: {videoFile.content_type})")
        save_upload_file(videoFile, video_path)
        
        job_id = str(uuid.uuid4())
        job_status[job_id] = {"status": "Pending", "message": "0/5: 작업 대기 중..."} # 
# ❗️ [수정] ❗️: 5단계로 변경
        
        background_tasks.add_task(run_analysis_task, job_id, video_path, frame_dir, video_dir)
        
        print(f"   > Job ID 발급: {job_id}")
        
        return {"job_id": job_id}

    except Exception as e:
        print(f"❌❌❌ [업로드 실패] 오류: {e}")
        cleanup_dirs(video_dir, frame_dir)
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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
    # .\venv\Scripts\activate    python main.py  http://127.0.0.1:8000