from pathlib import Path
import time as timer 

# 모든 처리 모듈을 여기서 임포트
from processing.video_analyzer import extract_all_frames, extract_audio
from processing.face_analyzer import analyze_image
from processing.audio_analyzer import transcribe_audio_with_timestamps, analyze_prosody_for_segments
from processing.ai_scorer import get_ai_score, is_gemini_configured # ⭐️ [수정] Gemini로 변경
from processing.data_combiner import align_data
from utils.helpers import cleanup_dirs

FRAME_RATE = 5
job_status = {} # 작업 상태를 main.py 대신 여기서 관리

def run_analysis_task(job_id: str, video_path: Path, frame_dir: Path, video_dir: Path):
    """
    [❗️ main.py에서 이동됨 ❗️]
    전체 분석 파이프라인을 실행하는 백그라운드 작업입니다.
    (총 6단계로 구성)
    """
    all_vision_results = []
    audio_path = frame_dir / "audio.wav" 
    
    try:
        # 1. 오디오 추출
        job_status[job_id] = {"status": "Analyzing", "message": "1/6: 오디오 트랙 추출 중..."}
        extract_audio(video_path, audio_path)
        
        # 2. 프레임 추출
        job_status[job_id] = {"status": "Analyzing", "message": "2/6: 비디오 프레임 추출 중..."}
        frame_paths = extract_all_frames(video_path, frame_dir, FRAME_RATE)
        
        if not frame_paths:
            raise Exception("비디오에서 프레임을 추출할 수 없습니다.")
        
        total_frames = len(frame_paths)
        
        # 3. 각 프레임 분석 (MediaPipe)
        job_status[job_id] = {"status": "Analyzing", "message": f"3/6: 얼굴 데이터 분석 중... (0/{total_frames})"}
        print(f"   > [3/6] 모든 프레임 분석 시작 (Job: {job_id})...")
        for i, path in enumerate(frame_paths):
            data = analyze_image(str(path))
            data["time"] = i / FRAME_RATE
            all_vision_results.append(data)
            
            if i % 20 == 0 or i == total_frames - 1:
                job_status[job_id] = {
                    "status": "Analyzing", 
                    "message": f"3/6: 얼굴 데이터 분석 중...",
                    "progress": i + 1,
                    "total": total_frames
                }
        print(f"   > [3/6] ✅ 프레임 분석 완료 (Job: {job_id}).")
        
        # 4. 음성 인식 (로컬 Whisper)
        job_status[job_id] = {"status": "Analyzing", "message": "4/6: ❗️로컬 음성 인식 실행 중... (시간 소요)❗️"}
        audio_segments, whisper_error = transcribe_audio_with_timestamps(str(audio_path))
        
        ai_report_message = "" # AI 채점 실패 시 사용할 기본 메시지
        if whisper_error:
            print(f"   > [4/6] ❗️ 음성 인식 오류: {whisper_error}")
            audio_segments = []
            ai_report_message = f"## 🤖 로컬 음성인식 오류\n\n**오류:** {whisper_error}\n\n시선/표정 분석 데이터는 정상적으로 추출되었습니다."
        else:
            print(f"   > [4/6] ✅ 음성 인식 완료 (Job: {job_id}).")

        # 5. 음성 운율 분석 (Praat)
        job_status[job_id] = {"status": "Analyzing", "message": "5/6: ❗️음성 운율(목소리 떨림) 분석 중...❗️"}
        audio_segments = analyze_prosody_for_segments(audio_path, audio_segments)

        # 6. 데이터 정렬 및 AI 채점
        job_status[job_id] = {"status": "Analyzing", "message": "6/6: 데이터 정렬 및 AI 채점 중..."}
        
        # 6-1. 정렬
        aligned_data = align_data(all_vision_results, audio_segments)
        
        # 6-2. AI 채점
        if is_gemini_configured(): # ⭐️ [수정] Gemini로 변경
            ai_result = get_ai_score(aligned_data)
        else:
            # Whisper는 성공했으나 Gemini 키가 없는 경우
            if not whisper_error:
                ai_report_message = "## 🤖 음성/표정/운율 분석 완료\n\nGemini API 키가 설정되지 않아 **AI 자동 채점 기능은 비활성화**되었습니다. \n\n대본, 시선/표정, 목소리 떨림 데이터 추출은 정상적으로 완료되었습니다." # ⭐️ [수정]
            
            ai_result = {"ai_feedback": ai_report_message} # whisper_error가 있을 경우 해당 메시지 사용
        
        print("   > [6/6] ✅ 데이터 정렬 및 AI 채점 완료.")

        final_result = {
            "ai_assessment": ai_result,
            "analysis_summary": {
                "total_frames_processed": len(all_vision_results),
                "duration_analyzed_sec": len(all_vision_results) / FRAME_RATE,
                "face_detected_frames": len([f for f in all_vision_results if "error" not in f]),
            },
            "raw_data": all_vision_results,
            "aligned_transcript_data": aligned_data
        }
        
        job_status[job_id] = {"status": "Complete", "result": final_result}
        print(f"\n✅✅✅ [작업 완료] (Job: {job_id})")

    except Exception as e:
        print(f"\n❌❌❌ [작업 실패] (Job: {job_id})")
        print(f"오류 내용: {e}")
        job_status[job_id] = {"status": "Error", "message": str(e)}
    
    finally:
        # 분석이 성공하든 실패하든 임시 파일 정리
        cleanup_dirs(video_dir, frame_dir)