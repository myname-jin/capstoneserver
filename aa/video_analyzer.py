import subprocess
import os
from pathlib import Path

# 
# ❗️ [추가] ❗️: FFmpeg로 오디오 트랙을 16khz mono wav 파일로 추출
def extract_audio(video_path: Path, output_audio_path: Path) -> Path:
    """
    FFmpeg를 사용하여 비디오에서 오디오 트랙을 추출합니다.
    Whisper AI가 가장 선호하는 16kHz, 16-bit, mono .wav 파일로 변환합니다.
    """
    print(f"   > [2/5] 오디오 트랙 추출 중...")
    
    try:
        # ffmpeg -i [입력] -vn (비디오X) -acodec pcm_s16le (16비트) -ar 16000 (16kHz) -ac 1 (모노) [출력]
        subprocess.run([
            'ffmpeg',
            '-i', str(video_path),
            '-vn',                         # 비디오 트랙 무시
            '-acodec', 'pcm_s16le',      # 오디오 코덱 (표준 WAV)
            '-ar', '16000',                # 샘플링 레이트 (Whisper 권장)
            '-ac', '1',                    # 오디오 채널 (모노)
            str(output_audio_path)
        ], check=True, capture_output=True, text=True)
        
        print(f"   > [2/5] ✅ 오디오 추출 완료: {output_audio_path.name}")
        return output_audio_path
        
    except subprocess.CalledProcessError as e:
        print("❌ FFmpeg 오디오 추출 오류!", e.stderr)
        raise Exception("FFmpeg 오디오 추출 실패")
    except FileNotFoundError:
        print("❌ 'ffmpeg' 명령을 찾을 수 없습니다.")
        raise Exception("FFmpeg가 설치되지 않았습니다.")


def extract_all_frames(video_path: Path, output_dir: Path, fps: int) -> list[Path]:
    """
    FFmpeg를 사용하여 비디오에서 프레임을 추출합니다.
    """
    print(f"   > [3/5] 비디오 프레임 추출 중... (초당 {fps} 프레임)")
    
    output_pattern = output_dir / "frame-%04d.jpg"
    
    try:
        subprocess.run([
            'ffmpeg',
            '-i', str(video_path),
            '-vf', f'fps={fps}',
            str(output_pattern)
        ], check=True, capture_output=True, text=True) 
        
    except subprocess.CalledProcessError as e:
        print("❌ FFmpeg 프레임 추출 오류!", e.stderr)
        raise Exception("FFmpeg 프레임 추출 실패")
    except FileNotFoundError:
        print("❌ 'ffmpeg' 명령을 찾을 수 없습니다.")
        raise Exception("FFmpeg가 설치되지 않았습니다.")

    frames = sorted([f for f in output_dir.glob('*.jpg')])
    print(f"   > [3/5] ✅ {len(frames)}개 프레임 추출 완료.")
    return frames