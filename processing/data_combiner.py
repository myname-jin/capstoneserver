# [신규 파일] processing/data_combiner.py
import numpy as np

def align_data(vision_data: list, audio_segments: list) -> list:
    """
    문장(audio_segments)별로 해당 시간대의 평균 시선/표정(vision_data) 및
    운율(prosody) 데이터를 계산하고 정렬합니다.
    """
    print(f"   > [6/6] 데이터 정렬 시작...")
    aligned_results = []
    
    # 얼굴이 검출된 유효한 프레임만 필터링
    valid_vision_data = [frame for frame in vision_data if "error" not in frame]
    if not valid_vision_data:
        # 얼굴 데이터가 아예 없어도 텍스트와 운율 데이터는 반환
        pass

    for segment in audio_segments:
        start_time = segment['start']
        end_time = segment['end']
        duration = end_time - start_time
        
        # 1. ⭐️ [추가] 발표 속도 (Speech Rate) 계산 (초당 글자 수)
        # (duration이 0인 오류 방지)
        speech_rate_cps = len(segment['text']) / duration if duration > 0 else 0

        # 2. ⭐️ [추가] 운율(Prosody) 데이터 추출
        prosody = {
            "jitter": round(segment.get('jitter', 0), 3),
            "shimmer": round(segment.get('shimmer', 0), 3)
        }
        
        # 'nan' 값이 들어오는 경우 0으로 처리
        if np.isnan(prosody['jitter']): prosody['jitter'] = 0
        if np.isnan(prosody['shimmer']): prosody['shimmer'] = 0
        
        # 3. (기존) 시선/표정 데이터 평균 계산
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
            "text": segment['text'],
            "speech_rate_cps": round(speech_rate_cps, 2), # ⭐️ [추가]
            "vision_avg": avg_vision,
            "prosody": prosody # ⭐️ [추가]
        })
        
    print(f"   > [6/6] ✅ 데이터 정렬 완료.")
    return aligned_results