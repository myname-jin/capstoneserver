import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2  # OpenCV 사용
import numpy as np
import os
from pathlib import Path

# MediaPipe 핵심 컴포넌트
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

# 모델을 전역 변수로 관리하여 한번만 로드
face_landmarker_instance = None
MODEL_PATH = Path(__file__).resolve().parent.parent / "face_landmarker.task"

def setup_face_landmarker():
    """
    서버 시작 시 MediaPipe FaceLandmarker 모델을 로드합니다.
    """
    global face_landmarker_instance
    if face_landmarker_instance:
        return face_landmarker_instance

    print("   > [1/5] AI 모델(MediaPipe) 로드 중...")
    
    if not MODEL_PATH.exists():
        print(f"❌ 모델 파일({MODEL_PATH})을 찾을 수 없습니다.")
        print("-> https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        print(f"-> 위 주소에서 'face_landmarker.task' 파일을 다운로드하여 {MODEL_PATH.parent} 폴더에 저장하세요.")
        raise FileNotFoundError(f"모델 파일({MODEL_PATH})을 찾을 수 없습니다. 다운로드가 필요합니다.")

    try:
        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True
        )
        face_landmarker_instance = FaceLandmarker.create_from_options(options)
        print("   > [1/5] ✅ MediaPipe 모델 로드 완료.")
        return face_landmarker_instance
    except Exception as e:
        print(f"❌ MediaPipe 모델 로드 중 심각한 오류 발생: {e}")
        raise

# 
# ❗️ [수정] ❗️: 더 많은 데이터를 반환하도록 함수 수정
def _process_blendshapes(blendshapes: list) -> dict:
    """JS 코드의 processBlendshapes 함수를 Python으로 변환합니다."""
    if not blendshapes:
        return {}
        
    # 카테고리 이름과 점수를 딕셔너리로 변환 (Top-10을 위해)
    cats = {c.category_name: c.score for c in blendshapes[0]}
    
    def pick(n):
        return cats.get(n, 0) # 해당 키가 없으면 0 반환

    # --- 기존 계산 ---
    gaze_h = ((pick('eyeLookOutLeft') - pick('eyeLookInLeft')) + (pick('eyeLookInRight') - pick('eyeLookOutRight'))) / 2
    gaze_v = ((pick('eyeLookUpLeft') - pick('eyeLookDownLeft')) + (pick('eyeLookUpRight') - pick('eyeLookDownRight'))) / 2
    smile = (pick('mouthSmileLeft') + pick('mouthSmileRight')) / 2
    frown = (pick('mouthFrownLeft') + pick('mouthFrownRight')) / 2
    brow_down = (pick('browDownLeft') + pick('browDownRight')) / 2
    jaw_open = pick('jawOpen')
    
    # --- ❗️ [추가] ❗️: 사진 UI에 있던 추가 데이터 ---
    brow_up = (pick('browInnerUp') + pick('browOuterUpLeft') + pick('browOuterUpRight')) / 3
    mouth_open = pick('mouthOpen') # jawOpen과 다름
    squint = (pick('eyeSquintLeft') + pick('eyeSquintRight')) / 2
    
    return {
        "gaze_h": gaze_h,
        "gaze_v": gaze_v,
        "smile": smile,
        "frown": frown,
        "brow_down": brow_down,
        "jaw_open": jaw_open,
        "brow_up": brow_up,      # 추가
        "mouth_open": mouth_open,  # 추가
        "squint": squint,        # 추가
        "all_blendshapes": cats  # ❗️ Top-10 리스트를 위한 전체 데이터
    }

def analyze_image(image_path: str) -> dict:
    """
    단일 이미지 파일을 분석하여 표정/시선 데이터를 반환합니다.
    """
    landmarker = setup_face_landmarker()
    if not landmarker:
        return {"error": "MediaPipe 모델이 로드되지 않았습니다."}
    
    try:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return {"error": "이미지 파일을 읽을 수 없습니다."}
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = landmarker.detect(mp_image)
        
        if results.face_blendshapes:
            return _process_blendshapes(results.face_blendshapes) # ❗️ 수정된 함수 호출
        else:
            return {"error": "얼굴 미검출"}
            
    except Exception as e:
        print(f"   > 이미지 분석 오류 ({image_path}): {e}")
        return {"error": str(e)}