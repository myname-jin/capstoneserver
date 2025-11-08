// --- [ 1. 라이브러리 임포트 (❗️핵심 수정❗️) ] ---
import { createRequire } from "module";
const require = createRequire(import.meta.url);

// ❗️ [수정] 모든 라이브러리를 Node.js 전용 'require' 방식으로 불러옵니다.
const { FaceLandmarker, FilesetResolver } = require("@mediapipe/tasks-vision/node");
const { extractFrames } = require("ffmpeg-extract-frames");
const sharp = require("sharp");
const { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync, unlinkSync } = require("fs");
const { join } = require("path");

// --- 설정 ---
const INPUT_VIDEO = './test.mp4';
const FRAME_DIR = './frames';
const OUTPUT_JSON = './analysis_results.json';
const FRAME_RATE = 5; 

let faceLandmarker = null;

async function setupFaceLandmarker() {
  console.log("   > [1/5] AI 모델 로드 중...");
  // ❗️ [수정] FilesetResolver도 Node.js 방식으로 생성
  const fileset = await FilesetResolver.forVisionTasks(
    "./node_modules/@mediapipe/tasks-vision/wasm" // 로컬에 설치된 wasm 파일 사용
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "CPU"
    },
    runningMode: "IMAGE", 
    numFaces: 1,
    outputFaceBlendshapes: true
  });
  console.log("   > [1/5] ✅ 모델 로드 완료.");
  return faceLandmarker;
}

async function extractAllFrames(inputFile, outputDir, fps) {
  console.log(`   > [2/5] 비디오 프레임 추출 중... (초당 ${fps} 프레임)`);
  if (existsSync(outputDir)) {
    readdirSync(outputDir).forEach(f => unlinkSync(join(outputDir, f)));
  } else {
    mkdirSync(outputDir, { recursive: true });
  }
  try {
    await extractFrames({
      input: inputFile,
      output: `${outputDir}/frame-%04d.jpg`,
      fps: fps
    });
  } catch (e) {
    console.error("❌ FFmpeg 오류!", e.message);
    console.error("-> 'test.mp4' 파일 이름이 정확한지, C:\OenMinPython 폴더에 파일이 있는지 확인하세요.");
    throw new Error("FFmpeg 프레임 추출 실패");
  }
  const frames = readdirSync(outputDir).filter(f => f.endsWith('.jpg'));
  console.log(`   > [2/5] ✅ ${frames.length}개 프레임 추출 완료.`);
  return frames.map(f => join(outputDir, f));
}

async function analyzeImage(imagePath) {
  try {
    const fileBuffer = readFileSync(imagePath);
    const image = sharp(fileBuffer);
    const metadata = await image.metadata();
    const pixelData = await image.removeAlpha().raw().toBuffer();
    const mpImage = { data: new Uint8Array(pixelData), width: metadata.width, height: metadata.height };
    const results = faceLandmarker.detect(mpImage); 
    if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
      return processBlendshapes(results.faceBlendshapes[0].categories);
    } else {
      return { error: '얼굴 미검출' }; 
    }
  } catch (e) {
    return { error: e.message };
  }
}

function processBlendshapes(blendshapes) {
  const pick = (n) => blendshapes.find(s => s.categoryName === n)?.score ?? 0;
  const gaze_h = ( (pick('eyeLookOutLeft') - pick('eyeLookInLeft')) + (pick('eyeLookInRight') - pick('eyeLookOutRight')) ) / 2;
  const gaze_v = ( (pick('eyeLookUpLeft') - pick('eyeLookDownLeft')) + (pick('eyeLookUpRight') - pick('eyeLookDownRight')) ) / 2;
  const smile = (pick('mouthSmileLeft') + pick('mouthSmileRight')) / 2;
  const frown = (pick('mouthFrownLeft') + pick('mouthFrownRight')) / 2;
  const brow_down = (pick('browDownLeft') + pick('browDownRight')) / 2;
  const jaw_open = pick('jawOpen');
  return { gaze_h, gaze_v, smile, frown, brow_down, jaw_open };
}

async function main() {
  try {
    await setupFaceLandmarker(); // 모델 먼저 로드
    const framePaths = await extractAllFrames(INPUT_VIDEO, FRAME_DIR, FRAME_RATE); 
    
    console.log("   > [3/5] 모든 프레임 분석 시작...");
    const allResults = [];
    
    for (let i = 0; i < framePaths.length; i++) {
      const path = framePaths[i];
      const time = i / FRAME_RATE;
      const data = await analyzeImage(path);
      data.time = time; 
      allResults.push(data);
      if (i % 20 === 0 || i === framePaths.length - 1) { 
        console.log(`     ... ${i+1}/${framePaths.length} 프레임 처리 중`);
      }
    }

    console.log("   > [4/5] 분석 완료. 결과 JSON 파일 저장 중...");
    writeFileSync(OUTPUT_JSON, JSON.stringify(allResults, null, 2));
    
    console.log(`\n✅✅✅ 완료! '${OUTPUT_JSON}' 파일을 열어 결과 데이터를 확인하세요.`);
    
  } catch (e) {
    console.error("❌ 치명적인 오류 발생:", e.message);
  }
}

main();