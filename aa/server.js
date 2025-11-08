// server.js
import express from "express";
import multer from "multer";
import sharp from "sharp";
import {
  readFileSync, existsSync, mkdirSync, readdirSync, unlinkSync
} from "fs";
import { join } from "path";

// MediaPipe (ESM OK)
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

// ⚠️ ffmpeg-extract-frames는 CJS default export → 함수 그대로 import
import extractFrames from "ffmpeg-extract-frames";

const app = express();
const port = 3000;

// 정적파일 서빙 (index.html 미리보기/테스트용)
app.use(express.static("."));

// 업로드 설정
const uploadDir = "./uploads";
if (!existsSync(uploadDir)) mkdirSync(uploadDir, { recursive: true });
const upload = multer({ dest: uploadDir });

let faceLandmarker = null;

// 모델 로드
async function setupFaceLandmarker() {
  console.log("1) MediaPipe 모델 로드 중…");
  const fileset = await FilesetResolver.forVisionTasks(
    "./node_modules/@mediapipe/tasks-vision/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "CPU",
    },
    runningMode: "IMAGE",
    numFaces: 1,
    outputFaceBlendshapes: true,
  });
  console.log("✅ 모델 로드 완료");
}

// 한 장 분석
async function analyzeImage(imagePath) {
  const fileBuffer = readFileSync(imagePath);
  const image = sharp(fileBuffer);
  const metadata = await image.metadata();
  const pixelData = await image.removeAlpha().raw().toBuffer();
  const mpImage = {
    data: new Uint8Array(pixelData),
    width: metadata.width,
    height: metadata.height,
  };
  const results = faceLandmarker.detect(mpImage);
  if (results.faceBlendshapes?.length > 0) {
    return processBlendshapes(results.faceBlendshapes[0].categories);
  }
  return { error: "얼굴 미검출" };
}

// 블렌드셰이프 → 지표
function processBlendshapes(blendshapes) {
  const pick = (n) => blendshapes.find((s) => s.categoryName === n)?.score ?? 0;
  const gaze_h =
    (pick("eyeLookOutLeft") - pick("eyeLookInLeft") +
      (pick("eyeLookInRight") - pick("eyeLookOutRight"))) /
    2;
  const gaze_v =
    (pick("eyeLookUpLeft") - pick("eyeLookDownLeft") +
      (pick("eyeLookUpRight") - pick("eyeLookDownRight"))) /
    2;
  const smile =
    (pick("mouthSmileLeft") + pick("mouthSmileRight")) / 2;
  const frown =
    (pick("mouthFrownLeft") + pick("mouthFrownRight")) / 2;
  const brow_down =
    (pick("browDownLeft") + pick("browDownRight")) / 2;
  const jaw_open = pick("jawOpen");
  return { gaze_h, gaze_v, smile, frown, brow_down, jaw_open };
}

// 비디오 전체 분석
async function analyzeVideoFile(videoPath) {
  const FRAME_DIR = "./frames";
  const FRAME_RATE = 5;
  const allResults = [];

  // 프레임 디렉토리 초기화
  if (!existsSync(FRAME_DIR)) mkdirSync(FRAME_DIR, { recursive: true });
  else readdirSync(FRAME_DIR).forEach((f) => unlinkSync(join(FRAME_DIR, f)));

  // ⚠️ ffmpeg-extract-frames: default 함수 (위에서 올바로 import 함)
  await extractFrames({
    input: videoPath,
    output: `${FRAME_DIR}/frame-%04d.jpg`,
    fps: FRAME_RATE,
  });
  const framePaths = readdirSync(FRAME_DIR)
    .filter((f) => f.endsWith(".jpg"))
    .map((f) => join(FRAME_DIR, f));

  for (let i = 0; i < framePaths.length; i++) {
    const path = framePaths[i];
    const time = i / FRAME_RATE;
    const data = await analyzeImage(path);
    data.time = time;
    allResults.push(data);
  }
  return allResults;
}

// 헬스체크/준비상태 확인
app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    modelLoaded: !!faceLandmarker,
  });
});

// 업로드 엔드포인트
app.post("/upload", upload.single("videoFile"), async (req, res) => {
  if (!req.file) return res.status(400).send("파일이 없습니다.");
  if (!faceLandmarker)
    return res.status(503).send("서버 AI 모델이 아직 로딩 중입니다.");

  const videoPath = req.file.path;
  try {
    const resultsJson = await analyzeVideoFile(videoPath);
    res.json(resultsJson);
  } catch (e) {
    res.status(500).send("서버 내부 처리 오류: " + e.message);
  } finally {
    try { unlinkSync(videoPath); } catch {}
  }
});

// 서버 시작
app.listen(port, async () => {
  try {
    await setupFaceLandmarker();
    console.log(`🚀 http://localhost:${port} 에서 실행 중`);
  } catch (e) {
    console.error("모델 로드 실패:", e);
    console.error("서버는 켜졌지만 /upload 요청은 막힐 수 있습니다.");
  }
});
