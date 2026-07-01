import {
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs";

export function createPoseRitualController({
  onGesture,
  onReady,
  onError,
} = {}) {
  let poseLandmarker = null;
  let video = null;
  let webcamRunning = false;
  let lastVideoTime = -1;

  let previousShoulderX = null;
  let swayHistory = [];

  const state = {
    poseReady: false,
    currentGesture: "none",
    lastGestureTime: 0,
  };

  async function init() {
    if (webcamRunning) return true;

    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm",
      );

      try {
        poseLandmarker = await createLandmarker(vision, "GPU");
        console.log("PoseLandmarker is running with GPU delegate.");
      } catch (gpuError) {
        console.warn("GPU delegate failed. Falling back to CPU.", gpuError);
        poseLandmarker = await createLandmarker(vision, "CPU");
        console.log("PoseLandmarker is running with CPU delegate.");
      }

      video = document.createElement("video");
      video.setAttribute("playsinline", "");
      video.muted = true;
      video.style.display = "none";
      document.body.appendChild(video);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
          facingMode: "user",
        },
        audio: false,
      });

      video.srcObject = stream;
      await video.play();

      state.poseReady = true;
      webcamRunning = true;

      onReady?.(state);
      requestAnimationFrame(predictLoop);

      return true;
    } catch (error) {
      console.error(error);
      onError?.(error);

      return false;
    }
  }

  async function createLandmarker(vision, delegate) {
    return await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        delegate,
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  }

  function predictLoop() {
    if (!webcamRunning || !poseLandmarker || !video) return;

    const now = performance.now();

    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;

      const result = poseLandmarker.detectForVideo(video, now);
      const gesture = classifyGesture(result);

      if (gesture !== state.currentGesture) {
        state.currentGesture = gesture;
        state.lastGestureTime = now;
        onGesture?.(gesture, result, state);
      }
    }

    requestAnimationFrame(predictLoop);
  }

  function classifyGesture(result) {
    const landmarks = result.landmarks?.[0];
    if (!landmarks) return "none";

    const nose = landmarks[0];
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];
    const leftWrist = landmarks[15];
    const rightWrist = landmarks[16];
    const leftHip = landmarks[23];
    const rightHip = landmarks[24];

    if (
      !nose ||
      !leftShoulder ||
      !rightShoulder ||
      !leftWrist ||
      !rightWrist ||
      !leftHip ||
      !rightHip
    ) {
      return "none";
    }

    const now = performance.now() / 1000;

    const shoulderX = (leftShoulder.x + rightShoulder.x) / 2;
    const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
    const hipY = (leftHip.y + rightHip.y) / 2;

    const torsoLength = Math.abs(hipY - shoulderY);
    const headToShoulder = nose.y - shoulderY;

    const leftHandRaised = leftWrist.y < leftShoulder.y - 0.08;
    const rightHandRaised = rightWrist.y < rightShoulder.y - 0.08;
    const bothHandsRaised = leftHandRaised && rightHandRaised;

    const bowing = headToShoulder > torsoLength * 0.12;
    const swaying = detectSway(shoulderX, now);
    const standingStill = detectStillness(shoulderX);

    if (bothHandsRaised) {
      return "offering";
    }

    if (leftHandRaised || rightHandRaised) {
      return "handRaised";
    }

    if (bowing) {
      return "bow";
    }

    if (swaying) {
      return "sway";
    }

    if (standingStill) {
      return "stillness";
    }

    return "none";
  }

  function detectSway(shoulderX, time) {
    swayHistory.push({ x: shoulderX, t: time });

    swayHistory = swayHistory.filter((item) => time - item.t < 1.6);

    if (swayHistory.length < 8) return false;

    const xs = swayHistory.map((item) => item.x);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);

    return maxX - minX > 0.08;
  }

  function detectStillness(shoulderX) {
    if (previousShoulderX === null) {
      previousShoulderX = shoulderX;
      return false;
    }

    const movement = Math.abs(shoulderX - previousShoulderX);
    previousShoulderX = shoulderX;

    return movement < 0.002;
  }

  function getState() {
    return { ...state };
  }

  function stop() {
    webcamRunning = false;

    if (video?.srcObject) {
      video.srcObject.getTracks().forEach((track) => track.stop());
    }

    video?.remove();
  }

  return {
    init,
    stop,
    getState,
  };
}
