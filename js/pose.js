const MODEL_URL = "https://teachablemachine.withgoogle.com/models/cLtqsj5MG/";

export function createPoseRitualController({
  onGesture,
  onReady,
  onError,
} = {}) {
  let model = null;
  let webcam = null;
  let webcamRunning = false;
  let maxPredictions = 0;

  const DEBUG = true;
  const CONFIDENCE_THRESHOLD = 0.55;

  const state = {
    poseReady: false,
    currentGesture: "none",
    lastGestureTime: 0,
  };

  async function init() {
    if (webcamRunning) return true;

    try {
      const modelURL = MODEL_URL + "model.json";
      const metadataURL = MODEL_URL + "metadata.json";

      model = await tmPose.load(modelURL, metadataURL);
      maxPredictions = model.getTotalClasses();

      const flip = true;
      webcam = new tmPose.Webcam(640, 480, flip);

      await webcam.setup();
      await webcam.play();

      webcamRunning = true;
      state.poseReady = true;

      onReady?.(state);
      requestAnimationFrame(predictLoop);

      return true;
    } catch (error) {
      console.error("Teachable Machine pose init failed:", error);
      onError?.(error);
      return false;
    }
  }

  async function predictLoop() {
    if (!webcamRunning || !model || !webcam) return;

    webcam.update();

    const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
    const predictions = await model.predict(posenetOutput);
    const best = getBestPrediction(predictions);

    if (best && DEBUG) {
      console.log(
        "gesture:",
        best.className,
        "probability:",
        best.probability.toFixed(2),
      );
    }

    if (best && best.probability > CONFIDENCE_THRESHOLD) {
      const gesture = best.className.trim().toLowerCase();

      state.currentGesture = gesture;
      state.lastGestureTime = performance.now();

      onGesture?.(
        gesture,
        {
          pose,
          predictions,
          best,
        },
        state,
      );
    }

    requestAnimationFrame(predictLoop);
  }

  function getBestPrediction(predictions) {
    if (!predictions || predictions.length === 0) return null;

    return predictions.reduce((best, item) => {
      return item.probability > best.probability ? item : best;
    }, predictions[0]);
  }

  function getState() {
    return { ...state };
  }

  function stop() {
    webcamRunning = false;
    state.poseReady = false;
    state.currentGesture = "none";

    if (webcam?.webcam?.srcObject) {
      webcam.webcam.srcObject.getTracks().forEach((track) => track.stop());
    }
  }

  return {
    init,
    stop,
    getState,
  };
}
