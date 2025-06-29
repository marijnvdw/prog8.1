import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
import "https://unpkg.com/ml5@1/dist/ml5.min.js";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

const demosSection = document.getElementById("demos");
const imageBlendShapes = document.getElementById("image-blend-shapes");
const videoBlendShapes = document.getElementById("video-blend-shapes");
const videoWidth = 480;

let faceLandmarker;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
let recordedData = [];
ml5.setBackend("webgl");
let emotionInterval;

const emotionCounts = {
    happy: 0,
    neutral: 0,
    sad: 0,
    angry: 0,
    surprised: 0,
};


// Demo en AI Vertaald van typescript naar JS
async function createFaceLandmarker() {
    const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        runningMode,
        numFaces: 1
    });
    demosSection.classList.remove("invisible");
}
createFaceLandmarker();

const imageContainers = document.getElementsByClassName("detectOnClick");

for (let imageContainer of imageContainers) {
    imageContainer.children[0].addEventListener("click", handleClick);
}

async function handleClick(event) {
    if (!faceLandmarker) {
        console.log("Wait for faceLandmarker to load before clicking!");
        return;
    }

    if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        await faceLandmarker.setOptions({ runningMode });
    }

    const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
    for (let i = allCanvas.length - 1; i >= 0; i--) {
        allCanvas[i].parentNode.removeChild(allCanvas[i]);
    }

    const faceLandmarkerResult = faceLandmarker.detect(event.target);
    const canvas = document.createElement("canvas");
    canvas.className = "canvas";
    canvas.width = event.target.naturalWidth;
    canvas.height = event.target.naturalHeight;
    canvas.style.left = "0px";
    canvas.style.top = "0px";
    canvas.style.width = event.target.width + "px";
    canvas.style.height = event.target.height + "px";

    event.target.parentNode.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    const drawingUtils = new DrawingUtils(ctx);

    for (const landmarks of faceLandmarkerResult.faceLandmarks) {
        drawAllConnectors(drawingUtils, landmarks);
    }

    drawBlendShapes(imageBlendShapes, faceLandmarkerResult.faceBlendshapes);
}

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
} else {
    console.warn("getUserMedia() is not supported by your browser");
}

function enableCam() {
    if (!faceLandmarker) {
        console.log("Wait! faceLandmarker not loaded yet.");
        return;
    }

    webcamRunning = !webcamRunning;
    enableWebcamButton.innerText = webcamRunning ? "Camera Uitzetten" : "Camera Aanzetten";

    const constraints = { video: true };

    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}

let lastVideoTime = -1;
let results;
const drawingUtils = new DrawingUtils(canvasCtx);

async function predictWebcam() {
    const ratio = video.videoHeight / video.videoWidth;
    video.style.width = videoWidth + "px";
    video.style.height = videoWidth * ratio + "px";
    canvasElement.style.width = videoWidth + "px";
    canvasElement.style.height = videoWidth * ratio + "px";
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await faceLandmarker.setOptions({ runningMode });
    }

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = faceLandmarker.detectForVideo(video, startTimeMs);
    }

    if (results.faceLandmarks) {
        for (const landmarks of results.faceLandmarks) {
            drawAllConnectors(drawingUtils, landmarks);
        }
    }

    drawBlendShapes(videoBlendShapes, results.faceBlendshapes);

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

function drawAllConnectors(drawer, landmarks) {
    drawer.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
    drawer.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030" });
    drawer.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF3030" });
    drawer.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30FF30" });
    drawer.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#30FF30" });
    drawer.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E0E0E0" });
    drawer.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#E0E0E0" });
    drawer.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: "#FF3030" });
    drawer.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: "#30FF30" });
}

function drawBlendShapes(el, blendShapes) {
    if (!blendShapes.length) return;

    let html = "";
    blendShapes[0].categories.forEach((shape) => {
        html += `
      <li class="blend-shapes-item">
        <span class="blend-shapes-label">${shape.displayName || shape.categoryName}</span>
        <span class="blend-shapes-value" style="width: calc(${(shape.score * 100).toFixed(2)}% - 120px)">${shape.score.toFixed(4)}</span>
      </li>
    `;
    });
    el.innerHTML = html;
}

// Einde Demo en AI Vertaald van typescript naar JS








document.getElementById("predictEmotion").addEventListener("click", async () => {
    if (!results || !results.faceBlendshapes || !results.faceBlendshapes.length) {
        alert("Geen blendshape data beschikbaar van de webcam.");
        return;
    }

    const resultDiv = document.getElementById("emotionResult");
    const nn = ml5.neuralNetwork({ task: 'classification', debug: true });

    const modelDetails = {
        model: "model/model.json",
        metadata: "model/model_meta.json",
        weights: "model/model.weights.bin"
    };

    nn.load(modelDetails, () => {
        if (emotionInterval) {
            clearInterval(emotionInterval);
        }

        emotionInterval = setInterval(async () => {
            if (!results || !results.faceBlendshapes || !results.faceBlendshapes.length) {
                return;
            }

            const Vector = results.faceBlendshapes[0].categories.map(cat => cat.score);
            try {
                const prediction = await nn.classify(Vector);
                const label = prediction[0].label;
                const confidence = (prediction[0].confidence * 100).toFixed(1);
                resultDiv.textContent = `Emotie: ${label} (${confidence}%)`;

                if (emotionCounts[label] !== undefined) {
                    emotionCounts[label]++;
                    updateEmotionCountDisplay();
                }

            } catch (err) {
                resultDiv.textContent = "Error";
                console.error("Error:", err);
            }
        }, 2000); // 2 seconden
    });
});

function updateEmotionCountDisplay() {
    const list = document.getElementById("emotionList");
    list.innerHTML = "";

    for (const [emotion, count] of Object.entries(emotionCounts)) {
        const emoji = {
            happy: "😊",
            neutral: "😐",
            sad: "😢",
            angry: "😠",
            surprised: "😲"
        }[emotion];

        const li = document.createElement("li");
        li.textContent = `${emoji} ${emotion}: ${count}`;
        list.appendChild(li);
    }
}

