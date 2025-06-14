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



let verificationData = [];
let verifierModel;
let verifierReady = false;

const storedData = localStorage.getItem("verificationData");
if (storedData) {
    verificationData = JSON.parse(storedData);
    updateAccuracyAndMatrix();
}

async function loadVerifierModel() {
    verifierModel = ml5.neuralNetwork({ task: 'classification', debug: false });
    await new Promise((resolve) => {
        verifierModel.load({
            model: "model/model.json",
            metadata: "model/model_meta.json",
            weights: "model/model.weights.bin"
        }, resolve);
    });
    verifierReady = true;
}

function updateAccuracyAndMatrix() {
    localStorage.setItem("verificationData", JSON.stringify(verificationData));

    const correctCount = verificationData.filter(d => d.actual === d.predicted).length;
    const accuracy = (correctCount / verificationData.length * 100).toFixed(1);

    document.getElementById("accuracyDisplay").textContent = `Accuracy: ${accuracy}% (${correctCount}/${verificationData.length})`;

    const labels = ["happy", "neutral", "sad", "angry", "surprised"];
    const matrix = {};

    labels.forEach(actual => {
        matrix[actual] = {};
        labels.forEach(predicted => {
            matrix[actual][predicted] = 0;
        });
    });

    verificationData.forEach(({ actual, predicted }) => {
        if (matrix[actual] && matrix[actual][predicted] !== undefined) {
            matrix[actual][predicted]++;
        }
    });

    const container = document.getElementById("confusionMatrixContainer");
    const table = document.createElement("table");
    table.className = "table-auto border-collapse border border-gray-300 mt-2";
    let html = "<thead><tr><th>Matrix</th>";

    labels.forEach(label => {
        html += `<th class="px-2 py-1 border border-gray-300">${label}</th>`;
    });
    html += "</tr></thead><tbody>";

    labels.forEach(actual => {
        html += `<tr><td class="font-bold px-2 py-1 border border-gray-300">${actual}</td>`;
        labels.forEach(predicted => {
            const count = matrix[actual][predicted];
            html += `<td class="border px-2 py-1 text-center">${count}</td>`;
        });
        html += "</tr>";
    });

    html += "</tbody>";
    table.innerHTML = html;

    container.innerHTML = "";
    container.appendChild(table);
}

document.getElementById("verifyData").addEventListener("click", async () => {
    if (!verifierReady) {
        await loadVerifierModel();
    }

    if (!results || !results.faceBlendshapes || !results.faceBlendshapes.length) {
        alert("Geen data beschikbaar.");
        return;
    }

    const vector = results.faceBlendshapes[0].categories.map(cat => cat.score);

    try {
        const prediction = await verifierModel.classify(vector);
        const predictedLabel = prediction[0].label;

        const correct = confirm(`AI voorspelt: ${predictedLabel}\nWas dit correct? Klik op 'OK' voor ja, 'Annuleren' voor nee.`);

        let actualLabel = predictedLabel;
        if (!correct) {
            actualLabel = prompt("Wat was de juiste emotie? (opties: happy, neutral, sad, angry, surprised)");
        }

        if (actualLabel) {
            verificationData.push({ actual: actualLabel, predicted: predictedLabel });
            updateAccuracyAndMatrix();
        }
    } catch (err) {
        console.error("Mislukt:", err);
        alert("Er ging iets mis.");
    }
});

