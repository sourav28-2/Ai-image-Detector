/*
  AI Image Detector — TFJS Model Version
  Requires: model.json + weight files in /model folder
*/

let model = null;

const input = document.getElementById("imageInput");
const preview = document.getElementById("previewImage");
const previewWrap = document.getElementById("previewWrap");
const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");
const loader = document.getElementById("loader");
const resultBox = document.getElementById("result");

let currentFile = null;

// 1. LOAD MODEL
(async function loadModel(){
  try {
    model = await tf.loadLayersModel("model/model.json");
    console.log("TFJS model loaded.");
  } catch (err) {
    console.error("Model load error:", err);
    alert("Failed to load AI detection model.");
  }
})();

// 2. Handle Image Upload
input.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (!file) return resetAll();

  currentFile = file;
  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    previewWrap.classList.remove("hidden");
    analyzeBtn.disabled = false;
    resetBtn.disabled = false;
    resultBox.classList.add("hidden");
  };
  reader.readAsDataURL(file);
});

// Reset
resetBtn.addEventListener("click", resetAll);

function resetAll(){
  input.value = "";
  preview.src = "";
  previewWrap.classList.add("hidden");
  analyzeBtn.disabled = true;
  resetBtn.disabled = true;
  resultBox.classList.add("hidden");
  currentFile = null;
}

// 3. Analyze Button
analyzeBtn.addEventListener("click", async () => {
  if (!currentFile || !model) return;

  loader.classList.remove("hidden");
  analyzeBtn.disabled = true;

  try {
    const imgBitmap = await createImageBitmap(currentFile);
    const score = await runTFJSModel(imgBitmap);
    showResult(score);
  } catch (err) {
    alert("Error analyzing image: " + err.message);
  } finally {
    loader.classList.add("hidden");
    analyzeBtn.disabled = false;
  }
});

// 4. Model Inference
async function runTFJSModel(bitmap){
  const tfImg = tf.browser.fromPixels(bitmap)
      .resizeNearestNeighbor([224,224])
      .toFloat()
      .div(255.0)
      .expandDims();

  const pred = model.predict(tfImg);
  const score = (await pred.data())[0]; 
  pred.dispose();
  tfImg.dispose();

  return score * 100;  // convert to %
}

// 5. Display Result
function showResult(score){
  resultBox.classList.remove("hidden");
  const isAI = score >= 90;

  resultBox.style.background = isAI ? "#fff1f0" : "#f0fff4";
  resultBox.style.color = isAI ? "#7f1d1d" : "#064e3b";

  const verdict = isAI ? "AI-generated" : "Likely real";

  resultBox.innerHTML = `
    <div>${verdict} — <strong>${score.toFixed(1)}%</strong></div>
    <div style="font-weight:500;font-size:13px;margin-top:8px;color:#475569">
      Scores ≥ 90% are flagged as AI-generated (Model-based).
    </div>
    <div style="margin-top:10px">
      <button id="downloadReport" class="btn">Download report (TXT)</button>
    </div>
  `;

  document.getElementById("downloadReport").addEventListener("click", () =>
    downloadReport(score)
  );
}

// 6. Report export
function downloadReport(score){
  const txt = [
    "AI Image Detector — TFJS Version",
    "Score: " + score.toFixed(2) + "%",
    "Flag: " + (score >= 90 ? "AI-generated" : "Likely real"),
    "Model: CNN binary classifier",
  ].join("\n");

  const blob = new Blob([txt], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "ai_detector_report.txt";
  a.click();
  URL.revokeObjectURL(url);
}
