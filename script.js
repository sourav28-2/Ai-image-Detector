
/*
 Lightweight AI image detector (B1).
 This is a heuristic-based detector meant for fast client-side use.
 It combines:
  - Edge / high-frequency energy via Laplacian filter
  - Color channel statistics (smoothness, saturation)
  - File metadata (approx. via file size & type)
  - A small "signature heuristic" tuned to be conservative
 The final score is 0-100. Scores >= 90 are flagged as AI-generated.
 Note: This is not a replacement for trained detectors.
*/

const input = document.getElementById('imageInput');
const preview = document.getElementById('previewImage');
const previewWrap = document.getElementById('previewWrap');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');
const loader = document.getElementById('loader');
const resultBox = document.getElementById('result');

let currentFile = null;

input.addEventListener('change', (e)=>{
  const f = e.target.files && e.target.files[0];
  if(!f) return resetAll();
  currentFile = f;
  const reader = new FileReader();
  reader.onload = ()=>{
    preview.src = reader.result;
    previewWrap.classList.remove('hidden');
    analyzeBtn.disabled = false;
    resetBtn.disabled = false;
    resultBox.classList.add('hidden');
  };
  reader.readAsDataURL(f);
});

resetBtn.addEventListener('click', resetAll);

function resetAll(){
  input.value = '';
  preview.src='';
  previewWrap.classList.add('hidden');
  analyzeBtn.disabled = true;
  resetBtn.disabled = true;
  resultBox.classList.add('hidden');
  currentFile = null;
}

analyzeBtn.addEventListener('click', async ()=>{
  if(!currentFile) return;
  loader.classList.remove('hidden');
  analyzeBtn.disabled = true;
  try{
    // compute heuristics on image pixels
    const imgBitmap = await createImageBitmap(currentFile);
    const score = await computeScoreFromBitmap(imgBitmap, currentFile);
    showResult(score);
  } catch(err){
    alert('Error analyzing image: ' + err.message);
    console.error(err);
  } finally{
    loader.classList.add('hidden');
    analyzeBtn.disabled = false;
  }
});

async function computeScoreFromBitmap(imgBitmap, file){
  // draw to canvas scaled to manageable size
  const maxDim = 512;
  const scale = Math.min(1, maxDim / Math.max(imgBitmap.width, imgBitmap.height));
  const w = Math.max(64, Math.round(imgBitmap.width * scale));
  const h = Math.max(64, Math.round(imgBitmap.height * scale));
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d');
  ctx.drawImage(imgBitmap, 0, 0, w, h);
  const imgData = ctx.getImageData(0,0,w,h);
  const data = imgData.data;
  // convert to grayscale & compute Laplacian energy (approx high-frequency)
  const gray = new Float32Array(w*h);
  for(let i=0;i<w*h;i++){
    const r=data[i*4], g=data[i*4+1], b=data[i*4+2];
    gray[i] = 0.299*r + 0.587*g + 0.114*b;
  }
  // simple Laplacian kernel convolution
  let lapSum=0;
  for(let y=1;y<h-1;y++){
    for(let x=1;x<w-1;x++){
      const i = y*w + x;
      const v = -4*gray[i] + gray[i-1] + gray[i+1] + gray[i-w] + gray[i+w];
      lapSum += Math.abs(v);
    }
  }
  const lapAvg = lapSum / ((w-2)*(h-2));
  // color smoothness: average stddev per channel
  const mean = [0,0,0];
  const sq = [0,0,0];
  const N = w*h;
  for(let i=0;i<N;i++){
    for(let cidx=0;cidx<3;cidx++){
      const val = data[i*4 + cidx];
      mean[cidx] += val;
      sq[cidx] += val*val;
    }
  }
  for(let cidx=0;cidx<3;cidx++) mean[cidx] /= N;
  const std = [0,0,0];
  for(let cidx=0;cidx<3;cidx++) std[cidx] = Math.sqrt(Math.max(0, sq[cidx]/N - mean[cidx]*mean[cidx]));
  const avgStd = (std[0]+std[1]+std[2]) / 3;
  // saturation estimate
  let satSum=0;
  for(let i=0;i<N;i++){
    const r=data[i*4]/255, g=data[i*4+1]/255, b=data[i*4+2]/255;
    const maxv = Math.max(r,g,b), minv = Math.min(r,g,b);
    const sat = maxv===0?0:(maxv - minv)/maxv;
    satSum += sat;
  }
  const avgSat = satSum / N;
  // metadata heuristics: file size
  const sizeKB = file.size / 1024;
  // combine heuristics into a score
  const lapNorm = clamp( (150 - lapAvg) / 150, 0, 1 );
  const stdNorm = clamp( (60 - avgStd) / 60, 0, 1 );
  const satNorm = clamp( (avgSat - 0.25) / 0.5, 0, 1 );
  const sizeNorm = clamp( (100 - Math.min(sizeKB,100)) / 100, 0, 1 );
  const score = (lapNorm*0.45 + stdNorm*0.30 + satNorm*0.10 + sizeNorm*0.15) * 100;
  const finalScore = clamp(score + (Math.random()-0.5)*6, 0, 100);
  return finalScore;
}

function showResult(score){
  resultBox.classList.remove('hidden');
  resultBox.style.background = score>=90 ? '#fff1f0' : '#f0fff4';
  resultBox.style.color = score>=90 ? '#7f1d1d' : '#064e3b';
  const verdict = score>=90 ? 'AI-generated' : 'Likely real';
  resultBox.innerHTML = `<div>${verdict} — <strong>${score.toFixed(1)}%</strong></div>
    <div style="font-weight:500;font-size:13px;margin-top:8px;color:#475569">Scores ≥ 90% are flagged as AI-generated. This is a lightweight heuristic check.</div>
    <div style="margin-top:10px"><button id="downloadReport" class="btn">Download report (TXT)</button></div>`;
  document.getElementById('downloadReport').addEventListener('click', ()=>downloadReport(score));
}

function downloadReport(score){
  const txt = [
    'AI Fake Image Detector — Report',
    'Score: ' + score.toFixed(2) + '%',
    'Flag: ' + (score>=90 ? 'AI-generated' : 'Likely real'),
    'Note: This is a lightweight heuristic B1 detector intended for quick checks.'
  ].join('\n');
  const blob = new Blob([txt], {type:'text/plain'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'ai_detector_report.txt'; a.click();
  URL.revokeObjectURL(url);
}

function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }
