"""
Pothole Detection & Road Damage Analysis - Flask Web Application
================================================================
Run: python app.py
Open: http://localhost:5000

Requires:
    pip install flask ultralytics opencv-python-headless pillow numpy
    Place your trained model as 'pothole_best.pt' in this directory.
"""

import os
import io
import cv2
import json
import time
import base64
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from flask import Flask, render_template_string, request, jsonify, Response

# Import AI agent
try:
    from agent import chat_with_agent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    print("⚠️  Agent module not available. Install langgraph and langchain-openai.")

# ── Optional: use YOLOv8 if available, else use mock detections ──────────────
try:
    from ultralytics import YOLO
    MODEL_PATH = "pothole_best_final.pt"
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        MODEL_LOADED = True
        print(f"✅ YOLOv8 model loaded from {MODEL_PATH}")
    else:
        model = None
        MODEL_LOADED = False
        print(f"⚠️  Model file not found at '{MODEL_PATH}'. Using demo mode.")
except ImportError:
    model = None
    MODEL_LOADED = False
    print("⚠️  ultralytics not installed. Using demo mode.")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024   # 50 MB upload limit
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)

CONF_THRESHOLD = float(os.environ.get('CONF_THRESHOLD', '0.25'))
IOU_THRESHOLD  = float(os.environ.get('IOU_THRESHOLD',  '0.45'))

# ── Helpers ───────────────────────────────────────────────────────────────────

SEVERITY_COLORS = {
    'High':   (0,   0,   220),
    'Medium': (0,   165, 255),
    'Low':    (0,   200, 100),
}

def classify_severity(area_pct: float) -> str:
    if area_pct > 5:  return 'High'
    if area_pct > 2:  return 'Medium'
    return 'Low'


def run_detection(img_bgr: np.ndarray, conf: float = CONF_THRESHOLD) -> tuple:
    """Run YOLOv8 inference and return (annotated_img, detections_list)."""
    h, w = img_bgr.shape[:2]

    if MODEL_LOADED and model is not None:
        results = model.predict(source=img_bgr, conf=conf, iou=IOU_THRESHOLD, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
            confidence = round(float(box.conf[0]), 3)
            cls_id     = int(box.cls[0])
            cls_name   = model.names.get(cls_id, 'pothole')
            area_pct   = round((x2-x1)*(y2-y1)/(w*h)*100, 2)
            severity   = classify_severity(area_pct)
            detections.append({
                'class': cls_name, 'confidence': confidence,
                'bbox': [x1, y1, x2, y2], 'area_pct': area_pct, 'severity': severity
            })
        annotated = results.plot()
    else:
        # Demo mode: draw fake detections for UI preview
        detections = _mock_detections(w, h)
        annotated  = img_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            color = SEVERITY_COLORS.get(d['severity'], (200, 200, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"{d['class']} {d['confidence']:.2f}"
            cv2.putText(annotated, label, (x1, max(y1-10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return annotated, detections


def _mock_detections(w, h):
    import random
    random.seed(42)
    dets = []
    for _ in range(random.randint(1, 4)):
        x1 = random.randint(0, w-100)
        y1 = random.randint(0, h-80)
        x2 = x1 + random.randint(60, 180)
        y2 = y1 + random.randint(50, 140)
        x2, y2 = min(x2, w), min(y2, h)
        area_pct = round((x2-x1)*(y2-y1)/(w*h)*100, 2)
        dets.append({
            'class': 'pothole',
            'confidence': round(random.uniform(0.45, 0.95), 3),
            'bbox': [x1, y1, x2, y2],
            'area_pct': area_pct,
            'severity': classify_severity(area_pct)
        })
    return dets


def img_to_base64(img_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buffer).decode('utf-8')


# ── HTML Template ─────────────────────────────────────────────────────────────

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pothole Detection · Road Damage Analysis</title>
<style>
  :root{
    --bg:#0d1117;--surface:#161b22;--surface2:#21262d;
    --accent:#f78166;--accent2:#79c0ff;--green:#56d364;
    --yellow:#e3b341;--red:#f85149;--border:#30363d;
    --text:#e6edf3;--muted:#8b949e;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}

  /* ── header ── */
  header{background:linear-gradient(135deg,#1a1f35 0%,#0d1117 100%);
    border-bottom:1px solid var(--border);padding:16px 28px;display:flex;
    align-items:center;justify-content:space-between;gap:12px}
  .logo{display:flex;align-items:center;gap:10px}
  .logo svg{width:36px;height:36px}
  .logo h1{font-size:1.35rem;font-weight:700;letter-spacing:.5px}
  .logo span{color:var(--accent)}
  .badge{background:var(--surface2);border:1px solid var(--border);border-radius:20px;
    padding:4px 14px;font-size:.75rem;color:var(--muted)}
  .badge.live{border-color:#56d36450;color:var(--green)}

  /* ── layout ── */
  .container{display:grid;grid-template-columns:340px 1fr;gap:0;height:calc(100vh - 65px)}
  @media(max-width:900px){.container{grid-template-columns:1fr;height:auto}}

  /* ── sidebar ── */
  aside{background:var(--surface);border-right:1px solid var(--border);
    padding:20px;overflow-y:auto;display:flex;flex-direction:column;gap:18px}
  .section-title{font-size:.7rem;font-weight:600;text-transform:uppercase;
    letter-spacing:1px;color:var(--muted);margin-bottom:10px}

  /* ── upload zone ── */
  .drop-zone{border:2px dashed var(--border);border-radius:12px;padding:28px 16px;
    text-align:center;cursor:pointer;transition:all .25s;position:relative}
  .drop-zone:hover,.drop-zone.dragover{border-color:var(--accent2);
    background:rgba(121,192,255,.05)}
  .drop-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
  .drop-icon{font-size:2.2rem;margin-bottom:8px}
  .drop-label{font-size:.85rem;color:var(--muted)}
  .drop-label strong{color:var(--text)}
  .file-types{font-size:.7rem;color:var(--muted);margin-top:6px}

  /* ── controls ── */
  .control-group{display:flex;flex-direction:column;gap:8px}
  label{font-size:.8rem;color:var(--muted)}
  input[type=range]{-webkit-appearance:none;width:100%;height:4px;
    background:var(--border);border-radius:2px;outline:none}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;
    width:16px;height:16px;border-radius:50%;background:var(--accent2);cursor:pointer}
  .range-val{float:right;color:var(--accent2);font-size:.8rem;font-weight:600}

  /* ── btn ── */
  .btn{display:block;width:100%;padding:11px;border:none;border-radius:8px;
    font-size:.9rem;font-weight:600;cursor:pointer;transition:all .2s;text-align:center}
  .btn-primary{background:linear-gradient(135deg,#1565c0,#1976d2);color:#fff}
  .btn-primary:hover{filter:brightness(1.15)}
  .btn-secondary{background:var(--surface2);border:1px solid var(--border);color:var(--text)}
  .btn-secondary:hover{border-color:var(--muted)}
  .btn:disabled{opacity:.45;cursor:not-allowed}

  /* ── stats cards ── */
  .stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .stat-card{background:var(--surface2);border:1px solid var(--border);
    border-radius:8px;padding:12px;text-align:center}
  .stat-val{font-size:1.5rem;font-weight:700}
  .stat-lbl{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
  .sev-high{color:var(--red)} .sev-med{color:var(--yellow)} .sev-low{color:var(--green)}

  /* ── detection list ── */
  .det-list{display:flex;flex-direction:column;gap:6px;max-height:260px;overflow-y:auto}
  .det-item{background:var(--surface2);border:1px solid var(--border);
    border-radius:8px;padding:10px 12px;display:flex;align-items:center;gap:10px}
  .det-badge{padding:2px 10px;border-radius:20px;font-size:.68rem;font-weight:700;white-space:nowrap}
  .det-badge.High{background:#f8514920;border:1px solid #f8514960;color:var(--red)}
  .det-badge.Medium{background:#e3b34120;border:1px solid #e3b34160;color:var(--yellow)}
  .det-badge.Low{background:#56d36420;border:1px solid #56d36460;color:var(--green)}
  .det-info{flex:1;font-size:.78rem}
  .det-class{font-weight:600;text-transform:capitalize}
  .det-meta{color:var(--muted);font-size:.7rem}

  /* ── main panel ── */
  main{display:flex;flex-direction:column;padding:20px;gap:16px;overflow-y:auto}
  .result-header{display:flex;align-items:center;justify-content:space-between}
  .result-header h2{font-size:1rem;font-weight:600}

  .canvas-wrap{background:var(--surface);border:1px solid var(--border);border-radius:12px;
    display:flex;align-items:center;justify-content:center;min-height:360px;
    position:relative;overflow:hidden}
  .canvas-wrap img{max-width:100%;max-height:60vh;border-radius:10px;display:block}
  .canvas-placeholder{text-align:center;color:var(--muted);padding:40px}
  .canvas-placeholder .ph-icon{font-size:4rem;margin-bottom:12px;opacity:.4}

  /* progress */
  .progress-wrap{display:none;position:absolute;bottom:0;left:0;right:0;height:3px;background:var(--border)}
  .progress-bar{height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent));
    width:0;transition:width .3s;border-radius:2px}

  /* ── tabs for image/video ── */
  .tabs{display:flex;gap:4px;background:var(--surface2);
    border:1px solid var(--border);border-radius:10px;padding:4px}
  .tab{flex:1;padding:8px;text-align:center;border-radius:7px;cursor:pointer;
    font-size:.82rem;font-weight:500;color:var(--muted);transition:all .2s;border:none;background:none}
  .tab.active{background:var(--bg);color:var(--text);box-shadow:0 1px 4px #0004}

  /* toast */
  .toast{position:fixed;bottom:24px;right:24px;background:var(--surface2);
    border:1px solid var(--border);border-radius:10px;padding:12px 20px;
    font-size:.85rem;opacity:0;transition:opacity .3s;pointer-events:none;z-index:999}
  .toast.show{opacity:1}
  .toast.error{border-color:var(--red);color:var(--red)}
  .toast.success{border-color:var(--green);color:var(--green)}

  /* spinner */
  .spinner{width:32px;height:32px;border:3px solid var(--border);
    border-top-color:var(--accent2);border-radius:50%;animation:spin .8s linear infinite;margin:auto}
  @keyframes spin{to{transform:rotate(360deg)}}

  /* severity bar */
  .sev-bar-wrap{margin-top:4px}
  .sev-row{display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:.75rem}
  .sev-label{width:55px;color:var(--muted)}
  .sev-bar-bg{flex:1;background:var(--border);border-radius:2px;height:6px}
  .sev-bar-fill{height:100%;border-radius:2px;transition:width .5s}
  .sev-count{width:24px;text-align:right;color:var(--muted)}

  /* chat panel */
  .chat-panel{background:var(--surface);border:1px solid var(--border);border-radius:12px;
    display:flex;flex-direction:column;height:400px}
  .chat-header{padding:12px 16px;border-bottom:1px solid var(--border);
    display:flex;align-items:center;gap:8px}
  .chat-header h3{font-size:.9rem;font-weight:600}
  .chat-messages{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px}
  .chat-msg{max-width:80%;padding:10px 14px;border-radius:12px;font-size:.85rem;line-height:1.4}
  .chat-msg.user{background:var(--accent2);color:#000;align-self:flex-end}
  .chat-msg.assistant{background:var(--surface2);color:var(--text);align-self:flex-start}
  .chat-input-area{padding:12px;border-top:1px solid var(--border);display:flex;gap:8px}
  .chat-input{flex:1;padding:10px 14px;border:1px solid var(--border);
    border-radius:8px;background:var(--bg);color:var(--text);font-size:.85rem;outline:none}
  .chat-input:focus{border-color:var(--accent2)}
  .chat-send{padding:10px 16px;background:var(--accent2);border:none;border-radius:8px;
    color:#000;font-weight:600;cursor:pointer;font-size:.85rem}
  .chat-send:hover{filter:brightness(1.1)}
  .chat-send:disabled{opacity:.5;cursor:not-allowed}
</style>
</head>
<body>

<header>
  <div class="logo">
    <svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="8" fill="#161b22"/>
      <path d="M4 24 Q8 16 14 20 Q18 24 22 14 Q26 4 32 12" stroke="#f78166" stroke-width="2.5" fill="none" stroke-linecap="round"/>
      <circle cx="14" cy="20" r="4" fill="#f7816640" stroke="#f78166" stroke-width="1.5"/>
      <circle cx="22" cy="14" r="3" fill="#79c0ff40" stroke="#79c0ff" stroke-width="1.5"/>
    </svg>
    <h1>Pothole<span>AI</span> &nbsp;·&nbsp; Road Damage Analysis</h1>
  </div>
  <div id="modelBadge" class="badge">{{ 'Model Loaded ✓' if model_loaded else 'Demo Mode' }}</div>
</header>

<div class="container">
  <aside>
    <!-- Mode tabs -->
    <div>
      <p class="section-title">Input Mode</p>
      <div class="tabs">
        <button class="tab active" onclick="switchTab('image',this)">🖼️ Image</button>
        <button class="tab" onclick="switchTab('video',this)">🎬 Video</button>
        <button class="tab" onclick="switchTab('webcam',this)">📷 Webcam</button>
      </div>
    </div>

    <!-- Upload zone -->
    <div id="uploadSection">
      <p class="section-title">Upload File</p>
      <div class="drop-zone" id="dropZone">
        <input type="file" id="fileInput" accept="image/*,video/*" onchange="handleFileSelect(event)">
        <div class="drop-icon">📁</div>
        <p class="drop-label"><strong>Click or drag & drop</strong></p>
        <p class="drop-label">to upload road imagery</p>
        <p class="file-types">JPG, PNG, BMP, MP4, AVI, MOV</p>
      </div>
    </div>

    <!-- Confidence control -->
    <div class="control-group">
      <p class="section-title">Detection Settings</p>
      <label>Confidence Threshold <span class="range-val" id="confVal">0.25</span></label>
      <input type="range" id="confSlider" min="0.1" max="0.95" step="0.05" value="0.25"
             oninput="document.getElementById('confVal').textContent=parseFloat(this.value).toFixed(2)">
      <label style="margin-top:8px">IoU Threshold <span class="range-val" id="iouVal">0.45</span></label>
      <input type="range" id="iouSlider" min="0.1" max="0.9" step="0.05" value="0.45"
             oninput="document.getElementById('iouVal').textContent=parseFloat(this.value).toFixed(2)">
    </div>

    <button class="btn btn-primary" id="analyzeBtn" onclick="analyzeFile()" disabled>
      🔍 Analyze Road
    </button>
    <button class="btn btn-secondary" onclick="clearResults()">🗑 Clear</button>

    <!-- Stats -->
    <div id="statsSection" style="display:none">
      <p class="section-title">Detection Summary</p>
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-val" id="statTotal">0</div>
          <div class="stat-lbl">Detections</div>
        </div>
        <div class="stat-card">
          <div class="stat-val sev-high" id="statHigh">0</div>
          <div class="stat-lbl">High Sev.</div>
        </div>
        <div class="stat-card">
          <div class="stat-val sev-med" id="statMed">0</div>
          <div class="stat-lbl">Medium Sev.</div>
        </div>
        <div class="stat-card">
          <div class="stat-val sev-low" id="statLow">0</div>
          <div class="stat-lbl">Low Sev.</div>
        </div>
      </div>
      <div class="sev-bar-wrap" id="sevBars" style="margin-top:12px"></div>
    </div>

    <!-- Detection list -->
    <div id="detListSection" style="display:none">
      <p class="section-title">Detected Defects</p>
      <div class="det-list" id="detList"></div>
    </div>

    <!-- Export -->
    <button class="btn btn-secondary" id="exportBtn" onclick="exportReport()" style="display:none">
      📊 Export JSON Report
    </button>
  </aside>

  <!-- Main result panel -->
  <main>
    <div class="result-header">
      <h2>Detection Output</h2>
      <span id="procTime" style="font-size:.78rem;color:var(--muted)"></span>
    </div>

    <div class="canvas-wrap" id="canvasWrap">
      <div class="canvas-placeholder" id="placeholder">
        <div class="ph-icon">🛣️</div>
        <p>Upload an image or video to start<br>road damage analysis</p>
        <p style="font-size:.75rem;margin-top:8px;color:#444">
          Powered by YOLOv8 · Real-time Detection
        </p>
      </div>
      <img id="resultImg" style="display:none" alt="Detection Result">
      <div id="loadingSpinner" style="display:none">
        <div class="spinner"></div>
        <p style="margin-top:12px;color:var(--muted);font-size:.85rem">Analyzing...</p>
      </div>
      <div class="progress-wrap" id="progressWrap">
        <div class="progress-bar" id="progressBar"></div>
      </div>
    </div>

    <!-- Avg confidence bar -->
    <div id="confBar" style="display:none;background:var(--surface);
         border:1px solid var(--border);border-radius:10px;padding:16px">
      <p class="section-title" style="margin-bottom:10px">Average Confidence</p>
      <div style="background:var(--border);border-radius:4px;height:10px;overflow:hidden">
        <div id="confBarFill" style="height:100%;background:linear-gradient(90deg,#1976d2,#79c0ff);
             border-radius:4px;transition:width .6s;width:0%"></div>
      </div>
      <p id="confBarLabel" style="font-size:.8rem;color:var(--muted);margin-top:6px"></p>
    </div>

    <!-- AI Chat Panel -->
    <div class="chat-panel">
      <div class="chat-header">
        <span>🤖</span>
        <h3>AI Assistant</h3>
      </div>
      <div class="chat-messages" id="chatMessages">
        <div class="chat-msg assistant">
          Hello! I'm your AI assistant for pothole detection. Ask me anything about road damage analysis!
        </div>
      </div>
      <div class="chat-input-area">
        <input type="text" class="chat-input" id="chatInput" placeholder="Ask about pothole detection..." onkeypress="if(event.key==='Enter')sendChat()">
        <button class="chat-send" id="chatSend" onclick="sendChat()">Send</button>
      </div>
    </div>
  </main>
</div>

<div class="toast" id="toast"></div>

<script>
let currentFile = null;
let lastDetections = [];
let currentMode = 'image';

// ── Drag & drop ───────────────────────────────────────────────────────────────
const dz = document.getElementById('dropZone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover') });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f) processFile(f);
});

function handleFileSelect(e) {
  if (e.target.files[0]) processFile(e.target.files[0]);
}

function processFile(file) {
  currentFile = file;
  document.getElementById('analyzeBtn').disabled = false;
  const dz = document.getElementById('dropZone');
  dz.querySelector('.drop-label').innerHTML = `<strong>${file.name}</strong>`;
  dz.querySelector('.file-types').textContent = `${(file.size/1024/1024).toFixed(2)} MB`;
  showToast(`📎 File ready: ${file.name}`, 'success');
}

function switchTab(mode, btn) {
  currentMode = mode;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  if (mode === 'webcam') {
    document.getElementById('uploadSection').style.display = 'none';
    showToast('📷 Webcam mode: Use the /webcam endpoint directly', 'success');
  } else {
    document.getElementById('uploadSection').style.display = '';
    document.getElementById('fileInput').accept = mode === 'image' ? 'image/*' : 'video/*';
  }
}

// ── Analyze ───────────────────────────────────────────────────────────────────
async function analyzeFile() {
  if (!currentFile) { showToast('Please select a file first', 'error'); return; }

  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Analyzing...';

  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('resultImg').style.display = 'none';
  document.getElementById('loadingSpinner').style.display = 'flex';
  document.getElementById('loadingSpinner').style.flexDirection = 'column';
  document.getElementById('loadingSpinner').style.alignItems = 'center';

  const formData = new FormData();
  formData.append('file', currentFile);
  formData.append('conf', document.getElementById('confSlider').value);
  formData.append('iou', document.getElementById('iouSlider').value);

  const t0 = performance.now();
  try {
    const res = await fetch('/detect', { method: 'POST', body: formData });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

    document.getElementById('loadingSpinner').style.display = 'none';
    const img = document.getElementById('resultImg');
    img.src = 'data:image/jpeg;base64,' + data.image_b64;
    img.style.display = 'block';

    document.getElementById('procTime').textContent = `⏱ ${elapsed}s · ${data.detections.length} detection(s)`;
    lastDetections = data.detections;
    renderStats(data.detections);
    showToast(`✅ Analysis complete — ${data.detections.length} defect(s) found`, 'success');
  } catch(err) {
    document.getElementById('loadingSpinner').style.display = 'none';
    document.getElementById('placeholder').style.display = '';
    showToast('❌ Error: ' + err.message, 'error');
  }

  btn.disabled = false;
  btn.textContent = '🔍 Analyze Road';
}

// ── Stats rendering ───────────────────────────────────────────────────────────
function renderStats(dets) {
  const high = dets.filter(d => d.severity === 'High').length;
  const med  = dets.filter(d => d.severity === 'Medium').length;
  const low  = dets.filter(d => d.severity === 'Low').length;

  document.getElementById('statTotal').textContent = dets.length;
  document.getElementById('statHigh').textContent  = high;
  document.getElementById('statMed').textContent   = med;
  document.getElementById('statLow').textContent   = low;
  document.getElementById('statsSection').style.display = '';

  // Severity bars
  const maxVal = Math.max(high, med, low, 1);
  document.getElementById('sevBars').innerHTML = [
    { label:'High',   val:high, color:'#f85149' },
    { label:'Medium', val:med,  color:'#e3b341' },
    { label:'Low',    val:low,  color:'#56d364' }
  ].map(s => `
    <div class="sev-row">
      <span class="sev-label">${s.label}</span>
      <div class="sev-bar-bg">
        <div class="sev-bar-fill" style="width:${(s.val/maxVal*100).toFixed(1)}%;background:${s.color}"></div>
      </div>
      <span class="sev-count">${s.val}</span>
    </div>`).join('');

  // Detection list
  document.getElementById('detListSection').style.display = '';
  document.getElementById('detList').innerHTML = dets.length === 0
    ? '<p style="font-size:.8rem;color:var(--muted);text-align:center">No defects detected</p>'
    : dets.map((d,i) => `
      <div class="det-item">
        <span style="font-size:1rem">🕳️</span>
        <div class="det-info">
          <div class="det-class">#${i+1} ${d.class}</div>
          <div class="det-meta">Conf: ${d.confidence} &nbsp;·&nbsp; Area: ${d.area_pct}%</div>
        </div>
        <span class="det-badge ${d.severity}">${d.severity}</span>
      </div>`).join('');

  // Confidence bar
  if (dets.length > 0) {
    const avgConf = dets.reduce((s,d) => s + d.confidence, 0) / dets.length;
    document.getElementById('confBar').style.display = '';
    document.getElementById('confBarFill').style.width = (avgConf * 100).toFixed(1) + '%';
    document.getElementById('confBarLabel').textContent =
      `Average confidence: ${(avgConf*100).toFixed(1)}% across ${dets.length} detection(s)`;
  }

  document.getElementById('exportBtn').style.display = '';
}

// ── Clear ─────────────────────────────────────────────────────────────────────
function clearResults() {
  currentFile = null; lastDetections = [];
  document.getElementById('fileInput').value = '';
  document.getElementById('analyzeBtn').disabled = true;
  document.getElementById('resultImg').style.display = 'none';
  document.getElementById('placeholder').style.display = '';
  document.getElementById('statsSection').style.display = 'none';
  document.getElementById('detListSection').style.display = 'none';
  document.getElementById('confBar').style.display = 'none';
  document.getElementById('exportBtn').style.display = 'none';
  document.getElementById('procTime').textContent = '';
  const dz = document.getElementById('dropZone');
  dz.querySelector('.drop-label').innerHTML = '<strong>Click or drag & drop</strong>';
  dz.querySelector('.file-types').textContent = 'JPG, PNG, BMP, MP4, AVI, MOV';
}

// ── Export ────────────────────────────────────────────────────────────────────
function exportReport() {
  const report = {
    timestamp: new Date().toISOString(),
    total_detections: lastDetections.length,
    summary: {
      high: lastDetections.filter(d => d.severity==='High').length,
      medium: lastDetections.filter(d => d.severity==='Medium').length,
      low: lastDetections.filter(d => d.severity==='Low').length
    },
    detections: lastDetections
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `pothole_report_${Date.now()}.json`;
  a.click();
  showToast('📊 Report exported!', 'success');
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let toastTimer;
function showToast(msg, type='') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show ' + type;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.className = 'toast', 3200);
}

// ── Chat ─────────────────────────────────────────────────────────────────────
async function sendChat() {
  const input = document.getElementById('chatInput');
  const msg = input.value.trim();
  if (!msg) return;

  const btn = document.getElementById('chatSend');
  btn.disabled = true;

  // Add user message
  const chatDiv = document.getElementById('chatMessages');
  chatDiv.innerHTML += `<div class="chat-msg user">${msg}</div>`;
  input.value = '';
  chatDiv.scrollTop = chatDiv.scrollHeight;

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        message: msg,
        detection_context: {detections: lastDetections}
      })
    });
    const data = await res.json();
    
    if (data.error) {
      chatDiv.innerHTML += `<div class="chat-msg assistant" style="color:var(--red)">Error: ${data.error}</div>`;
    } else {
      chatDiv.innerHTML += `<div class="chat-msg assistant">${data.response}</div>`;
    }
  } catch(err) {
    chatDiv.innerHTML += `<div class="chat-msg assistant" style="color:var(--red)">Error: ${err.message}</div>`;
  }

  chatDiv.scrollTop = chatDiv.scrollHeight;
  btn.disabled = false;
}
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML, model_loaded=MODEL_LOADED)


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    conf = float(request.form.get('conf', CONF_THRESHOLD))
    iou  = float(request.form.get('iou',  IOU_THRESHOLD))

    # Decode image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    # Run detection
    start = time.time()
    annotated, detections = run_detection(img, conf=conf)
    elapsed = round(time.time() - start, 3)

    image_b64 = img_to_base64(annotated)
    return jsonify({
        'detections': detections,
        'total': len(detections),
        'inference_time_s': elapsed,
        'image_b64': image_b64,
        'summary': {
            'high':   sum(1 for d in detections if d['severity'] == 'High'),
            'medium': sum(1 for d in detections if d['severity'] == 'Medium'),
            'low':    sum(1 for d in detections if d['severity'] == 'Low'),
        }
    })


@app.route('/webcam')
def webcam_stream():
    """Simple MJPEG stream from the default camera."""
    def generate():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            yield b''
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, _ = run_detection(frame)
            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL_LOADED,
        'model_path': MODEL_PATH if MODEL_LOADED else None,
        'conf_threshold': CONF_THRESHOLD,
    })


@app.route('/chat', methods=['POST'])
def chat():
    """AI Agent chat endpoint"""
    if not AGENT_AVAILABLE:
        return jsonify({'error': 'AI agent not available. Install dependencies first.'}), 400
    
    data = request.get_json()
    message = data.get('message', '')
    detection_context = data.get('detection_context', {})
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        response = chat_with_agent(message, detection_context)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 55)
    print("  Pothole Detection · Flask Web App")
    print("=" * 55)
    print(f"  Model loaded : {MODEL_LOADED}")
    print(f"  Mode         : {'Inference' if MODEL_LOADED else 'Demo (mock detections)'}")
    print(f"  URL          : http://localhost:5000")
    print("=" * 55)
    
    # Production-ready configuration
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
