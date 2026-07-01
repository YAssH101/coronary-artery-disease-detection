// ── Element refs ──────────────────────────────────────
const dropZone          = document.getElementById("dropZone");
const fileInput         = document.getElementById("fileInput");
const fileList          = document.getElementById("fileList");
const fileListSection   = document.getElementById("fileListSection");
const fileCountLabel    = document.getElementById("fileCountLabel");
const sliceBadgeCount   = document.getElementById("sliceBadgeCount");
const clearBtn          = document.getElementById("clearBtn");
const analyseBtn        = document.getElementById("analyseBtn");
const btnText           = document.getElementById("btnText");
const btnSpinner        = document.getElementById("btnSpinner");

const viewerPlaceholder = document.getElementById("viewerPlaceholder");
const slideStage        = document.getElementById("slideStage");
const slideImg          = document.getElementById("slideImg");
const ovlFilename       = document.getElementById("ovlFilename");
const ovlCounter        = document.getElementById("ovlCounter");
const sliceResultBadge  = document.getElementById("sliceResultBadge");
const slideCounter      = document.getElementById("slideCounter");
const thumbBar          = document.getElementById("thumbBar");
const thumbStrip        = document.getElementById("thumbStrip");
const prevBtn           = document.getElementById("prevBtn");
const nextBtn           = document.getElementById("nextBtn");
const autoPlayBtn       = document.getElementById("autoPlayBtn");
const heatmapBtn        = document.getElementById("heatmapBtn");
const heatmapLoading    = document.getElementById("heatmapLoading");

const analysisPlaceholder = document.getElementById("analysisPlaceholder");
const resultCard          = document.getElementById("resultCard");
const errorCard           = document.getElementById("errorCard");
const errorMsg            = document.getElementById("errorMsg");
const reportTag           = document.getElementById("reportTag");

// Status bar
const sbModel = document.getElementById("sbModel");
const sbStudy = document.getElementById("sbStudy");
const sbSlice = document.getElementById("sbSlice");
const modelDot  = document.getElementById("modelDot");
const modelStatus = document.getElementById("modelStatus");

// ── State ──────────────────────────────────────────────
let files        = [];
let objectURLs   = [];
let currentIdx   = 0;
let autoInterval = null;
let perImageData = [];
let heatmapMode  = false;
let heatmapCache = {};   // index → base64 data URL

// ── Check model health on load ─────────────────────────
(async () => {
  try {
    const r = await fetch("/health");
    const d = await r.json();
    if (d.model_loaded) {
      modelDot.classList.add("active");
      modelStatus.textContent = "Model: Ready";
      sbModel.textContent = "● Model: Ready";
      sbModel.className = "loaded";
    } else {
      modelDot.classList.add("error");
      modelStatus.textContent = "Model: Not loaded";
      sbModel.textContent = "● Model: Not loaded";
      sbModel.className = "error";
    }
  } catch {
    modelDot.classList.add("error");
    modelStatus.textContent = "Model: Offline";
  }
})();

// ── File selection ─────────────────────────────────────
const folderInput = document.getElementById("folderInput");

// Labels stop propagation so their click doesn't also trigger the zone handler
document.querySelector('label[for="folderInput"]').addEventListener("click", e => e.stopPropagation());
document.querySelector('label[for="fileInput"]').addEventListener("click",   e => e.stopPropagation());

// Clicking anywhere else in the drop zone opens the folder picker
dropZone.addEventListener("click", e => {
  if (!e.target.closest("label")) folderInput.click();
});

fileInput.addEventListener("change",   () => handleFiles(Array.from(fileInput.files)));
folderInput.addEventListener("change", () => handleFiles(Array.from(folderInput.files)));

dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const items = Array.from(e.dataTransfer.items || []);
  if (items.length && items[0].webkitGetAsEntry) {
    collectFromItems(items).then(fs => handleFiles(fs));
  } else {
    handleFiles(Array.from(e.dataTransfer.files));
  }
});

async function collectFromItems(items) {
  const entries = items.map(i => i.webkitGetAsEntry()).filter(Boolean);
  const nested  = await Promise.all(entries.map(readEntry));
  return nested.flat()
    .filter(f => f && /\.(jpe?g|png|bmp)$/i.test(f.name))
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
}

async function readEntry(entry) {
  if (entry.isFile) {
    return new Promise(resolve => entry.file(resolve, () => resolve(null)));
  }
  const reader  = entry.createReader();
  const entries = await readAllEntries(reader);
  const nested  = await Promise.all(entries.map(readEntry));
  return nested.flat();
}

async function readAllEntries(reader) {
  return new Promise(resolve => {
    const results = [];
    const read = () => reader.readEntries(batch => {
      if (!batch.length) { resolve(results); return; }
      results.push(...batch);
      read();
    }, () => resolve(results));
    read();
  });
}

clearBtn.addEventListener("click", clearAll);

function handleFiles(newFiles) {
  const valid = newFiles.filter(f => f && /\.(jpe?g|png|bmp)$/i.test(f.name));
  if (!valid.length) return;

  objectURLs.forEach(u => URL.revokeObjectURL(u));
  files = valid;
  objectURLs = files.map(f => URL.createObjectURL(f));
  perImageData = [];
  heatmapCache = {};
  setHeatmapMode(false);

  buildFileList();
  buildThumbs();
  showViewer();
  hideResults();
  updateHeader();
  analyseBtn.disabled = false;
  clearBtn.hidden = false;
}

function clearAll() {
  objectURLs.forEach(u => URL.revokeObjectURL(u));
  files = []; objectURLs = []; perImageData = [];
  heatmapCache = {};
  setHeatmapMode(false);
  fileInput.value = "";
  folderInput.value = "";
  fileList.innerHTML = "";
  thumbStrip.innerHTML = "";
  fileListSection.hidden = true;
  thumbBar.hidden = true;
  clearBtn.hidden = true;
  analyseBtn.disabled = true;
  slideStage.hidden = true;
  viewerPlaceholder.hidden = false;
  hideResults();
  stopAutoPlay();
  prevBtn.disabled = nextBtn.disabled = autoPlayBtn.disabled = heatmapBtn.disabled = true;
  slideCounter.textContent = "— / —";
  sliceBadgeCount.hidden = true;
  sbStudy.textContent = "No study loaded";
  sbSlice.textContent = "—";
}

function updateHeader() {
  const n = files.length;
  fileCountLabel.textContent = `${n} slice${n !== 1 ? "s" : ""}`;
  sliceBadgeCount.textContent = n;
  sliceBadgeCount.hidden = false;
  sbStudy.textContent = `Study loaded — ${n} slice${n !== 1 ? "s" : ""}`;
}

// ── File list (left panel) ─────────────────────────────
function buildFileList() {
  fileList.innerHTML = "";
  files.forEach((f, i) => {
    const row = document.createElement("div");
    row.className = "file-row" + (i === 0 ? " active" : "");
    row.dataset.idx = i;

    const thumb = document.createElement("img");
    thumb.className = "file-thumb";
    thumb.src = objectURLs[i];
    thumb.alt = f.name;

    const info = document.createElement("div");
    info.className = "file-info";
    info.innerHTML = `<div class="file-name" title="${f.name}">${f.name}</div>
                      <div class="file-pred pred-wait" id="fpred-${i}">PENDING</div>`;

    const dot = document.createElement("div");
    dot.className = "file-dot";
    dot.id = `fdot-${i}`;

    row.appendChild(thumb);
    row.appendChild(info);
    row.appendChild(dot);
    row.addEventListener("click", () => goTo(i));
    fileList.appendChild(row);
  });
  fileListSection.hidden = false;
}

function updateFileListPredictions() {
  perImageData.forEach((d, i) => {
    const pred = document.getElementById(`fpred-${i}`);
    const dot  = document.getElementById(`fdot-${i}`);
    const row  = fileList.children[i];
    if (!pred || !dot || !row) return;
    if (d.error) {
      pred.textContent = "ERROR";
      pred.className = "file-pred pred-wait";
    } else {
      const sick = d.prediction === "CAD Detected";
      pred.textContent = sick ? `CAD ${d.probability_sick}%` : `NORMAL ${d.probability_normal}%`;
      pred.className   = "file-pred " + (sick ? "pred-sick" : "pred-normal");
      dot.className    = "file-dot "  + (sick ? "dot-sick"  : "dot-normal");
      row.classList.toggle("fr-sick",   sick);
      row.classList.toggle("fr-normal", !sick);
    }
  });
}

// ── Viewer ─────────────────────────────────────────────
function showViewer() {
  viewerPlaceholder.hidden = true;
  slideStage.hidden = false;
  thumbBar.hidden = false;
  prevBtn.disabled = nextBtn.disabled = autoPlayBtn.disabled = heatmapBtn.disabled = false;
  goTo(0);
}

function buildThumbs() {
  thumbStrip.innerHTML = "";
  files.forEach((f, i) => {
    const thumb = document.createElement("div");
    thumb.className = "thumb" + (i === 0 ? " active" : "");
    thumb.dataset.idx = i;
    const img = document.createElement("img");
    img.src = objectURLs[i];
    img.alt = f.name;
    thumb.appendChild(img);
    thumb.addEventListener("click", () => goTo(i));
    thumbStrip.appendChild(thumb);
  });
}

function goTo(idx) {
  currentIdx = idx;
  slideImg.src = objectURLs[idx];
  ovlFilename.textContent = files[idx].name;
  ovlCounter.textContent  = `${idx + 1} / ${files.length}`;
  slideCounter.textContent = `${idx + 1} / ${files.length}`;
  sbSlice.textContent = `Slice ${idx + 1} of ${files.length} — ${files[idx].name}`;

  // Update active states
  document.querySelectorAll(".file-row").forEach((r, i) => r.classList.toggle("active", i === idx));
  document.querySelectorAll(".thumb").forEach((t, i) => t.classList.toggle("active", i === idx));

  // Scroll file row into view
  const activeRow = fileList.children[idx];
  if (activeRow) activeRow.scrollIntoView({ block: "nearest" });

  // Scroll thumb into view
  const activeThumb = thumbStrip.children[idx];
  if (activeThumb) activeThumb.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });

  // Per-slice badge on image
  if (perImageData[idx] && !perImageData[idx].error) {
    const d = perImageData[idx];
    const sick = d.prediction === "CAD Detected";
    sliceResultBadge.textContent = sick ? `⚠ CAD ${d.probability_sick}%` : `✓ NORMAL ${d.probability_normal}%`;
    sliceResultBadge.className   = sick ? "badge-sick" : "badge-normal";
    sliceResultBadge.hidden      = false;
  } else {
    sliceResultBadge.hidden = true;
  }

  prevBtn.disabled = idx === 0;
  nextBtn.disabled = idx === files.length - 1;

  if (heatmapMode) fetchHeatmap(idx);
}

prevBtn.addEventListener("click", () => { if (currentIdx > 0)              goTo(currentIdx - 1); });
nextBtn.addEventListener("click", () => { if (currentIdx < files.length-1) goTo(currentIdx + 1); });

autoPlayBtn.addEventListener("click", () => autoInterval ? stopAutoPlay() : startAutoPlay());

function startAutoPlay() {
  autoPlayBtn.classList.add("active");
  autoPlayBtn.textContent = "⏸ PAUSE";
  autoInterval = setInterval(() => goTo((currentIdx + 1) % files.length), 1200);
}
function stopAutoPlay() {
  clearInterval(autoInterval);
  autoInterval = null;
  autoPlayBtn.classList.remove("active");
  autoPlayBtn.textContent = "▶ PLAY";
}

document.addEventListener("keydown", e => {
  if (slideStage.hidden) return;
  if (e.key === "ArrowLeft")  prevBtn.click();
  if (e.key === "ArrowRight") nextBtn.click();
  if (e.key === " ") { e.preventDefault(); autoPlayBtn.click(); }
});

// ── Analysis ───────────────────────────────────────────
analyseBtn.addEventListener("click", async () => {
  if (!files.length) return;
  stopAutoPlay();
  setLoading(true);
  hideResults();

  const formData = new FormData();
  files.forEach(f => formData.append("files", f));

  try {
    const res  = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok || data.error) {
      showError(data.error || "Unexpected server error.");
    } else {
      perImageData = data.per_image;
      applyThumbBadges();
      updateFileListPredictions();
      showResult(data.aggregate, data.per_image);
      goTo(currentIdx);
    }
  } catch {
    showError("Cannot reach server. Is the Flask app running?");
  } finally {
    setLoading(false);
  }
});

function applyThumbBadges() {
  document.querySelectorAll(".thumb").forEach((thumb, i) => {
    const d = perImageData[i];
    if (!d || d.error) return;
    const sick = d.prediction === "CAD Detected";
    thumb.classList.remove("t-sick", "t-normal");
    thumb.classList.add(sick ? "t-sick" : "t-normal");
    let dot = thumb.querySelector(".thumb-dot");
    if (!dot) { dot = document.createElement("div"); thumb.appendChild(dot); }
    dot.className = "thumb-dot " + (sick ? "d-sick" : "d-normal");
  });
}

// ── Results ────────────────────────────────────────────
function showResult(agg, perImage) {
  const sick = agg.prediction !== "Normal";

  // Diagnosis header
  const hdr = document.getElementById("diagHeader");
  hdr.className = "diag-header " + (sick ? "dh-sick" : "dh-normal");
  document.getElementById("diagIcon").textContent  = sick ? "⚠" : "✓";
  document.getElementById("diagLabel").textContent = agg.prediction.toUpperCase();
  document.getElementById("diagRisk").textContent  =
    `RISK LEVEL: ${agg.risk_level.toUpperCase()}  ·  CONFIDENCE: ${agg.confidence}%`;

  // Metrics
  document.getElementById("metricConfidence").textContent = `${agg.confidence}%`;
  document.getElementById("metricSlices").textContent     = agg.total_slices;
  document.getElementById("metricCad").textContent        = agg.cad_slices;
  document.getElementById("metricNormal").textContent     = agg.normal_slices;

  // Gauges
  document.getElementById("probSick").textContent   = `${agg.probability_sick}%`;
  document.getElementById("probNormal").textContent = `${agg.probability_normal}%`;
  setTimeout(() => {
    document.getElementById("barSick").style.width   = `${agg.probability_sick}%`;
    document.getElementById("barNormal").style.width = `${agg.probability_normal}%`;
  }, 60);

  // Slice breakdown
  const list = document.getElementById("perSliceList");
  list.innerHTML = "";
  perImage.forEach(d => {
    const row = document.createElement("div");
    if (d.error) {
      row.className = "bk-row";
      row.innerHTML = `<div class="bk-dot"></div><span class="bk-name">${d.filename}</span><span class="bk-val" style="color:var(--text-muted)">ERR</span>`;
    } else {
      const s = d.prediction === "CAD Detected";
      row.className = `bk-row ${s ? "bk-sick" : "bk-normal"}`;
      row.innerHTML = `
        <div class="bk-dot ${s ? "d-sick" : "d-normal"}"></div>
        <span class="bk-name" title="${d.filename}">${d.filename}</span>
        <span class="bk-val">${s ? d.probability_sick + "%" : d.probability_normal + "%"}</span>`;
    }
    list.appendChild(row);
  });

  analysisPlaceholder.hidden = true;
  reportTag.hidden = false;
  resultCard.hidden = false;
  errorCard.hidden  = true;
}

// ── Grad-CAM heatmap ───────────────────────────────────
heatmapBtn.addEventListener("click", () => setHeatmapMode(!heatmapMode));

function setHeatmapMode(on) {
  heatmapMode = on;
  heatmapBtn.classList.toggle("active", on);
  heatmapBtn.textContent = on ? "◈ HEATMAP" : "HEATMAP";
  if (!on) {
    slideImg.src = objectURLs[currentIdx] || "";
  } else if (files.length) {
    fetchHeatmap(currentIdx);
  }
}

async function fetchHeatmap(idx) {
  if (heatmapCache[idx]) {
    slideImg.src = heatmapCache[idx];
    return;
  }

  heatmapLoading.hidden = false;
  slideImg.style.opacity = "0.3";

  const fd = new FormData();
  fd.append("file", files[idx]);

  try {
    const res  = await fetch("/gradcam", { method: "POST", body: fd });
    const data = await res.json();
    if (data.heatmap) {
      heatmapCache[idx] = data.heatmap;
      if (heatmapMode && currentIdx === idx) slideImg.src = data.heatmap;
    }
  } catch {
    // silently fall back to original
  } finally {
    heatmapLoading.hidden = true;
    slideImg.style.opacity = "";
  }
}

// ── UI helpers ─────────────────────────────────────────
function setLoading(on) {
  analyseBtn.disabled  = on;
  btnText.textContent  = on ? "PROCESSING…" : "ANALYSE STUDY";
  btnSpinner.hidden    = !on;
}
function hideResults() {
  resultCard.hidden = true;
  errorCard.hidden  = true;
  reportTag.hidden  = true;
  analysisPlaceholder.hidden = false;
}
function showError(msg) {
  errorMsg.textContent = msg;
  errorCard.hidden     = false;
  resultCard.hidden    = true;
  analysisPlaceholder.hidden = true;
}
