"use strict";
const $ = (id) => document.getElementById(id);
let queue = [], modelReady = false, busy = false, archiveUrl = null, currentRequest = null;
const photoUrls = [];
const types = new Set(["image/jpeg", "image/png", "image/webp"]);

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}
function renderQueue() {
  $("queue").replaceChildren();
  if (!queue.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No photos selected.";
    $("queue").append(empty);
  }
  queue.forEach((file, index) => {
    const row = document.createElement("li");
    const icon = document.createElement("span");
    icon.className = "file-icon"; icon.textContent = "IMG";
    const name = document.createElement("span");
    name.className = "file-name"; name.textContent = file.name; name.title = file.name;
    const size = document.createElement("span");
    size.className = "file-size"; size.textContent = `${(file.size / 1048576).toFixed(1)} MiB`;
    const remove = document.createElement("button");
    remove.type = "button"; remove.className = "remove"; remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${file.name}`); remove.disabled = busy;
    remove.addEventListener("click", () => { queue.splice(index, 1); renderQueue(); });
    row.append(icon, name, size, remove); $("queue").append(row);
  });
  $("queue-count").textContent = `${queue.length} / 30`;
  $("clear").disabled = busy || !queue.length;
  $("submit").disabled = busy || !queue.length || !modelReady;
}
function addFiles(files) {
  if (busy) return;
  const errors = [];
  for (const file of files) {
    if ((!types.has(file.type) && file.type) || !/\.(jpe?g|png|webp)$/i.test(file.name)) {
      errors.push(`${file.name}: use JPEG, PNG or WebP.`); continue;
    }
    if (file.size > 20 * 1048576) { errors.push(`${file.name}: exceeds 20 MiB.`); continue; }
    if (queue.some((f) => f.name === file.name && f.size === file.size && f.lastModified === file.lastModified)) continue;
    if (queue.length >= 30) { errors.push("A batch can contain up to 30 images."); break; }
    queue.push(file);
  }
  showError(errors.join(" ")); renderQueue();
}
$("files").addEventListener("change", (event) => { addFiles(event.target.files); event.target.value = ""; });
$("clear").addEventListener("click", () => { queue = []; renderQueue(); });
for (const event of ["dragenter", "dragover"]) {
  $("drop-zone").addEventListener(event, (e) => { e.preventDefault(); if (!busy) $("drop-zone").classList.add("dragging"); });
}
for (const event of ["dragleave", "drop"]) {
  $("drop-zone").addEventListener(event, (e) => {
    e.preventDefault(); $("drop-zone").classList.remove("dragging");
    if (event === "drop") addFiles(e.dataTransfer.files);
  });
}
// Prevent accidental navigation if a file lands outside the drop zone.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());
for (const id of ["confidence", "intensity", "padding"]) {
  $(id).addEventListener("input", () => {
    const value = Number($(id).value);
    $(`${id}-value`).textContent = id === "confidence" ? value.toFixed(2) : id === "padding" ? `${Math.round(value * 100)}%` : String(value);
  });
}
function updateSettings() {
  $("intensity").disabled = $("style").value === "blackout";
  $("max-resolution").disabled = !$("downscale").checked;
  $("intensity-hint").textContent = $("style").value === "blur" ? "Gaussian kernel width in pixels; higher means stronger blur." : $("style").value === "pixelate" ? "Higher intensity produces larger mosaic blocks." : "Used by blur and mosaic styles.";
}
$("style").addEventListener("change", updateSettings);
$("downscale").addEventListener("change", updateSettings);
updateSettings();

async function checkHealth() {
  try {
    const response = await fetch("/health", {cache: "no-store"});
    const data = await response.json();
    modelReady = response.ok && data.ready;
    $("model-status").textContent = modelReady ? "Detection model ready" : `Model unavailable: ${data.error || "check server logs"}`;
  } catch { modelReady = false; $("model-status").textContent = "Cannot reach the server. Check that it is running."; }
  $("model-status").classList.toggle("unavailable", !modelReady);
  renderQueue();
}

function clearResults() {
  $("result").hidden = true;
  $("result-grid").replaceChildren();
  $("rejected-files").replaceChildren();
  $("rejected-files").hidden = true;
  $("preview-error").hidden = true;
  $("preview-image").removeAttribute("src");
  $("preview-download").removeAttribute("href");
  $("download-zip").removeAttribute("href");
  if (archiveUrl) URL.revokeObjectURL(archiveUrl);
  archiveUrl = null;
  photoUrls.splice(0).forEach((url) => URL.revokeObjectURL(url));
}

async function readStoredEntries(archive) {
  // main.py writes seekable, uncompressed ZIP entries. Slice JPEGs directly from the Blob.
  const entries = new Map();
  let offset = 0;
  while (offset + 30 <= archive.size) {
    const header = new DataView(await archive.slice(offset, offset + 30).arrayBuffer());
    const signature = header.getUint32(0, true);
    if (signature === 0x02014b50) return entries;
    if (signature !== 0x04034b50 || (header.getUint16(6, true) & 9) || header.getUint16(8, true)) break;
    const length = header.getUint32(18, true);
    const nameStart = offset + 30;
    const nameEnd = nameStart + header.getUint16(26, true);
    const start = nameEnd + header.getUint16(28, true);
    offset = start + length;
    if (offset > archive.size) break;
    const name = await archive.slice(nameStart, nameEnd).text();
    entries.set(name, archive.slice(start, offset, name.endsWith(".jpg") ? "image/jpeg" : ""));
  }
  throw new Error("Unsupported or incomplete result archive");
}

function previewCard(file, blob) {
  const url = URL.createObjectURL(blob);
  photoUrls.push(url);
  const filename = file.output.split("/").pop();
  const card = document.createElement("li");
  const open = document.createElement("button");
  open.type = "button";
  open.className = "photo-preview";
  open.setAttribute("aria-label", "Enlarge " + file.input);
  const image = new Image();
  image.alt = "";
  image.loading = "lazy";
  image.src = url;
  open.append(image);
  const name = document.createElement("strong");
  name.textContent = file.input;
  const status = document.createElement("span");
  if (file.detections === 0) {
    status.className = "review";
    status.textContent = "Review required · No plates detected";
  } else {
    status.textContent = file.detections + (file.detections === 1 ? " plate masked" : " plates masked");
  }
  const download = document.createElement("a");
  download.className = "button secondary";
  download.href = url;
  download.download = filename;
  download.textContent = "Download JPEG";
  download.setAttribute("aria-label", "Download JPEG for " + file.input);
  open.addEventListener("click", () => {
    $("preview-title").textContent = file.input;
    $("preview-image").src = url;
    $("preview-image").alt = "Processed preview of " + file.input;
    $("preview-download").href = url;
    $("preview-download").download = filename;
    $("preview-dialog").showModal();
  });
  card.append(open, name, status, download);
  return card;
}

async function renderResults(xhr) {
  const archive = xhr.response;
  archiveUrl = URL.createObjectURL(archive);
  $("download-zip").href = archiveUrl;
  const count = (name) => xhr.getResponseHeader("X-" + name + "-Count") || "0";
  $("result-summary").textContent = count("Processed") + " processed · " + count("Review") + " with no detections · " + count("Rejected") + " rejected";
  try {
    const entries = await readStoredEntries(archive);
    const manifest = JSON.parse(await entries.get("manifest.json").text());
    for (const file of manifest.files) {
      if (file.status === "rejected") {
        const row = document.createElement("li");
        row.textContent = file.input + ": " + file.error;
        $("rejected-files").append(row);
        $("rejected-files").hidden = false;
      } else {
        $("result-grid").append(previewCard(file, entries.get(file.output)));
      }
    }
  } catch {
    $("preview-error").hidden = false;
  }
  $("result").hidden = false;
  $("result-title").focus();
}

$("preview-dialog").addEventListener("close", () => {
  $("preview-image").removeAttribute("src");
  $("preview-download").removeAttribute("href");
});

$("batch-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy || !queue.length || !modelReady) return;
  const data = new FormData();
  queue.forEach((file) => data.append("files", file, file.name));
  for (const id of ["confidence", "style", "intensity", "padding"]) data.append(id, $(id).value);
  data.append("downscale", String($("downscale").checked));
  data.append("max_resolution", $("max-resolution").value);
  showError("");
  clearResults();
  busy = true; $("settings").disabled = true; $("files").disabled = true; renderQueue();
  $("progress-area").hidden = false; $("progress").value = 0;
  $("progress-label").textContent = "Uploading photos…"; $("progress-value").textContent = "0%";
  const xhr = new XMLHttpRequest(); currentRequest = xhr;
  xhr.open("POST", "/api/process"); xhr.responseType = "blob"; xhr.timeout = 30 * 60 * 1000;
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const percent = Math.round(e.loaded / e.total * 100);
      $("progress").value = percent; $("progress-value").textContent = `${percent}%`;
    }
  };
  xhr.upload.onload = () => {
    $("progress").removeAttribute("value"); $("progress-value").textContent = "";
    $("progress-label").textContent = "Processing photos & preparing ZIP…";
  };
  xhr.onload = async () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      $("progress-label").textContent = "Preparing previews…";
      $("cancel").disabled = true;
      await renderResults(xhr);
    } else {
      try {
        const error = JSON.parse(await xhr.response.text());
        showError(typeof error.detail === "string" ? error.detail : "Invalid settings or upload. Check your files and try again.");
      } catch { showError(`Request failed (${xhr.status}). Check the server or reverse proxy.`); }
    }
    finishRequest();
  };
  xhr.onerror = () => showError("Connection lost. Check the server and try again.");
  xhr.ontimeout = () => showError("The request timed out. Try fewer or smaller photos.");
  xhr.onabort = () => showError("Request cancelled. The server may finish the current batch before accepting another.");
  function finishRequest() {
    busy = false; currentRequest = null; $("settings").disabled = false; $("files").disabled = false;
    $("progress-area").hidden = true; $("cancel").disabled = false;
    updateSettings(); renderQueue();
  }
  xhr.onloadend = () => { if (!xhr.status) finishRequest(); };
  xhr.send(data);
});
$("cancel").addEventListener("click", () => currentRequest?.abort());
checkHealth();
