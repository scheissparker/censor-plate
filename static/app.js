"use strict";
const $ = (id) => document.getElementById(id);
let queue = [], modelReady = false, busy = false, archiveUrl = null, currentRequest = null;
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
    empty.textContent = "Your next batch starts here.";
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
    $("model-status").textContent = modelReady ? "● Plate detection model ready · CPU only" : `Model unavailable: ${data.error || "check server logs"}`;
  } catch { modelReady = false; $("model-status").textContent = "Cannot reach the server. Check that it is running."; }
  $("model-status").classList.toggle("unavailable", !modelReady);
  renderQueue();
}

$("batch-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy || !queue.length || !modelReady) return;
  const data = new FormData();
  queue.forEach((file) => data.append("files", file, file.name));
  for (const id of ["confidence", "style", "intensity", "padding"]) data.append(id, $(id).value);
  data.append("downscale", String($("downscale").checked));
  data.append("max_resolution", $("max-resolution").value);
  showError(""); $("result").hidden = true;
  if (archiveUrl) { URL.revokeObjectURL(archiveUrl); archiveUrl = null; }
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
      archiveUrl = URL.createObjectURL(xhr.response);
      const processed = xhr.getResponseHeader("X-Processed-Count") || "0";
      const rejected = xhr.getResponseHeader("X-Rejected-Count") || "0";
      const review = xhr.getResponseHeader("X-Review-Count") || "0";
      $("result").replaceChildren();
      const message = document.createElement("div");
      message.textContent = `Your ZIP is ready. ${processed} images processed · ${rejected} rejected · ${review} with no detections. Inspect all results before sharing.`;
      const link = document.createElement("a");
      link.href = archiveUrl; link.download = "censored-plates.zip"; link.textContent = "Download censored photos ↓";
      $("result").append(message, link); $("result").hidden = false;
    } else {
      try {
        const error = JSON.parse(await xhr.response.text());
        showError(typeof error.detail === "string" ? error.detail : "Invalid settings or upload. Check your files and try again.");
      } catch { showError(`Request failed (${xhr.status}). Check the server or reverse proxy.`); }
    }
  };
  xhr.onerror = () => showError("Connection lost. Check the server and try again.");
  xhr.ontimeout = () => showError("The request timed out. Try fewer or smaller photos.");
  xhr.onabort = () => showError("Request cancelled. The server may finish the current batch before accepting another.");
  xhr.onloadend = () => {
    busy = false; currentRequest = null; $("settings").disabled = false; $("files").disabled = false;
    $("progress-area").hidden = true; updateSettings(); renderQueue();
  };
  xhr.send(data);
});
$("cancel").addEventListener("click", () => currentRequest?.abort());
window.addEventListener("beforeunload", () => { if (archiveUrl) URL.revokeObjectURL(archiveUrl); });
checkHealth();
