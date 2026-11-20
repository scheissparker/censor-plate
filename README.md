## censor-plate

A CPU-only FastAPI app for batches of **up to 30 European/German plate photos**.
The browser provides drag and drop, upload progress, processing state, confidence,
blur/mosaic/blackout, intensity, coverage padding, and output downscaling.
No CDN, PyTorch, OCR, GPU, Node build step, or external image-processing service.

```text
censor_plates/
├── main.py
├── download_model.py
├── requirements.txt
├── requirements-dev.txt
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── app.js
├── models/
│   ├── NOTICE.md
│   ├── MODEL_LICENSE.txt
│   └── plate_detector.onnx   # downloaded; excluded from Git/Docker build
├── tests/
│   └── test_app.py
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Windows: plain Python

Use **64-bit Python 3.10–3.13**; Python 3.12 is recommended and was used for local
verification. Python 3.14/3.15 are not supported by these pinned native wheels.
Open PowerShell in this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_model.py
.\.venv\Scripts\python.exe main.py
```

Open **http://localhost:8000**. Stop with Ctrl+C. No venv activation or PowerShell
execution-policy change is necessary. With an activated venv, `python main.py`
does the same thing. If ONNX Runtime reports a missing DLL, install Microsoft's
[Visual C++ 2015–2022 x64 redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

The model downloader uses only the standard library. It fetches a pinned ~10 MiB
YOLO11 nano ONNX file and verifies SHA-256. Internet is required for initial
package/model downloads; processing works offline. Missing or incompatible
weights keep the UI available with an error; `/health` returns 503 and processing
is disabled. Restart after installing/changing the model.

## Ubuntu ARM64 or x86-64: Docker Compose

Install Docker Engine with the Compose v2 plugin. On Ubuntu, download the weights
with system Python (no pip installation needed), then build and start:

```bash
mkdir -p models
python3 download_model.py
docker compose up -d --build
docker compose logs -f app
```

The same Dockerfile builds natively on **linux/amd64 and linux/arm64 (aarch64)**,
including a Raspberry Pi running a 64-bit Ubuntu OS. 32-bit ARM is unsupported.
The image uses `python:3.10-slim` and headless OpenCV with only `libgomp1` added.
The pinned NumPy, OpenCV, Pillow and ONNX Runtime releases publish Linux ARM64
wheels; no native ML compilation is intended. Allow more build/startup time on
small ARM machines. Docker builds/ARM execution require validation on your host.

Compose applies 2 CPU / 2 GiB limits, disables swap beyond that limit, runs as a
non-root user with a read-only root filesystem, mounts weights read-only, and
uses a disk-backed named volume for temporary data. **Keep at least 3 GiB free
on the Docker volume filesystem**; large full-resolution outputs can need more.
Temporary image data is removed after successful downloads and handled errors.
Forced termination can leave multipart/result temp files on the volume; to purge
them after stopping the app: `docker compose down -v`. This removes its temporary
volume, not the host `models` directory. It is not secure erasure.

The port binds to server loopback. To use it from your Windows PC:

```bash
ssh -N -L 8000:127.0.0.1:8000 your-user@your-server
```

Then open http://localhost:8000 on your PC. For a public/shared deployment, put an
authenticated HTTPS reverse proxy in front. This app has no built-in accounts.
Set the proxy request-body limit to at least **601 MiB**, allow long request
timeouts (e.g. 30 minutes), and preserve the original Host header for same-origin
checks. It is deliberately a small, single-worker service, not a public upload
service with user isolation or distributed job storage.

Optional multi-architecture publishing (requires Docker Buildx and a registry):

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t YOUR_REGISTRY/plate-censor:1.0 --push .
```

Model weights remain a runtime mount and are not in the published image.

## Using the app

1. Choose or drop up to 30 JPEG, PNG, or still WebP files, each at most 20 MiB.
2. Choose the style. Blackout is the default; blur/mosaic may leave text readable.
3. Lower confidence to detect more candidates; higher values reduce false positives.
4. Choose output downscaling or preserve the decoded original dimensions.
5. Click **Censor photos**, then review the processed photos in the results grid. Select a preview to enlarge it, download a single JPEG, or download the whole batch as a ZIP.

The upload percentage measures bytes sent. Processing uses an indeterminate
indicator; there is no background job queue or persistent server gallery.
Cancelling disconnects the request; an already-running batch may finish first.

The ZIP contains:

- `censored/`: JPEGs with one or more detected regions censored.
- `review_required/`: JPEGs where no plates were detected; they can still contain plates.
- `manifest.json`: input/output names, detection counts, dimensions, options, and rejected inputs.
- `READ_ME.txt`: review reminder.

Corrupt/unsupported/oversized files are omitted and reported in the manifest;
other valid files still finish. Internal inference or storage errors abort the
request rather than producing a seemingly successful partial result. Filename
paths are sanitized, and sequence prefixes avoid collisions. All outputs apply
EXIF rotation, flatten transparency on white, and strip source EXIF/GPS metadata.
Animated images, HEIC, AVIF, GIF, TIFF and RAW files are not supported.

**Inspect every image before sharing.** There is no guarantee of complete plate
removal, including in `censored/`. Small, distant, angled or occluded plates can
be missed. Output resizing does not increase the detector input resolution.
Gaussian kernel size is an absolute pixel width; large plates can need blackout.

## Model and CPU configuration

The included downloader selects the
[MorseTech Lab YOLO11 nano plate model](https://huggingface.co/morsetechlab/yolov11-license-plate-detection).
The model is a general plate detector; the app is configured for European/German
photos with padding, EXIF support and adjustable thresholds. A German photograph
regression test is included, but there is no measured German-plate recall.
See [model provenance and AGPL-3.0 terms](models/NOTICE.md). The upstream model
card documents train/test contamination, so its metrics are not a reliable
held-out benchmark. Validate with your own representative photos.

Custom weights must be **single-class plate detectors**, float32 NCHW ONNX input
`[1,3,H,W]`, with raw output `[1,5,N]` or `[1,N,5]` containing pixel-space
`cx,cy,w,h,confidence`. Export with `batch=1`, `half=False`, `nms=False` and
preferably `imgsz=640`. YOLOv5/objectness outputs, COCO models, end-to-end/NMS
exports, normalized boxes and multiclass models are unsupported. Shape checks
cannot prove model quality or class semantics. Use trusted weights only.

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `models/plate_detector.onnx` beside main.py | Alternative trusted weights |
| `ORT_THREADS` | `2` | ONNX intra-op threads (1–32); match CPU budget |
| `MAX_IMAGE_PIXELS` | `40000000` | Decoded source pixel limit, checked before decoding |
| `HOST` | `127.0.0.1` (`0.0.0.0` in image) | Listen address |
| `PORT` | `8000` | Listen port |
| `TMPDIR` (Linux), `TEMP` (Windows) | OS default | Writable disk for upload/result temp files |

The pixel limit includes images that will be downscaled: increasing it raises
peak memory. A 48 MP phone photo exceeds the default; export a smaller JPEG or
increase the limit together with the memory budget. OpenCV and BLAS/OpenMP
threads are fixed to one; ONNX uses sequential execution, a bounded intra-op
pool, and no thread spinning. Inference is warmed up once during lifespan startup.
Only one batch is admitted, including upload/download; another receives 429.
Health/UI requests remain responsive while CPU work runs in a worker thread.

Uploads are first spooled to disk under a 601 MiB raw-body limit, including
chunked requests, then parsed as `list[UploadFile]`. Only one decoded image is
held at a time. The raw spool is discarded before inference. ZIPs are built on
disk with `ZIP_STORED` (JPEGs are already compressed), streamed in chunks, and
deleted afterwards, including handled download failures. Browser downloads use
a Blob, so the browser needs enough memory for the final archive.

Run **one Uvicorn worker**. Additional workers each load their own model and
multiply admission/memory limits. For direct Uvicorn invocation use
`uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1`.
Thread management follows [ONNX Runtime's documentation](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover multipart batches, ZIP/report behavior, bad files, settings and size
limits, EXIF orientation/metadata, letterbox/NMS coordinate mapping, all censorship
styles, admission and temporary cleanup. A real model smoke test runs when the
weights exist. Blank-image controls and a German photograph regression check
basic behavior; they are not a representative accuracy benchmark.

GitHub Actions runs the suite on Windows/Python 3.12 and Ubuntu/Python 3.10,
including a pinned-model download, and builds both Docker architectures.
