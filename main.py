"""CPU-only plate censorship. Run with `python main.py` (one worker)."""
from __future__ import annotations

import os

# Set these BEFORE importing NumPy, OpenCV or ONNX Runtime.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import io
import json
import logging
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Annotated, Literal
from urllib.parse import urlsplit
import warnings
import zipfile

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "plate_detector.onnx")))
MAX_FILES = 30
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_FILES * MAX_FILE_BYTES + 1024 * 1024
MAX_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
ORT_THREADS = int(os.getenv("ORT_THREADS", "2"))
if MAX_PIXELS < 1 or not 1 <= ORT_THREADS <= 32:
    raise ValueError("MAX_IMAGE_PIXELS must be positive; ORT_THREADS must be 1–32.")
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
logger = logging.getLogger("plate_censor")


class InvalidImage(ValueError):
    pass


@dataclass(frozen=True)
class Options:
    confidence: float = 0.30
    style: str = "blackout"
    intensity: int = 61
    downscale: bool = True
    max_resolution: int = 2048
    padding: float = 0.15


class PlateDetector:
    """Single-class YOLOv8/11/12 raw ONNX output, WITHOUT embedded NMS.

    Explicitly reject COCO/multiclass and other output formats instead of
    silently treating cars, people, etc. as license plates.
    """

    def __init__(self, path: Path):
        if not path.is_file():
            raise ValueError("Model missing. Run: python download_model.py, then restart.")
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = ORT_THREADS
        settings.inter_op_num_threads = 1
        settings.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        settings.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        settings.add_session_config_entry("session.intra_op.allow_spinning", "0")
        settings.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(path), sess_options=settings, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise ValueError("Model must have one float32 NCHW input.")
        self.input_name = inputs[0].name
        shape = inputs[0].shape
        if len(shape) != 4 or shape[1] != 3 or (isinstance(shape[0], int) and shape[0] != 1):
            raise ValueError("Model input must have shape [1, 3, height, width].")
        self.height = shape[2] if isinstance(shape[2], int) else 640
        self.width = shape[3] if isinstance(shape[3], int) else 640
        if not (32 <= self.height <= 1280 and 32 <= self.width <= 1280):
            raise ValueError("Model input dimensions must be between 32 and 1280.")
        if len(outputs) != 1:
            raise ValueError("Use a single-class YOLO raw export with nms=False.")
        # Warm up AND validate the actual output once at startup.
        warmup = np.zeros((1, 3, self.height, self.width), dtype=np.float32)
        self._rows(self.session.run(None, {self.input_name: warmup})[0])

    @staticmethod
    def _rows(output: np.ndarray) -> np.ndarray:
        if output.ndim != 3 or output.shape[0] != 1:
            raise ValueError("Unsupported model output: expected [1, 5, N] or [1, N, 5].")
        if output.shape[1] == 5:
            rows = output[0].T
        elif output.shape[2] == 5:
            rows = output[0]
        else:
            raise ValueError(
                "Model is not a single-class raw YOLO plate detector. "
                "Expected 5 values (cx, cy, w, h, plate confidence) per prediction."
            )
        return rows

    def detect(self, image: np.ndarray, confidence: float) -> list[tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        scale = min(self.width / width, self.height / height)
        resized_w, resized_h = round(width * scale), round(height * scale)
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        left, top = (self.width - resized_w) // 2, (self.height - resized_h) // 2
        letterbox = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        letterbox[top:top + resized_h, left:left + resized_w] = resized
        tensor = np.ascontiguousarray(letterbox[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= 255.0
        rows = self._rows(self.session.run(None, {self.input_name: tensor})[0])
        rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] >= confidence)
                    & (rows[:, 2] > 0) & (rows[:, 3] > 0)]
        if not len(rows):
            return []
        # Bound NMS work even with pathological model outputs.
        rows = rows[np.argsort(rows[:, 4])[::-1][:3000]]
        boxes = np.column_stack((rows[:, 0] - rows[:, 2] / 2,
                                 rows[:, 1] - rows[:, 3] / 2, rows[:, 2:4]))
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), rows[:, 4].tolist(), confidence, 0.45)
        result = []
        for index in np.asarray(indices).reshape(-1)[:300]:
            x, y, w, h = boxes[index]
            # Undo rounded resize dimensions independently to avoid edge drift.
            x1 = int(np.floor((x - left) * width / resized_w))
            y1 = int(np.floor((y - top) * height / resized_h))
            x2 = int(np.ceil((x + w - left) * width / resized_w))
            y2 = int(np.ceil((y + h - top) * height / resized_h))
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                result.append((x1, y1, x2, y2))
        return result


def decode_image(stream, options: Options) -> np.ndarray:
    """Check headers BEFORE allocating image pixels; honor smartphone EXIF rotation."""
    stream.seek(0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(stream) as original:
                if original.format not in {"JPEG", "PNG", "WEBP"}:
                    raise InvalidImage("Only JPEG, PNG and still WebP images are supported.")
                if original.width * original.height > MAX_PIXELS:
                    raise InvalidImage(f"Image exceeds the {MAX_PIXELS:,}-pixel safety limit.")
                if getattr(original, "n_frames", 1) != 1:
                    raise InvalidImage("Animated or multi-frame images are not supported.")
                # JPEG draft decoding can save substantial memory before full decode.
                if options.downscale:
                    original.draft("RGB", (options.max_resolution, options.max_resolution))
                with ImageOps.exif_transpose(original) as oriented:
                    if options.downscale:
                        oriented.thumbnail((options.max_resolution, options.max_resolution),
                                           Image.Resampling.LANCZOS)
                    with oriented.convert("RGBA") as rgba:
                        with Image.new("RGB", rgba.size, "white") as rgb:
                            rgb.paste(rgba, mask=rgba.getchannel("A"))
                            return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImage("Invalid, truncated, or oversized image.") from exc


def censor(image: np.ndarray, boxes: list[tuple[int, int, int, int]], options: Options) -> None:
    height, width = image.shape[:2]
    for x1, y1, x2, y2 in boxes:
        pad_x = max(2, int(np.ceil((x2 - x1) * options.padding)))
        pad_y = max(2, int(np.ceil((y2 - y1) * options.padding)))
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
        region = image[y1:y2, x1:x2]
        if options.style == "blackout":
            region[:] = 0
        elif options.style == "blur":
            kernel = options.intensity | 1  # Gaussian kernels must be odd.
            region[:] = cv2.GaussianBlur(region, (kernel, kernel), 0)
        else:
            block = max(4, options.intensity // 2)
            small = cv2.resize(region, (max(1, region.shape[1] // block),
                                        max(1, region.shape[0] // block)),
                               interpolation=cv2.INTER_AREA)
            region[:] = cv2.resize(small, (region.shape[1], region.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)


def safe_stem(filename: str) -> str:
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(basename).stem).strip("_")[:80] or "image"


class TemporaryZipResponse(FileResponse):
    """Delete the archive even when a client disconnects during download."""

    def __init__(self, path: Path, directory: Path, **kwargs):
        super().__init__(path, **kwargs)
        self.directory = directory

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await run_in_threadpool(shutil.rmtree, self.directory, True)


class BatchGuard:
    """Admit ONE batch before multipart parsing; bound chunked bodies too.

    Raw uploads spool to disk, not RAM. Replaying them into the multipart parser
    costs extra disk I/O but ensures rejected requests cannot fill parser temp
    files first. The raw spool is closed before image processing starts.
    """

    def __init__(self, app):
        self.app = app
        self.busy = False

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        response_started = False

        async def secure_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                message["headers"] = list(message.get("headers", [])) + [
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"content-security-policy", b"default-src 'self'; script-src 'self'; "
                     b"style-src 'self'; img-src 'self' blob:; connect-src 'self'; "
                     b"frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
                ]
            await send(message)

        if scope["method"] != "POST" or scope["path"] != "/api/process":
            return await self.app(scope, receive, secure_send)
        headers = dict(scope.get("headers", []))

        async def reject(status, detail, extra=None):
            await JSONResponse({"detail": detail}, status_code=status, headers=extra)(
                scope, receive, secure_send)

        # Prevent a third-party web page from submitting batches to a local server.
        origin = headers.get(b"origin")
        if origin:
            try:
                parsed = urlsplit(origin.decode("latin1"))
            except ValueError:
                return await reject(403, "Invalid upload origin.")
            if parsed.netloc.lower() != headers.get(b"host", b"").decode("latin1").lower():
                return await reject(403, "Cross-origin uploads are not allowed.")
        if self.busy:
            return await reject(429, "Another batch is active. Please try again shortly.",
                                {"Retry-After": "5"})
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await reject(400, "Invalid Content-Length.")
        if length < 0 or length > MAX_REQUEST_BYTES:
            return await reject(413, "Batch exceeds the 600 MiB upload limit.")
        # No await between testing and setting: atomic within this event loop.
        self.busy = True
        try:
            with tempfile.TemporaryFile(prefix="plate-upload-") as spool:
                size = 0
                started = time.monotonic()
                while True:
                    if time.monotonic() - started > 600:
                        return await reject(408, "Upload timed out.")
                    try:
                        message = await asyncio.wait_for(receive(), timeout=60)
                    except asyncio.TimeoutError:
                        return await reject(408, "Upload timed out.")
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    size += len(chunk)
                    if size > MAX_REQUEST_BYTES:
                        return await reject(413, "Batch exceeds the 600 MiB upload limit.")
                    await run_in_threadpool(spool.write, chunk)
                    if not message.get("more_body", False):
                        break
                spool.seek(0)
                replayed = 0

                async def replay():
                    nonlocal replayed
                    if replayed >= size:
                        return await receive()
                    data = await run_in_threadpool(spool.read, min(1024 * 1024, size - replayed))
                    replayed += len(data)
                    more = replayed < size
                    if not more:
                        spool.close()
                    return {"type": "http.request", "body": data, "more_body": more}

                if not size:
                    return await reject(400, "Upload body is empty.")
                await self.app(scope, replay, secure_send)
        except OSError:
            logger.exception("Temporary storage failure")
            if not response_started:
                return await reject(507, "Temporary storage unavailable. Check server disk space.")
            # A failed download after headers have been sent must end the stream.
            raise
        finally:
            self.busy = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = None
    app.state.model_error = None
    try:
        app.state.detector = await run_in_threadpool(PlateDetector, MODEL_PATH)
        logger.info("Plate model loaded on CPU with %s ORT threads", ORT_THREADS)
    except Exception as exc:
        logger.exception("Model unavailable")
        app.state.model_error = str(exc)
    yield
    app.state.detector = None


app = FastAPI(title="Plate Censor", lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(BatchGuard)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8"))


@app.get("/static/app.js", include_in_schema=False)
def javascript():
    return FileResponse(BASE_DIR / "static" / "app.js", media_type="text/javascript")


@app.get("/static/style.css", include_in_schema=False)
def stylesheet():
    return FileResponse(BASE_DIR / "static" / "style.css", media_type="text/css")


@app.get("/health")
def health(request: Request):
    ready = request.app.state.detector is not None
    return JSONResponse({"ready": ready, "model": MODEL_PATH.name,
                         "error": request.app.state.model_error,
                         "max_files": MAX_FILES, "max_file_mb": MAX_FILE_BYTES // 1024**2,
                         "max_pixels": MAX_PIXELS}, status_code=200 if ready else 503)


@app.post("/api/process")
def process_batch(
    request: Request,
    files: Annotated[list[UploadFile], File()],
    confidence: Annotated[float, Form(ge=0.1, le=0.9, allow_inf_nan=False)] = 0.30,
    style: Annotated[Literal["blur", "pixelate", "blackout"], Form()] = "blackout",
    intensity: Annotated[int, Form(ge=15, le=151)] = 61,
    downscale: Annotated[bool, Form()] = True,
    max_resolution: Annotated[int, Form(ge=640, le=4096)] = 2048,
    padding: Annotated[float, Form(ge=0, le=0.5, allow_inf_nan=False)] = 0.15,
):
    detector = request.app.state.detector
    if detector is None:
        raise HTTPException(503, "Model unavailable. Check /health and server logs; restart after installing weights.")
    if not 1 <= len(files) <= MAX_FILES:
        raise HTTPException(400, f"Select between 1 and {MAX_FILES} images.")
    options = Options(confidence, style, intensity, downscale, max_resolution, padding)
    directory = Path(tempfile.mkdtemp(prefix="plate-results-"))
    archive = directory / "censored-plates.zip"
    success = False
    manifest = []
    try:
        # JPEGs are compressed already. ZIP_STORED saves CPU and memory.
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
            for index, upload in enumerate(files, start=1):
                entry = {"input": (upload.filename or "image")[:255]}
                try:
                    upload.file.seek(0, io.SEEK_END)
                    if upload.file.tell() > MAX_FILE_BYTES:
                        raise InvalidImage("File exceeds 20 MiB.")
                    image = decode_image(upload.file, options)
                    try:
                        boxes = detector.detect(image, options.confidence)
                        censor(image, boxes, options)
                        encoded_ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if not encoded_ok:
                            raise RuntimeError("JPEG encoding failed")
                        group = "censored" if boxes else "review_required"
                        name = f"{group}/{index:03d}_{safe_stem(upload.filename or 'image')}.jpg"
                        try:
                            output.writestr(name, encoded.tobytes())
                        finally:
                            del encoded
                        entry.update(status="censored" if boxes else "no_plates_detected",
                                     output=name, detections=len(boxes),
                                     width=image.shape[1], height=image.shape[0])
                    finally:
                        del image
                except InvalidImage as exc:
                    entry.update(status="rejected", error=str(exc))
                # Inference/storage failures abort the batch instead of returning
                # a successful-looking archive containing uncensored failures.
                finally:
                    upload.file.close()
                manifest.append(entry)
            output.writestr("manifest.json", json.dumps({
                "notice": "Inspect every output before sharing. No detection is NOT proof that no plate exists.",
                "options": options.__dict__, "files": manifest,
            }, indent=2, ensure_ascii=False))
            output.writestr("READ_ME.txt", "Inspect ALL images before sharing.\n"
                            "review_required/: no plates detected; these images may still contain readable plates.\n"
                            "Rejected inputs are omitted. See manifest.json.\n"
                            "JPEG outputs have orientation applied and EXIF/GPS metadata removed.\n")
        response = TemporaryZipResponse(
            archive, directory, media_type="application/zip", filename="censored-plates.zip",
            headers={"X-Processed-Count": str(sum(e["status"] != "rejected" for e in manifest)),
                     "X-Rejected-Count": str(sum(e["status"] == "rejected" for e in manifest)),
                     "X-Review-Count": str(sum(e["status"] == "no_plates_detected" for e in manifest))},
        )
        success = True
        return response
    except Exception as exc:
        logger.exception("Batch processing failed")
        raise HTTPException(500, "Processing failed. No archive returned. Check server logs and temporary disk space.") from exc
    finally:
        for upload in files:
            upload.file.close()
        if not success:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")),
                workers=1, timeout_keep_alive=5)
