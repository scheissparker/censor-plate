import asyncio
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

import main


def photo(size=(160, 100), exif=None):
    image = Image.new("RGB", size, "#d0d8e0")
    output = io.BytesIO()
    image.save(output, "JPEG", exif=exif or b"")
    return output.getvalue()


class FakeDetector:
    def detect(self, image, confidence):
        return [(30, 35, 90, 60)]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "PlateDetector", lambda path: FakeDetector())
    with TestClient(main.app) as instance:
        yield instance


def unzip(response):
    assert response.status_code == 200, response.text if response.status_code != 200 else ""
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    return archive, json.loads(archive.read("manifest.json"))


def test_batch_30_files_duplicate_names_blackout_and_cleanup(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main.tempfile, "tempdir", str(tmp_path))
    files = [("files", ("../../car.jpg", photo(), "image/jpeg")) for _ in range(30)]
    response = client.post("/api/process", files=files, data={"downscale": "false"})
    archive, manifest = unzip(response)
    assert len(manifest["files"]) == 30
    outputs = [entry["output"] for entry in manifest["files"]]
    assert len(set(outputs)) == 30
    assert all(name.startswith("censored/") and ".." not in name for name in outputs)
    image = np.array(Image.open(io.BytesIO(archive.read(outputs[0]))))
    assert image[45:50, 45:65].max() < 10
    assert response.headers["X-Processed-Count"] == "30"
    assert response.headers["X-Rejected-Count"] == "0"
    assert list(tmp_path.iterdir()) == []


def test_bad_images_reported_and_no_detection_review(client):
    main.app.state.detector = SimpleNamespace(detect=lambda *_: [])
    response = client.post("/api/process", files=[
        ("files", ("ok.jpg", photo(), "application/octet-stream")),
        ("files", ("bad.jpg", b"not an image", "image/jpeg")),
    ])
    archive, manifest = unzip(response)
    assert manifest["files"][0]["status"] == "no_plates_detected"
    assert manifest["files"][0]["output"].startswith("review_required/")
    assert manifest["files"][1]["status"] == "rejected"
    assert response.headers["X-Review-Count"] == "1"
    assert response.headers["X-Rejected-Count"] == "1"
    assert len(archive.namelist()) == 3


@pytest.mark.parametrize("settings", [{"confidence": "nan"}, {"confidence": "0.99"},
                                     {"intensity": "0"}, {"style": "unknown"},
                                     {"padding": "inf"}, {"max_resolution": "99999"}])
def test_invalid_settings(client, settings):
    assert client.post("/api/process", files={"files": ("car.jpg", photo())}, data=settings).status_code == 422


def test_file_count_limit(client):
    response = client.post("/api/process", files=[("files", ("a.jpg", photo()))] * 31)
    assert response.status_code == 400


def test_size_limits(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_FILE_BYTES", 10)
    _, manifest = unzip(client.post("/api/process", files={"files": ("a.jpg", photo())}))
    assert manifest["files"][0]["status"] == "rejected"
    monkeypatch.setattr(main, "MAX_REQUEST_BYTES", 50)
    assert client.post("/api/process", content=b"a" * 51).status_code == 413
    assert client.get("/health").status_code == 200


def test_model_missing_and_security_headers(client):
    main.app.state.detector = None
    main.app.state.model_error = "Model missing"
    assert client.get("/health").status_code == 503
    assert client.post("/api/process", files={"files": ("a.jpg", photo())}).status_code == 503
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert client.post("/api/process", headers={"Origin": "https://elsewhere.example"}).status_code == 403


def test_processing_failure_cleans_files(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main.tempfile, "tempdir", str(tmp_path))
    def broken(*_):
        raise RuntimeError("inference failure")
    main.app.state.detector = SimpleNamespace(detect=broken)
    response = client.post("/api/process", files={"files": ("a.jpg", photo())})
    assert response.status_code == 500
    assert list(tmp_path.iterdir()) == []
    main.app.state.detector = FakeDetector()
    assert client.post("/api/process", files={"files": ("a.jpg", photo())}).status_code == 200


def test_orientation_and_metadata(client):
    exif = Image.Exif()
    exif[274] = 6  # 90 degrees clockwise
    exif[270] = "private metadata"
    response = client.post("/api/process", files={"files": ("rotated.jpg", photo((160, 100), exif))})
    archive, manifest = unzip(response)
    with Image.open(io.BytesIO(archive.read(manifest["files"][0]["output"]))) as output:
        assert output.size == (100, 160)
        assert not output.getexif()


def test_pixel_limit_checked_before_decode(monkeypatch):
    monkeypatch.setattr(main, "MAX_PIXELS", 100)
    with pytest.raises(main.InvalidImage, match="pixel"):
        main.decode_image(io.BytesIO(photo()), main.Options())


def test_downscale_and_transparent_background():
    image = Image.new("RGBA", (1500, 800), (100, 100, 100, 0))
    stream = io.BytesIO()
    image.save(stream, "PNG")
    result = main.decode_image(stream, main.Options(max_resolution=640))
    assert result.shape == (341, 640, 3)
    assert result.min() == 255


@pytest.mark.parametrize("style", ["blackout", "blur", "pixelate"])
def test_styles_modify_only_padded_region(style):
    image = np.random.default_rng(0).integers(0, 256, (100, 160, 3), dtype=np.uint8)
    before = image.copy()
    main.censor(image, [(40, 30, 100, 60)], main.Options(style=style))
    assert not np.array_equal(image[30:60, 40:100], before[30:60, 40:100])
    assert np.array_equal(image[:20], before[:20])
    if style == "blackout":
        assert image[30:60, 40:100].max() == 0


def test_letterbox_nms_and_model_contract():
    detector = object.__new__(main.PlateDetector)
    detector.width = detector.height = 640
    detector.input_name = "images"
    # 1280x640 source: scale .5, top pad 160. Box in source is 200,100,600,300.
    predictions = np.array([[[200, 200, 400], [260, 260, 260], [200, 200, 40],
                             [100, 100, 40], [.9, .8, .05]]], dtype=np.float32)
    def inference(_, inputs):
        assert inputs["images"].shape == (1, 3, 640, 640)
        assert inputs["images"].dtype == np.float32
        return [predictions]
    detector.session = SimpleNamespace(run=inference)
    result = detector.detect(np.zeros((640, 1280, 3), dtype=np.uint8), .3)
    assert result == [(200, 100, 600, 300)]
    assert detector._rows(predictions.transpose(0, 2, 1)).shape == (3, 5)
    with pytest.raises(ValueError, match="single-class"):
        detector._rows(np.zeros((1, 84, 8400)))


def test_guard_chunked_limit_and_admission(monkeypatch):
    monkeypatch.setattr(main, "MAX_REQUEST_BYTES", 5)
    async def scenario():
        called = []
        async def downstream(*_):
            called.append(True)
        guard = main.BatchGuard(downstream)
        scope = {"type": "http", "method": "POST", "path": "/api/process", "headers": []}
        messages = iter([{"type": "http.request", "body": b"123", "more_body": True},
                         {"type": "http.request", "body": b"456", "more_body": False}])
        output = []
        async def receive():
            return next(messages)
        async def send(message):
            output.append(message)
        await guard(scope, receive, send)
        assert output[0]["status"] == 413
        assert not called and not guard.busy
        guard.busy = True
        output.clear()
        await guard(scope, receive, send)
        assert output[0]["status"] == 429
    asyncio.run(scenario())


def test_zip_removed_on_send_failure(tmp_path):
    directory = tmp_path / "results"
    directory.mkdir()
    archive = directory / "test.zip"
    archive.write_bytes(b"test")
    response = main.TemporaryZipResponse(archive, directory)
    async def run():
        async def send(_):
            raise OSError("client disconnected")
        with pytest.raises(OSError):
            await response({"type": "http", "method": "GET", "headers": []}, None, send)
    asyncio.run(run())
    assert not directory.exists()


@pytest.mark.skipif(not main.MODEL_PATH.is_file(), reason="Download the optional model to run real inference")
def test_real_model_smoke():
    detector = main.PlateDetector(main.MODEL_PATH)
    for color in (0, 114, 255):
        assert detector.detect(np.full((480, 640, 3), color, dtype=np.uint8), .3) == []


@pytest.mark.skipif(not main.MODEL_PATH.is_file(), reason="Download the optional model to run real inference")
def test_real_german_plate_end_to_end():
    photo_path = Path(__file__).parent / "assets" / "german-car.jpg"
    with TestClient(main.app) as client:
        response = client.post("/api/process", files={"files": ("german-car.jpg", photo_path.read_bytes(), "image/jpeg")})
    archive, manifest = unzip(response)
    entry = manifest["files"][0]
    assert entry["status"] == "censored" and entry["detections"] >= 1
    with Image.open(io.BytesIO(archive.read(entry["output"]))) as result:
        pixels = np.array(result)
        # The annotated center of the visible German plate must be blacked out.
        assert pixels[620:675, 450:590].max() < 10
        assert pixels[100:200, 100:200].std() > 10  # retain the surrounding scene
