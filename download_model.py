"""Download pinned plate weights with the Python standard library only."""
import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request

REVISION = "251a30d7daedca065f56e04b0af04052c907c68f"
REPOSITORY = "morsetechlab/yolov11-license-plate-detection"
SHA256 = "693133a1db97a3ba1e90068986f80afb72c3fcddb681e57181a89a9a3dc351d6"
URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/license-plate-finetune-v1n.onnx"


def download(destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with destination.open("rb") as current:
            digest = hashlib.file_digest(current, "sha256").hexdigest() if sys.version_info >= (3, 11) else _digest(current)
        if digest == SHA256:
            print(f"Verified existing model: {destination}")
            return
        raise RuntimeError(f"{destination} already exists with a different checksum. Move it aside to download the pinned model.")
    partial = destination.with_suffix(".onnx.part")
    try:
        checksum = hashlib.sha256()
        request = urllib.request.Request(URL, headers={"User-Agent": "PlateCensor/1.0"})
        size = 0
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 25 * 1024 * 1024:
                    raise RuntimeError("Unexpectedly large model download.")
                checksum.update(chunk)
                target.write(chunk)
        if checksum.hexdigest() != SHA256:
            raise RuntimeError("Model checksum mismatch. The downloaded file was discarded.")
        partial.replace(destination)
        print(f"Verified and saved {size / 1024**2:.1f} MiB: {destination}")
    finally:
        partial.unlink(missing_ok=True)


def _digest(stream):
    checksum = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        checksum.update(chunk)
    return checksum.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "models" / "plate_detector.onnx")
    args = parser.parse_args()
    try:
        download(args.output)
    except Exception as exc:
        parser.exit(1, f"Download failed: {exc}\n")
