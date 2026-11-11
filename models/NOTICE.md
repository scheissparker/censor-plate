# Model provenance

The optional download is MorseTech Lab's YOLO11 nano plate detector:
https://huggingface.co/morsetechlab/yolov11-license-plate-detection
Revision: `251a30d7daedca065f56e04b0af04052c907c68f`
File: `license-plate-finetune-v1n.onnx`

The model is trained for license plate detection, not exclusively German plates.
The app uses only detection with ONNX Runtime; it includes no OCR. Padding,
EXIF handling and configurable thresholds support European/German photo use.
Tests include a German plate photograph and blank-image negative controls.
These checks are not a representative German/European recall evaluation.

The model card and ONNX metadata declare AGPL-3.0. MODEL_LICENSE.txt reproduces
the upstream repository's license:
https://github.com/morsetechlab/Yolov11-License-Plate-Detection
This app does not relabel the weights or grant additional model rights.

The upstream model card documents contamination between its training/test splits.
Its published metrics are not a reliable held-out benchmark; validate on your
own photos. No accuracy or throughput figures are claimed by this application.

Download SHA-256:
`693133a1db97a3ba1e90068986f80afb72c3fcddb681e57181a89a9a3dc351d6`

Weights are not automatically fetched at server startup, sent to a third party,
or included in the Docker image. `download_model.py` is an explicit setup step.
