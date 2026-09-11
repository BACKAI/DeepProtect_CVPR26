#!/usr/bin/env python3
"""Align raw face images to the 1024x1024 FFHQ/e4e coordinate system."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from deep_protect.utils import iter_image_paths


def align_face(path: Path, predictor, output_size: int = 1024):
    import dlib
    import scipy.ndimage

    image = dlib.load_rgb_image(str(path))
    detector = dlib.get_frontal_face_detector()
    detections = detector(image, 1)
    if not detections:
        raise RuntimeError("no face detected")
    detection = max(detections, key=lambda box: box.width() * box.height())
    landmarks = predictor(image, detection)
    lm = np.asarray([[point.x, point.y] for point in landmarks.parts()], dtype=np.float32)

    eye_left = lm[36:42].mean(axis=0)
    eye_right = lm[42:48].mean(axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left, mouth_right = lm[48], lm[54]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    pil = Image.open(path).convert("RGB")
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        pil = pil.resize((int(round(pil.width / shrink)), int(round(pil.height / shrink))), Image.Resampling.LANCZOS)
        quad /= shrink
        qsize /= shrink

    border = max(int(round(qsize * 0.1)), 3)
    crop = (
        int(np.floor(quad[:, 0].min())) - border,
        int(np.floor(quad[:, 1].min())) - border,
        int(np.ceil(quad[:, 0].max())) + border,
        int(np.ceil(quad[:, 1].max())) + border,
    )
    crop = (max(crop[0], 0), max(crop[1], 0), min(crop[2], pil.width), min(crop[3], pil.height))
    if crop[2] - crop[0] < pil.width or crop[3] - crop[1] < pil.height:
        pil = pil.crop(crop)
        quad -= crop[:2]

    pad = (
        max(-int(np.floor(quad[:, 0].min())) + border, 0),
        max(-int(np.floor(quad[:, 1].min())) + border, 0),
        max(int(np.ceil(quad[:, 0].max())) - pil.width + border, 0),
        max(int(np.ceil(quad[:, 1].max())) - pil.height + border, 0),
    )
    if max(pad) > border - 4:
        pad = tuple(np.maximum(pad, int(round(qsize * 0.3))))
        array = np.pad(np.asarray(pil, dtype=np.float32), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect")
        h, w, _ = array.shape
        yy, xx = np.ogrid[:h, :w]
        mask = np.maximum(
            1.0 - np.minimum(xx / max(pad[0], 1), (w - 1 - xx) / max(pad[2], 1)),
            1.0 - np.minimum(yy / max(pad[1], 1), (h - 1 - yy) / max(pad[3], 1)),
        )
        blur = qsize * 0.02
        array += (scipy.ndimage.gaussian_filter(array, [blur, blur, 0]) - array) * np.clip(mask[..., None] * 3 + 1, 0, 1)
        array += (np.median(array, axis=(0, 1)) - array) * np.clip(mask[..., None], 0, 1)
        pil = Image.fromarray(np.uint8(np.clip(np.rint(array), 0, 255)), "RGB")
        quad += np.asarray(pad[:2])

    return pil.transform(
        (output_size, output_size), Image.Transform.QUAD, tuple((quad + 0.5).flatten()), Image.Resampling.BILINEAR
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shape-predictor", type=Path, required=True, help="dlib 68-point shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    import dlib

    predictor = dlib.shape_predictor(str(args.shape_predictor))
    paths = list(iter_image_paths(args.input_dir))
    for index, path in enumerate(paths, start=1):
        try:
            aligned = align_face(path, predictor, args.size)
            relative = path.relative_to(args.input_dir).with_suffix(".jpg")
            destination = args.output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            aligned.save(destination, quality=95)
            print(f"[{index}/{len(paths)}] {path} -> {destination}")
        except Exception as exc:
            print(f"[skip] {path}: {exc}")


if __name__ == "__main__":
    main()

