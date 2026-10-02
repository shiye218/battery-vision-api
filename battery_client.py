#!/usr/bin/env python3
"""Battery signal for every frame host.py publishes, from the battery API on ailab2.

Pulls each new frame from the camera host (host.py), posts its colour channels
to battery_api.py on ailab2 and prints one JSON line per frame on stdout:

    {"signal": 1, "frame": 570, "time": 1790000000.123, "votes": 1, "models": {"dart": 1},
     "score": 0.82, "count": 1, "skipped": 0, "api_ms": 84.2, "age_ms": 312.5}

signal 1 = a battery in the frame, 0 = none, null = the API gave no answer
(the line then carries "error"). Every frame gets a line, not only changes:
host.py's frames are consecutive stretches of belt, so each is its own answer.
"models" is each model's own signal, "score" and "count" the highest over
them, "skipped" the frames the host published that this one replaced, and
"age_ms" how old the frame was when its line was printed. stdout carries
nothing else; logs go to stderr. --udp also sends each line as one datagram.

    python3 battery_client.py                            # DART, until Ctrl-C
    python3 battery_client.py --model all --save         # majority of three; ailab2 keeps the images
    python3 battery_client.py --images captures/*.png    # files instead of the camera

Dependencies: numpy, Pillow.
"""

import argparse
import io
import json
import socket
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

CONTENT_TYPES = {"png": "image/png", "jpg": "image/jpeg", "npy": "application/octet-stream"}


def log(message):
    print(message, file=sys.stderr, flush=True)


def get_json(url, timeout=10):
    with urlopen(url, timeout=timeout) as response:
        return json.load(response)


def encode(bgr, kind):
    buffer = io.BytesIO()
    if kind == "npy":
        np.save(buffer, bgr)  # the API reads .npy as B,G,R, like host.py
    else:
        image = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
        if kind == "png":
            image.save(buffer, format="PNG", compress_level=1)
        else:
            image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def camera_frames(base, kind):
    """Each new frame as (number, body, content type, monotonic birth time, skipped), forever."""
    last = None
    while True:
        try:
            if last is None:
                status = get_json(f"{base}/status")
                if status.get("not_ready"):
                    log(f"Camera host not ready: {status['not_ready']}; retrying in 2 s")
                    time.sleep(2)
                    continue
                last = int(status.get("frame_number") or 0)
                log(f"Camera {status.get('camera')}: {status.get('fps')} frames/s, frame {last}")
            query = urlencode({"since": last, "timeout": 30})
            with urlopen(f"{base}/frame.npy?{query}", timeout=40) as response:
                if response.status == 204:
                    # Nothing new for 30 s. Re-read /status: a restarted host
                    # counts from 0 again and would never pass `last`.
                    last = None
                    continue
                number = int(response.headers["X-Frame-Number"])
                age = float(response.headers.get("X-Frame-Age-Seconds") or 0)
                frame = np.load(io.BytesIO(response.read()), allow_pickle=False)
            born = time.monotonic() - age
        except (OSError, ValueError) as exc:  # URLError and timeouts are OSErrors
            log(f"Camera host: {exc}; retrying in 2 s")
            last = None
            time.sleep(2)
            continue

        if frame.ndim != 3 or frame.shape[2] < 3 or frame.dtype != np.uint8:
            raise RuntimeError(f"Unexpected frame: shape={frame.shape}, dtype={frame.dtype}")
        skipped = max(0, number - last - 1)
        last = number
        # The host supplies B, G, R, NIR; only the colour channels go to the API.
        yield number, encode(frame[:, :, :3], kind), CONTENT_TYPES[kind], born, skipped


def file_frames(paths):
    for path in paths:
        kind = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        yield path.stem, path.read_bytes(), kind, None, 0


def detect(api, body, content_type, params):
    request = Request(f"{api}/detect?{urlencode(params)}", data=body,
                      headers={"Content-Type": content_type}, method="POST")
    # Generous: a request that arrives while the models load waits for them.
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def parse_address(text):
    host, _, port = text.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError(f"expected HOST:PORT, got {text!r}")
    return host, int(port)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--camera", default="http://10.80.24.190:8090", help="host.py address")
    parser.add_argument("--api", default="http://ailab3.samk.fi:2793",
                        help="battery_api.py address (ailab2 port 6006)")
    parser.add_argument("--model", default="dart",
                        help="dart, yoloe, locateanything, all, or a comma list")
    parser.add_argument("--vote", choices=("majority", "any", "unanimous"), default="majority",
                        help="how several models combine into one signal")
    parser.add_argument("--conf", type=float,
                        help="score threshold for dart and yoloe (default: the API's)")
    parser.add_argument("--save", action="store_true",
                        help="have ailab2 keep each frame, its annotated image and the result")
    parser.add_argument("--encode", choices=CONTENT_TYPES, default="png",
                        help="how frames travel to the API: png and npy are lossless")
    parser.add_argument("--images", type=Path, nargs="+",
                        help="send these image files instead of reading the camera")
    parser.add_argument("--count", type=int, default=0, help="stop after this many frames; 0 = never")
    parser.add_argument("--udp", type=parse_address, metavar="HOST:PORT",
                        help="also send each line as a UDP datagram")
    args = parser.parse_args()
    api = args.api.rstrip("/")

    try:
        health = get_json(f"{api}/health")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Battery API unreachable at {api}: {exc}\n")
    log("API models: " + ", ".join(f"{name} {model['state']}" for name, model in health["models"].items()))

    params = {"model": args.model, "vote": args.vote, "save": int(args.save)}
    if args.conf is not None:
        params["conf"] = args.conf
    frames = file_frames(args.images) if args.images else camera_frames(args.camera.rstrip("/"), args.encode)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.udp else None

    done = 0
    try:
        for frame_id, body, content_type, born, skipped in frames:
            sent = time.monotonic()
            line = {"signal": None, "frame": frame_id, "time": None}
            try:
                result = detect(api, body, content_type, dict(params, frame_id=frame_id))
            except HTTPError as exc:
                line["error"] = f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}"
            except (OSError, ValueError) as exc:
                line["error"] = str(exc)
            else:
                answered = [r for r in result["results"].values() if "error" not in r]
                line.update(
                    signal=result["signal"],
                    votes=result["votes"],
                    models={name: r.get("signal") for name, r in result["results"].items()},
                    score=round(max((r["score"] for r in answered), default=0.0), 3),
                    count=max((r["count"] for r in answered), default=0),
                )
                if "saved" in result:
                    line["saved"] = result["saved"]["annotated"]
            now = time.monotonic()
            line["time"] = round(time.time(), 3)
            line.update(skipped=skipped, api_ms=round((now - sent) * 1000, 1))
            if born is not None:
                line["age_ms"] = round((now - born) * 1000, 1)
            if "error" in line:
                log(f"Frame {frame_id}: {line['error']}")

            text = json.dumps(line)
            print(text, flush=True)
            if udp:
                udp.sendto(text.encode(), args.udp)
            done += 1
            if args.count and done >= args.count:
                break
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
