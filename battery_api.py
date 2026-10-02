#!/usr/bin/env python3
"""

The three detectors need three incompatible Python environments, so this
process loads none of them. It starts one worker per model -- this same file
run with --worker under that model's Python -- keeps every model resident on
the GPU, and passes each request's image to the workers it names over a pipe.
Detection is battery_signal.py's detector classes with its default prompts,
thresholds and input sizes, so tuning there applies here too.

  GET  /health   JSON: per model "loading" | "ready" | "failed", load time, error
  POST /detect   body: one image -- PNG, JPEG, or .npy as host.py serves it

/detect query parameters:
  model     dart | yoloe | locateanything | all, or a comma list (default: first of --models)
  vote      how several models combine: majority (default), any, unanimous
  conf      score threshold for dart and yoloe, instead of battery_signal.py's
  save      1 = keep the input, an annotated JPEG and a JSONL record under --save-dir
  frame_id  any text; echoed back and put in saved file names

    {"signal": 1, "frame_id": "570", "models": ["dart"], "vote": "majority",
     "votes": 1, "answered": 1, ..., "results": {"dart": {"signal": 1, "count": 1,
     "score": 0.82, "boxes": [[x1, y1, x2, y2]], "scores": [0.82], "infer_ms": 31.2}}}

signal 1 = battery, 0 = none. A model that fails on a request reports "error"
instead and does not vote; when no model answers the request fails with 502.
Images are BGR, as OpenCV and host.py have them: a .npy body is read as
B,G,R[,NIR] and NIR is dropped; PNG and JPEG decode the usual way.

"""

import argparse
import asyncio
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
PYTHONS = {
    "dart": ROOT / ".conda-envs" / "dart" / "bin" / "python",
    "yoloe": ROOT / ".conda-envs" / "yoloe26" / "bin" / "python",
    "locateanything": ROOT / "locate-anything" / ".conda" / "bin" / "python",
}
VOTES = ("majority", "any", "unanimous")
# BGR box colour and short label per model, for annotated images.
STYLE = {
    "dart": ((0, 200, 255), "dart"),
    "yoloe": ((255, 160, 0), "yoloe"),
    "locateanything": ((220, 60, 255), "LA"),
}


def log(message):
    print(message, file=sys.stderr, flush=True)


def run_worker(args):
    """One model in its own Python: a JSON header line plus raw BGR bytes in, a JSON line out."""
    # Model libraries print to stdout. Keep the real stdout for replies and
    # send everything else to the log.
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    requests = sys.stdin.buffer

    def reply(message):
        replies.write(json.dumps(message) + "\n")

    import battery_signal

    name = args.worker
    weights, prompts, conf, imgsz = battery_signal.DEFAULTS[name]
    prompts = args.prompts or prompts
    started = time.monotonic()
    try:
        if name == "locateanything":
            detect = battery_signal.LocateAnythingDetector(
                weights, prompts, args.distractors, conf, imgsz, args.locate_mode
            )
        else:
            detector = battery_signal.YoloeDetector if name == "yoloe" else battery_signal.DartDetector
            detect = detector(weights, prompts, args.distractors, conf, imgsz)
        detect(np.zeros((64, 64, 3), np.uint8))  # warm up CUDA before the first real request
    except Exception as exc:
        reply({"error": f"{type(exc).__name__}: {exc}"})
        raise
    reply({"ready": True, "load_ms": round((time.monotonic() - started) * 1000),
           "prompts": prompts, "conf": conf, "imgsz": imgsz})

    while header := requests.readline():
        request = json.loads(header)
        height, width = request["shape"]
        frame = np.frombuffer(requests.read(height * width * 3), np.uint8)
        frame = frame.reshape(height, width, 3).copy()
        # The detectors read .conf on every call; LocateAnything ignores it.
        detect.conf = conf if request.get("conf") is None else request["conf"]
        started = time.perf_counter()
        try:
            boxes, scores = detect(frame)
        except Exception as exc:
            log(f"[{name}] inference failed: {type(exc).__name__}: {exc}")
            reply({"error": f"{type(exc).__name__}: {exc}"})
            continue
        reply({
            "boxes": [[round(value, 1) for value in box] for box in boxes],
            "scores": [round(score, 4) for score in scores],
            "infer_ms": round((time.perf_counter() - started) * 1000, 1),
        })


class Worker:
    """The API's handle on one worker process; one request at a time goes down its pipe."""

    def __init__(self, name, args):
        self.name, self.args = name, args
        self.state, self.error, self.info = "loading", None, {}
        self.proc = None
        self.ready = asyncio.Event()
        self.lock = asyncio.Lock()

    async def start(self):
        args = self.args
        command = [str(PYTHONS[self.name]), str(Path(__file__).resolve()), "--worker", self.name,
                   "--locate-mode", args.locate_mode, "--distractors", *args.distractors]
        if args.prompts:
            command += ["--prompts", *args.prompts]
        log(f"[api] loading {self.name} with {PYTHONS[self.name]}")
        try:
            # A session of its own: Ctrl-C stops the API, which then closes the
            # workers in order instead of all of them dying mid-request.
            self.proc = await asyncio.create_subprocess_exec(
                *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                cwd=ROOT, limit=1 << 24, start_new_session=True,
            )
            reply = await self._read()
        except Exception as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
        else:
            if reply.get("ready"):
                self.state, self.info = "ready", reply
                log(f"[api] {self.name} ready in {reply['load_ms'] / 1000:.1f} s")
            else:
                self._fail(reply.get("error", f"unexpected first message {reply}"))
        finally:
            self.ready.set()

    async def detect(self, frame, conf):
        await self.ready.wait()
        if self.state != "ready":
            raise RuntimeError(f"{self.name} {self.state}: {self.error}")
        # Shielded: a request cancelled between writing and reading would
        # leave its reply in the pipe for the next request to read.
        return await asyncio.shield(self._exchange(frame, conf))

    async def _exchange(self, frame, conf):
        async with self.lock:
            if self.proc.returncode is not None:
                self._fail(f"worker exited with code {self.proc.returncode}")
                raise RuntimeError(f"{self.name} failed: {self.error}")
            header = json.dumps({"shape": frame.shape[:2], "conf": conf}).encode() + b"\n"
            try:
                self.proc.stdin.write(header + frame.tobytes())
                await self.proc.stdin.drain()
                return await self._read()
            except Exception as exc:
                self._fail(f"{type(exc).__name__}: {exc}")
                raise RuntimeError(f"{self.name} failed: {self.error}") from exc

    async def _read(self):
        line = await self.proc.stdout.readline()
        if not line:
            raise EOFError(f"worker exited with code {await self.proc.wait()}")
        return json.loads(line)

    def _fail(self, error):
        self.state, self.error = "failed", error
        log(f"[api] {self.name} failed: {error}")

    async def stop(self):
        if self.proc is None or self.proc.returncode is not None:
            return
        self.proc.stdin.close()  # the worker exits at end of input
        try:
            await asyncio.wait_for(self.proc.wait(), 10)
        except asyncio.TimeoutError:
            self.proc.kill()
            await self.proc.wait()

    def status(self):
        return {"state": self.state, "error": self.error,
                "pid": self.proc.pid if self.proc else None, **self.info}


def decode_image(body):
    if not body:
        raise ValueError("empty body; POST the image bytes")
    if body.startswith(b"\x93NUMPY"):
        frame = np.load(io.BytesIO(body), allow_pickle=False)
    else:
        frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise ValueError("body is not a PNG, JPEG or .npy image")
    if frame.dtype != np.uint8:
        raise ValueError(f"expected 8-bit pixels, got {frame.dtype}")
    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"unexpected image shape {frame.shape}")
    return np.ascontiguousarray(frame[:, :, :3])


def annotate(frame, record):
    for name, result in record["results"].items():
        colour, label = STYLE[name]
        for (x1, y1, x2, y2), score in zip(result.get("boxes", []), result.get("scores", [])):
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
            cv2.putText(frame, f"{label} {score:.2f}", (int(x1), max(40, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    votes = "  ".join(f"{STYLE[name][1]}:{result.get('signal', 'err')}"
                      for name, result in record["results"].items())
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 160, 0) if record["signal"] else (80, 80, 80), -1)
    cv2.putText(frame, f"BATTERY {record['signal']}   {votes}", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


class Saver:
    """Writes saved requests on one thread: replies never wait, the JSONL stays in order."""

    def __init__(self, root):
        self.root = root
        self.pool = ThreadPoolExecutor(max_workers=1)

    def submit(self, frame, response):
        now = datetime.now()
        folder = self.root / now.strftime("%Y%m%d")
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", response["frame_id"])[:64]
        stem = now.strftime("%H%M%S_%f") + (f"_{tag}" if tag else "")
        paths = {
            "image": folder / f"{stem}.png",
            "annotated": folder / f"{stem}_pred.jpg",
            "predictions": folder / "predictions.jsonl",
        }
        saved = {key: str(path) for key, path in paths.items()}
        self.pool.submit(self._write, frame, dict(response, saved=saved), paths)
        return saved

    @staticmethod
    def _write(frame, record, paths):
        try:
            paths["image"].parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(paths["image"]), frame)
            cv2.imwrite(str(paths["annotated"]), annotate(frame.copy(), record),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            with paths["predictions"].open("a") as out:
                out.write(json.dumps(record) + "\n")
        except Exception as exc:
            log(f"[api] saving {paths['image'].name} failed: {exc}")

    def close(self):
        self.pool.shutdown(wait=True)


def serve(args):
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request

    workers = {name: Worker(name, args) for name in dict.fromkeys(args.models)}
    saver = Saver(args.save_dir)
    started_at = time.time()

    @asynccontextmanager
    async def lifespan(app):
        loading = [asyncio.create_task(worker.start()) for worker in workers.values()]
        yield
        for task in loading:
            task.cancel()
        await asyncio.gather(*(worker.stop() for worker in workers.values()))
        saver.close()

    app = FastAPI(title="Battery detection API", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {
            "ready": all(worker.state == "ready" for worker in workers.values()),
            "models": {name: worker.status() for name, worker in workers.items()},
            "default_model": args.models[0],
            "save_dir": str(args.save_dir),
            "save_all": args.save_all,
            "uptime_s": round(time.time() - started_at),
        }

    @app.post("/detect")
    async def detect(request: Request, model: str = args.models[0], vote: str = "majority",
                     conf: float | None = None, save: bool = False, frame_id: str = ""):
        names = list(workers) if model == "all" else [n.strip() for n in model.split(",") if n.strip()]
        names = list(dict.fromkeys(names))
        unknown = [name for name in names if name not in workers]
        if not names or unknown:
            raise HTTPException(400, f"unknown or unloaded model {unknown or model!r}; "
                                     f"loaded: {', '.join(workers)}, or all")
        if vote not in VOTES:
            raise HTTPException(400, f"vote must be one of {', '.join(VOTES)}")
        try:
            frame = decode_image(await request.body())
        except Exception as exc:
            raise HTTPException(400, f"cannot read image: {exc}") from exc

        started = time.perf_counter()
        replies = await asyncio.gather(*(workers[name].detect(frame, conf) for name in names),
                                       return_exceptions=True)
        results = {}
        for name, reply in zip(names, replies):
            if isinstance(reply, BaseException):
                results[name] = {"error": str(reply)}
            elif "error" in reply:
                results[name] = {"error": reply["error"]}
            else:
                scores = reply["scores"]
                results[name] = {"signal": int(bool(scores)), "count": len(scores),
                                 "score": max(scores, default=0.0), **reply}
        answered = [result["signal"] for result in results.values() if "error" not in result]
        if not answered:
            raise HTTPException(502, {"error": "no model answered", "results": results})
        votes = sum(answered)
        signal = {"majority": votes * 2 > len(answered), "any": votes > 0,
                  "unanimous": votes == len(answered)}[vote]

        response = {
            "signal": int(signal),
            "frame_id": frame_id,
            "models": names,
            "vote": vote,
            "votes": votes,
            "answered": len(answered),
            "width": frame.shape[1],
            "height": frame.shape[0],
            "time": round(time.time(), 3),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "results": results,
        }
        if save or args.save_all:
            response["saved"] = saver.submit(frame, response)
        return response

    log(f"[api] http://{args.host}:{args.port}/detect  (outside: http://ailab3.samk.fi:2793)")
    uvicorn.run(app, host=args.host, port=args.port, access_log=False, log_level="warning")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--models", nargs="+", choices=PYTHONS, default=list(PYTHONS),
                        help="models to load; the first is the default for requests")
    parser.add_argument("--locate-mode", choices=("fast", "hybrid", "slow"), default="fast",
                        help="LocateAnything decoding (see battery_signal.py --mode)")
    parser.add_argument("--prompts", nargs="+",
                        help="prompts that count as a battery, for every model "
                             "(default: battery_signal.py's, per model)")
    parser.add_argument("--distractors", nargs="*", default=[],
                        help="extra prompts for look-alikes; their detections never count")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "temp" / "api_results",
                        help="saved requests go to <save-dir>/<YYYYMMDD>/")
    parser.add_argument("--save-all", action="store_true",
                        help="save every request, whatever its save= says")
    parser.add_argument("--worker", choices=PYTHONS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
