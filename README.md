# Battery Detection

This project detects batteries in camera frames or image files. `battery_api.py` provides a detection API with three models: **DART, YOLOE, and LocateAnything**. `battery_client.py` reads new frames from a camera server, sends them to the API, and outputs one JSON line per frame. When using multiple models, results can be combined by majority vote (the default), any positive vote, or unanimous vote.

## Requirements

- **API server:** Requires `battery_signal.py`, model weights, and a separate Python environment for each model. To run all three models simultaneously, deploy `battery_api.py` on a computer with **at least 16 GB of GPU memory**.
- **Client:** Install `numpy` and `Pillow` (for example, `pip install numpy Pillow`). The computer running the client must be able to reach **both the camera server and the API server**.

The API address currently configured in the scripts is **http://ailab3.samk.fi:2793**. Check model status at the [health endpoint](http://ailab3.samk.fi:2793/health). The default camera server address is `http://10.80.24.190:8090`. If either address changes, specify it with `--api` or `--camera`.

## Usage

Start the API on the server with the model environments installed. By default, it listens on `0.0.0.0:6006` and loads all three models:

```bash
python3 battery_api.py
# To load only one model: python3 battery_api.py --models dart
```

Run the client on a computer that can reach the camera and API servers:

```bash
python3 battery_client.py                                  # Use DART and read camera frames continuously
python3 battery_client.py --model all --vote majority      # Combine all three models by majority vote
python3 battery_client.py --model all --save --count 10    # Process 10 frames and save results on the API server
python3 battery_client.py --images captures/*.png          # Detect local images; no camera connection needed
python3 battery_client.py --camera http://CAMERA_HOST:PORT --api http://API_HOST:PORT
```

The client writes one JSON line per frame to standard output. `signal` is `1` when a battery is detected, `0` when none is detected, and `null` when the request fails. `models` contains the result from each model. Logs go to standard error.

With `--save`, the **API server** stores the original image, an annotated image, and a `predictions.jsonl` record. The default location is `temp/api_results/` under the script directory; change it with the API's `--save-dir` option. The repository's `api_results/` directory contains existing example results and is not the default save location.

For more options, run `python3 battery_api.py --help` or `python3 battery_client.py --help`.
