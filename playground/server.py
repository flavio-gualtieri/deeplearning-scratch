# playground/server.py
#
# Minimal local backend for the drag-and-drop playground. Stdlib-only:
# serves the static frontend, reports CSV columns, writes config.yaml from
# the assembled pipeline, and runs entrypoint.py as a subprocess while
# streaming its stdout back to the browser.

import json
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepscratch.data.tabular import infer_feature_type  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
STATIC_DIR = Path(__file__).resolve().parent / "static"
CONFIG_PATH = REPO_ROOT / "config.yaml"

EPOCH_LINE = re.compile(
    r"Epoch (?P<epoch>\d+)/(?P<total>\d+) - train_loss: (?P<train_loss>[\d.]+)"
    r"(?: - val_loss: (?P<val_loss>[\d.]+))?"
)

state_lock = threading.Lock()
run_state = {
    "status": "idle",  # idle | running | done | error
    "points": [],
    "error": None,
    "log_tail": [],
}


def find_csv() -> Path:
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {DATA_DIR}")
    return csv_files[0]


def read_csv_frame(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def compute_column_types(df: pd.DataFrame) -> dict:
    return {column: infer_feature_type(str(dtype)) for column, dtype in df.dtypes.items()}


def build_config(payload: dict) -> dict:
    csv_path = find_csv()
    target = payload["target"]
    encoder_groups = payload["encoder_groups"]

    if not encoder_groups:
        raise ValueError("Add at least one encoder group.")

    features = [column for group in encoder_groups for column in group["feature_columns"]]

    if not features:
        raise ValueError("Select at least one feature column (everything but the target).")

    if target in features:
        raise ValueError("The target column cannot also be a feature column.")

    head = payload["head"]
    training = payload["training"]

    if head["type"] == "regression":
        task_type = "regression"
        output_dim = int(head.get("output_dim", 1))
    elif head["type"] == "classification":
        task_type = "classification"
        df = read_csv_frame(csv_path)
        classes = set(df[target].astype(str)) if target in df.columns else set()
        output_dim = max(len(classes), 2)
    else:
        # multilabel / projection / sequence_tagging: this playground only ever
        # selects a single target column, so none of these get a "real" target
        # encoding of their own -- they ride on the classification target path
        # with an explicitly-sized output instead.
        task_type = "classification"
        output_dim = int(head.get("output_dim", 1))

    total_embedding_dim = sum(group["encoder"]["embedding_dim"] for group in encoder_groups)

    splits = training["splits"]
    total = sum(splits)
    if total <= 0:
        splits, total = [0.7, 0.15, 0.15], 1.0
    normalized_splits = [split / total for split in splits]

    return {
        "general": {
            "input_path": str(csv_path.relative_to(REPO_ROOT)),
            "output_path": "runs",
            "task_type": task_type,
        },
        "run": {
            "name": f"playground-{int(time.time())}",
            "seed": 42,
        },
        "data": {
            "type": "tabular_csv",
            "feature_columns": features,
            "target_column": target,
            "batch_size": training["batch_size"],
            "splits": normalized_splits,
            "shuffle": True,
        },
        "components": {
            "encoder": {
                # Pass each group's encoder dict through as-is -- different
                # encoder types need different hyperparameter fields (e.g.
                # cnn's `channels`, rnn's `hidden_dim`), and entrypoint.py's
                # build_leaf_encoder() is the place that knows which fields
                # a given type actually reads.
                "groups": [
                    {
                        "feature_columns": group["feature_columns"],
                        "encoder": dict(group["encoder"]),
                    }
                    for group in encoder_groups
                ],
            },
            "head": {
                "type": head["type"],
                "input_dim": total_embedding_dim,
                "output_dim": output_dim,
                "hidden_dims": head["hidden_dims"],
                "dropout": head.get("dropout", 0.0),
                "normalize": head.get("normalize", True),
            },
            "trainer": {
                "type": "basic",
                "epochs": training["epochs"],
                "learning_rate": training["learning_rate"],
                "loss_fn": training["loss_fn"],
                "optimizer": training["optimizer"],
                "device": training["device"] or None,
                "metrics_every": 1,
            },
        },
        "logging": {"type": "jsonl", "log_every": 1},
        "checkpointing": {
            "enabled": True,
            "save_every": 1,
            "save_best": True,
            "monitor": "val_loss",
            "mode": "min",
        },
    }


def run_training(payload: dict) -> None:
    with state_lock:
        if run_state["status"] == "running":
            raise RuntimeError("A run is already in progress.")

        run_state.update(status="running", points=[], error=None, log_tail=[])

    config = build_config(payload)

    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    process = subprocess.Popen(
        [sys.executable, "entrypoint.py"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    threading.Thread(target=_stream_process, args=(process,), daemon=True).start()


def _stream_process(process: subprocess.Popen) -> None:
    for line in process.stdout:
        line = line.rstrip("\n")
        match = EPOCH_LINE.search(line)

        with state_lock:
            run_state["log_tail"].append(line)
            run_state["log_tail"][:] = run_state["log_tail"][-20:]

            if match:
                run_state["points"].append(
                    {
                        "epoch": int(match.group("epoch")),
                        "total": int(match.group("total")),
                        "train_loss": float(match.group("train_loss")),
                        "val_loss": float(match.group("val_loss"))
                        if match.group("val_loss")
                        else None,
                    }
                )

    return_code = process.wait()

    with state_lock:
        if return_code == 0:
            run_state["status"] = "done"
        else:
            run_state["status"] = "error"
            run_state["error"] = "\n".join(run_state["log_tail"]) or f"exited with code {return_code}"


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html")
        elif self.path == "/app.js":
            self._send_file(STATIC_DIR / "app.js", "application/javascript")
        elif self.path == "/style.css":
            self._send_file(STATIC_DIR / "style.css", "text/css")
        elif self.path == "/api/csv":
            try:
                csv_path = find_csv()
                df = read_csv_frame(csv_path)
                self._send_json(
                    {
                        "filename": csv_path.name,
                        "columns": list(df.columns),
                        "column_types": compute_column_types(df),
                    }
                )
            except FileNotFoundError as error:
                self._send_json({"error": str(error)}, status=404)
        elif self.path == "/api/status":
            with state_lock:
                self._send_json(
                    {
                        "status": run_state["status"],
                        "points": run_state["points"],
                        "error": run_state["error"],
                    }
                )
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self._send_json({"error": "not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")

        try:
            run_training(payload)
            self._send_json({"status": "started"})
        except RuntimeError as error:
            self._send_json({"error": str(error)}, status=409)
        except Exception as error:
            self._send_json({"error": str(error)}, status=400)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"

    print(f"DeepScratch playground running at {url}")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
