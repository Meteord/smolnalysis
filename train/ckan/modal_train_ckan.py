from __future__ import annotations

import modal


APP_NAME = "smolnalysis-ckan-minicpm5-lora"
VOLUME_NAME = "smolnalysis-ckan-training"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("train/ckan/requirements-train.txt")
    .add_local_file("train/ckan/train_minicpm_lora.py", remote_path="/root/train_minicpm_lora.py")
    .add_local_file("train/ckan/data/generated/valid_train_1000_repaired.jsonl", remote_path="/root/data/train.jsonl")
    .add_local_file("train/ckan/data/generated/valid_eval_golden_60_repaired.jsonl", remote_path="/root/data/eval.jsonl")
)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 4,
    volumes={"/outputs": volume},
)
def train_ckan_lora(smoke: bool = True) -> None:
    import subprocess

    cmd = [
        "python",
        "/root/train_minicpm_lora.py",
        "--train-data",
        "/root/data/train.jsonl",
        "--eval-data",
        "/root/data/eval.jsonl",
        "--output-dir",
        "/outputs/smolnalysis-ckan-retrieval-minicpm5-lora-smoke" if smoke else "/outputs/smolnalysis-ckan-retrieval-minicpm5-lora",
        "--per-device-train-batch-size",
        "1",
        "--gradient-accumulation-steps",
        "8",
        "--eval-steps",
        "10" if smoke else "25",
        "--save-steps",
        "10" if smoke else "25",
    ]
    if smoke:
        cmd.extend(["--train-limit", "24", "--eval-limit", "12", "--num-train-epochs", "1"])
    subprocess.run(cmd, check=True)
    volume.commit()


@app.local_entrypoint()
def main(smoke: bool = True) -> None:
    train_ckan_lora.remote(smoke=smoke)
