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
    .add_local_file("train/ckan/evaluate_lora.py", remote_path="/root/evaluate_lora.py")
    .add_local_file("train/ckan/ckan_dataset_tools.py", remote_path="/root/ckan_dataset_tools.py")
    .add_local_file("train/ckan/data/generated/valid_examples_multitool_200_repaired.jsonl", remote_path="/root/data/smoke_train.jsonl")
    .add_local_file("train/ckan/data/generated/valid_examples_multitool_train_1600_repaired.jsonl", remote_path="/root/data/train.jsonl")
    .add_local_file("train/ckan/data/generated/valid_examples_multitool_eval_160.jsonl", remote_path="/root/data/eval.jsonl")
    .add_local_file("train/ckan/data/multitool_eval_golden.jsonl", remote_path="/root/data/hand_eval.jsonl")
)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 4,
    volumes={"/outputs": volume},
)
def train_ckan_lora(smoke: bool = True, challenge: bool = False) -> None:
    import subprocess

    cmd = [
        "python",
        "/root/train_minicpm_lora.py",
        "--train-data",
        "/root/data/smoke_train.jsonl" if smoke else "/root/data/train.jsonl",
        "--eval-data",
        "/root/data/eval.jsonl",
        "--output-dir",
        "/outputs/smolnalysis-ckan-retrieval-minicpm5-lora-challenge" if challenge and not smoke else "/outputs/smolnalysis-ckan-retrieval-minicpm5-lora-smoke" if smoke else "/outputs/smolnalysis-ckan-retrieval-minicpm5-lora",
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
        cmd.extend(["--train-limit", "48", "--eval-limit", "24", "--num-train-epochs", "1"])
    subprocess.run(cmd, check=True)
    volume.commit()


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    volumes={"/outputs": volume},
)
def evaluate_ckan_lora(smoke: bool = False, challenge: bool = False, adapter_variant: str = "base") -> None:
    import subprocess

    if smoke:
        adapter_path = "/outputs/smolnalysis-ckan-retrieval-minicpm5-lora-smoke"
    elif adapter_variant == "challenge":
        adapter_path = "/outputs/smolnalysis-ckan-retrieval-minicpm5-lora-challenge"
    else:
        adapter_path = "/outputs/smolnalysis-ckan-retrieval-minicpm5-lora"
    eval_data = "/root/data/hand_eval.jsonl" if challenge else "/root/data/eval.jsonl"
    output_dir = (
        "/outputs/eval-challenge-adapter-challenge"
        if challenge and adapter_variant == "challenge" and not smoke
        else "/outputs/eval-challenge-smoke"
        if smoke and challenge
        else "/outputs/eval-challenge"
        if challenge
        else "/outputs/eval-smoke"
        if smoke
        else "/outputs/eval"
    )
    cmd = [
        "python",
        "/root/evaluate_lora.py",
        "--eval-data",
        eval_data,
        "--adapter-path",
        adapter_path,
        "--output-dir",
        output_dir,
        "--max-new-tokens",
        "256",
    ]
    subprocess.run(cmd, check=True)
    volume.commit()


@app.local_entrypoint()
def main(mode: str = "train", smoke: bool = True, challenge: bool = False, adapter_variant: str = "base") -> None:
    if mode == "train":
        train_ckan_lora.remote(smoke=smoke, challenge=challenge)
    elif mode == "evaluate":
        evaluate_ckan_lora.remote(smoke=smoke, challenge=challenge, adapter_variant=adapter_variant)
    else:
        raise ValueError("mode must be 'train' or 'evaluate'")
