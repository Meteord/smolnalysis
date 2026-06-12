#!/usr/bin/env python3
"""Convenience launcher for the OpenUI MiniCPM LoRA trainer."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    trainer = Path(__file__).resolve().parent / "train" / "openui_lang" / "train_minicpm_lora.py"
    runpy.run_path(str(trainer), run_name="__main__")
