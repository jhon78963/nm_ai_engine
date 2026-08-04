#!/usr/bin/env python3
"""Pipeline completo: exportar datos → entrenar demanda (Prophet) → entrenar precio (Ridge)."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
PYTHON = sys.executable


def _run(script_name: str) -> None:
    module = f"scripts.{script_name.replace('.py', '')}"
    logger.info("=== Ejecutando %s ===", module)
    subprocess.run([PYTHON, "-m", module], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    _run("export_training_data.py")
    _run("train_demand.py")
    _run("train_price.py")
    logger.info("Pipeline ML completado.")


if __name__ == "__main__":
    main()
