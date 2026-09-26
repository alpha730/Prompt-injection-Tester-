from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    load_dotenv(ROOT / ".env")   # GROQ_API_KEY; never hard-coded
    return yaml.safe_load(path.read_text(encoding="utf-8"))
