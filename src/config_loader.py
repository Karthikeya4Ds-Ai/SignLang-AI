"""
Configuration Loader Module
Provides typed and safe access to project settings and class vocabularies.
"""

from pathlib import Path
from typing import Any, Dict, List
import yaml
import numpy as np


def get_project_root() -> Path:
    """Returns the absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """Loads YAML configuration file with safe defaults."""
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_all_classes(config: Dict[str, Any] = None) -> np.ndarray:
    """
    Returns the canonical ordered list of all 47 classes.
    Order MUST exactly match the training data:
    1. Static Numbers (0-9)
    2. Static Alphabet (A-Z)
    3. Dynamic Gestures (Hello, Goodbye, Thanks, Please, Yes, No, Food, Water, Help, Stop)
    4. Special Token (_idle_)
    """
    if config is None:
        config = load_config()

    classes_cfg = config.get("classes", {})
    numbers = classes_cfg.get("static_numbers", [str(i) for i in range(10)])
    alphabet = classes_cfg.get("static_alphabet", [chr(c) for c in range(ord('A'), ord('Z') + 1)])
    dynamic = classes_cfg.get("dynamic_gestures", [
        'Hello', 'Goodbye', 'Thanks', 'Please', 'Yes', 'No',
        'Food', 'Water', 'Help', 'Stop'
    ])
    idle = classes_cfg.get("special_tokens", {}).get("idle", "_idle_")

    all_classes = list(numbers) + list(alphabet) + list(dynamic)
    if idle not in all_classes:
        all_classes.append(idle)

    return np.array(all_classes)


def get_label_map(classes: np.ndarray = None) -> Dict[str, int]:
    """Maps class names to numeric target indices."""
    if classes is None:
        classes = get_all_classes()
    return {name: idx for idx, name in enumerate(classes)}
