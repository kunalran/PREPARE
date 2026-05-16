from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_root_module(current_file: str) -> ModuleType:
    current_path = Path(current_file).resolve()
    repo_root = current_path.parents[1]
    root_script = repo_root / current_path.name
    module_name = f"_root_{current_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, root_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load root module from {root_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def reexport(module: ModuleType, namespace: dict[str, object]) -> None:
    public_names = getattr(module, "__all__", None)
    if public_names is None:
        public_names = [name for name in dir(module) if not name.startswith("__")]
    for name in public_names:
        namespace[name] = getattr(module, name)
