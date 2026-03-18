"""Pytest test suite bootstrap for the local AirSim package.

Tests can run without a live UE5 / CosysAirSim instance. This initializer
forces imports to resolve to the repository's ``src/airsim`` package so test
collection does not accidentally mix stale or shadowed module objects.
"""

import importlib.util
import sys
from pathlib import Path


_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def _bootstrap_local_airsim_package() -> None:
    existing = sys.modules.get("airsim")
    if existing is not None:
        module_file = getattr(existing, "__file__", "") or ""
        if Path(module_file).resolve() == (_SRC_DIR / "airsim" / "__init__.py").resolve():
            return

    for module_name in [
        name for name in list(sys.modules.keys())
        if name == "airsim" or name.startswith("airsim.")
    ]:
        del sys.modules[module_name]

    airsim_init = _SRC_DIR / "airsim" / "__init__.py"
    airsim_spec = importlib.util.spec_from_file_location(
        "airsim",
        airsim_init,
        submodule_search_locations=[str(_SRC_DIR / "airsim")],
    )
    airsim_module = importlib.util.module_from_spec(airsim_spec)
    sys.modules["airsim"] = airsim_module
    assert airsim_spec.loader is not None
    airsim_spec.loader.exec_module(airsim_module)


_bootstrap_local_airsim_package()
