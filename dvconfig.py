"""Configuration loader shared by the dvnanima films (matplotlib/numpy films with a ``config.toml`` next to the script).

Every number that changes the picture, the timing or the physics of a film lives in the ``config.toml`` of its project;
the scripts only contain the algorithms.  Usage in a script::

    from dvconfig import load_config
    CFG = load_config(HERE)                  # HERE / "config.toml"
    CFG.layout.lattice_axes                  # attribute access, nested tables, lists stay lists

Overrides (handy for experiments, nothing has to be edited):

    python film.py --config other.toml       # another file instead of config.toml
    python film.py --set camera.elevation_deg=35 --set style.dot_scale=20
    DVN_CONFIG=other.toml python film.py

``--config`` and ``--set`` are read from ``sys.argv`` when the module is loaded (so that modules that read the
configuration at import, such as physics modules, see them too); the scripts declare the same options in their
``argparse`` so that ``--help`` shows them.  A missing key raises ``KeyError`` with the full path of the key.
"""

from __future__ import annotations

import ast
import os
import sys
import tomllib
from pathlib import Path
from typing import Any


class Cfg:
    """Read-only attribute view of a nested dictionary (a TOML table)."""

    def __init__(self, data: dict, path: str = "") -> None:
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_path", path)

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        path = object.__getattribute__(self, "_path")
        if name not in data:
            raise KeyError(f"config key '{path + name}' is missing")
        v = data[name]
        return Cfg(v, f"{path}{name}.") if isinstance(v, dict) else v

    __getitem__ = __getattr__

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("the configuration is read-only; use --set or a config file")

    def __contains__(self, name: str) -> bool:
        return name in object.__getattribute__(self, "_data")

    def keys(self):
        return object.__getattribute__(self, "_data").keys()

    def to_dict(self) -> dict:
        return object.__getattribute__(self, "_data")


def _peek(flag: str) -> list[str]:
    """Values of ``--flag value`` / ``--flag=value`` in sys.argv (without consuming them)."""
    out, argv = [], sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            out.append(argv[i + 1])
        elif a.startswith(flag + "="):
            out.append(a.split("=", 1)[1])
    return out


def _set_path(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    if keys[-1] not in d and not isinstance(value, dict):
        raise KeyError(f"--set: config key '{dotted}' does not exist")
    d[keys[-1]] = value


def _parse_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def load_texts(project_dir: str | Path, lang: str, name: str = "texts.toml") -> dict:
    """The table ``[lang]`` of ``project_dir/texts.toml`` (all the words and formulas of a film, one table per language)."""
    with open(Path(project_dir) / name, "rb") as fh:
        return tomllib.load(fh)[lang]


def load_config(project_dir: str | Path, name: str = "config.toml") -> Cfg:
    """Load ``project_dir/config.toml`` (or the file given by ``--config`` / ``DVN_CONFIG``) and apply ``--set`` overrides."""
    given = _peek("--config")
    path = Path(given[-1]) if given else (Path(os.environ["DVN_CONFIG"]) if "DVN_CONFIG" in os.environ else Path(project_dir) / name)
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    for item in _peek("--set"):
        if "=" not in item:
            raise ValueError(f"--set expects section.key=value, got '{item}'")
        key, text = item.split("=", 1)
        _set_path(data, key.strip(), _parse_value(text.strip()))
    return Cfg(data)
