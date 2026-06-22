# SPDX-License-Identifier: Apache-2.0
"""HTTP serving utilities."""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    'create_app': ('sglang_omni.serve.openai_api', 'create_app'),
    'launch_server': ('sglang_omni.serve.launcher', 'launch_server'),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
