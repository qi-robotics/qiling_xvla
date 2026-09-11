#!/usr/bin/env python3
"""Keep LeRobot 0.5.0 importable when only the XVLA extra is installed.

policies/__init__.py and factory.py import every policy package. Groot's
package __init__ then imports modeling_groot, which crashes on Python 3.12.
"""

from __future__ import annotations

import site
from pathlib import Path

GROOT_INIT = '''\
from .configuration_groot import GrootConfig

__all__ = ["GrootConfig"]
'''

POLICIES_INIT = '''\
from .xvla.configuration_xvla import XVLAConfig as XVLAConfig

__all__ = ["XVLAConfig"]
'''


def main() -> None:
    site_pkg = Path(site.getsitepackages()[0])
    policies = site_pkg / "lerobot" / "policies"
    if not policies.is_dir():
        raise SystemExit(f"lerobot.policies not found under {site_pkg}")
    (policies / "groot" / "__init__.py").write_text(GROOT_INIT)
    (policies / "__init__.py").write_text(POLICIES_INIT)
    print(f"patched lerobot.policies for XVLA-only import at {policies}")


if __name__ == "__main__":
    main()
