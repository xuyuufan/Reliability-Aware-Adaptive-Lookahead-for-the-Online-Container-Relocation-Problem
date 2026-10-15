"""R modules with explicit read-only access to EM and H20 shared modules."""

from pathlib import Path
import sys
from pkgutil import extend_path

R_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = R_ROOT.parent
for root in (WORKSPACE_ROOT / "EM", WORKSPACE_ROOT / "Mijn_RL", WORKSPACE_ROOT):
    value = str(root)
    if root.is_dir() and value not in sys.path:
        sys.path.insert(0, value)

__path__ = extend_path(__path__, __name__)
