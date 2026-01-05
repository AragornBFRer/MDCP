"""Helper wrapper to download WILDS datasets when the host certificate is invalid.

The upstream CodaLab mirror occasionally ships an expired TLS certificate,
which breaks urllib's default HTTPS verification. This loader disables
certificate checks before delegating to wilds.download_datasets' CLI so we can
still fetch the archives. Use only on trusted mirrors.
"""

import ssl
import sys
from pathlib import Path


_LOCAL_WILDS_PATH = Path(__file__).resolve().parent.parent / "get_data" / "get_wilds_data"
if _LOCAL_WILDS_PATH.exists():
    sys.path.insert(0, str(_LOCAL_WILDS_PATH))

from wilds.download_datasets import main as wilds_main  # type: ignore

# Disable HTTPS verification explicitly for urllib.
ssl._create_default_https_context = ssl._create_unverified_context


if __name__ == "__main__":
    wilds_main()
