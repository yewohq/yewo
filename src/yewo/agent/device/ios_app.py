"""Read identity metadata from an iOS app bundle or IPA."""

import plistlib
import zipfile
from pathlib import Path
from typing import Optional


def _identifier(data: bytes) -> Optional[str]:
    try:
        value = plistlib.loads(data).get("CFBundleIdentifier")
    except (plistlib.InvalidFileException, ValueError, TypeError):
        return None
    return value if isinstance(value, str) and value else None


def bundle_identifier(app_path: str) -> Optional[str]:
    """The app's CFBundleIdentifier, without requiring a driver echo."""
    path = Path(app_path)
    if path.is_dir():
        info = path / "Info.plist"
        try:
            return _identifier(info.read_bytes())
        except OSError:
            return None
    if path.suffix.lower() != ".ipa":
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            names = [
                name
                for name in archive.namelist()
                if name.startswith("Payload/") and name.endswith(".app/Info.plist")
            ]
            if len(names) != 1:
                return None
            return _identifier(archive.read(names[0]))
    except (OSError, zipfile.BadZipFile, KeyError):
        return None
