"""Provenance: every committed corpus/reference file must be fingerprinted in MANIFEST.md."""

import hashlib
import re

from tests.conftest import CORPUS_DIR, REFERENCE_DIR

MANIFEST = CORPUS_DIR.parent / "MANIFEST.md"


def _manifest_hashes() -> dict[str, str]:
    rows = re.findall(r"^\| `([^`]+)` \|.*\| `([0-9a-f]{64})` \|$", MANIFEST.read_text(), re.M)
    return dict(rows)


def test_manifest_matches_committed_files():
    hashes = _manifest_hashes()
    files = [*CORPUS_DIR.glob("*.md"), REFERENCE_DIR / "thresholds.yaml"]
    for path in files:
        assert path.name in hashes, f"{path.name} is missing from MANIFEST.md"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashes[path.name] == actual, f"{path.name} changed; update MANIFEST.md"
