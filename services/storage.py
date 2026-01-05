
from __future__ import annotations
from pathlib import Path
import hashlib
import shutil

class StorageService:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.landing = data_dir / "landing"
        self.staging = data_dir / "staging"
        self.curated = data_dir / "curated"
        self.quarantine = data_dir / "quarantine"

        for d in [self.landing, self.staging, self.curated, self.quarantine]:
            d.mkdir(parents=True, exist_ok=True)

    def list_pdfs(self):
        candidates = []
        # Old location (kept for backward compatibility)
        candidates.extend(self.landing.glob("*.pdf"))
        # Actual location used in this repo
        candidates.extend((self.data_dir / "pdf" / "landing" / "raw").rglob("*.pdf"))

        return sorted(set(candidates))

    def sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def quarantine_file(self, path: Path, reason: str):
        target = self.quarantine / path.name
        shutil.move(str(path), str(target))
        (target.with_suffix(".reason.txt")).write_text(reason, encoding="utf-8")
