from __future__ import annotations
from pathlib import Path
import shutil
import hashlib


class Feeder:
    """
    Copies new PDFs from raw storage to landing buffer.
    Uses sha256 to prevent re-feeding the same file again.
    """

    def __init__(self, raw_dir: Path, landing_dir: Path, state_dir: Path):
        self.raw_dir = raw_dir
        self.landing_dir = landing_dir
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.seen_file = self.state_dir / "seen_hashes.txt"

        if not self.seen_file.exists():
            self.seen_file.write_text("", encoding="utf-8")

    def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _load_seen(self) -> set[str]:
        return set(self.seen_file.read_text(encoding="utf-8").splitlines())

    def _save_seen(self, hashes: set[str]) -> None:
        self.seen_file.write_text("\n".join(sorted(hashes)) + "\n", encoding="utf-8")

    def feed(self, limit: int | None = None) -> dict:
        seen = self._load_seen()
        fed = 0
        skipped = 0

        pdfs = sorted(self.raw_dir.rglob("*.pdf"))

        for pdf in pdfs:
            file_hash = self._sha256(pdf)
            if file_hash in seen:
                skipped += 1
                continue

            dest = self.landing_dir / pdf.name
            # prevent name collision in landing
            if dest.exists():
                dest = self.landing_dir / f"{pdf.stem}_{file_hash[:8]}.pdf"

            shutil.copy2(str(pdf), str(dest))
            seen.add(file_hash)
            fed += 1

            if limit is not None and fed >= limit:
                break

        self._save_seen(seen)
        return {"fed": fed, "skipped": skipped, "raw_found": len(pdfs)}
