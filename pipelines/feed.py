from pathlib import Path
from services.feeder import Feeder


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"

    raw_dir = data_dir / "pdf" / "raw"
    landing_dir = data_dir / "landing"
    state_dir = data_dir / ".state"

    feeder = Feeder(raw_dir=raw_dir, landing_dir=landing_dir, state_dir=state_dir)
    stats = feeder.feed(limit=5)  # feed 5 files per run (safe default)
    print("FEED DONE:", stats)


if __name__ == "__main__":
    main()
