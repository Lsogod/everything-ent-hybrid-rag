from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.indexer import IndexerService


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronously index one or more files with current env settings.")
    parser.add_argument("files", nargs="+", help="file paths to index")
    args = parser.parse_args()

    service = IndexerService()
    service.store.ensure_index()

    results = []
    for raw_path in args.files:
        file_path = str(Path(raw_path).resolve())
        results.append(service.index_file(file_path))

    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
