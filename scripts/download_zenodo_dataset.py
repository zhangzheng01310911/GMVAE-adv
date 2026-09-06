#!/usr/bin/env python3
"""Download and verify the published Hong Kong dataset from Zenodo."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_RECORD = "22195838"
DEFAULT_FILENAME = "hk_traffic_20250601_20250621.npz"


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # Zenodo currently publishes an MD5 file checksum.
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the published HK-Traffic-GMVAE NPZ from Zenodo"
    )
    parser.add_argument("--record", default=DEFAULT_RECORD)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument(
        "--output",
        default=f"data/hong_kong/{DEFAULT_FILENAME}",
        help="destination file",
    )
    args = parser.parse_args()

    api_url = f"https://zenodo.org/api/records/{args.record}"
    request = Request(api_url, headers={"User-Agent": "GMVAE-Zenodo-Downloader/1.0"})
    with urlopen(request, timeout=120) as response:
        record = json.load(response)

    candidates = [item for item in record.get("files", []) if item["key"] == args.filename]
    if not candidates:
        available = ", ".join(item["key"] for item in record.get("files", []))
        raise FileNotFoundError(
            f"{args.filename!r} is not in Zenodo record {args.record}. "
            f"Available files: {available}"
        )

    metadata = candidates[0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")

    expected = metadata["checksum"].removeprefix("md5:")
    if output.is_file() and md5sum(output) == expected:
        print(f"Already downloaded and verified: {output}")
        return

    print(f"Downloading {args.filename} from Zenodo record {args.record} ...")
    request = Request(
        metadata["links"]["self"],
        headers={"User-Agent": "GMVAE-Zenodo-Downloader/1.0"},
    )
    with urlopen(request, timeout=300) as download:
        with partial.open("wb") as stream:
            while True:
                chunk = download.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)

    actual = md5sum(partial)
    if actual != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch: expected {expected}, obtained {actual}")

    partial.replace(output)
    print(f"Downloaded and verified: {output}")
    print(f"Dataset DOI: {record['links']['doi']}")


if __name__ == "__main__":
    main()
