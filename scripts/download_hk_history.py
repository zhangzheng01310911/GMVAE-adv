#!/usr/bin/env python3
"""Download official DATA.GOV.HK historical traffic detector snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://app.data.gov.hk/v1/historical-archive"
DEFAULT_SOURCE = "https://resource.data.one.gov.hk/td/traffic-detectors/rawSpeedVol-all.xml"


def build_session() -> requests.Session:
    """Create a session that retries transient HTTP, connection, and TLS failures."""
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        status=8,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=2)
    session = requests.Session()
    session.headers.update({"User-Agent": "HK-Traffic-Research-Downloader/1.0"})
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_json(session, path, params):
    response = session.get(f"{API}/{path}", params=params, timeout=(30, 90))
    response.raise_for_status()
    return response.json()


def looks_complete(path: Path) -> bool:
    """Skip only non-empty files that look like XML, enabling safe resume."""
    if not path.is_file() or path.stat().st_size < 100:
        return False
    with path.open("rb") as stream:
        prefix = stream.read(256).lstrip()
    return prefix.startswith(b"<?xml") or prefix.startswith(b"<")


def dates_between(start: str, end: str):
    current = datetime.strptime(start, "%Y%m%d")
    final = datetime.strptime(end, "%Y%m%d")
    while current <= final:
        yield current.strftime("%Y%m%d")
        current += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--delay-min", type=float, default=0.20)
    parser.add_argument("--delay-max", type=float, default=0.60)
    args = parser.parse_args()
    if args.delay_min < 0 or args.delay_max < args.delay_min:
        parser.error("require 0 <= --delay-min <= --delay-max")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    session = build_session()
    items = []
    responses = []
    # A one-minute resource can exceed the API's 10,000-version response cap
    # in a month, so query one calendar day at a time.
    for date in dates_between(args.start, args.end):
        payload = get_json(
            session,
            "list-file-versions", {"url": args.source, "start": date, "end": date}
        )
        responses.append(payload)
        timestamps = payload.get("timestamps", [])
        print(f"{date}: {len(timestamps)} archived versions")
        items.extend(timestamps)
    (output / "versions.json").write_text(
        json.dumps(responses, indent=2), encoding="utf-8"
    )
    if not items:
        raise RuntimeError(
            "No archived versions returned. Verify the source URL with the "
            "list-files API and print versions.json for the server response."
        )
    manifest = []
    failed = []
    for index, stamp in enumerate(items, start=1):
        target = output / f"rawSpeedVol_{stamp}.xml"
        if not looks_complete(target):
            partial = target.with_suffix(target.suffix + ".part")
            last_error = None
            for attempt in range(1, args.max_attempts + 1):
                try:
                    response = session.get(
                        f"{API}/get-file",
                        params={"url": args.source, "time": stamp},
                        timeout=(30, 180),
                        allow_redirects=True,
                    )
                    response.raise_for_status()
                    content = response.content
                    if len(content) < 100 or not content.lstrip().startswith(b"<"):
                        raise ValueError(
                            f"response is not a plausible XML file ({len(content)} bytes)"
                        )
                    partial.write_bytes(content)
                    partial.replace(target)
                    last_error = None
                    break
                except (requests.RequestException, ValueError) as error:
                    last_error = error
                    if partial.exists():
                        partial.unlink()
                    wait = min(90.0, 2.0 ** (attempt - 1)) + random.uniform(0, 1)
                    print(
                        f"Retry {attempt}/{args.max_attempts} for {stamp} "
                        f"after {type(error).__name__}; waiting {wait:.1f}s",
                        flush=True,
                    )
                    time.sleep(wait)
                    # A fresh connection pool helps after repeated TLS EOF errors.
                    if isinstance(error, requests.exceptions.SSLError):
                        session.close()
                        session = build_session()
            if last_error is not None:
                failed.append(
                    {"time": stamp, "error": f"{type(last_error).__name__}: {last_error}"}
                )
                (output / "failed.json").write_text(
                    json.dumps(failed, indent=2), encoding="utf-8"
                )
                print(f"FAILED {stamp}; recorded in failed.json and continuing", flush=True)
                continue
            time.sleep(random.uniform(args.delay_min, args.delay_max))
        if index == 1 or index % 100 == 0 or index == len(items):
            print(f"Downloaded/verified {index}/{len(items)}: {target.name}")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest.append({"time": stamp, "file": target.name, "sha256": digest})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "failed.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")
    print(
        f"Finished: {len(manifest)} verified files, {len(failed)} failures. "
        "Rerun the same command to retry only missing/invalid files."
    )


if __name__ == "__main__":
    main()
