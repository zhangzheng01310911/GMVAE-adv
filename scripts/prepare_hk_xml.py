#!/usr/bin/env python3
"""Convert archived Hong Kong rawSpeedVol XML snapshots to a training NPZ.

Aggregation contract
--------------------
* Node: detector_id.
* Invalid/offline lanes: missing.
* Volume: sum over valid lanes, both 30-second periods, and all snapshots in a
  requested output bin.
* Speed: volume-weighted mean; falls back to an unweighted valid-lane mean
  when every corresponding volume is zero/missing.
* Occupancy: mean of valid lane observations.
* Graph: symmetric top-k positive speed-correlation graph calculated from the
  first 70% of output timestamps only. This is leakage-free but is not a road
  connectivity graph; the manuscript must describe it as such.
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


STAMP = re.compile(r"rawSpeedVol_(\d{8}-\d{4})\.xml$")


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def child_text(element: ET.Element, name: str) -> str | None:
    wanted = name.lower()
    for child in element.iter():
        if local(child.tag) == wanted and child.text is not None:
            return child.text.strip()
    return None


def number(text: str | None) -> float:
    if text in (None, "", "-", "NA", "N/A"):
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def snapshot_time(path: Path) -> datetime:
    match = STAMP.search(path.name)
    if not match:
        raise ValueError(f"unexpected filename: {path.name}")
    return datetime.strptime(match.group(1), "%Y%m%d-%H%M")


def parse_snapshot(path: Path):
    root = ET.parse(path).getroot()
    aggregate = defaultdict(lambda: {"speed_num": 0.0, "speed_den": 0.0, "speed_sum": 0.0, "speed_n": 0, "volume": 0.0, "volume_n": 0, "occ_sum": 0.0, "occ_n": 0})
    for detector in root.iter():
        if local(detector.tag) != "detector":
            continue
        detector_id = child_text(detector, "detector_id")
        if not detector_id:
            continue
        lanes = [node for node in detector.iter() if local(node.tag) == "lane"]
        for lane in lanes:
            valid = child_text(lane, "valid")
            if valid is not None and valid.upper() != "Y":
                continue
            speed = number(child_text(lane, "speed"))
            volume = number(child_text(lane, "volume"))
            occupancy = number(child_text(lane, "occupancy"))
            item = aggregate[detector_id]
            if np.isfinite(volume):
                item["volume"] += volume
                item["volume_n"] += 1
            if np.isfinite(speed):
                item["speed_sum"] += speed
                item["speed_n"] += 1
                if np.isfinite(volume) and volume > 0:
                    item["speed_num"] += speed * volume
                    item["speed_den"] += volume
            if np.isfinite(occupancy):
                item["occ_sum"] += occupancy
                item["occ_n"] += 1
    result = {}
    for detector_id, item in aggregate.items():
        if item["speed_den"] > 0:
            speed = item["speed_num"] / item["speed_den"]
        elif item["speed_n"]:
            speed = item["speed_sum"] / item["speed_n"]
        else:
            speed = float("nan")
        volume = item["volume"] if item["volume_n"] else float("nan")
        occupancy = item["occ_sum"] / item["occ_n"] if item["occ_n"] else float("nan")
        result[detector_id] = (speed, volume, occupancy)
    return result


def correlation_graph(speed: np.ndarray, k: int, train_ratio: float = 0.7) -> np.ndarray:
    train = speed[: max(2, int(len(speed) * train_ratio))].astype(np.float64)
    means = np.nanmean(train, axis=0)
    global_mean = np.nanmean(train)
    means = np.where(np.isfinite(means), means, global_mean)
    filled = np.where(np.isnan(train), means[None, :], train)
    corr = np.corrcoef(filled, rowvar=False)
    corr = np.nan_to_num(corr, nan=-1.0)
    np.fill_diagonal(corr, -1.0)
    adjacency = np.zeros_like(corr, dtype=np.float32)
    k = min(k, corr.shape[0] - 1)
    for node in range(corr.shape[0]):
        neighbours = np.argpartition(corr[node], -k)[-k:]
        neighbours = neighbours[corr[node, neighbours] > 0]
        adjacency[node, neighbours] = 1.0
    return np.maximum(adjacency, adjacency.T)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Hong Kong rawSpeedVol XML for GMVAE")
    parser.add_argument("--input", required=True, help="directory containing rawSpeedVol_*.xml")
    parser.add_argument("--output", required=True, help="output .npz path")
    parser.add_argument("--frequency-minutes", type=int, default=5)
    parser.add_argument("--min-coverage", type=float, default=0.50, help="retain detectors present in at least this fraction of snapshots")
    parser.add_argument("--graph-k", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=None, help="software smoke test only")
    args = parser.parse_args()

    paths = sorted(Path(args.input).glob("rawSpeedVol_*.xml"))
    if args.max_files:
        paths = paths[: args.max_files]
    if not paths:
        raise FileNotFoundError(f"no rawSpeedVol_*.xml files in {args.input}")
    print(f"Found {len(paths)} XML files")

    # Pass 1 determines detector coverage without retaining every parsed row.
    coverage = defaultdict(int)
    good_paths = []
    failures = []
    for index, path in enumerate(paths, 1):
        try:
            record = parse_snapshot(path)
            for detector_id in record:
                coverage[detector_id] += 1
            good_paths.append(path)
        except (ET.ParseError, OSError, ValueError) as exc:
            failures.append({"file": path.name, "error": str(exc)})
        if index == 1 or index % 500 == 0 or index == len(paths):
            print(f"Coverage pass: {index}/{len(paths)}")
    threshold = args.min_coverage * len(good_paths)
    detector_ids = sorted(key for key, count in coverage.items() if count >= threshold)
    if len(detector_ids) < 2:
        raise RuntimeError("fewer than two detectors passed the coverage threshold")
    detector_index = {key: value for value, key in enumerate(detector_ids)}
    print(f"Retained {len(detector_ids)} detectors at coverage >= {args.min_coverage:.1%}")

    start = snapshot_time(good_paths[0])
    end = snapshot_time(good_paths[-1])
    frequency_seconds = args.frequency_minutes * 60
    bins = int((end - start).total_seconds() // frequency_seconds) + 1
    nodes = len(detector_ids)
    speed_num = np.zeros((bins, nodes), dtype=np.float64)
    speed_den = np.zeros((bins, nodes), dtype=np.float64)
    speed_sum = np.zeros((bins, nodes), dtype=np.float64)
    speed_count = np.zeros((bins, nodes), dtype=np.uint16)
    volume_sum = np.zeros((bins, nodes), dtype=np.float64)
    volume_count = np.zeros((bins, nodes), dtype=np.uint16)
    occupancy_sum = np.zeros((bins, nodes), dtype=np.float64)
    occupancy_count = np.zeros((bins, nodes), dtype=np.uint16)

    for index, path in enumerate(good_paths, 1):
        stamp = snapshot_time(path)
        bin_index = int((stamp - start).total_seconds() // frequency_seconds)
        record = parse_snapshot(path)
        for detector_id, (speed, volume, occupancy) in record.items():
            node = detector_index.get(detector_id)
            if node is None:
                continue
            if np.isfinite(speed):
                speed_sum[bin_index, node] += speed
                speed_count[bin_index, node] += 1
                if np.isfinite(volume) and volume > 0:
                    speed_num[bin_index, node] += speed * volume
                    speed_den[bin_index, node] += volume
            if np.isfinite(volume):
                volume_sum[bin_index, node] += volume
                volume_count[bin_index, node] += 1
            if np.isfinite(occupancy):
                occupancy_sum[bin_index, node] += occupancy
                occupancy_count[bin_index, node] += 1
        if index == 1 or index % 500 == 0 or index == len(good_paths):
            print(f"Aggregation pass: {index}/{len(good_paths)}")

    speed = np.full((bins, nodes), np.nan, dtype=np.float32)
    weighted = speed_den > 0
    fallback = (~weighted) & (speed_count > 0)
    speed[weighted] = (speed_num[weighted] / speed_den[weighted]).astype(np.float32)
    speed[fallback] = (speed_sum[fallback] / speed_count[fallback]).astype(np.float32)
    volume = np.full((bins, nodes), np.nan, dtype=np.float32)
    volume[volume_count > 0] = volume_sum[volume_count > 0].astype(np.float32)
    occupancy = np.full((bins, nodes), np.nan, dtype=np.float32)
    valid_occ = occupancy_count > 0
    occupancy[valid_occ] = (occupancy_sum[valid_occ] / occupancy_count[valid_occ]).astype(np.float32)
    values = np.stack([speed, volume, occupancy], axis=-1)
    adjacency = correlation_graph(speed, args.graph_k)
    timestamps = np.array([
        np.datetime64(start) + np.timedelta64(index * args.frequency_minutes, "m")
        for index in range(bins)
    ])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        values=values,
        adjacency=adjacency,
        timestamps=timestamps,
        detector_ids=np.asarray(detector_ids),
        channel_names=np.asarray(["speed_kmh", "volume_vehicles", "occupancy_percent"]),
    )
    report = {
        "source_files": len(paths),
        "parsed_files": len(good_paths),
        "failed_files": len(failures),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "frequency_minutes": args.frequency_minutes,
        "time_bins": bins,
        "detectors": nodes,
        "channels": ["speed_kmh", "volume_vehicles", "occupancy_percent"],
        "missing_ratio_total": float(np.isnan(values).mean()),
        "missing_ratio_by_channel": {
            "speed": float(np.isnan(speed).mean()),
            "volume": float(np.isnan(volume).mean()),
            "occupancy": float(np.isnan(occupancy).mean()),
        },
        "undirected_edges": int(np.count_nonzero(np.triu(adjacency, 1))),
        "graph": f"symmetric top-{args.graph_k} positive speed-correlation graph from first 70% timestamps",
        "failures": failures[:100],
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
