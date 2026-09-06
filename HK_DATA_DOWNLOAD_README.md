# Downloading and Preparing the Hong Kong Traffic Dataset

This document explains how to reconstruct the Hong Kong traffic dataset used
by the GMVAE experiments. The workflow uses two Python programs:

1. `scripts/download_hk_history.py` queries the official DATA.GOV.HK
   Historical Archive API and downloads archived XML snapshots.
2. `scripts/prepare_hk_xml.py` validates and aggregates the XML snapshots into
   a compressed NumPy (`.npz`) dataset suitable for model training.

## 1. Official data sources

The observations are provided by the Transport Department of the Government
of the Hong Kong Special Administrative Region through DATA.GOV.HK.

- Open-data portal: <https://data.gov.hk/>
- Live speed--volume--occupancy XML resource:
  <https://resource.data.one.gov.hk/td/traffic-detectors/rawSpeedVol-all.xml>
- Detector metadata:
  <https://static.data.gov.hk/td/traffic-data-strategic-major-roads/info/traffic_speed_volume_occ_info.csv>
- Historical Archive API documentation:
  <https://data.gov.hk/en/help/api-spec>
- DATA.GOV.HK Terms and Conditions:
  <https://data.gov.hk/en/terms-and-conditions>

The live XML URL contains the latest snapshot. Historical observations are
obtained through the official Historical Archive API.

## 2. Requirements

Python 3.10 or later is recommended. Install the two required packages:

```bash
python -m pip install numpy requests
```

Run all subsequent commands from the root directory of this repository.

## 3. Download historical XML snapshots

The following command downloads snapshots from June 1 through June 21, 2025,
inclusive:

```bash
python -u scripts/download_hk_history.py \
  --start 20250601 \
  --end 20250621 \
  --output data/hong_kong/raw_20250601_20250621
```

Dates must be written in `YYYYMMDD` format. The downloader queries one day at
a time because the Historical Archive API returns at most 10,000 versions in
one request.

The output directory contains:

- `rawSpeedVol_YYYYMMDD-HHMM.xml`: archived traffic snapshots;
- `versions.json`: unmodified version-list responses returned by the API;
- `manifest.json`: timestamp, filename, and SHA-256 checksum of every verified
  XML file;
- `failed.json`: snapshots that could not be downloaded after all retries.

The downloader is resumable. Re-running the same command skips existing XML
files that pass its basic validity check and retries missing or invalid files.
Temporary downloads use the `.part` suffix and are atomically renamed only
after a plausible XML response has been received.

To download a different period, change `--start` and `--end`. For example:

```bash
python -u scripts/download_hk_history.py \
  --start 20250801 \
  --end 20250831 \
  --output data/hong_kong/raw_20250801_20250831
```

Downloading many thousands of snapshots can take several hours. Progress is
printed after every 100 files.

## 4. Check the downloaded files

Count plausible XML files:

```bash
find data/hong_kong/raw_20250601_20250621 \
  -type f -name 'rawSpeedVol_*.xml' -size +100c | wc -l
```

Inspect any failures:

```bash
python -m json.tool \
  data/hong_kong/raw_20250601_20250621/failed.json
```

If `failed.json` is non-empty, run the download command again. The script will
retry only files that are missing or fail validation.

## 5. Convert XML snapshots to a training dataset

After downloading the raw snapshots, run:

```bash
python -u scripts/prepare_hk_xml.py \
  --input data/hong_kong/raw_20250601_20250621 \
  --output data/hong_kong/hk_traffic_20250601_20250621.npz \
  --frequency-minutes 5 \
  --min-coverage 0.50 \
  --graph-k 3 \
  2>&1 | tee logs/prepare_hk_20250601_20250621.log
```

Create the log directory first if necessary:

```bash
mkdir -p logs
```

The preprocessing program performs two passes over the XML collection. The
first pass measures detector coverage and excludes malformed files. The second
pass aggregates the retained observations into fixed five-minute intervals.

The processing rules are:

- invalid or offline lane records are treated as missing;
- detector speed is the volume-weighted mean of valid lane speeds, with the
  unweighted mean used when valid positive volume weights are unavailable;
- detector volume is the sum of valid lane volumes;
- detector occupancy is the mean of valid lane occupancy observations;
- detectors must be present in at least 50% of successfully parsed snapshots;
- missing observations remain IEEE `NaN` in the processed file;
- a symmetric top-3 positive speed-correlation graph is estimated using only
  the first 70% of timestamps.

The graph is a data-driven correlation graph, not a physical road-connectivity
graph. Restricting graph estimation to the first 70% of timestamps prevents
validation and test observations from leaking into graph construction.

## 6. Processed outputs

The command creates:

- `hk_traffic_20250601_20250621.npz`: processed arrays;
- `hk_traffic_20250601_20250621.report.json`: preprocessing statistics,
  missingness, graph size, and parsing failures;
- `logs/prepare_hk_20250601_20250621.log`: complete console log.

The NPZ archive contains:

| Key | Description |
| --- | --- |
| `values` | Array with shape `[time, detector, channel]` |
| `adjacency` | Symmetric detector adjacency matrix |
| `timestamps` | Five-minute timestamps in Hong Kong local time |
| `detector_ids` | Detector identifiers in array order |
| `channel_names` | `speed_kmh`, `volume_vehicles`, and `occupancy_percent` |

For the June 1--21, 2025 reconstruction used in our data preparation, the
quality report recorded 22,813 source files, 22,812 successfully parsed files,
6,048 five-minute time bins, 774 retained detectors, an overall missing ratio
of approximately 4.66%, and 1,926 undirected graph edges. One truncated XML
snapshot (`rawSpeedVol_20250617-0300.xml`) could not be parsed and was recorded
in the report rather than silently discarded.

## 7. Inspect the processed dataset

```bash
python -c "import numpy as np; d=np.load('data/hong_kong/hk_traffic_20250601_20250621.npz'); print({k: d[k].shape for k in d.files}); print(d['channel_names'])"
```

Expected principal dimensions for the June 1--21 reconstruction are:

```text
values:      (6048, 774, 3)
adjacency:   (774, 774)
timestamps:  (6048,)
detector_ids:(774,)
```

## 8. Reproducibility and attribution

Preserve the raw `versions.json`, download `manifest.json`, preprocessing
report, command-line arguments, and source URLs with every dataset release.
The raw observations should be attributed to:

> Transport Department, The Government of the Hong Kong Special
> Administrative Region, via DATA.GOV.HK.

The download and preprocessing scripts document how the processed arrays were
constructed; they do not change the ownership or licensing status of the
underlying government data. Users should review the current DATA.GOV.HK Terms
and Conditions before redistributing raw files.

## 9. Common problems

### No progress is printed

Use `python -u` as shown above to disable buffered console output. During XML
preprocessing, the first progress message is printed immediately and then
after every 500 files.

### A preprocessing process appears suspended

Check its status and resource usage:

```bash
pgrep -af 'prepare_hk_xml.py'
ps -p PROCESS_ID -o pid,etime,%cpu,%mem,rss,stat,cmd
```

Run only one preprocessing process for a given output path.

### One or more XML files are malformed

The preprocessing script records malformed files in the JSON quality report
and continues with the remaining snapshots. Do not manually conceal such
failures; retain the report as part of the reproducibility record.

### The output path cannot be found

Paths are interpreted relative to the directory from which the command is
executed. Run the command from the repository root or use absolute paths.
