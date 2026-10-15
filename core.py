
from __future__ import annotations

import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

FEATURE_NAMES = [
    "height_nA",
    "fwhm_s",
    "height_at_fwhm_nA",
    "area_nA_s",
    "width_s",
    "skew",
    "kurtosis",
]

FEATURE_LABELS = {
    "height_nA": "Height / blockade (nA)",
    "fwhm_s": "FWHM (s)",
    "height_at_fwhm_nA": "Height at FWHM (nA)",
    "area_nA_s": "Area (nA·s)",
    "width_s": "Width / dwell-like duration (s)",
    "skew": "Skewness",
    "kurtosis": "Kurtosis",
}


def read_uploaded_bytes(uploaded):
    if uploaded is None:
        return None
    if hasattr(uploaded, "getvalue"):
        return uploaded.getvalue()
    if isinstance(uploaded, (bytes, bytearray)):
        return bytes(uploaded)
    raise TypeError("Expected Streamlit UploadedFile or bytes.")


def npz_to_dict(blob, allow_pickle=True):
    blob = read_uploaded_bytes(blob) if not isinstance(blob, (bytes, bytearray)) else bytes(blob)
    with np.load(io.BytesIO(blob), allow_pickle=allow_pickle) as z:
        return {k: z[k] for k in z.files}


def npz_bytes(arrays):
    f = io.BytesIO()
    np.savez_compressed(f, **arrays)
    return f.getvalue()


def jsonable_settings(x):
    try:
        if isinstance(x, np.ndarray) and x.shape == ():
            x = x.item()
    except Exception:
        pass
    if isinstance(x, bytes):
        x = x.decode(errors="replace")
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {"raw": x}
    if isinstance(x, dict):
        return x
    return {"raw": repr(x)}


def load_dataset(blob):
    arrays = npz_to_dict(blob, allow_pickle=True)
    if "X" not in arrays:
        raise ValueError("dataset.npz must contain an X matrix.")
    X = np.asarray(arrays["X"], float)
    if X.ndim != 2 or X.shape[1] < 7:
        raise ValueError(f"Expected X[event, feature] with >=7 columns; got {X.shape}.")
    return X, arrays


def dataset_table(X):
    df = pd.DataFrame(X[:, :7], columns=FEATURE_NAMES)
    df["fwhm_ms"] = df["fwhm_s"] * 1e3
    df["width_ms"] = df["width_s"] * 1e3
    df["area_nA_ms"] = df["area_nA_s"] * 1e3
    with np.errstate(divide="ignore", invalid="ignore"):
        df["fwhm_fraction"] = df["fwhm_s"] / df["width_s"]
    if X.shape[1] > 7:
        df["baseline_nA"] = X[:, 7]
    if X.shape[1] > 8:
        df["event_time_s"] = X[:, 8]
    if X.shape[1] > 9:
        df["event_time_alt_s"] = X[:, 9]
    df["dataset_row"] = np.arange(len(df), dtype=int)
    return df


def event_ids_from_keyed_arrays(arrays):
    ids = set()
    patterns = [
        r"EVENT_DATA_(\d+)_part_\d+",
        r"SEGMENT_INFO_(\d+)_.*",
        r"EVENT_ANALYSIS_(\d+)",
        r"INPUT_FIT_(\d+)",
        r"INPUT_LEVELS_(\d+)",
        r"INPUT_WIDTHS_(\d+)",
        r"REFINED_(\d+)_.*",
    ]
    for key in arrays:
        for pat in patterns:
            m = re.fullmatch(pat, key)
            if m:
                ids.add(int(m.group(1)))
                break
    return sorted(ids)


def fitting_event_starts(arrays):
    starts = {}
    for i in event_ids_from_keyed_arrays(arrays):
        bkey = f"EVENT_DATA_{i}_part_2"
        tkey = f"EVENT_DATA_{i}_part_0"
        if bkey in arrays:
            b = np.asarray(arrays[bkey], float).ravel()
            if len(b) >= 1 and np.isfinite(b[0]):
                starts[i] = float(b[0])
        elif tkey in arrays:
            t = np.asarray(arrays[tkey], float).ravel()
            if len(t):
                starts[i] = float(t[0])
    return starts


def map_fitting_to_dataset(fitting_arrays, X, tolerance=1e-7, start_column=8):
    starts = fitting_event_starts(fitting_arrays)
    mapping = {}
    rows_used = set()

    if X.shape[1] <= start_column:
        return mapping, pd.DataFrame(
            [{"event_id": i, "dataset_row": None, "status": "dataset has no event-time column"} for i in starts]
        )

    times = np.asarray(X[:, start_column], float)
    order = np.argsort(times)
    ordered = times[order]
    status = []

    for event_id, start in starts.items():
        lo = np.searchsorted(ordered, start - tolerance, side="left")
        hi = np.searchsorted(ordered, start + tolerance, side="right")
        candidates = [int(x) for x in order[lo:hi] if int(x) not in rows_used]

        if len(candidates) == 1:
            row = candidates[0]
            mapping[event_id] = row
            rows_used.add(row)
            note = "matched"
        elif len(candidates) == 0:
            row = None
            note = "no matching dataset timestamp"
        else:
            row = None
            note = "ambiguous dataset timestamp"

        status.append(
            dict(event_id=int(event_id), event_start_s=float(start), dataset_row=row, status=note)
        )

    return mapping, pd.DataFrame(status)


def _object_record_start(record):
    if isinstance(record, dict):
        for key in ("start_time", "event_start", "start"):
            if key in record:
                try:
                    return float(record[key])
                except Exception:
                    pass
    return np.nan


def raw_event_starts(eventdata_arrays):
    if "events" in eventdata_arrays:
        records = eventdata_arrays["events"]
        return {int(i): _object_record_start(r) for i, r in enumerate(records)}

    starts = {}
    ids = event_ids_from_keyed_arrays(eventdata_arrays)
    for i in ids:
        bkey = f"EVENT_DATA_{i}_part_2"
        if bkey in eventdata_arrays:
            b = np.asarray(eventdata_arrays[bkey], float).ravel()
            if len(b):
                starts[i] = float(b[0])
    return starts


def map_fitting_to_raw(fitting_arrays, eventdata_arrays, tolerance=1e-7):
    fstarts = fitting_event_starts(fitting_arrays)
    rstarts = raw_event_starts(eventdata_arrays)
    raw_ids = np.array(list(rstarts.keys()), int)
    raw_times = np.array([rstarts[i] for i in raw_ids], float)

    mapping = {}
    used = set()
    status = []

    for fid, start in fstarts.items():
        diff = np.abs(raw_times - start)
        candidates = [int(raw_ids[j]) for j in np.flatnonzero(diff <= tolerance) if int(raw_ids[j]) not in used]
        if len(candidates) == 1:
            rid = candidates[0]
            mapping[fid] = rid
            used.add(rid)
            note = "matched"
        elif len(candidates) == 0:
            rid = None
            note = "no matching eventdata timestamp"
        else:
            rid = None
            note = "ambiguous eventdata timestamp"
        status.append(dict(event_id=int(fid), raw_event_id=rid, status=note))

    return mapping, pd.DataFrame(status)


def minmax_scale(X):
    X = np.asarray(X, float)
    lo = np.nanmin(X, axis=0)
    hi = np.nanmax(X, axis=0)
    span = hi - lo
    constant = span == 0
    span[constant] = 1.0
    Z = -1.0 + 2.0 * (X - lo) / span
    return Z, lo, hi, constant


def kmeans_numpy(X, k, n_init=100, max_iter=2000, random_state=42):
    X = np.asarray(X, float)
    if len(X) <= k:
        raise ValueError("Need more events than clusters.")
    if not np.isfinite(X).all():
        raise ValueError("K-means matrix contains non-finite values.")

    master = np.random.default_rng(random_state)
    best_labels = None
    best_centers = None
    best_inertia = np.inf

    for _ in range(int(n_init)):
        rng = np.random.default_rng(master.integers(0, 2**32 - 1))

        centers = [X[int(rng.integers(len(X)))].copy()]
        while len(centers) < k:
            C = np.asarray(centers)
            d2 = np.min(np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2), axis=1)
            total = d2.sum()
            if not np.isfinite(total) or total <= 0:
                next_idx = int(rng.integers(len(X)))
            else:
                next_idx = int(rng.choice(len(X), p=d2 / total))
            centers.append(X[next_idx].copy())

        centers = np.asarray(centers)

        for _iteration in range(int(max_iter)):
            d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(d2, axis=1)

            new_centers = centers.copy()
            nearest = d2.min(axis=1)

            for c in range(k):
                members = X[labels == c]
                if len(members):
                    new_centers[c] = members.mean(axis=0)
                else:
                    farthest = int(np.argmax(nearest))
                    new_centers[c] = X[farthest]

            if np.allclose(new_centers, centers, rtol=1e-8, atol=1e-10):
                centers = new_centers
                break
            centers = new_centers

        d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        inertia = float(np.sum(d2[np.arange(len(X)), labels]))

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers, best_inertia


def clustering_diagnostics(X, labels):
    labels = np.asarray(labels, int)
    if len(np.unique(labels)) < 2:
        return dict(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
    return dict(
        silhouette=float(silhouette_score(X, labels)),
        davies_bouldin=float(davies_bouldin_score(X, labels)),
        calinski_harabasz=float(calinski_harabasz_score(X, labels)),
    )


def scan_k(X, k_min=2, k_max=8, n_init=100, max_iter=2000, random_state=42):
    rows = []
    upper = min(int(k_max), len(X) - 1)
    for k in range(int(k_min), upper + 1):
        labels, centers, inertia = kmeans_numpy(
            X, k, n_init=n_init, max_iter=max_iter, random_state=random_state
        )
        d = clustering_diagnostics(X, labels)
        counts = np.bincount(labels, minlength=k)
        rows.append(
            dict(
                k=int(k),
                silhouette=d["silhouette"],
                davies_bouldin=d["davies_bouldin"],
                calinski_harabasz=d["calinski_harabasz"],
                inertia=float(inertia),
                smallest_cluster=int(counts.min()),
                largest_cluster=int(counts.max()),
            )
        )
    return pd.DataFrame(rows)


def reorder_labels_by_height(df_valid, labels):
    temp = pd.DataFrame(
        {"height_nA": df_valid["height_nA"].to_numpy(float), "old_cluster": labels}
    )
    order = (
        temp.groupby("old_cluster")["height_nA"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    mapping = {int(old): int(new) for new, old in enumerate(order)}
    out = np.asarray([mapping[int(x)] for x in labels], int)
    return out, mapping


def pca_numpy(X):
    X = np.asarray(X, float)
    centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ Vt.T
    eig = S**2 / max(1, len(centered) - 1)
    explained = eig / eig.sum() if eig.sum() > 0 else np.zeros_like(eig)
    return scores, explained, Vt


def eta_squared(values, groups):
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    finite = np.isfinite(values)
    values = values[finite]
    groups = groups[finite]
    if not len(values):
        return np.nan
    grand = values.mean()
    ss_total = np.sum((values - grand) ** 2)
    if ss_total <= 0:
        return np.nan
    ss_between = 0.0
    for g in np.unique(groups):
        x = values[groups == g]
        ss_between += len(x) * (x.mean() - grand) ** 2
    return float(ss_between / ss_total)


def fitting_trace(fitting_arrays, event_id):
    tkey = f"EVENT_DATA_{event_id}_part_0"
    ykey = f"EVENT_DATA_{event_id}_part_1"
    bkey = f"EVENT_DATA_{event_id}_part_2"
    basekey = f"EVENT_DATA_{event_id}_part_4"
    if not all(k in fitting_arrays for k in (tkey, ykey, bkey)):
        return None
    t = np.asarray(fitting_arrays[tkey], float).ravel()
    y = np.asarray(fitting_arrays[ykey], float).ravel()
    bounds = np.asarray(fitting_arrays[bkey], float).ravel()
    if len(t) != len(y) or len(bounds) < 2:
        return None
    if basekey in fitting_arrays:
        baseline = np.asarray(fitting_arrays[basekey], float)
        if baseline.ndim == 0:
            baseline = np.full_like(y, float(baseline))
        baseline = baseline.ravel()
        if len(baseline) != len(y):
            return None
    else:
        pad = y[(t < bounds[0]) | (t >= bounds[1])]
        if len(pad) < 4:
            return None
        baseline = np.full_like(y, np.median(pad))
    blockade = baseline - y
    return dict(time=t, current=y, baseline=baseline, blockade=blockade, bounds=bounds[:2])


def centered_profiles(fitting_arrays, event_ids, labels_by_event, window_samples=850):
    rows = []
    dts = []
    valid = []

    for eid in event_ids:
        tr = fitting_trace(fitting_arrays, eid)
        if tr is None:
            continue
        t = tr["time"]
        b = tr["bounds"]
        idx = np.flatnonzero((t >= b[0]) & (t < b[1]))
        if len(idx) < 2:
            continue
        dt = float(np.median(np.diff(t)))
        dts.append(dt)
        valid.append((eid, tr, int(idx[0]), int(idx[-1])))

    if not valid:
        return None

    dt = float(np.median(dts))
    total = max(50, int(window_samples))
    center = total // 2
    profiles = []
    labels = []
    eids = []

    for eid, tr, start, end in valid:
        blockade = tr["blockade"]
        midpoint = int(round((start + end) / 2))
        src_left = midpoint - center
        src_right = src_left + total
        src0 = max(0, src_left)
        src1 = min(len(blockade), src_right)
        arr = np.full(total, np.nan)
        if src1 > src0:
            dst0 = src0 - src_left
            dst1 = dst0 + (src1 - src0)
            arr[dst0:dst1] = blockade[src0:src1]
        profiles.append(arr)
        labels.append(labels_by_event[eid])
        eids.append(eid)

    return dict(
        profiles=np.asarray(profiles, float),
        labels=np.asarray(labels, int),
        event_ids=np.asarray(eids, int),
        time_ms=(np.arange(total) - center) * dt * 1e3,
        data_index=np.arange(total),
        midpoint_index=center,
        sampling_rate_hz=1.0 / dt if dt > 0 else np.nan,
    )


def pointwise_cluster_medians(profile_info, min_coverage_fraction=0.5):
    if profile_info is None:
        return {}
    P = profile_info["profiles"]
    labels = profile_info["labels"]
    out = {}
    for c in sorted(np.unique(labels)):
        M = P[labels == c]
        coverage = np.sum(np.isfinite(M), axis=0)
        med = np.nanmedian(M, axis=0)
        med[coverage < max(1, math.ceil(min_coverage_fraction * len(M)))] = np.nan
        out[int(c)] = dict(
            median=med,
            members=M,
            coverage=coverage,
            n=int(len(M)),
        )
    return out


def segment_metrics(fitting_arrays, mapping_fit_to_dataset=None):
    rows = []
    for eid in event_ids_from_keyed_arrays(fitting_arrays):
        lkey = f"SEGMENT_INFO_{eid}_segment_mean_diffs"
        wkey = f"SEGMENT_INFO_{eid}_segment_widths_time"
        if lkey not in fitting_arrays or wkey not in fitting_arrays:
            continue

        levels = np.asarray(fitting_arrays[lkey], float).ravel()
        widths = np.asarray(fitting_arrays[wkey], float).ravel()
        n = min(len(levels), len(widths))
        levels = levels[:n]
        widths = widths[:n]

        ok = np.isfinite(levels) & np.isfinite(widths) & (widths > 0)
        levels = levels[ok]
        widths = widths[ok]
        if not len(levels):
            continue

        total = float(widths.sum())
        duration_weighted_blockade = float(np.sum(levels * widths) / total)
        duration_weighted_abs_blockade = float(np.sum(np.abs(levels) * widths) / total)
        ecd_signed = float(np.sum(levels * widths))
        ecd_abs = float(np.sum(np.abs(levels) * widths))

        duration_weighted_segment = float(np.sum(widths**2) / total)
        amp_weights = np.abs(levels)
        blockade_weighted_segment = (
            float(np.sum(amp_weights * widths) / amp_weights.sum())
            if amp_weights.sum() > 0
            else np.nan
        )

        rows.append(
            dict(
                event_id=int(eid),
                dataset_row=(
                    int(mapping_fit_to_dataset[eid])
                    if mapping_fit_to_dataset and eid in mapping_fit_to_dataset
                    else np.nan
                ),
                n_segments=int(len(levels)),
                total_segment_dwell_ms=total * 1e3,
                duration_weighted_mean_blockade_nA=duration_weighted_blockade,
                duration_weighted_mean_abs_blockade_nA=duration_weighted_abs_blockade,
                segment_ecd_signed_nA_ms=ecd_signed * 1e3,
                segment_ecd_abs_nA_ms=ecd_abs * 1e3,
                duration_weighted_segment_duration_ms=duration_weighted_segment * 1e3,
                blockade_weighted_segment_duration_ms=blockade_weighted_segment * 1e3,
                min_segment_blockade_nA=float(np.min(levels)),
                max_segment_blockade_nA=float(np.max(levels)),
            )
        )
    return pd.DataFrame(rows)


def _subset_keyed_npz(arrays, selected_ids):
    selected_ids = {int(x) for x in selected_ids}
    patterns = [
        r"EVENT_DATA_(\d+)_part_\d+",
        r"SEGMENT_INFO_(\d+)_.*",
        r"EVENT_ANALYSIS_(\d+)",
        r"INPUT_FIT_(\d+)",
        r"INPUT_LEVELS_(\d+)",
        r"INPUT_WIDTHS_(\d+)",
        r"REFINED_(\d+)_.*",
    ]

    out = {}
    for key, value in arrays.items():
        event_id = None
        for pat in patterns:
            m = re.fullmatch(pat, key)
            if m:
                event_id = int(m.group(1))
                break
        if event_id is None or event_id in selected_ids:
            out[key] = value
    return out


def subset_dataset_arrays(dataset_arrays, selected_rows):
    selected_rows = np.asarray(sorted(set(int(x) for x in selected_rows)), int)
    X = np.asarray(dataset_arrays["X"])
    n = len(X)
    out = {}
    for key, value in dataset_arrays.items():
        arr = np.asarray(value)
        if key == "X":
            out[key] = arr[selected_rows]
        elif arr.ndim >= 1 and len(arr) == n and key != "settings":
            out[key] = arr[selected_rows]
        else:
            out[key] = value
    out["original_dataset_rows"] = selected_rows
    return out


def subset_eventdata_arrays(eventdata_arrays, raw_ids):
    raw_ids = sorted(set(int(x) for x in raw_ids))
    if "events" in eventdata_arrays:
        out = dict(eventdata_arrays)
        records = eventdata_arrays["events"]
        out["events"] = records[raw_ids]
        out["original_eventdata_rows"] = np.asarray(raw_ids, int)
        return out
    return _subset_keyed_npz(eventdata_arrays, raw_ids)


def subset_fitting_arrays(fitting_arrays, fitting_ids):
    return _subset_keyed_npz(fitting_arrays, fitting_ids)


def export_group_zip(
    label,
    selected_dataset_rows,
    dataset_arrays,
    fitting_arrays,
    eventdata_arrays,
    fit_to_dataset,
    fit_to_raw,
    event_table=None,
    source_names=None,
):
    selected_dataset_rows = sorted(set(int(x) for x in selected_dataset_rows))
    selected_rows_set = set(selected_dataset_rows)

    fitting_ids = [
        int(fid)
        for fid, row in fit_to_dataset.items()
        if int(row) in selected_rows_set
    ]
    raw_ids = [
        int(fit_to_raw[fid])
        for fid in fitting_ids
        if fid in fit_to_raw
    ]

    dataset_subset = subset_dataset_arrays(dataset_arrays, selected_dataset_rows)
    fitting_subset = subset_fitting_arrays(fitting_arrays, fitting_ids)
    eventdata_subset = subset_eventdata_arrays(eventdata_arrays, raw_ids)

    source_names = source_names or {}
    ds_name = source_names.get("dataset", "selected.dataset.npz")
    fit_name = source_names.get("fitting", "selected.event_fitting.npz")
    raw_name = source_names.get("eventdata", "selected.event_data.npz")

    # Add label while preserving suffix.
    def labelled(name):
        p = Path(name)
        base_name = p.name
        if base_name.endswith(".npz"):
            base_name = base_name[:-4]
        return f"{base_name}_{label}.npz"

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(labelled(ds_name), npz_bytes(dataset_subset))
        z.writestr(labelled(fit_name), npz_bytes(fitting_subset))
        z.writestr(labelled(raw_name), npz_bytes(eventdata_subset))
        if event_table is not None:
            z.writestr(f"{label}_events.csv", event_table.to_csv(index=False))
        meta = dict(
            label=str(label),
            n_dataset_rows=len(selected_dataset_rows),
            n_fitting_events=len(fitting_ids),
            n_eventdata_events=len(raw_ids),
            original_dataset_rows=selected_dataset_rows,
            original_fitting_event_ids=fitting_ids,
            original_eventdata_ids=raw_ids,
        )
        z.writestr(f"{label}_provenance.json", json.dumps(meta, indent=2))
    return out.getvalue(), meta


def export_many_groups_zip(
    groups,
    dataset_arrays,
    fitting_arrays,
    eventdata_arrays,
    fit_to_dataset,
    fit_to_raw,
    full_table,
    source_names=None,
):
    outer = io.BytesIO()
    manifests = []
    with zipfile.ZipFile(outer, "w", zipfile.ZIP_DEFLATED) as z:
        for label, selected_rows in groups.items():
            rows_set = set(int(x) for x in selected_rows)
            table = full_table[full_table["dataset_row"].isin(rows_set)].copy()
            blob, meta = export_group_zip(
                label,
                selected_rows,
                dataset_arrays,
                fitting_arrays,
                eventdata_arrays,
                fit_to_dataset,
                fit_to_raw,
                event_table=table,
                source_names=source_names,
            )
            z.writestr(f"{label}.zip", blob)
            manifests.append(meta)
        z.writestr("manifest.json", json.dumps(manifests, indent=2))
    return outer.getvalue()


def parse_edges(text):
    vals = []
    for token in re.split(r"[,;\s]+", str(text).strip()):
        if token:
            vals.append(float(token))
    vals = sorted(set(vals))
    if len(vals) < 2:
        raise ValueError("Enter at least two numeric edges.")
    return vals


def gate_rows(values, edges):
    values = np.asarray(values, float)
    groups = {}
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i == len(edges) - 2:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        label = f"gate_{i+1}_{lo:g}_to_{hi:g}"
        groups[label] = np.flatnonzero(mask).astype(int).tolist()
    return groups
