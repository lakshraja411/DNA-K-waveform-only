import io
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from core import (
    load_dataset,
    dataset_table,
    npz_to_dict,
    map_fitting_to_dataset,
    raw_event_starts,
    kmeans_numpy,
    reorder_labels_by_height,
    fitting_trace,
    centered_profiles,
    pointwise_cluster_medians,
)


# =====================================================================
# Page + plotting defaults
# =====================================================================
st.set_page_config(page_title="Whole-Waveform K-Means", layout="wide")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "savefig.bbox": "tight",
})

st.title("Whole-Waveform K-Means")
st.caption(
    "dataset + event_data → whole-waveform K-means → blockade–dwell scatter → "
    "optional real-baseline family plots from event_fitting"
)

for key, default in {"result": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =====================================================================
# General helpers
# =====================================================================
def read_bytes(uploaded):
    return None if uploaded is None else uploaded.getvalue()


def figure_bytes(fig, fmt="png", dpi=600):
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format=fmt,
        dpi=dpi if fmt == "png" else None,
        bbox_inches="tight",
    )
    return buf.getvalue()


def apply_optional_limits(ax, enabled, xmin, xmax, ymin, ymax):
    if enabled:
        ax.set_xlim(float(xmin), float(xmax))
        ax.set_ylim(float(ymin), float(ymax))


def eventdata_sampling_rate(eventdata_arrays):
    if "sampling_rate" in eventdata_arrays:
        try:
            fs = float(np.asarray(eventdata_arrays["sampling_rate"]).item())
            if np.isfinite(fs) and fs > 0:
                return fs
        except Exception:
            pass
    return np.nan


def raw_blockade_from_eventdata(eventdata_arrays, raw_id):
    """Return the positive blockade waveform used for clustering."""
    raw_id = int(raw_id)

    # Main NanoSense object-array layout.
    if "events" in eventdata_arrays:
        records = eventdata_arrays["events"]
        if raw_id < 0 or raw_id >= len(records):
            return None
        rec = records[raw_id]
        if not isinstance(rec, dict) or "event_data" not in rec:
            return None

        # In this layout event_data = current - baseline.
        delta = np.asarray(rec["event_data"], float).ravel()
        delta = delta[np.isfinite(delta)]
        if len(delta) < 2:
            return None
        return -delta

    # Fallback keyed layout.
    ykey = f"EVENT_DATA_{raw_id}_part_1"
    basekey = f"EVENT_DATA_{raw_id}_part_4"

    if ykey not in eventdata_arrays:
        return None

    y = np.asarray(eventdata_arrays[ykey], float).ravel()
    if len(y) < 2:
        return None

    if basekey in eventdata_arrays:
        base = np.asarray(eventdata_arrays[basekey], float)
        if base.ndim == 0:
            blockade = float(base) - y
        else:
            base = base.ravel()
            if len(base) != len(y):
                return None
            blockade = base - y
    else:
        blockade = -y

    return blockade if np.isfinite(blockade).all() else None


# =====================================================================
# Direct event_data <-> dataset timestamp matching
# This removes event_fitting from the CLUSTERING pathway.
# =====================================================================
def _match_raw_to_dataset_column(eventdata_arrays, X, column, tolerance):
    raw_starts = raw_event_starts(eventdata_arrays)
    raw_ids = np.asarray(sorted(raw_starts), dtype=int)
    raw_times = np.asarray([raw_starts[i] for i in raw_ids], dtype=float)

    finite_raw = np.isfinite(raw_times)
    raw_ids = raw_ids[finite_raw]
    raw_times = raw_times[finite_raw]

    if X.shape[1] <= column:
        return {}, pd.DataFrame(), 0

    ds_times = np.asarray(X[:, column], float)
    order = np.argsort(ds_times)
    ordered = ds_times[order]

    mapping = {}
    used_rows = set()
    status = []

    for rid, start in zip(raw_ids, raw_times):
        lo = np.searchsorted(ordered, start - tolerance, side="left")
        hi = np.searchsorted(ordered, start + tolerance, side="right")
        candidates = [int(r) for r in order[lo:hi] if int(r) not in used_rows]

        if len(candidates) == 1:
            row = candidates[0]
            mapping[int(rid)] = row
            used_rows.add(row)
            note = "matched"
        elif len(candidates) == 0:
            row = None
            note = "no matching dataset timestamp"
        else:
            row = None
            note = "ambiguous dataset timestamp"

        status.append({
            "raw_event_id": int(rid),
            "event_start_s": float(start),
            "dataset_row": row,
            "status": note,
            "dataset_time_column": int(column),
        })

    return mapping, pd.DataFrame(status), len(mapping)


def map_raw_to_dataset(eventdata_arrays, X, tolerance=1e-7):
    """Use dataset X[:,8] when available; try X[:,9] too and retain the better mapping."""
    candidates = []
    for col in (8, 9):
        if X.shape[1] > col:
            candidates.append(
                _match_raw_to_dataset_column(eventdata_arrays, X, col, float(tolerance))
            )

    if not candidates:
        raise ValueError("dataset.npz has no event-time column (expected X[:,8] or X[:,9]).")

    best = max(candidates, key=lambda item: item[2])
    mapping, status, _ = best
    return mapping, status


# =====================================================================
# Whole-waveform matrix
# =====================================================================
@st.cache_data(show_spinner=False)
def build_waveform_matrix_cached(
    dataset_blob,
    eventdata_blob,
    tolerance,
    vector_length,
    scaling,
):
    X, dataset_arrays = load_dataset(dataset_blob)
    eventdata_arrays = npz_to_dict(eventdata_blob, allow_pickle=True)
    df = dataset_table(X)

    raw_to_dataset, matching_status = map_raw_to_dataset(
        eventdata_arrays, X, tolerance=float(tolerance)
    )

    vectors = []
    rows = []
    raw_ids_used = []
    original_lengths = []

    n = max(20, int(vector_length))

    for raw_id in sorted(raw_to_dataset):
        dsrow = int(raw_to_dataset[raw_id])
        blockade = raw_blockade_from_eventdata(eventdata_arrays, raw_id)

        if blockade is None or len(blockade) < 2:
            continue
        if not np.isfinite(blockade).all():
            continue

        original_lengths.append(len(blockade))

        # Native-time midpoint pad/crop used ONLY to make equal-length vectors
        # for K-means. The recorded blockade samples themselves are not altered.
        vec = np.zeros(n, dtype=float)
        center = n // 2
        m = len(blockade)

        if m >= n:
            src0 = max(0, m // 2 - center)
            src1 = src0 + n
            if src1 > m:
                src1 = m
                src0 = m - n
            vec[:] = blockade[src0:src1]
        else:
            dst0 = center - m // 2
            dst1 = dst0 + m
            vec[dst0:dst1] = blockade

        vectors.append(vec)
        rows.append(dsrow)
        raw_ids_used.append(int(raw_id))

    if len(vectors) < 3:
        raise ValueError(
            "Too few directly matched dataset ↔ event_data waveforms for clustering. "
            "Check the timestamp tolerance."
        )

    X_wave = np.asarray(vectors, float)

    if scaling == "L2-normalised shape only":
        norms = np.linalg.norm(X_wave, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_cluster = X_wave / norms
    else:
        X_cluster = X_wave.copy()

    valid_df = df.set_index("dataset_row").loc[rows].reset_index()
    valid_df["raw_event_id"] = raw_ids_used

    fs = eventdata_sampling_rate(eventdata_arrays)

    meta = {
        "sampling_rate_hz": float(fs) if np.isfinite(fs) else np.nan,
        "vector_length": int(n),
        "n_events": int(len(X_cluster)),
        "fraction_cropped": float(np.mean(np.asarray(original_lengths) > n)),
        "matched_raw_events": int(len(raw_to_dataset)),
    }

    return (
        X,
        dataset_arrays,
        eventdata_arrays,
        df,
        X_cluster,
        valid_df,
        matching_status,
        meta,
    )


# =====================================================================
# K-means + silhouette
# =====================================================================
def diagnostics_fast(X, labels, silhouette_sample_size, random_state):
    labels = np.asarray(labels, int)
    if len(np.unique(labels)) < 2:
        return {
            "silhouette": np.nan,
            "silhouette_n": 0,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }

    n = len(X)
    if silhouette_sample_size <= 0 or silhouette_sample_size >= n:
        sil_n = n
        sil = float(silhouette_score(X, labels))
    else:
        sil_n = int(min(silhouette_sample_size, n))
        sil = float(
            silhouette_score(
                X,
                labels,
                sample_size=sil_n,
                random_state=int(random_state),
            )
        )

    return {
        "silhouette": sil,
        "silhouette_n": sil_n,
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def scan_k_fast(X, k_max, scan_n_init, max_iter, random_state, silhouette_sample_size):
    rows = []
    upper = min(int(k_max), len(X) - 1)

    for k in range(2, upper + 1):
        labels, centers, inertia = kmeans_numpy(
            X,
            k,
            n_init=int(scan_n_init),
            max_iter=int(max_iter),
            random_state=int(random_state),
        )

        d = diagnostics_fast(
            X, labels, int(silhouette_sample_size), int(random_state)
        )
        counts = np.bincount(labels, minlength=k)

        rows.append({
            "k": int(k),
            "silhouette": d["silhouette"],
            "silhouette_n": d["silhouette_n"],
            "davies_bouldin": d["davies_bouldin"],
            "calinski_harabasz": d["calinski_harabasz"],
            "inertia": float(inertia),
            "smallest_cluster": int(counts.min()),
            "largest_cluster": int(counts.max()),
        })

    return pd.DataFrame(rows)


def cluster_whole_waveforms(
    X_cluster,
    valid_df,
    mode,
    manual_k,
    k_max,
    scan_n_init,
    final_n_init,
    max_iter,
    random_state,
    silhouette_sample_size,
):
    scan = scan_k_fast(
        X_cluster,
        int(k_max),
        int(scan_n_init),
        int(max_iter),
        int(random_state),
        int(silhouette_sample_size),
    )

    if mode == "Silhouette":
        k = int(scan.loc[scan["silhouette"].idxmax(), "k"])
    else:
        k = int(manual_k)

    labels, centers, inertia = kmeans_numpy(
        X_cluster,
        k,
        n_init=int(final_n_init),
        max_iter=int(max_iter),
        random_state=int(random_state),
    )

    # Only the DISPLAY IDs are reordered shallow -> deep.
    labels, label_mapping = reorder_labels_by_height(valid_df, labels)

    diag = diagnostics_fast(
        X_cluster, labels, int(silhouette_sample_size), int(random_state)
    )

    out = valid_df.copy()
    out["cluster"] = labels

    return {
        "k": k,
        "labels": labels,
        "centers": centers,
        "inertia": inertia,
        "scan": scan,
        "diagnostics": diag,
        "valid_df": out,
        "label_mapping": label_mapping,
    }


# =====================================================================
# Physical family-display functions (event_fitting only)
# =====================================================================
def labels_by_fitting_event(fitting_arrays, X, valid_df, tolerance):
    fit_to_dataset, status = map_fitting_to_dataset(
        fitting_arrays, X, tolerance=float(tolerance)
    )

    row_to_cluster = {
        int(row["dataset_row"]): int(row["cluster"])
        for _, row in valid_df.iterrows()
    }

    labels = {}
    for fid, dsrow in fit_to_dataset.items():
        if int(dsrow) in row_to_cluster:
            labels[int(fid)] = row_to_cluster[int(dsrow)]

    return labels, status


def start_aligned_profiles(
    fitting_arrays,
    event_ids,
    labels_by_event,
    pre_ms=0.5,
    post_ms=0.5,
    duration_percentile=95,
):
    """Start-align real fitting traces while preserving real baseline and dwell.

    For each event we use:
        real pre-event samples -> real event -> real post-event samples

    The event start is placed at t=0. No stretching/compression is performed and
    no baseline is fabricated. Because events have different durations, shorter
    traces naturally finish earlier; the unused tail of the common display array
    is NaN. The pointwise median is later restricted by a coverage threshold.

    Events above the chosen duration percentile are omitted from THIS DISPLAY
    only. Cluster membership and the blockade-dwell scatter are unchanged.
    """
    candidates = []
    dts = []
    durations_samples = []

    for eid in event_ids:
        tr = fitting_trace(fitting_arrays, eid)
        if tr is None:
            continue

        t = tr["time"]
        bounds = tr["bounds"]
        idx = np.flatnonzero((t >= bounds[0]) & (t < bounds[1]))
        if len(idx) < 2:
            continue

        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            continue

        start = int(idx[0])
        end = int(idx[-1])
        duration_samples = end - start + 1

        candidates.append((eid, tr, start, end, duration_samples))
        dts.append(dt)
        durations_samples.append(duration_samples)

    if not candidates:
        return None

    dt = float(np.median(dts))
    pre_samples = max(1, int(round(float(pre_ms) * 1e-3 / dt)))
    post_samples = max(1, int(round(float(post_ms) * 1e-3 / dt)))
    event_samples = max(
        2,
        int(np.ceil(np.percentile(durations_samples, float(duration_percentile)))),
    )

    total_samples = pre_samples + event_samples + post_samples

    profiles = []
    labels = []
    used_ids = []
    excluded_long = 0
    excluded_edges = 0

    for eid, tr, start, end, duration_samples in candidates:
        if duration_samples > event_samples:
            excluded_long += 1
            continue

        blockade = np.asarray(tr["blockade"], float)

        # Extract REAL baseline immediately before the event, the full real event,
        # and REAL baseline immediately after its own event end.
        src0 = start - pre_samples
        src1 = end + 1 + post_samples

        if src0 < 0 or src1 > len(blockade):
            excluded_edges += 1
            continue

        segment = blockade[src0:src1]
        if not np.isfinite(segment).all():
            continue

        # Common time axis is start-aligned. A short event therefore ends earlier.
        # After its real post-event baseline finishes, the remainder is unknown
        # (NaN), NOT artificially padded with zero.
        arr = np.full(total_samples, np.nan, dtype=float)
        ncopy = min(len(segment), total_samples)
        arr[:ncopy] = segment[:ncopy]

        profiles.append(arr)
        labels.append(labels_by_event[eid])
        used_ids.append(eid)

    if not profiles:
        return None

    time_ms = (np.arange(total_samples) - pre_samples) * dt * 1e3

    return {
        "profiles": np.asarray(profiles, float),
        "labels": np.asarray(labels, int),
        "event_ids": np.asarray(used_ids, int),
        "time_ms": time_ms,
        "event_start_index": int(pre_samples),
        "sampling_rate_hz": 1.0 / dt,
        "pre_ms": float(pre_ms),
        "post_ms": float(post_ms),
        "event_window_ms": event_samples * dt * 1e3,
        "duration_percentile": float(duration_percentile),
        "excluded_long": int(excluded_long),
        "excluded_edges": int(excluded_edges),
        "candidate_count": int(len(candidates)),
    }


# =====================================================================
# INPUTS
# =====================================================================
st.subheader("1 · Files")

c1, c2, c3 = st.columns(3)
dataset_file = c1.file_uploader("dataset.npz · required", type=["npz"])
eventdata_file = c2.file_uploader("event_data.npz · required", type=["npz"])
fitting_file = c3.file_uploader(
    "event_fitting.npz · optional",
    type=["npz"],
    help=(
        "Not used for K-means. It is only used to draw family waveforms with "
        "real pre-event and post-event baseline."
    ),
)

st.info(
    "Clustering now uses dataset.npz + event_data.npz directly. "
    "event_fitting.npz is completely optional for clustering."
)

tolerance = st.number_input(
    "Timestamp matching tolerance (s)",
    min_value=1e-10,
    max_value=1e-3,
    value=1e-7,
    format="%.1e",
)


# =====================================================================
# CLUSTERING CONTROLS
# =====================================================================
st.subheader("2 · Whole-waveform clustering")

p1, p2, p3 = st.columns(3)
vector_length = p1.slider(
    "Native-time vector length (samples)", 100, 3000, 850, 50
)
scaling = p2.selectbox(
    "Waveform scaling",
    ["Physical blockade (nA)", "L2-normalised shape only"],
)
mode = p3.radio("Choose k", ["Silhouette", "Manual"])

q1, q2, q3, q4 = st.columns(4)
manual_k = q1.number_input("Manual k", 2, 12, 4, 1)
k_max = q2.number_input("Maximum k to test", 3, 12, 8, 1)
scan_n_init = q3.number_input(
    "Initialisations during k scan", 1, 100, 10, 1,
    help="Use 10 for speed; the final selected k is refit more robustly below.",
)
final_n_init = q4.number_input("Final K-means initialisations", 10, 300, 100, 10)

r1, r2 = st.columns(2)
silhouette_sample_size = r1.number_input(
    "Silhouette sample size",
    0,
    20000,
    3000,
    500,
    help=(
        "0 = exact full silhouette. 3000 is a reproducible approximation for large datasets. "
        "K-means itself still uses every matched event."
    ),
)
random_state = r2.number_input("Random seed", 0, 1_000_000, 42, 1)


# =====================================================================
# FAMILY-DISPLAY CONTROLS
# =====================================================================
st.subheader("3 · Waveform family display")

if fitting_file is not None:
    d1, d2, d3, d4 = st.columns(4)
    display_alignment = d1.selectbox(
        "Alignment",
        ["Start-aligned + real baseline", "Midpoint-aligned"],
    )
    pre_ms = d2.number_input("Pre-event baseline (ms)", 0.05, 5.0, 0.5, 0.05)
    post_ms = d3.number_input("Post-event baseline (ms)", 0.05, 5.0, 0.5, 0.05)
    duration_percentile = d4.slider("Event-duration percentile shown", 80, 100, 95, 1)

    median_coverage = st.slider(
        "Minimum family coverage for red median",
        min_value=0.30,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "At each time point, the red median is drawn only if at least this fraction "
            "of that cluster still has real recorded signal available there."
        ),
    )

    display_window = st.slider(
        "Midpoint display window (samples) · used only in midpoint mode",
        200,
        3000,
        850,
        50,
    )
else:
    display_alignment = "Start-aligned + real baseline"
    pre_ms = 0.5
    post_ms = 0.5
    duration_percentile = 95
    median_coverage = 0.50
    display_window = 850
    st.caption(
        "No event_fitting file uploaded: clustering and blockade–dwell plotting will still work. "
        "Upload event_fitting if you want physical family waveforms with real baseline before and after each event."
    )


# =====================================================================
# PLOT EDITOR
# =====================================================================
default_waveform_xlabel = (
    "Time relative to event start (ms)"
    if display_alignment == "Start-aligned + real baseline"
    else "Time relative to event midpoint (ms)"
)

with st.expander("Plot editing options"):
    e1, e2 = st.columns(2)
    show_legends = e1.checkbox("Show legends", value=True)
    max_member_traces = e2.slider("Grey traces per cluster", 0, 100, 50, 5)

    st.markdown("**Blockade–dwell scatter**")
    a, b, c = st.columns(3)
    scatter_title = a.text_input("Scatter title", "Whole-waveform K-means: blockade vs dwell")
    scatter_xlabel = b.text_input("Scatter x label", "Event width / dwell-like duration (ms)")
    scatter_ylabel = c.text_input("Scatter y label", "Blockade height (nA)")

    scatter_custom_limits = st.checkbox("Custom scatter axis limits", value=False)
    s1, s2, s3, s4 = st.columns(4)
    scatter_xmin = s1.number_input("Scatter x min", value=0.0)
    scatter_xmax = s2.number_input("Scatter x max", value=5.0)
    scatter_ymin = s3.number_input("Scatter y min", value=0.0)
    scatter_ymax = s4.number_input("Scatter y max", value=5.0)

    st.markdown("**Waveform family panels**")
    a, b, c = st.columns(3)
    family_title = a.text_input("Family plot title", "Waveform families")
    family_xlabel = b.text_input("Family x label", default_waveform_xlabel)
    family_ylabel = c.text_input("Family y label", "Blockade (nA)")

    family_custom_limits = st.checkbox("Custom family axis limits", value=False)
    f1, f2, f3, f4 = st.columns(4)
    family_xmin = f1.number_input("Family x min", value=-0.5)
    family_xmax = f2.number_input("Family x max", value=5.0)
    family_ymin = f3.number_input("Family y min", value=-1.0)
    family_ymax = f4.number_input("Family y max", value=5.0)

    st.markdown("**Median overlay**")
    a, b, c = st.columns(3)
    overlay_title = a.text_input("Overlay title", "Overlay of median waveform families")
    overlay_xlabel = b.text_input("Overlay x label", default_waveform_xlabel)
    overlay_ylabel = c.text_input("Overlay y label", "Median blockade (nA)")

    overlay_custom_limits = st.checkbox("Custom overlay axis limits", value=False)
    o1, o2, o3, o4 = st.columns(4)
    overlay_xmin = o1.number_input("Overlay x min", value=-0.5)
    overlay_xmax = o2.number_input("Overlay x max", value=5.0)
    overlay_ymin = o3.number_input("Overlay y min", value=-1.0)
    overlay_ymax = o4.number_input("Overlay y max", value=5.0)


# =====================================================================
# RUN
# =====================================================================
if st.button("Run whole-waveform clustering", type="primary"):
    if dataset_file is None or eventdata_file is None:
        st.error("Upload dataset.npz and event_data.npz first.")
    else:
        try:
            dataset_blob = read_bytes(dataset_file)
            eventdata_blob = read_bytes(eventdata_file)
            fitting_blob = read_bytes(fitting_file) if fitting_file is not None else None

            with st.spinner("Matching dataset ↔ event_data and building waveform matrix..."):
                (
                    X,
                    dataset_arrays,
                    eventdata_arrays,
                    df,
                    X_cluster,
                    valid_df,
                    matching_status,
                    meta,
                ) = build_waveform_matrix_cached(
                    dataset_blob,
                    eventdata_blob,
                    float(tolerance),
                    int(vector_length),
                    scaling,
                )

            with st.spinner("Scanning k and fitting K-means..."):
                result = cluster_whole_waveforms(
                    X_cluster,
                    valid_df,
                    mode,
                    int(manual_k),
                    int(k_max),
                    int(scan_n_init),
                    int(final_n_init),
                    2000,
                    int(random_state),
                    int(silhouette_sample_size),
                )

            result.update({
                "X": X,
                "dataset_arrays": dataset_arrays,
                "eventdata_arrays": eventdata_arrays,
                "df": df,
                "X_cluster": X_cluster,
                "matching_status": matching_status,
                "meta": meta,
                "fitting_arrays": (
                    npz_to_dict(fitting_blob, allow_pickle=True)
                    if fitting_blob is not None else None
                ),
                "display_alignment": display_alignment,
                "pre_ms": float(pre_ms),
                "post_ms": float(post_ms),
                "duration_percentile": int(duration_percentile),
                "median_coverage": float(median_coverage),
                "display_window": int(display_window),
            })

            st.session_state.result = result
            st.success(f"Complete · selected k = {result['k']}")

        except Exception as exc:
            st.exception(exc)


# =====================================================================
# RESULTS
# =====================================================================
result = st.session_state.result

if result is not None:
    valid = result["valid_df"]
    scan = result["scan"]
    diag = result["diagnostics"]
    meta = result["meta"]

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matched events", len(valid))
    m2.metric("Selected k", result["k"])
    m3.metric("Silhouette", f"{diag['silhouette']:.3f}")
    m4.metric("Cropped in clustering vector", f"{100*meta['fraction_cropped']:.2f}%")

    fs = meta["sampling_rate_hz"]
    if np.isfinite(fs):
        st.caption(
            f"Clustering source: event_data only · {fs/1000:.1f} kHz · "
            f"{meta['vector_length']} samples = {meta['vector_length']/fs*1e3:.3f} ms · "
            f"silhouette evaluated on n={diag['silhouette_n']} events."
        )

    with st.expander("Direct dataset ↔ event_data matching audit"):
        st.dataframe(result["matching_status"], use_container_width=True)

    # -------------------------------------------------------------
    # k diagnostics
    # -------------------------------------------------------------
    st.subheader("k selection")
    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.dataframe(scan.round(4), use_container_width=True)
    with c2:
        fig_k, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(scan["k"], scan["silhouette"], marker="o", linewidth=2)
        ax.axvline(result["k"], linestyle="--", linewidth=1.2)
        ax.set_xlabel("Number of clusters, k")
        ax.set_ylabel("Silhouette score")
        ax.set_title("Silhouette-based k selection")
        fig_k.tight_layout()
        st.pyplot(fig_k, use_container_width=True)

    # -------------------------------------------------------------
    # Plot 1: blockade vs dwell
    # -------------------------------------------------------------
    st.subheader("Plot 1 · Blockade vs dwell")
    fig_bd, ax = plt.subplots(figsize=(8.5, 5.7))

    for c, g in valid.groupby("cluster"):
        ax.scatter(
            g["width_ms"],
            g["height_nA"],
            s=18,
            alpha=0.45,
            label=f"Cluster {c} · n={len(g)}",
        )

    ax.set_xlabel(scatter_xlabel)
    ax.set_ylabel(scatter_ylabel)
    ax.set_title(scatter_title)
    apply_optional_limits(
        ax,
        scatter_custom_limits,
        scatter_xmin,
        scatter_xmax,
        scatter_ymin,
        scatter_ymax,
    )
    if show_legends:
        ax.legend(frameon=False)
    fig_bd.tight_layout()
    st.pyplot(fig_bd, use_container_width=True)

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download scatter PNG",
        figure_bytes(fig_bd, "png", 600),
        "waveform_kmeans_blockade_dwell.png",
        "image/png",
    )
    d2.download_button(
        "Download scatter PDF",
        figure_bytes(fig_bd, "pdf"),
        "waveform_kmeans_blockade_dwell.pdf",
        "application/pdf",
    )

    # -------------------------------------------------------------
    # Optional fitting-based physical family plots
    # -------------------------------------------------------------
    fitting_arrays = result.get("fitting_arrays")

    if fitting_arrays is None:
        st.subheader("Waveform family plots")
        st.info(
            "K-means is complete without event_fitting. Upload event_fitting and rerun if you want "
            "the baseline → blockade → baseline family panels and median overlay."
        )
    else:
        labels_map, fitting_status = labels_by_fitting_event(
            fitting_arrays,
            result["X"],
            valid,
            float(tolerance),
        )

        if result["display_alignment"] == "Start-aligned + real baseline":
            info = start_aligned_profiles(
                fitting_arrays,
                sorted(labels_map),
                labels_map,
                pre_ms=result["pre_ms"],
                post_ms=result["post_ms"],
                duration_percentile=result["duration_percentile"],
            )

            if info is not None:
                medians = pointwise_cluster_medians(info, float(result["median_coverage"]))
                x = info["time_ms"]
                default_xlabel = "Time relative to event start (ms)"
            else:
                medians = {}
                x = None
                default_xlabel = "Time relative to event start (ms)"
        else:
            info = centered_profiles(
                fitting_arrays,
                sorted(labels_map),
                labels_map,
                window_samples=result["display_window"],
            )
            medians = pointwise_cluster_medians(info, float(result["median_coverage"])) if info is not None else {}
            x = info["time_ms"] if info is not None else None
            default_xlabel = "Time relative to event midpoint (ms)"

        if not medians:
            st.warning("No valid fitting traces were available for the family waveform display.")
        else:
            clusters = sorted(medians)
            n_clusters = len(clusters)
            ncols = 2 if n_clusters > 1 else 1
            nrows = int(math.ceil(n_clusters / ncols))

            st.subheader("Plot 2 · Waveform families")

            if result["display_alignment"] == "Start-aligned + real baseline":
                st.caption(
                    f"Start aligned at t=0 with REAL fitting-trace baseline. "
                    f"Requested pre/post baseline: {info['pre_ms']:.2f}/{info['post_ms']:.2f} ms. "
                    f"Display candidates: {info['candidate_count']}; used: {len(info['event_ids'])}; "
                    f"excluded as longest tail: {info['excluded_long']}; "
                    f"excluded for insufficient real pre/post signal: {info['excluded_edges']}. "
                    f"Red median requires ≥{100*float(result['median_coverage']):.0f}% cluster coverage at each time point."
                )

            fig_w, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(7 * ncols, 3.8 * nrows),
                squeeze=False,
            )
            axes = axes.ravel()

            for ax in axes[n_clusters:]:
                ax.axis("off")

            for ax, c in zip(axes, clusters):
                item = medians[c]
                members = item["members"]

                show_n = min(int(max_member_traces), len(members))
                if show_n:
                    step = max(1, len(members) // show_n)
                    sample = members[::step][:show_n]
                    for prof in sample:
                        ax.plot(x, prof, color="gray", alpha=0.10, linewidth=0.8)

                ax.plot(
                    x,
                    item["median"],
                    color="red",
                    linewidth=2.3,
                    label="Pointwise median",
                )

                if result["display_alignment"] == "Start-aligned + real baseline":
                    ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)

                ax.set_title(f"Cluster {c} · n={item['n']}")
                ax.set_xlabel(family_xlabel if family_xlabel else default_xlabel)
                ax.set_ylabel(family_ylabel)
                apply_optional_limits(
                    ax,
                    family_custom_limits,
                    family_xmin,
                    family_xmax,
                    family_ymin,
                    family_ymax,
                )
                if show_legends:
                    ax.legend(frameon=False)

            fig_w.suptitle(family_title, y=1.01)
            fig_w.tight_layout()
            st.pyplot(fig_w, use_container_width=True)

            d1, d2 = st.columns(2)
            d1.download_button(
                "Download waveform PNG",
                figure_bytes(fig_w, "png", 600),
                "waveform_kmeans_family_profiles.png",
                "image/png",
            )
            d2.download_button(
                "Download waveform PDF",
                figure_bytes(fig_w, "pdf"),
                "waveform_kmeans_family_profiles.pdf",
                "application/pdf",
            )

            # ---------------------------------------------------------
            # Plot 3: overlay
            # ---------------------------------------------------------
            st.subheader("Plot 3 · Median waveform overlay")
            fig_overlay, ax = plt.subplots(figsize=(9, 5.2))

            for c in clusters:
                ax.plot(
                    x,
                    medians[c]["median"],
                    linewidth=2.2,
                    label=f"Cluster {c} · n={medians[c]['n']}",
                )

            if result["display_alignment"] == "Start-aligned + real baseline":
                ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)

            ax.set_xlabel(overlay_xlabel if overlay_xlabel else default_xlabel)
            ax.set_ylabel(overlay_ylabel)
            ax.set_title(overlay_title)
            apply_optional_limits(
                ax,
                overlay_custom_limits,
                overlay_xmin,
                overlay_xmax,
                overlay_ymin,
                overlay_ymax,
            )
            if show_legends:
                ax.legend(frameon=False)
            fig_overlay.tight_layout()
            st.pyplot(fig_overlay, use_container_width=True)

            o1, o2 = st.columns(2)
            o1.download_button(
                "Download overlay PNG",
                figure_bytes(fig_overlay, "png", 600),
                "waveform_kmeans_median_overlay.png",
                "image/png",
            )
            o2.download_button(
                "Download overlay PDF",
                figure_bytes(fig_overlay, "pdf"),
                "waveform_kmeans_median_overlay.pdf",
                "application/pdf",
            )
