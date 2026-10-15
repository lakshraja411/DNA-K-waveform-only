
import io
import math
from pathlib import Path

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
    map_fitting_to_raw,
    kmeans_numpy,
    reorder_labels_by_height,
    centered_profiles,
    pointwise_cluster_medians,
)

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
    "event_data waveform → silhouette k selection → K-means → blockade–dwell scatter → midpoint-aligned waveform families"
)

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
for key, default in {
    "recording": None,
    "result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def read_bytes(uploaded):
    return uploaded.getvalue()


def load_recording(dataset_blob, eventdata_blob, fitting_blob, tolerance=1e-7):
    X, dataset_arrays = load_dataset(dataset_blob)
    eventdata_arrays = npz_to_dict(eventdata_blob, allow_pickle=True)
    fitting_arrays = npz_to_dict(fitting_blob, allow_pickle=True)
    df = dataset_table(X)

    fit_to_dataset, ds_status = map_fitting_to_dataset(
        fitting_arrays, X, tolerance=float(tolerance)
    )
    fit_to_raw, raw_status = map_fitting_to_raw(
        fitting_arrays, eventdata_arrays, tolerance=float(tolerance)
    )

    return dict(
        X=X,
        df=df,
        dataset_arrays=dataset_arrays,
        eventdata_arrays=eventdata_arrays,
        fitting_arrays=fitting_arrays,
        fit_to_dataset=fit_to_dataset,
        fit_to_raw=fit_to_raw,
        ds_status=ds_status,
        raw_status=raw_status,
    )


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
    """Positive blockade waveform for one NanoSense event_data event."""
    raw_id = int(raw_id)

    # Main NanoSense object-array layout.
    if "events" in eventdata_arrays:
        records = eventdata_arrays["events"]
        if raw_id < 0 or raw_id >= len(records):
            return None
        rec = records[raw_id]
        if not isinstance(rec, dict) or "event_data" not in rec:
            return None

        # event_data = current - baseline, so a downward event is negative.
        delta = np.asarray(rec["event_data"], float).ravel()
        delta = delta[np.isfinite(delta)]
        if len(delta) < 2:
            return None
        return -delta

    # Fallback keyed numeric layout.
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


@st.cache_data(show_spinner=False)
def build_waveform_matrix_cached(
    dataset_blob,
    eventdata_blob,
    fitting_blob,
    tolerance,
    vector_length,
    scaling,
):
    recording = load_recording(
        dataset_blob, eventdata_blob, fitting_blob, tolerance=float(tolerance)
    )

    df = recording["df"].copy()
    fit_to_dataset = recording["fit_to_dataset"]
    fit_to_raw = recording["fit_to_raw"]
    eventdata_arrays = recording["eventdata_arrays"]

    dataset_to_raw = {}
    for fid, dsrow in fit_to_dataset.items():
        if fid in fit_to_raw:
            dataset_to_raw[int(dsrow)] = int(fit_to_raw[fid])

    vectors = []
    rows = []
    original_lengths = []

    n = max(20, int(vector_length))

    for dsrow in sorted(dataset_to_raw):
        raw_id = dataset_to_raw[dsrow]
        blockade = raw_blockade_from_eventdata(eventdata_arrays, raw_id)

        if blockade is None or len(blockade) < 2:
            continue
        if not np.isfinite(blockade).all():
            continue

        original_lengths.append(len(blockade))

        # Native-time midpoint pad/crop.
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

    if len(vectors) < 3:
        raise ValueError(
            "Too few dataset ↔ event_data matched waveforms. "
            "Check the matching audit and timestamp tolerance."
        )

    X_wave = np.asarray(vectors, float)

    if scaling == "L2-normalised shape only":
        norms = np.linalg.norm(X_wave, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_cluster = X_wave / norms
    else:
        X_cluster = X_wave.copy()

    valid_df = df.set_index("dataset_row").loc[rows].reset_index()
    valid_rows = np.asarray(rows, int)

    fs = eventdata_sampling_rate(eventdata_arrays)

    meta = {
        "sampling_rate_hz": float(fs) if np.isfinite(fs) else np.nan,
        "vector_length": int(n),
        "n_events": int(len(X_cluster)),
        "fraction_cropped": float(
            np.mean(np.asarray(original_lengths) > n)
        ) if original_lengths else np.nan,
    }

    return recording, X_cluster, valid_df, valid_rows, meta


def diagnostics_fast(X, labels, silhouette_sample_size, random_state):
    labels = np.asarray(labels, int)
    if len(np.unique(labels)) < 2:
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }

    n = len(X)

    # Full silhouette if sample size is 0 or >= N; otherwise reproducible sample.
    if silhouette_sample_size <= 0 or silhouette_sample_size >= n:
        sil = float(silhouette_score(X, labels))
        sil_n = n
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


def scan_k_fast(
    X,
    k_min,
    k_max,
    scan_n_init,
    max_iter,
    random_state,
    silhouette_sample_size,
):
    rows = []
    upper = min(int(k_max), len(X) - 1)

    for k in range(int(k_min), upper + 1):
        labels, centers, inertia = kmeans_numpy(
            X,
            k,
            n_init=int(scan_n_init),
            max_iter=int(max_iter),
            random_state=int(random_state),
        )

        d = diagnostics_fast(
            X,
            labels,
            silhouette_sample_size=int(silhouette_sample_size),
            random_state=int(random_state),
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
        k_min=2,
        k_max=int(k_max),
        scan_n_init=int(scan_n_init),
        max_iter=int(max_iter),
        random_state=int(random_state),
        silhouette_sample_size=int(silhouette_sample_size),
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

    # Display labels only: shallow -> deep by median NanoSense height.
    labels, label_mapping = reorder_labels_by_height(valid_df, labels)

    final_diag = diagnostics_fast(
        X_cluster,
        labels,
        silhouette_sample_size=int(silhouette_sample_size),
        random_state=int(random_state),
    )

    out = valid_df.copy()
    out["cluster"] = labels

    return {
        "k": k,
        "labels": labels,
        "centers": centers,
        "inertia": inertia,
        "scan": scan,
        "diagnostics": final_diag,
        "valid_df": out,
        "label_mapping": label_mapping,
    }


def labels_by_fitting_event(recording, valid_df):
    row_to_fit = {
        int(dataset_row): int(fitting_id)
        for fitting_id, dataset_row in recording["fit_to_dataset"].items()
    }

    out = {}
    for _, row in valid_df.iterrows():
        dsrow = int(row["dataset_row"])
        if dsrow in row_to_fit:
            out[row_to_fit[dsrow]] = int(row["cluster"])
    return out


def figure_bytes(fig, fmt="png", dpi=600):
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format=fmt,
        dpi=dpi if fmt == "png" else None,
        bbox_inches="tight",
    )
    return buf.getvalue()


# ---------------------------------------------------------------------
# 1. Files
# ---------------------------------------------------------------------
st.subheader("1 · Load files")

c1, c2, c3 = st.columns(3)
dataset_file = c1.file_uploader("dataset.npz", type=["npz"])
eventdata_file = c2.file_uploader("event_data.npz", type=["npz"])
fitting_file = c3.file_uploader("event_fitting.npz", type=["npz"])

tolerance = st.number_input(
    "Timestamp matching tolerance (s)",
    min_value=1e-10,
    max_value=1e-3,
    value=1e-7,
    format="%.1e",
)

# ---------------------------------------------------------------------
# 2. Clustering settings
# ---------------------------------------------------------------------
st.subheader("2 · Whole-waveform clustering")

p1, p2, p3 = st.columns(3)

vector_length = p1.slider(
    "Native-time vector length (samples)",
    min_value=100,
    max_value=3000,
    value=850,
    step=50,
)

scaling = p2.selectbox(
    "Waveform scaling",
    ["Physical blockade (nA)", "L2-normalised shape only"],
)

mode = p3.radio(
    "Choose k",
    ["Silhouette", "Manual"],
)

q1, q2, q3, q4 = st.columns(4)

manual_k = q1.number_input(
    "Manual k",
    min_value=2,
    max_value=12,
    value=4,
    step=1,
)

k_max = q2.number_input(
    "Maximum k to test",
    min_value=3,
    max_value=12,
    value=8,
    step=1,
)

scan_n_init = q3.number_input(
    "Initialisations during k scan",
    min_value=1,
    max_value=100,
    value=10,
    step=1,
    help=(
        "10 is much faster than 100 for the silhouette scan. "
        "The final selected k is refit using the separate final initialisations below."
    ),
)

final_n_init = q4.number_input(
    "Final K-means initialisations",
    min_value=10,
    max_value=300,
    value=100,
    step=10,
)

r1, r2, r3 = st.columns(3)

silhouette_sample_size = r1.number_input(
    "Silhouette sample size",
    min_value=0,
    max_value=20000,
    value=3000,
    step=500,
    help=(
        "0 = use every event (slow for large datasets). "
        "3000 gives a reproducible approximate silhouette while K-means itself still uses ALL events."
    ),
)

random_state = r2.number_input(
    "Random seed",
    min_value=0,
    max_value=1_000_000,
    value=42,
    step=1,
)

display_window = r3.slider(
    "Waveform-family display window (samples)",
    min_value=200,
    max_value=3000,
    value=850,
    step=50,
)

st.caption(
    "For large datasets, speed comes mainly from sampling the silhouette calculation and using fewer initialisations during the k scan. "
    "The final K-means fit still uses every matched event."
)

# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
if st.button("Run whole-waveform clustering", type="primary"):
    if not (dataset_file and eventdata_file and fitting_file):
        st.error("Upload all three files first.")
    else:
        try:
            dataset_blob = read_bytes(dataset_file)
            eventdata_blob = read_bytes(eventdata_file)
            fitting_blob = read_bytes(fitting_file)

            with st.spinner("Building whole-waveform matrix..."):
                recording, X_cluster, valid_df, valid_rows, meta = build_waveform_matrix_cached(
                    dataset_blob,
                    eventdata_blob,
                    fitting_blob,
                    float(tolerance),
                    int(vector_length),
                    scaling,
                )

            with st.spinner("Scanning k and fitting final K-means..."):
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

            result["recording"] = recording
            result["X_cluster"] = X_cluster
            result["meta"] = meta
            result["display_window"] = int(display_window)

            st.session_state.recording = recording
            st.session_state.result = result

            st.success(f"Complete · selected k = {result['k']}")

        except Exception as exc:
            st.exception(exc)


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------
result = st.session_state.result

if result is not None:
    valid = result["valid_df"]
    scan = result["scan"]
    meta = result["meta"]
    diag = result["diagnostics"]

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matched events", len(valid))
    m2.metric("Selected k", result["k"])
    m3.metric("Silhouette", f"{diag['silhouette']:.3f}")
    m4.metric(
        "Cropped by clustering window",
        f"{100 * meta['fraction_cropped']:.2f}%"
        if np.isfinite(meta["fraction_cropped"])
        else "—",
    )

    fs = meta["sampling_rate_hz"]
    if np.isfinite(fs):
        st.caption(
            f"Sampling rate: {fs/1000:.1f} kHz · "
            f"{meta['vector_length']} samples = {meta['vector_length']/fs*1e3:.3f} ms · "
            f"silhouette evaluated on n={diag.get('silhouette_n', len(valid))} events."
        )

    # -------------------------------------------------------------
    # k diagnostics
    # -------------------------------------------------------------
    st.subheader("k selection")

    c1, c2 = st.columns([1.1, 1])

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

    ax.set_xlabel("Event width / dwell-like duration (ms)")
    ax.set_ylabel("Blockade height (nA)")
    ax.set_title("Whole-waveform K-means: blockade vs dwell")
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
    # Plot 2: waveform families
    # -------------------------------------------------------------
    st.subheader("Plot 2 · Midpoint-aligned waveform families")

    recording = result["recording"]
    labels_map = labels_by_fitting_event(recording, valid)

    info = centered_profiles(
        recording["fitting_arrays"],
        sorted(labels_map),
        labels_map,
        window_samples=int(result["display_window"]),
    )

    medians = pointwise_cluster_medians(info, 0.50) if info is not None else {}

    if not medians:
        st.warning("No matched event-fitting traces were available for the waveform-family plot.")
    else:
        x = info["time_ms"]
        clusters = sorted(medians)
        n_clusters = len(clusters)

        ncols = 2 if n_clusters > 1 else 1
        nrows = int(math.ceil(n_clusters / ncols))

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

            show_n = min(50, len(members))
            if show_n:
                step = max(1, len(members) // show_n)
                sample = members[::step][:show_n]

                for prof in sample:
                    ax.plot(
                        x,
                        prof,
                        color="gray",
                        alpha=0.10,
                        linewidth=0.8,
                    )

            ax.plot(
                x,
                item["median"],
                color="red",
                linewidth=2.3,
                label="Pointwise median",
            )

            ax.set_title(f"Cluster {c} · n={item['n']}")
            ax.set_xlabel("Time relative to event midpoint (ms)")
            ax.set_ylabel("Blockade (nA)")
            ax.legend(frameon=False)

        fig_w.suptitle("Midpoint-aligned waveform families", y=1.01)
        fig_w.tight_layout()

        st.pyplot(fig_w, use_container_width=True)
        st.caption(
            "Grey = measured member traces. Red = pointwise median. "
            "The median is shown only where at least 50% of the family has data."
        )

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
