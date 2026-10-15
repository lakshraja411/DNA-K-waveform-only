
import io
import re
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from core import (
    FEATURE_NAMES,
    read_uploaded_bytes,
    load_dataset,
    dataset_table,
    npz_to_dict,
    map_fitting_to_dataset,
    map_fitting_to_raw,
    minmax_scale,
    kmeans_numpy,
    clustering_diagnostics,
    scan_k,
    reorder_labels_by_height,
    pca_numpy,
    eta_squared,
    centered_profiles,
    pointwise_cluster_medians,
    segment_metrics,
    export_many_groups_zip,
    parse_edges,
    gate_rows,
)

st.set_page_config(page_title="DNA K-Means Workbench", layout="wide")

# Publication-oriented Matplotlib defaults. Display figures are rendered at a
# higher DPI, while downloadable raster figures default to 600 dpi. PDF/SVG
# downloads remain vector graphics.
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

st.title("DNA K-Means Workbench")
st.caption(
    "Choose scalar-feature or whole-waveform K-means → waveform families → physical plots → current gates → segment-weighted analysis"
)

with st.sidebar:
    st.subheader("Figure output")
    st.select_slider(
        "PNG export resolution (dpi)",
        options=[300, 600, 900],
        value=600,
        key="figure_export_dpi",
        help="PDF and SVG are vector formats and are not limited by raster DPI.",
    )
    st.checkbox(
        "Transparent export background",
        value=False,
        key="figure_export_transparent",
    )
    st.caption(
        "Every displayed Matplotlib figure has its own Save this figure panel with PNG, PDF and SVG downloads."
    )
    st.caption(
        "Mean/median overlay controls appear inside Edit plot appearance wherever a statistical centre is meaningful. "
        "The red median waveform remains the waveform-family representative."
    )

APP_STATE_VERSION = "4.1"
if st.session_state.get("_app_state_version") != APP_STATE_VERSION:
    # Results from older deployed code can have a different summary-table schema.
    # Clear only computed analysis state; uploaded widgets themselves remain in the page.
    for _key in [
        "loaded",
        "cluster_result",
        "multi_compare",
        "cluster_export_blob",
        "gate_export_blob",
        "gate_groups",
        "gate_table",
    ]:
        st.session_state.pop(_key, None)
    st.session_state["_app_state_version"] = APP_STATE_VERSION

# -----------------------------
# session state
# -----------------------------
for key, default in {
    "loaded": None,
    "cluster_result": None,
    "multi_compare": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# -----------------------------
# helpers
# -----------------------------
def load_recording(dataset_blob, eventdata_blob, fitting_blob, dataset_name, eventdata_name, fitting_name, tolerance):
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

    row_to_fit = {int(row): int(fid) for fid, row in fit_to_dataset.items()}
    df["event_id"] = [row_to_fit.get(int(i), np.nan) for i in df["dataset_row"]]

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
        source_names=dict(
            dataset=dataset_name,
            eventdata=eventdata_name,
            fitting=fitting_name,
        ),
    )


def _eventdata_sampling_rate(eventdata_arrays):
    if "sampling_rate" in eventdata_arrays:
        try:
            fs = float(np.asarray(eventdata_arrays["sampling_rate"]).item())
            if np.isfinite(fs) and fs > 0:
                return fs
        except Exception:
            pass
    return np.nan


def _raw_blockade_from_eventdata(eventdata_arrays, raw_id):
    """Return positive blockade samples for one NanoSense event_data record.

    In the inspected NanoSense object-array layout, record['event_data'] is
    current minus baseline, so a downward event is negative and blockade is -delta.
    """
    raw_id = int(raw_id)
    if "events" in eventdata_arrays:
        records = eventdata_arrays["events"]
        if raw_id < 0 or raw_id >= len(records):
            return None
        rec = records[raw_id]
        if not isinstance(rec, dict) or "event_data" not in rec:
            return None
        delta = np.asarray(rec["event_data"], float).ravel()
        delta = delta[np.isfinite(delta)]
        if len(delta) < 2:
            return None
        return -delta

    # Fallback for keyed numeric layouts.
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
        # If no baseline is available, assume this keyed vector is already
        # baseline-relative current and invert dips to positive blockade.
        blockade = -y
    return blockade if np.isfinite(blockade).all() else None


def build_eventdata_waveform_matrix(
    recording,
    representation="Native-time midpoint pad/crop",
    native_samples=850,
    resample_points=200,
    waveform_scaling="Physical blockade (nA)",
):
    """Create one fixed-length event_data waveform vector per matched dataset row.

    Native-time mode midpoint-centres the original samples and pads shorter events
    with physical baseline (0 nA blockade); longer events are centre-cropped.

    Duration-normalised mode linearly resamples the complete event to a fixed
    number of positions from 0 to 100% event duration. It preserves amplitude but
    intentionally removes absolute dwell time from the clustering coordinates.
    """
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

    fs = _eventdata_sampling_rate(eventdata_arrays)

    for dsrow in sorted(dataset_to_raw):
        raw_id = dataset_to_raw[dsrow]
        blockade = _raw_blockade_from_eventdata(eventdata_arrays, raw_id)
        if blockade is None or len(blockade) < 2:
            continue

        original_lengths.append(len(blockade))

        if representation == "Duration-normalised resample":
            n = max(16, int(resample_points))
            old_x = np.linspace(0.0, 1.0, len(blockade))
            new_x = np.linspace(0.0, 1.0, n)
            vec = np.interp(new_x, old_x, blockade)
        else:
            n = max(20, int(native_samples))
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
                if dst0 < 0:
                    blockade = blockade[-dst0:]
                    dst0 = 0
                    dst1 = len(blockade)
                if dst1 > n:
                    blockade = blockade[: n - dst0]
                    dst1 = n
                vec[dst0:dst1] = blockade

        if not np.isfinite(vec).all():
            continue
        vectors.append(vec)
        rows.append(dsrow)

    if len(vectors) < 3:
        raise ValueError(
            "Too few dataset ↔ event_data matched waveforms for clustering. "
            "Check the timestamp matching audit in Step 1."
        )

    X_wave = np.asarray(vectors, float)

    if waveform_scaling == "L2-normalised shape only":
        norms = np.linalg.norm(X_wave, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_cluster = X_wave / norms
    else:
        X_cluster = X_wave.copy()

    valid_df = df.set_index("dataset_row").loc[rows].reset_index()
    valid_rows = np.asarray(rows, int)

    if representation == "Duration-normalised resample":
        axis = np.linspace(0.0, 100.0, X_cluster.shape[1])
        axis_label = "Normalised event position (%)"
    else:
        if np.isfinite(fs) and fs > 0:
            axis = (np.arange(X_cluster.shape[1]) - X_cluster.shape[1] // 2) / fs * 1e3
            axis_label = "Time relative to event midpoint (ms)"
        else:
            axis = np.arange(X_cluster.shape[1]) - X_cluster.shape[1] // 2
            axis_label = "Centered sample index"

    meta = dict(
        representation=representation,
        waveform_scaling=waveform_scaling,
        sampling_rate_hz=float(fs) if np.isfinite(fs) else np.nan,
        vector_length=int(X_cluster.shape[1]),
        original_length_median=float(np.median(original_lengths)),
        original_length_min=int(np.min(original_lengths)),
        original_length_max=int(np.max(original_lengths)),
        axis=axis,
        axis_label=axis_label,
    )
    return X_cluster, valid_df, valid_rows, meta


def run_kmeans_pipeline(
    recording,
    mode,
    manual_k,
    kmax,
    n_init,
    random_state,
    clustering_input="NanoSense 7 scalar features",
    waveform_representation="Native-time midpoint pad/crop",
    waveform_native_samples=850,
    waveform_resample_points=200,
    waveform_scaling="Physical blockade (nA)",
):
    df = recording["df"].copy()

    if clustering_input == "event_data whole waveform":
        X_cluster, valid_df, valid_rows, waveform_meta = build_eventdata_waveform_matrix(
            recording,
            representation=waveform_representation,
            native_samples=int(waveform_native_samples),
            resample_points=int(waveform_resample_points),
            waveform_scaling=waveform_scaling,
        )
        finite_mask = np.zeros(len(df), dtype=bool)
        finite_mask[valid_rows] = True
        feature_min = feature_max = constant = None
        input_type = "waveform"
    else:
        X_features = df[FEATURE_NAMES].to_numpy(float)
        finite_mask = np.all(np.isfinite(X_features), axis=1)
        X_valid = X_features[finite_mask]
        valid_df = df.loc[finite_mask].copy().reset_index(drop=True)
        valid_rows = valid_df["dataset_row"].to_numpy(int)
        if len(valid_df) < 3:
            raise ValueError("Too few valid events for clustering.")
        X_cluster, feature_min, feature_max, constant = minmax_scale(X_valid)
        waveform_meta = None
        input_type = "scalar"

    if len(valid_df) < 3:
        raise ValueError("Too few valid events for clustering.")

    scan = scan_k(
        X_cluster,
        k_min=2,
        k_max=int(kmax),
        n_init=int(n_init),
        max_iter=2000,
        random_state=int(random_state),
    )
    if len(scan) == 0:
        raise ValueError("k scan failed. Check that enough events remain after filtering.")

    if mode == "Silhouette":
        k = int(scan.loc[scan["silhouette"].idxmax(), "k"])
    else:
        k = int(manual_k)

    if k >= len(valid_df):
        raise ValueError(f"k={k} requires more than {k} valid events; only {len(valid_df)} are available.")

    labels, centers, inertia = kmeans_numpy(
        X_cluster,
        k,
        n_init=int(n_init),
        max_iter=2000,
        random_state=int(random_state),
    )

    # Cluster IDs themselves are arbitrary. Reorder them from shallow to deep
    # using the independent NanoSense height descriptor only for readable labels.
    labels, label_mapping = reorder_labels_by_height(valid_df, labels)
    diag = clustering_diagnostics(X_cluster, labels)

    assignments = df.copy()
    assignments["cluster"] = -1
    for row, label in zip(valid_rows, labels):
        assignments.loc[assignments["dataset_row"] == int(row), "cluster"] = int(label)

    scores, explained, loadings = pca_numpy(X_cluster)

    return dict(
        k=k,
        labels=labels,
        assignments=assignments,
        valid_mask=finite_mask,
        valid_rows=valid_rows,
        valid_df=valid_df,
        X_scaled=X_cluster,
        X_cluster=X_cluster,
        scan=scan,
        diagnostics=diag,
        inertia=inertia,
        scores=scores,
        explained=explained,
        loadings=loadings,
        feature_min=feature_min,
        feature_max=feature_max,
        constant=constant,
        n_init=int(n_init),
        random_state=int(random_state),
        mode=mode,
        input_type=input_type,
        clustering_input=clustering_input,
        waveform_meta=waveform_meta,
        label_mapping=label_mapping,
    )



def _parse_optional_float(text):
    text = str(text).strip()
    if not text:
        return None
    return float(text)


def plot_editor(
    key,
    default_title,
    default_xlabel,
    default_ylabel,
    default_bins=None,
    allow_log=False,
    allow_summary_markers=False,
    default_show_mean=True,
    default_show_median=True,
):
    """Reusable controls for labels, limits, typography and optional mean/median overlays."""
    with st.expander("Edit plot appearance", expanded=False):
        title = st.text_input("Plot title", value=default_title, key=f"{key}_title")
        c1, c2 = st.columns(2)
        xlabel = c1.text_input("X-axis title", value=default_xlabel, key=f"{key}_xlabel")
        ylabel = c2.text_input("Y-axis title", value=default_ylabel, key=f"{key}_ylabel")

        c1, c2, c3, c4 = st.columns(4)
        xmin = c1.text_input("X min (blank = auto)", value="", key=f"{key}_xmin")
        xmax = c2.text_input("X max (blank = auto)", value="", key=f"{key}_xmax")
        ymin = c3.text_input("Y min (blank = auto)", value="", key=f"{key}_ymin")
        ymax = c4.text_input("Y max (blank = auto)", value="", key=f"{key}_ymax")

        bins = default_bins
        if default_bins is not None:
            bins = st.slider(
                "Histogram bins",
                min_value=5,
                max_value=120,
                value=int(default_bins),
                step=1,
                key=f"{key}_bins",
            )

        logx = logy = False
        if allow_log:
            c1, c2 = st.columns(2)
            logx = c1.checkbox("Log x-axis", value=False, key=f"{key}_logx")
            logy = c2.checkbox("Log y-axis", value=False, key=f"{key}_logy")

        show_mean = bool(default_show_mean)
        show_median = bool(default_show_median)
        if allow_summary_markers:
            st.markdown("**Mean / median overlays**")
            c1, c2 = st.columns(2)
            show_mean = c1.checkbox(
                "Show mean",
                value=bool(default_show_mean),
                key=f"{key}_show_mean",
            )
            show_median = c2.checkbox(
                "Show median",
                value=bool(default_show_median),
                key=f"{key}_show_median",
            )

        st.markdown("**Typography**")
        c1, c2, c3, c4 = st.columns(4)
        title_fontsize = c1.slider(
            "Title size", 9, 24, 13, 1, key=f"{key}_title_fontsize"
        )
        label_fontsize = c2.slider(
            "Axis-label size", 8, 22, 11, 1, key=f"{key}_label_fontsize"
        )
        tick_fontsize = c3.slider(
            "Tick size", 7, 20, 10, 1, key=f"{key}_tick_fontsize"
        )
        legend_fontsize = c4.slider(
            "Legend size", 7, 20, 9, 1, key=f"{key}_legend_fontsize"
        )
        show_grid = st.checkbox(
            "Show light grid",
            value=False,
            key=f"{key}_grid",
        )

    return dict(
        key=key,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xmin=_parse_optional_float(xmin),
        xmax=_parse_optional_float(xmax),
        ymin=_parse_optional_float(ymin),
        ymax=_parse_optional_float(ymax),
        bins=bins,
        logx=logx,
        logy=logy,
        show_mean=show_mean,
        show_median=show_median,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=legend_fontsize,
        show_grid=show_grid,
    )


def apply_plot_editor(ax, opts, set_title=True):
    if set_title:
        ax.set_title(opts["title"], fontsize=opts.get("title_fontsize", 13))
    ax.set_xlabel(opts["xlabel"], fontsize=opts.get("label_fontsize", 11))
    ax.set_ylabel(opts["ylabel"], fontsize=opts.get("label_fontsize", 11))
    ax.tick_params(axis="both", labelsize=opts.get("tick_fontsize", 10))
    if opts["xmin"] is not None or opts["xmax"] is not None:
        lo, hi = ax.get_xlim()
        ax.set_xlim(opts["xmin"] if opts["xmin"] is not None else lo, opts["xmax"] if opts["xmax"] is not None else hi)
    if opts["ymin"] is not None or opts["ymax"] is not None:
        lo, hi = ax.get_ylim()
        ax.set_ylim(opts["ymin"] if opts["ymin"] is not None else lo, opts["ymax"] if opts["ymax"] is not None else hi)
    if opts.get("logx"):
        ax.set_xscale("log")
    if opts.get("logy"):
        ax.set_yscale("log")
    if opts.get("show_grid"):
        ax.grid(True, alpha=0.20, linewidth=0.7)
        ax.set_axisbelow(True)
    legend = ax.get_legend()
    if legend is not None:
        for item in legend.get_texts():
            item.set_fontsize(opts.get("legend_fontsize", 9))


def apply_figure_editor(fig, opts, title_as_suptitle=True, skip_colorbar=True):
    axes = list(fig.axes)
    main_axes = []
    for ax in axes:
        if skip_colorbar and ax.get_label() == "<colorbar>":
            continue
        main_axes.append(ax)
    for ax in main_axes:
        ax.set_xlabel(opts["xlabel"], fontsize=opts.get("label_fontsize", 11))
        ax.set_ylabel(opts["ylabel"], fontsize=opts.get("label_fontsize", 11))
        ax.tick_params(axis="both", labelsize=opts.get("tick_fontsize", 10))
        if opts["xmin"] is not None or opts["xmax"] is not None:
            lo, hi = ax.get_xlim()
            ax.set_xlim(opts["xmin"] if opts["xmin"] is not None else lo, opts["xmax"] if opts["xmax"] is not None else hi)
        if opts["ymin"] is not None or opts["ymax"] is not None:
            lo, hi = ax.get_ylim()
            ax.set_ylim(opts["ymin"] if opts["ymin"] is not None else lo, opts["ymax"] if opts["ymax"] is not None else hi)
        if opts.get("logx"):
            ax.set_xscale("log")
        if opts.get("logy"):
            ax.set_yscale("log")
        if opts.get("show_grid"):
            ax.grid(True, alpha=0.20, linewidth=0.7)
            ax.set_axisbelow(True)
        legend = ax.get_legend()
        if legend is not None:
            for item in legend.get_texts():
                item.set_fontsize(opts.get("legend_fontsize", 9))
    if title_as_suptitle:
        fig.suptitle(opts["title"], y=1.01, fontsize=opts.get("title_fontsize", 13))
    elif main_axes:
        main_axes[0].set_title(opts["title"], fontsize=opts.get("title_fontsize", 13))
    fig.tight_layout()
    return fig


def _safe_figure_stem(text):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip()).strip("_.")
    return stem[:120] or "figure"


def _figure_title(fig):
    if getattr(fig, "_suptitle", None) is not None:
        title = fig._suptitle.get_text().strip()
        if title:
            return title
    for ax in fig.axes:
        title = ax.get_title().strip()
        if title:
            return title
    return "figure"


def _figure_signature(fig):
    """Small deterministic signature so prepared downloads invalidate after figure edits."""
    h = hashlib.sha1()
    for ax in fig.axes:
        h.update(str((ax.get_title(), ax.get_xlabel(), ax.get_ylabel(), ax.get_xlim(), ax.get_ylim())).encode())
        for line in ax.lines:
            try:
                h.update(np.asarray(line.get_xdata(), dtype=float).tobytes())
                h.update(np.asarray(line.get_ydata(), dtype=float).tobytes())
            except Exception:
                h.update(repr(line).encode())
        for coll in ax.collections:
            try:
                h.update(np.asarray(coll.get_offsets(), dtype=float).tobytes())
            except Exception:
                h.update(repr(coll).encode())
        for image in ax.images:
            try:
                h.update(np.asarray(image.get_array()).tobytes())
            except Exception:
                h.update(repr(image).encode())
        for patch in ax.patches[:500]:
            vals = []
            for attr in ("get_x", "get_y", "get_width", "get_height"):
                if hasattr(patch, attr):
                    try:
                        vals.append(float(getattr(patch, attr)()))
                    except Exception:
                        pass
            h.update(repr(vals).encode())
    return h.hexdigest()[:12]


def _figure_bytes(fig, fmt, dpi=600, transparent=False):
    buf = io.BytesIO()
    kwargs = dict(
        format=fmt,
        bbox_inches="tight",
        pad_inches=0.08,
        transparent=bool(transparent),
    )
    if fmt == "png":
        kwargs["dpi"] = int(dpi)
    if not transparent:
        kwargs["facecolor"] = "white"
    fig.savefig(buf, **kwargs)
    return buf.getvalue()


_FIGURE_RENDER_COUNTER = 0


def render_figure(fig, use_container_width=True, filename=None):
    """Render a high-quality figure and provide on-demand PNG/PDF/SVG downloads."""
    global _FIGURE_RENDER_COUNTER
    _FIGURE_RENDER_COUNTER += 1
    st.pyplot(fig, use_container_width=use_container_width)

    stem = _safe_figure_stem(filename or _figure_title(fig))
    signature = _figure_signature(fig)
    uid = f"fig_{_FIGURE_RENDER_COUNTER}_{stem}_{signature}"
    cache_key = f"_{uid}_exports"

    with st.expander("Save this figure", expanded=False):
        dpi = int(st.session_state.get("figure_export_dpi", 600))
        transparent = bool(st.session_state.get("figure_export_transparent", False))
        st.caption(
            f"PNG exports at {dpi} dpi. PDF and SVG are vector files. "
            "After changing plot labels/limits/markers, prepare the figure again."
        )
        if st.button("Prepare PNG / PDF / SVG", key=f"{uid}_prepare"):
            st.session_state[cache_key] = {
                "png": _figure_bytes(fig, "png", dpi=dpi, transparent=transparent),
                "pdf": _figure_bytes(fig, "pdf", dpi=dpi, transparent=transparent),
                "svg": _figure_bytes(fig, "svg", dpi=dpi, transparent=transparent),
            }
        exports = st.session_state.get(cache_key)
        if exports:
            c1, c2, c3 = st.columns(3)
            c1.download_button(
                f"PNG · {dpi} dpi",
                exports["png"],
                f"{stem}.png",
                "image/png",
                key=f"{uid}_png",
            )
            c2.download_button(
                "PDF · vector",
                exports["pdf"],
                f"{stem}.pdf",
                "application/pdf",
                key=f"{uid}_pdf",
            )
            c3.download_button(
                "SVG · vector",
                exports["svg"],
                f"{stem}.svg",
                "image/svg+xml",
                key=f"{uid}_svg",
            )


def add_1d_summary_markers(ax, values, opts):
    vals = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if not len(vals):
        return
    if opts.get("show_mean", False):
        mean = float(np.mean(vals))
        ax.axvline(mean, color="black", linestyle="-", linewidth=1.5, label=f"Mean = {mean:.3g}")
    if opts.get("show_median", False):
        median = float(np.median(vals))
        ax.axvline(median, color="red", linestyle="--", linewidth=1.5, label=f"Median = {median:.3g}")


def add_2d_summary_markers(ax, x, y, opts, label_prefix=None):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return
    x = x[finite]
    y = y[finite]
    if opts.get("show_mean", False):
        ax.scatter(
            np.mean(x), np.mean(y), marker="+", s=125, linewidths=1.8,
            color="black", zorder=8,
            label=(f"{label_prefix} mean" if label_prefix else None),
        )
    if opts.get("show_median", False):
        ax.scatter(
            np.median(x), np.median(y), marker="X", s=90,
            color="red", zorder=8,
            label=(f"{label_prefix} median" if label_prefix else None),
        )


def summary_marker_caption(opts):
    parts = []
    if opts.get("show_mean", False):
        parts.append("mean = black + / solid line")
    if opts.get("show_median", False):
        parts.append("median = red × / dashed line")
    return "; ".join(parts)


def cluster_summary_table(assignments):
    valid = assignments[assignments["cluster"] >= 0].copy()
    rows = []
    for c, g in valid.groupby("cluster"):
        def stats(col):
            x = g[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
            n = len(x)
            if not n:
                return dict(mean=np.nan, median=np.nan, std=np.nan, sem=np.nan, q1=np.nan, q3=np.nan)
            std = float(np.std(x, ddof=1)) if n > 1 else 0.0
            return dict(
                mean=float(np.mean(x)),
                median=float(np.median(x)),
                std=std,
                sem=float(std / np.sqrt(n)) if n else np.nan,
                q1=float(np.percentile(x, 25)),
                q3=float(np.percentile(x, 75)),
            )

        dwell = stats("width_ms")
        blockade = stats("height_nA")
        fwhm = stats("fwhm_ms")
        area = stats("area_nA_ms")
        frac = stats("fwhm_fraction")
        skew = stats("skew")
        kurt = stats("kurtosis")

        rows.append(dict(
            cluster=int(c),
            n=int(len(g)),
            mean_blockade_nA=blockade["mean"],
            median_blockade_nA=blockade["median"],
            std_blockade_nA=blockade["std"],
            sem_blockade_nA=blockade["sem"],
            q1_blockade_nA=blockade["q1"],
            q3_blockade_nA=blockade["q3"],
            mean_dwell_ms=dwell["mean"],
            median_dwell_ms=dwell["median"],
            std_dwell_ms=dwell["std"],
            sem_dwell_ms=dwell["sem"],
            q1_dwell_ms=dwell["q1"],
            q3_dwell_ms=dwell["q3"],
            mean_fwhm_ms=fwhm["mean"],
            median_fwhm_ms=fwhm["median"],
            mean_area_nA_ms=area["mean"],
            median_area_nA_ms=area["median"],
            mean_fwhm_fraction=frac["mean"],
            median_fwhm_fraction=frac["median"],
            mean_skew=skew["mean"],
            median_skew=skew["median"],
            mean_kurtosis=kurt["mean"],
            median_kurtosis=kurt["median"],
        ))

    summary = pd.DataFrame(rows)
    if len(summary):
        summary["population_pct"] = 100 * summary["n"] / summary["n"].sum()
    return summary


def build_labels_by_event(recording, assignments):
    row_to_fit = {int(row): int(fid) for fid, row in recording["fit_to_dataset"].items()}
    valid = assignments[assignments["cluster"] >= 0].copy()
    labels_by_event = {}
    for _, row in valid.iterrows():
        dsrow = int(row["dataset_row"])
        if dsrow in row_to_fit:
            labels_by_event[row_to_fit[dsrow]] = int(row["cluster"])
    return labels_by_event


def get_waveform_family_data(recording, assignments, window_samples=850, min_coverage_fraction=0.5):
    labels_by_event = build_labels_by_event(recording, assignments)
    if not labels_by_event:
        return None, None
    info = centered_profiles(
        recording["fitting_arrays"],
        sorted(labels_by_event),
        labels_by_event,
        window_samples=int(window_samples),
    )
    medians = pointwise_cluster_medians(info, float(min_coverage_fraction)) if info is not None else {}
    return info, medians


def plot_waveform_family_panels(info, medians, x_mode="time", max_members_per_cluster=50):
    """Real member traces plus the pointwise median representative only."""
    if info is None or not medians:
        return None, None

    x = info["time_ms"] if x_mode == "time" else info["data_index"]
    xlabel = "Time relative to event midpoint (ms)" if x_mode == "time" else "Centered data index"
    clusters = sorted(medians)
    n = len(clusters)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 3.6 * nrows),
        squeeze=False
    )
    axes = axes.ravel()

    for ax in axes[n:]:
        ax.axis("off")

    for ax, c in zip(axes, clusters):
        item = medians[c]
        members = item["members"]
        show_n = min(max_members_per_cluster, len(members))

        if show_n > 0:
            step = max(1, len(members) // show_n)
            sample = members[::step][:show_n]
            for prof in sample:
                ax.plot(x, prof, color="gray", alpha=0.10, linewidth=0.8)

        ax.plot(
            x,
            item["median"],
            color="red",
            linewidth=2.2,
            label="Median representative"
        )
        ax.set_title(f"Cluster {c} · n={item['n']}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Blockade (nA)")
        ax.legend(frameon=False)

    fig.tight_layout()

    overlay, ax = plt.subplots(figsize=(10, 4.8))
    for c in clusters:
        ax.plot(
            x,
            medians[c]["median"],
            linewidth=2.0,
            label=f"Cluster {c}"
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Median blockade (nA)")
    ax.set_title("Median representative profiles overlaid")
    ax.legend(frameon=False)
    overlay.tight_layout()

    return fig, overlay


def plot_histograms_by_cluster(
    valid,
    column,
    xlabel,
    bins=30,
    title="Cluster distributions",
    ylabel="Event count",
    xlim=None,
    ylim=None,
    show_mean=True,
    show_median=True,
):
    clusters = sorted(valid["cluster"].unique())
    n = len(clusters)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.6 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, c in zip(axes, clusters):
        vals = valid.loc[valid["cluster"] == c, column].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(vals, bins=bins)
        marker_opts = {"show_mean": bool(show_mean), "show_median": bool(show_median)}
        add_1d_summary_markers(ax, vals, marker_opts)
        ax.set_title(f"Cluster {c} · n={len(vals)}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if show_mean or show_median:
            ax.legend(frameon=False)
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def group_uploaded_triplets(files):
    grouped = {}
    pat = re.compile(r"(.+?)(\.dataset|\.event_data|\.event_fitting)\.npz$", flags=re.IGNORECASE)
    for f in files:
        m = pat.match(f.name)
        if not m:
            continue
        key, kind = m.group(1), m.group(2).lower()
        grouped.setdefault(key, {})
        grouped[key][kind] = f
    return grouped


def guess_salt_label(group_key):
    token = re.split(r"[_\-]", group_key)[0]
    return token


def lineplot_summary_across_salts(summary_df, ycol, ylabel, title):
    families = sorted(summary_df["cluster"].unique())
    salts = list(dict.fromkeys(summary_df["salt"].tolist()))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for fam in families:
        g = summary_df[summary_df["cluster"] == fam]
        g = g.set_index("salt").reindex(salts).reset_index()
        ax.plot(g["salt"], g[ycol], marker="o", label=f"Family {chr(65 + int(fam))}")
    ax.set_xlabel("Electrolyte")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def compare_waveform_profiles(results_by_salt, x_mode="time", min_common_families=None):
    salts = list(results_by_salt.keys())
    ks = []
    for salt, item in results_by_salt.items():
        med = item.get("medians") or {}
        ks.append(len(med))
    if not ks or min(ks) == 0:
        return None
    n_fam = min(ks) if min_common_families is None else min(min_common_families, min(ks))
    ncols = 2 if n_fam > 1 else 1
    nrows = int(np.ceil(n_fam / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.5 * nrows), squeeze=False, sharey=True)
    axes = axes.ravel()
    for ax in axes[n_fam:]:
        ax.axis("off")
    for fam in range(n_fam):
        ax = axes[fam]
        for salt, item in results_by_salt.items():
            info = item["profile_info"]
            med = item["medians"]
            if fam not in med:
                continue
            x = info["time_ms"] if x_mode == "time" else info["data_index"]
            ax.plot(x, med[fam]["median"], linewidth=2, label=salt)
        ax.set_title(f"Family {chr(65 + fam)}")
        ax.set_xlabel("Time relative to event midpoint (ms)" if x_mode == "time" else "Centered data index")
        ax.set_ylabel("Median blockade (nA)")
        ax.legend(frameon=False)
    fig.suptitle("Matched DNA event families across salts", y=1.01)
    fig.tight_layout()
    return fig


def cross_salt_cluster_stats(assignments, salt):
    """Per-cluster mean/median plus SD, SEM and IQR for cross-salt comparison."""
    d = assignments[assignments["cluster"] >= 0].copy()
    rows = []
    for c, g in d.groupby("cluster"):
        def stats(col):
            x = g[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
            n = len(x)
            if not n:
                return dict(mean=np.nan, median=np.nan, std=np.nan, sem=np.nan, q1=np.nan, q3=np.nan)
            std = float(np.std(x, ddof=1)) if n > 1 else 0.0
            return dict(
                mean=float(np.mean(x)),
                median=float(np.median(x)),
                std=std,
                sem=float(std / np.sqrt(n)) if n else np.nan,
                q1=float(np.percentile(x, 25)),
                q3=float(np.percentile(x, 75)),
            )
        dwell = stats("width_ms")
        blockade = stats("height_nA")
        fwhm = stats("fwhm_ms")
        area = stats("area_nA_ms")
        rows.append(dict(
            salt=salt,
            cluster=int(c),
            n=int(len(g)),
            population_pct=100.0 * len(g) / len(d),
            mean_dwell_ms=dwell["mean"],
            median_dwell_ms=dwell["median"],
            std_dwell_ms=dwell["std"],
            sem_dwell_ms=dwell["sem"],
            q1_dwell_ms=dwell["q1"],
            q3_dwell_ms=dwell["q3"],
            mean_blockade_nA=blockade["mean"],
            median_blockade_nA=blockade["median"],
            std_blockade_nA=blockade["std"],
            sem_blockade_nA=blockade["sem"],
            q1_blockade_nA=blockade["q1"],
            q3_blockade_nA=blockade["q3"],
            mean_fwhm_ms=fwhm["mean"],
            median_fwhm_ms=fwhm["median"],
            mean_area_nA_ms=area["mean"],
            median_area_nA_ms=area["median"],
        ))
    return pd.DataFrame(rows)


def segment_weighted_salt_summary(results_by_salt):
    all_rows = []
    summary_rows = []
    for salt, item in results_by_salt.items():
        rec = item["recording"]
        seg = segment_metrics(rec["fitting_arrays"], rec["fit_to_dataset"])
        if not len(seg):
            continue
        seg = seg.copy()
        seg.insert(0, "salt", salt)
        all_rows.append(seg)

        for metric, prefix in [
            ("duration_weighted_mean_blockade_nA", "weighted_deltaI_nA"),
            ("total_segment_dwell_ms", "total_dwell_ms"),
            ("duration_weighted_segment_duration_ms", "weighted_segment_duration_ms"),
        ]:
            x = seg[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
            n = len(x)
            if not n:
                continue
            std = float(np.std(x, ddof=1)) if n > 1 else 0.0
            summary_rows.append(dict(
                salt=salt,
                metric=metric,
                metric_label=prefix,
                n=n,
                mean=float(np.mean(x)),
                median=float(np.median(x)),
                std=std,
                sem=float(std / np.sqrt(n)),
                q1=float(np.percentile(x, 25)),
                q3=float(np.percentile(x, 75)),
            ))

    all_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return all_df, summary_df


def overlay_histogram_by_salt(
    df,
    column,
    bins=40,
    density=True,
    title="",
    xlabel="",
    ylabel="",
    show_mean=False,
    show_median=False,
):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    finite = df[["salt", column]].replace([np.inf, -np.inf], np.nan).dropna()
    if not len(finite):
        return fig
    global_edges = np.histogram_bin_edges(finite[column].to_numpy(float), bins=bins)
    for salt, g in finite.groupby("salt"):
        vals = g[column].to_numpy(float)
        _, _, patches = ax.hist(
            vals,
            bins=global_edges,
            histtype="step",
            linewidth=1.8,
            density=density,
            label=salt,
        )
        color = None
        try:
            if len(patches):
                color = patches[0].get_edgecolor()
        except Exception:
            color = None
        if show_mean:
            ax.axvline(np.mean(vals), color=color, linestyle="-", linewidth=1.0, alpha=0.75)
        if show_median:
            ax.axvline(np.median(vals), color=color, linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def salt_errorbar_summary(summary, metric, center_mode, title, ylabel, plot_style="Line + error bars"):
    q = summary[summary["metric"] == metric].copy()
    order = [s for s in ["LiCl", "NaCl", "KCl", "RbCl", "CsCl"] if s in set(q["salt"])]
    order += [s for s in q["salt"].tolist() if s not in order]
    q = q.set_index("salt").reindex(order).reset_index()

    if center_mode == "Mean ± SD":
        y = q["mean"].to_numpy(float)
        yerr = q["std"].to_numpy(float)
    elif center_mode == "Mean ± SEM":
        y = q["mean"].to_numpy(float)
        yerr = q["sem"].to_numpy(float)
    else:
        y = q["median"].to_numpy(float)
        low = y - q["q1"].to_numpy(float)
        high = q["q3"].to_numpy(float) - y
        yerr = np.vstack([low, high])

    xpos = np.arange(len(q))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if plot_style == "Bar plot":
        ax.bar(xpos, y, yerr=yerr, capsize=4, alpha=0.85, width=0.68)
    else:
        ax.errorbar(xpos, y, yerr=yerr, marker="o", linewidth=1.5, capsize=4)
    ax.set_xticks(xpos, q["salt"])
    ax.set_title(title)
    ax.set_xlabel("Electrolyte")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def mapped_cross_salt_stats(results_by_salt, mapping):
    """Recompute the current summary schema from assignments, then apply family mapping."""
    rows = []
    for salt, item in results_by_salt.items():
        stats = cross_salt_cluster_stats(item["result"]["assignments"], salt)
        item["stats"] = stats
        m = mapping.get(salt, {})
        stats["family"] = stats["cluster"].map(lambda c: m.get(int(c), "Unmapped"))
        rows.append(stats)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def cross_salt_profile_figure(results_by_salt, mapping, x_mode="time"):
    """Matched pointwise-median waveform families across salts."""
    families = sorted(
        {
            fam
            for salt, m in mapping.items()
            for fam in m.values()
            if fam != "Unmapped"
        }
    )
    if not families:
        return None

    ncols = 2 if len(families) > 1 else 1
    nrows = int(np.ceil(len(families) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 3.5 * nrows),
        squeeze=False,
        sharey=True,
    )
    axes = axes.ravel()

    for ax in axes[len(families):]:
        ax.axis("off")

    for ax, fam in zip(axes, families):
        for salt, item in results_by_salt.items():
            inverse = {
                family: int(cluster)
                for cluster, family in mapping.get(salt, {}).items()
                if family != "Unmapped"
            }
            if fam not in inverse:
                continue

            cluster = inverse[fam]
            info = item.get("profile_info")
            medians = item.get("medians") or {}
            if info is None or cluster not in medians:
                continue

            x = info["time_ms"] if x_mode == "time" else info["data_index"]
            ax.plot(
                x,
                medians[cluster]["median"],
                linewidth=2.0,
                label=salt,
            )

        ax.set_title(f"Family {fam}")
        ax.set_xlabel(
            "Time relative to event midpoint (ms)"
            if x_mode == "time"
            else "Centered data index"
        )
        ax.set_ylabel("Median blockade (nA)")
        ax.legend(frameon=False)

    fig.suptitle("Matched median DNA event families across salts", y=1.01)
    fig.tight_layout()
    return fig


def cross_salt_metric_figure(
    stats,
    metric,
    q1,
    q3,
    ylabel,
    title,
):
    families = sorted([f for f in pd.unique(stats["family"]) if f != "Unmapped"])
    salts = [s for s in ["LiCl", "NaCl", "KCl", "RbCl", "CsCl"] if s in set(stats["salt"])]
    salts += [s for s in pd.unique(stats["salt"]) if s not in salts]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for fam in families:
        xs, ys, loerr, hierr = [], [], [], []
        for i, salt in enumerate(salts):
            q = stats[(stats["salt"] == salt) & (stats["family"] == fam)]
            if not len(q):
                continue
            row = q.iloc[0]
            value = float(row[metric])
            low = float(row[q1])
            high = float(row[q3])
            xs.append(i)
            ys.append(value)
            loerr.append(max(0.0, value - low))
            hierr.append(max(0.0, high - value))
        if xs:
            ax.errorbar(
                xs,
                ys,
                yerr=[loerr, hierr],
                marker="o",
                linewidth=1.5,
                capsize=3,
                label=f"Family {fam}",
            )

    ax.set_xticks(range(len(salts)), salts)
    ax.set_xlabel("Electrolyte")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def cross_salt_family_metric_figure(stats, quantity="dwell", summary_mode="Median + IQR", title="", ylabel=""):
    families = sorted([f for f in pd.unique(stats["family"]) if f != "Unmapped"])
    salts = [s for s in ["LiCl", "NaCl", "KCl", "RbCl", "CsCl"] if s in set(stats["salt"])]
    salts += [s for s in pd.unique(stats["salt"]) if s not in salts]

    if quantity == "dwell":
        mean_col, med_col = "mean_dwell_ms", "median_dwell_ms"
        std_col, sem_col = "std_dwell_ms", "sem_dwell_ms"
        q1_col, q3_col = "q1_dwell_ms", "q3_dwell_ms"
    else:
        mean_col, med_col = "mean_blockade_nA", "median_blockade_nA"
        std_col, sem_col = "std_blockade_nA", "sem_blockade_nA"
        q1_col, q3_col = "q1_blockade_nA", "q3_blockade_nA"

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for fam in families:
        xs, ys, lows, highs = [], [], [], []
        for i, salt in enumerate(salts):
            q = stats[(stats["salt"] == salt) & (stats["family"] == fam)]
            if not len(q):
                continue
            row = q.iloc[0]
            xs.append(i)
            if summary_mode == "Mean ± SD":
                v = float(row[mean_col]); e = float(row[std_col])
                ys.append(v); lows.append(e); highs.append(e)
            elif summary_mode == "Mean ± SEM":
                v = float(row[mean_col]); e = float(row[sem_col])
                ys.append(v); lows.append(e); highs.append(e)
            else:
                v = float(row[med_col])
                ys.append(v)
                lows.append(max(0.0, v - float(row[q1_col])))
                highs.append(max(0.0, float(row[q3_col]) - v))
        if xs:
            ax.errorbar(xs, ys, yerr=[lows, highs], marker="o", linewidth=1.5, capsize=3, label=f"Family {fam}")

    ax.set_xticks(range(len(salts)), salts)
    ax.set_xlabel("Electrolyte")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def cross_salt_population_figure(stats):
    salts = [s for s in ["LiCl", "NaCl", "KCl", "RbCl", "CsCl"] if s in set(stats["salt"])]
    salts += [s for s in pd.unique(stats["salt"]) if s not in salts]
    families = sorted([f for f in pd.unique(stats["family"]) if f != "Unmapped"])
    if "Unmapped" in set(stats["family"]):
        families += ["Unmapped"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bottom = np.zeros(len(salts))
    for fam in families:
        vals = []
        for salt in salts:
            q = stats[(stats["salt"] == salt) & (stats["family"] == fam)]
            vals.append(float(q["population_pct"].sum()) if len(q) else 0.0)
        ax.bar(salts, vals, bottom=bottom, label=("Unmapped" if fam == "Unmapped" else f"Family {fam}"))
        bottom += np.asarray(vals)

    ax.set_ylim(0, 100)
    ax.set_xlabel("Electrolyte")
    ax.set_ylabel("Population (%)")
    ax.set_title("Event-family population fractions across salts")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def cross_salt_heatmap(stats, metric, title, value_label):
    families = sorted([f for f in pd.unique(stats["family"]) if f != "Unmapped"])
    salts = [s for s in ["LiCl", "NaCl", "KCl", "RbCl", "CsCl"] if s in set(stats["salt"])]
    salts += [s for s in pd.unique(stats["salt"]) if s not in salts]

    matrix = np.full((len(families), len(salts)), np.nan)
    for i, fam in enumerate(families):
        for j, salt in enumerate(salts):
            q = stats[(stats["salt"] == salt) & (stats["family"] == fam)]
            if len(q):
                matrix[i, j] = float(q.iloc[0][metric])

    fig, ax = plt.subplots(figsize=(1.3 * max(5, len(salts)), 1.0 * max(3, len(families) + 1)))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(salts)), salts)
    ax.set_yticks(range(len(families)), [f"Family {f}" for f in families])
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_label)

    for i in range(len(families)):
        for j in range(len(salts)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")

    fig.tight_layout()
    return fig


def cross_salt_blockade_dwell_panels(results_by_salt, show_mean=False, show_median=False):
    salts = list(results_by_salt.keys())
    if not salts:
        return None

    all_valid = [
        item["result"]["assignments"].query("cluster >= 0")
        for item in results_by_salt.values()
    ]
    xmin = min(float(d["width_ms"].min()) for d in all_valid)
    xmax = max(float(d["width_ms"].max()) for d in all_valid)
    ymin = min(float(d["height_nA"].min()) for d in all_valid)
    ymax = max(float(d["height_nA"].max()) for d in all_valid)

    ncols = 2 if len(salts) > 1 else 1
    nrows = int(np.ceil(len(salts) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.4 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax in axes[len(salts):]:
        ax.axis("off")

    marker_opts = {"show_mean": bool(show_mean), "show_median": bool(show_median)}
    for ax, salt in zip(axes, salts):
        valid = results_by_salt[salt]["result"]["assignments"]
        valid = valid[valid["cluster"] >= 0]
        for c, g in valid.groupby("cluster"):
            ax.scatter(g["width_ms"], g["height_nA"], s=14, alpha=0.42, label=f"Cluster {c}")
            add_2d_summary_markers(ax, g["width_ms"], g["height_nA"], marker_opts)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(salt)
        ax.set_xlabel("Event width / dwell-like duration (ms)")
        ax.set_ylabel("Blockade height (nA)")
        ax.legend(frameon=False)

    fig.suptitle("Blockade–dwell distributions with shared axes", y=1.01)
    fig.tight_layout()
    return fig


# -----------------------------
# UI tabs
# -----------------------------
tabs = st.tabs(
    [
        "1 · Load files",
        "2 · K-means clustering",
        "3 · ΔI gates",
        "4 · Segment-weighted analysis",
        "5 · Export",
        "6 · Multi-salt compare",
    ]
)

# -------------------------------------------------------------------
# 1. LOAD
# -------------------------------------------------------------------
with tabs[0]:
    st.header("Load one recording")

    c1, c2, c3 = st.columns(3)
    dataset_file = c1.file_uploader("dataset.npz", type=["npz"], key="dataset_upload")
    eventdata_file = c2.file_uploader("event_data.npz", type=["npz"], key="eventdata_upload")
    fitting_file = c3.file_uploader("event_fitting.npz", type=["npz"], key="fitting_upload")

    tolerance = st.number_input(
        "Timestamp matching tolerance (s)",
        min_value=1e-10,
        max_value=1e-3,
        value=1e-7,
        format="%.1e",
    )

    if st.button("Load and validate", type="primary"):
        if not (dataset_file and eventdata_file and fitting_file):
            st.error("Upload all three files first.")
        else:
            try:
                recording = load_recording(
                    read_uploaded_bytes(dataset_file),
                    read_uploaded_bytes(eventdata_file),
                    read_uploaded_bytes(fitting_file),
                    dataset_file.name,
                    eventdata_file.name,
                    fitting_file.name,
                    float(tolerance),
                )
                st.session_state.loaded = recording
                st.session_state.cluster_result = None
                st.success("Files loaded.")
            except Exception as exc:
                st.exception(exc)

    p = st.session_state.loaded
    if p:
        df = p["df"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Dataset events", len(df))
        m2.metric("Dataset ↔ fitting matched", len(p["fit_to_dataset"]))
        m3.metric("Fitting ↔ eventdata matched", len(p["fit_to_raw"]))
        m4.metric("Dataset columns", p["X"].shape[1])

        st.subheader("NanoSense feature table")
        st.dataframe(df.head(30), use_container_width=True)

        with st.expander("Dataset ↔ event-fitting matching audit"):
            st.dataframe(p["ds_status"], use_container_width=True)

        with st.expander("Event-fitting ↔ event-data matching audit"):
            st.dataframe(p["raw_status"], use_container_width=True)

        st.info(
            "Step 2 now lets you choose either the seven NanoSense scalar descriptors from dataset.npz "
            "or the complete event waveform stored in event_data.npz. The waveform route clusters the signal itself."
        )

# -------------------------------------------------------------------
# 2. K-MEANS
# -------------------------------------------------------------------
with tabs[1]:
    st.header("Reproducible K-means clustering")
    p = st.session_state.loaded

    if not p:
        st.info("Load the three files in Step 1 first.")
    else:
        df = p["df"].copy()

        clustering_input = st.radio(
            "What should K-means cluster?",
            ["NanoSense 7 scalar features", "event_data whole waveform"],
            horizontal=True,
            help=(
                "Scalar mode uses height, FWHM, height-at-FWHM, area, width, skew and kurtosis. "
                "Waveform mode uses the measured event_data signal itself."
            ),
        )

        waveform_representation = "Native-time midpoint pad/crop"
        waveform_native_samples = 850
        waveform_resample_points = 200
        waveform_scaling = "Physical blockade (nA)"

        if clustering_input == "event_data whole waveform":
            st.info(
                "Whole-waveform mode: each event becomes a vector of measured blockade samples. "
                "K-means therefore groups events by their full signal morphology rather than by seven summary numbers."
            )
            cwa, cwb, cwc = st.columns(3)
            waveform_representation = cwa.selectbox(
                "Waveform representation",
                ["Native-time midpoint pad/crop", "Duration-normalised resample"],
                help=(
                    "Native-time mode preserves real dwell by midpoint-centering and padding with 0 nA baseline. "
                    "Duration-normalised mode resamples every complete event to the same number of positions and therefore focuses more on shape."
                ),
            )
            if waveform_representation == "Native-time midpoint pad/crop":
                waveform_native_samples = cwb.slider(
                    "Vector length (samples)", 100, 3000, 850, 50,
                    help="This is the clustering vector length, not resampling. Shorter events are baseline-padded; longer events are centre-cropped.",
                )
                fs = _eventdata_sampling_rate(p["eventdata_arrays"])
                if np.isfinite(fs):
                    cwb.caption(f"At {fs/1000:.1f} kHz: {waveform_native_samples/fs*1e3:.3f} ms total window")
            else:
                waveform_resample_points = cwb.slider(
                    "Resampled waveform points", 64, 512, 200, 8,
                    help="Every complete event is linearly resampled from 0 to 100% of its own duration.",
                )
            waveform_scaling = cwc.selectbox(
                "Waveform scaling",
                ["Physical blockade (nA)", "L2-normalised shape only"],
                help=(
                    "Physical blockade retains amplitude differences. L2 normalisation removes most overall amplitude information and emphasises shape."
                ),
            )
            if waveform_scaling == "L2-normalised shape only":
                st.warning("Shape-only normalisation removes much of the blockade-amplitude information. Use it as a comparison, not the default physical analysis.")
        else:
            X_features = df[FEATURE_NAMES].to_numpy(float)
            finite_mask = np.all(np.isfinite(X_features), axis=1)
            if np.any(finite_mask):
                _, _, _, constant = minmax_scale(X_features[finite_mask])
                if np.any(constant):
                    names = [FEATURE_NAMES[i] for i in np.flatnonzero(constant)]
                    st.warning("Constant feature(s): " + ", ".join(names))

        c1, c2, c3, c4 = st.columns(4)
        mode = c1.radio("Choose k", ["Manual", "Silhouette"], horizontal=False)
        manual_k = c2.number_input("Manual k", min_value=2, max_value=12, value=4, step=1)
        kmax = c3.number_input("Maximum k to test", min_value=3, max_value=12, value=8, step=1)
        n_init = c4.number_input("K-means initialisations", min_value=10, max_value=300, value=100, step=10)
        random_state = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)

        if st.button("Run K-means", type="primary"):
            try:
                st.session_state.cluster_result = run_kmeans_pipeline(
                    p, mode, int(manual_k), int(kmax), int(n_init), int(random_state),
                    clustering_input=clustering_input,
                    waveform_representation=waveform_representation,
                    waveform_native_samples=int(waveform_native_samples),
                    waveform_resample_points=int(waveform_resample_points),
                    waveform_scaling=waveform_scaling,
                )
                st.success(f"K-means complete: k={st.session_state.cluster_result['k']}")
            except Exception as exc:
                st.exception(exc)

        r = st.session_state.cluster_result
        if r:
            assignments = r["assignments"]
            valid = assignments[assignments["cluster"] >= 0].copy()
            scan = r["scan"]

            if r.get("input_type") == "waveform":
                wm = r.get("waveform_meta") or {}
                st.success(
                    f"Clustering space: event_data whole waveform · {wm.get('representation','')} · "
                    f"{wm.get('vector_length','?')} dimensions · {len(valid)} matched events"
                )
                if wm.get("representation") == "Native-time midpoint pad/crop":
                    st.caption(
                        "The waveform vectors retain native sample spacing. Short events are centred and padded with 0 nA blockade; "
                        "events longer than the selected vector are centre-cropped."
                    )
                else:
                    st.caption(
                        "Each complete event is resampled to a common 0–100% duration axis. This preserves waveform amplitude/shape but removes absolute dwell from the clustering coordinates."
                    )
            else:
                st.success(f"Clustering space: seven scaled NanoSense scalar descriptors · {len(valid)} events")

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Selected k", r["k"])
            d2.metric("Silhouette", f'{r["diagnostics"]["silhouette"]:.3f}')
            d3.metric("Davies–Bouldin", f'{r["diagnostics"]["davies_bouldin"]:.3f}')
            d4.metric("Calinski–Harabasz", f'{r["diagnostics"]["calinski_harabasz"]:.1f}')

            if np.isfinite(r["diagnostics"]["silhouette"]) and r["diagnostics"]["silhouette"] < 0.25:
                st.warning(
                    "Silhouette is below 0.25. Treat these as candidate signal families rather than proven discrete populations."
                )

            st.subheader("k diagnostics")
            st.dataframe(scan.round(4), use_container_width=True)

            opts = plot_editor(
                "k_diagnostics",
                "K selection diagnostics",
                "Number of clusters, k",
                "Silhouette score",
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(scan["k"], scan["silhouette"], marker="o", label="Silhouette")
            ax2 = ax.twinx()
            ax2.plot(scan["k"], scan["inertia"], marker="s", label="Inertia")
            apply_plot_editor(ax, opts)
            ax2.set_ylabel("Within-cluster dispersion", fontsize=opts.get("label_fontsize", 11))
            ax2.tick_params(axis="y", labelsize=opts.get("tick_fontsize", 10))
            render_figure(fig, use_container_width=True)

            st.subheader("Cluster summary")
            summary = cluster_summary_table(valid)
            st.dataframe(summary.round(4), use_container_width=True)

            st.subheader("Physical plots")
            plot_choice = st.selectbox(
                "Plot",
                [
                    "Blockade vs dwell",
                    "Dwell-time histograms by cluster",
                    "Blockade histograms by cluster",
                    "FWHM vs total width",
                    "FWHM / width",
                    "Area vs dwell",
                    "Population fraction",
                    "Waveform family panels + overlay",
                ],
            )

            if plot_choice == "Blockade vs dwell":
                opts = plot_editor(
                    "single_bd",
                    "Blockade amplitude versus event duration",
                    "Event width / dwell-like duration (ms)",
                    "Blockade height (nA)",
                    allow_log=True,
                    allow_summary_markers=True,
                )
                fig, ax = plt.subplots(figsize=(8, 5))
                for c, g in valid.groupby("cluster"):
                    ax.scatter(g["width_ms"], g["height_nA"], s=16, alpha=0.45, label=f"Cluster {c}")
                    add_2d_summary_markers(ax, g["width_ms"], g["height_nA"], opts)
                apply_plot_editor(ax, opts)
                ax.legend()
                render_figure(fig, use_container_width=True)
                if summary_marker_caption(opts):
                    st.caption(summary_marker_caption(opts))

            elif plot_choice == "Dwell-time histograms by cluster":
                opts = plot_editor(
                    "single_dwell_hist",
                    "Dwell-time distributions by cluster",
                    "Event width / dwell-like duration (ms)",
                    "Event count",
                    default_bins=30,
                    allow_summary_markers=True,
                )
                fig = plot_histograms_by_cluster(
                    valid, "width_ms", opts["xlabel"], bins=opts["bins"],
                    title=opts["title"], ylabel=opts["ylabel"],
                    xlim=(opts["xmin"], opts["xmax"]) if opts["xmin"] is not None and opts["xmax"] is not None else None,
                    ylim=(opts["ymin"], opts["ymax"]) if opts["ymin"] is not None and opts["ymax"] is not None else None,
                    show_mean=opts["show_mean"],
                    show_median=opts["show_median"],
                )
                render_figure(fig, use_container_width=True)
                if summary_marker_caption(opts):
                    st.caption(summary_marker_caption(opts))

            elif plot_choice == "Blockade histograms by cluster":
                opts = plot_editor(
                    "single_block_hist",
                    "Blockade distributions by cluster",
                    "Blockade height (nA)",
                    "Event count",
                    default_bins=30,
                    allow_summary_markers=True,
                )
                fig = plot_histograms_by_cluster(
                    valid, "height_nA", opts["xlabel"], bins=opts["bins"],
                    title=opts["title"], ylabel=opts["ylabel"],
                    xlim=(opts["xmin"], opts["xmax"]) if opts["xmin"] is not None and opts["xmax"] is not None else None,
                    ylim=(opts["ymin"], opts["ymax"]) if opts["ymin"] is not None and opts["ymax"] is not None else None,
                    show_mean=opts["show_mean"],
                    show_median=opts["show_median"],
                )
                render_figure(fig, use_container_width=True)
                if summary_marker_caption(opts):
                    st.caption(summary_marker_caption(opts))

            elif plot_choice == "FWHM vs total width":
                opts = plot_editor(
                    "single_fwhm_width",
                    "FWHM versus total event width",
                    "Total event width (ms)",
                    "FWHM (ms)",
                    allow_log=True,
                    allow_summary_markers=True,
                )
                fig, ax = plt.subplots(figsize=(8, 5))
                for c, g in valid.groupby("cluster"):
                    ax.scatter(g["width_ms"], g["fwhm_ms"], s=16, alpha=0.45, label=f"Cluster {c}")
                    add_2d_summary_markers(ax, g["width_ms"], g["fwhm_ms"], opts)
                mx = max(valid["width_ms"].max(), valid["fwhm_ms"].max())
                ax.plot([0, mx], [0, mx], "--", linewidth=1)
                apply_plot_editor(ax, opts)
                ax.legend()
                render_figure(fig, use_container_width=True)

            elif plot_choice == "FWHM / width":
                opts = plot_editor(
                    "single_fwhm_ratio",
                    "Fraction of event spent above half maximum",
                    "Cluster",
                    "FWHM / event width",
                    allow_summary_markers=True,
                )
                fig, ax = plt.subplots(figsize=(8, 5))
                groups = sorted(valid["cluster"].unique())
                arr = [
                    valid.loc[valid.cluster == c, "fwhm_fraction"]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .to_numpy()
                    for c in groups
                ]
                ax.boxplot(
                    arr,
                    labels=[f"Cluster {c}" for c in groups],
                    showfliers=False,
                    showmeans=opts["show_mean"],
                    meanprops={"marker": "+", "markeredgecolor": "black", "markersize": 9},
                    medianprops={"color": "red", "linewidth": 1.5, "visible": opts["show_median"]},
                )
                apply_plot_editor(ax, opts)
                render_figure(fig, use_container_width=True)

            elif plot_choice == "Area vs dwell":
                opts = plot_editor(
                    "single_area_dwell",
                    "Integrated blockade versus event duration",
                    "Event width / dwell-like duration (ms)",
                    "Integrated area (nA·ms)",
                    allow_log=True,
                    allow_summary_markers=True,
                )
                fig, ax = plt.subplots(figsize=(8, 5))
                for c, g in valid.groupby("cluster"):
                    ax.scatter(g["width_ms"], g["area_nA_ms"], s=16, alpha=0.45, label=f"Cluster {c}")
                    add_2d_summary_markers(ax, g["width_ms"], g["area_nA_ms"], opts)
                apply_plot_editor(ax, opts)
                ax.legend()
                render_figure(fig, use_container_width=True)

            elif plot_choice == "Population fraction":
                opts = plot_editor(
                    "single_population",
                    "Cluster population fractions",
                    "Cluster",
                    "Population (%)",
                )
                pop = valid["cluster"].value_counts().sort_index()
                pct = 100 * pop / pop.sum()
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar([f"Cluster {c}" for c in pct.index], pct.values)
                for i, value in enumerate(pct.values):
                    ax.text(i, value, f"{value:.1f}%", ha="center", va="bottom")
                apply_plot_editor(ax, opts)
                render_figure(fig, use_container_width=True)

            else:
                c1, c2 = st.columns([1, 1])
                x_mode = c1.radio("Waveform x-axis", ["time", "data_index"], horizontal=True)
                win = c2.slider("Centered window (samples)", 200, 3000, 850, 50)
                info, medians = get_waveform_family_data(p, assignments, window_samples=win)
                if not medians:
                    st.warning("No matched event-fitting traces available for family waveforms.")
                else:
                    xlab = "Time relative to event midpoint (ms)" if x_mode == "time" else "Centered data index"
                    opts = plot_editor(
                        "single_waveform_panels",
                        "Midpoint-aligned cluster families",
                        xlab,
                        "Blockade (nA)",
                    )
                    fig_panels, fig_overlay = plot_waveform_family_panels(
                        info, medians, x_mode=x_mode
                    )
                    apply_figure_editor(fig_panels, opts, title_as_suptitle=True)
                    render_figure(fig_panels, use_container_width=True)
                    st.caption(
                        "Grey curves are real member traces; the pointwise median representative profile is overlaid."
                    )
                    opts2 = plot_editor(
                        "single_waveform_overlay",
                        "Median representative profiles overlaid",
                        xlab,
                        "Median blockade (nA)",
                    )
                    apply_plot_editor(fig_overlay.axes[0], opts2)
                    render_figure(fig_overlay, use_container_width=True)

            st.subheader("PCA diagnostic view of the clustering space")
            scores = r["scores"]
            explained = r["explained"]
            loadings = r["loadings"]

            c1, c2 = st.columns(2)

            with c1:
                pc2_pct = 100 * explained[1] if len(explained) > 1 else 0.0
                opts = plot_editor(
                    "pca_scatter",
                    "PCA projection of K-means clusters",
                    f"PC1 ({100*explained[0]:.1f}% variance)",
                    f"PC2 ({pc2_pct:.1f}% variance)",
                    allow_summary_markers=True,
                )
                fig, ax = plt.subplots(figsize=(7, 5))
                pc2 = scores[:, 1] if scores.shape[1] > 1 else np.zeros(len(scores))
                for c in sorted(np.unique(r["labels"])):
                    mask = r["labels"] == c
                    ax.scatter(scores[mask, 0], pc2[mask], s=15, alpha=0.5, label=f"Cluster {c}")
                    add_2d_summary_markers(ax, scores[mask, 0], pc2[mask], opts)
                apply_plot_editor(ax, opts)
                ax.legend()
                render_figure(fig, use_container_width=True)

            if r.get("input_type") == "scalar":
                with c2:
                    opts = plot_editor(
                        "pca_loadings",
                        "PCA loadings",
                        "NanoSense feature",
                        "PCA loading",
                    )
                    fig, ax = plt.subplots(figsize=(8, 5))
                    x = np.arange(len(FEATURE_NAMES))
                    w = 0.35
                    ax.bar(x - w / 2, loadings[0], width=w, label="PC1")
                    if loadings.shape[0] > 1:
                        ax.bar(x + w / 2, loadings[1], width=w, label="PC2")
                    ax.axhline(0, linewidth=0.8)
                    ax.set_xticks(x)
                    ax.set_xticklabels(FEATURE_NAMES, rotation=35, ha="right")
                    apply_plot_editor(ax, opts)
                    ax.legend()
                    render_figure(fig, use_container_width=True)

                if loadings.shape[0] > 1:
                    contribution = explained[0] * loadings[0] ** 2 + explained[1] * loadings[1] ** 2
                else:
                    contribution = explained[0] * loadings[0] ** 2
                contribution_pct = 100 * contribution / contribution.sum()
                eta = [
                    eta_squared(valid[col].to_numpy(), valid["cluster"].to_numpy())
                    for col in FEATURE_NAMES
                ]
                contribution_table = pd.DataFrame(
                    {
                        "feature": FEATURE_NAMES,
                        "PC1_PC2_contribution_pct": contribution_pct,
                        "eta_squared_cluster_separation": eta,
                    }
                ).sort_values("eta_squared_cluster_separation", ascending=False)
                st.dataframe(contribution_table.round(4), use_container_width=True)
            else:
                with c2:
                    wm = r.get("waveform_meta") or {}
                    xaxis = np.asarray(wm.get("axis", np.arange(loadings.shape[1])), float)
                    xlabel = wm.get("axis_label", "Waveform position")
                    opts = plot_editor(
                        "waveform_pca_loadings",
                        "Waveform PCA loading profiles",
                        xlabel,
                        "PCA loading",
                    )
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(xaxis, loadings[0], label="PC1")
                    if loadings.shape[0] > 1:
                        ax.plot(xaxis, loadings[1], label="PC2")
                    ax.axhline(0, linewidth=0.8)
                    apply_plot_editor(ax, opts)
                    ax.legend(frameon=False)
                    render_figure(fig, use_container_width=True)
                st.caption(
                    "PCA here is diagnostic only. K-means used the whole waveform vector directly; the loading curves show which waveform positions dominate the first two visualisation components."
                )

            st.download_button(
                "Download cluster assignments CSV",
                assignments.to_csv(index=False),
                "cluster_assignments.csv",
                "text/csv",
            )

# -------------------------------------------------------------------
# 3. DELTA-I GATES
# -------------------------------------------------------------------
with tabs[2]:
    st.header("ΔI histogram, current gates and dwell-time histograms")
    p = st.session_state.loaded

    if not p:
        st.info("Load the files first.")
    else:
        df = p["df"].copy()
        metric_options = {
            "NanoSense height / ΔI": "height_nA",
            "Height at FWHM": "height_at_fwhm_nA",
        }

        segment_df = segment_metrics(p["fitting_arrays"], p["fit_to_dataset"])
        if len(segment_df):
            weighted_map = segment_df.dropna(subset=["dataset_row"]).set_index("dataset_row")
            df["weighted_mean_deltaI_nA"] = df["dataset_row"].map(
                weighted_map["duration_weighted_mean_blockade_nA"]
            )
            metric_options["Segment-duration-weighted mean ΔI"] = "weighted_mean_deltaI_nA"

        metric_label = st.selectbox("Quantity used to gate events", list(metric_options))
        metric = metric_options[metric_label]
        values = df[metric].to_numpy(float)
        finite = np.isfinite(values)

        opts = plot_editor(
            "gate_original_hist",
            "ΔI distribution",
            metric_label + " (nA)",
            "Event count",
            default_bins=45,
            allow_summary_markers=True,
        )
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(values[finite], bins=opts["bins"])
        if np.any(finite):
            add_1d_summary_markers(ax, values[finite], opts)
        apply_plot_editor(ax, opts)
        if opts["show_mean"] or opts["show_median"]:
            ax.legend(frameon=False)
        render_figure(fig, use_container_width=True)

        if np.any(finite):
            vmin = float(np.nanmin(values))
            vmax = float(np.nanmax(values))
            default_edges = np.linspace(vmin, vmax, 5)
            default_text = ", ".join(f"{x:.3g}" for x in default_edges)
        else:
            default_text = "0, 1, 2, 3"

        edge_text = st.text_input(
            "Gate edges (nA)",
            value=default_text,
            help="Example: 0, 1, 2, 3, 4 creates four current ranges.",
        )

        try:
            edges = parse_edges(edge_text)
            groups = gate_rows(values, edges)

            gate_opts = plot_editor(
                "gate_edges_hist",
                "ΔI gates",
                metric_label + " (nA)",
                "Event count",
                default_bins=opts["bins"],
                allow_summary_markers=True,
            )
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(values[finite], bins=gate_opts["bins"])
            for edge in edges:
                ax.axvline(edge, linestyle=":", linewidth=1)
            add_1d_summary_markers(ax, values[finite], gate_opts)
            apply_plot_editor(ax, gate_opts)
            if gate_opts["show_mean"] or gate_opts["show_median"]:
                ax.legend(frameon=False)
            render_figure(fig, use_container_width=True)

            gate_rows_table = []
            for label, rows in groups.items():
                sub = df.iloc[rows]
                xdi = sub[metric].replace([np.inf, -np.inf], np.nan).dropna()
                xdt = sub["width_ms"].replace([np.inf, -np.inf], np.nan).dropna()
                gate_rows_table.append(
                    dict(
                        gate=label,
                        n=len(sub),
                        mean_deltaI_nA=float(np.mean(xdi)) if len(xdi) else np.nan,
                        median_deltaI_nA=float(np.median(xdi)) if len(xdi) else np.nan,
                        mean_dwell_ms=float(np.mean(xdt)) if len(xdt) else np.nan,
                        median_dwell_ms=float(np.median(xdt)) if len(xdt) else np.nan,
                    )
                )
            gate_summary = pd.DataFrame(gate_rows_table)
            st.dataframe(gate_summary.round(4), use_container_width=True)

            selected_gate = st.selectbox("Inspect gate", list(groups))
            selected_rows = groups[selected_gate]
            gate_df = df.iloc[selected_rows].copy()

            c1, c2 = st.columns(2)

            with c1:
                gate_di_opts = plot_editor(
                    "gate_selected_deltaI",
                    f"ΔI distribution · {selected_gate}",
                    metric_label + " (nA)",
                    "Event count",
                    default_bins=min(45, max(5, len(gate_df) // 5 + 1)),
                    allow_summary_markers=True,
                )
                vals = gate_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.hist(vals, bins=gate_di_opts["bins"])
                if len(vals):
                    add_1d_summary_markers(ax, vals, gate_di_opts)
                apply_plot_editor(ax, gate_di_opts)
                if gate_di_opts["show_mean"] or gate_di_opts["show_median"]:
                    ax.legend(frameon=False)
                render_figure(fig, use_container_width=True)

            with c2:
                gate_dt_opts = plot_editor(
                    "gate_selected_dwell",
                    f"Dwell-time distribution · {selected_gate}",
                    "Event width / dwell-like duration (ms)",
                    "Event count",
                    default_bins=min(45, max(5, len(gate_df) // 5 + 1)),
                    allow_summary_markers=True,
                )
                vals = gate_df["width_ms"].replace([np.inf, -np.inf], np.nan).dropna()
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.hist(vals, bins=gate_dt_opts["bins"])
                if len(vals):
                    add_1d_summary_markers(ax, vals, gate_dt_opts)
                apply_plot_editor(ax, gate_dt_opts)
                if gate_dt_opts["show_mean"] or gate_dt_opts["show_median"]:
                    ax.legend(frameon=False)
                render_figure(fig, use_container_width=True)

            st.session_state["gate_groups"] = groups
            st.session_state["gate_table"] = df

        except Exception as exc:
            st.warning(str(exc))

# -------------------------------------------------------------------
# 4. SEGMENT WEIGHTED
# -------------------------------------------------------------------
with tabs[3]:
    st.header("Segment-weighted analysis on the original undivided recording")
    p = st.session_state.loaded

    if not p:
        st.info("Load the files first.")
    else:
        seg = segment_metrics(p["fitting_arrays"], p["fit_to_dataset"])

        if not len(seg):
            st.warning(
                "No SEGMENT_INFO_*_segment_mean_diffs / segment_widths_time arrays were found."
            )
        else:
            st.write(
                "For each event, the duration-weighted blockade is Σ(ΔIⱼτⱼ)/Στⱼ. "
                "The app now treats this mainly as a distribution across events, rather than as a replacement for clustering."
            )

            # --------------------- summary stats ---------------------
            def _summary_row(series):
                x = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
                n = len(x)
                std = float(np.std(x, ddof=1)) if n > 1 else 0.0
                return dict(
                    n=n,
                    mean=float(np.mean(x)) if n else np.nan,
                    median=float(np.median(x)) if n else np.nan,
                    std=std if n else np.nan,
                    sem=float(std / np.sqrt(n)) if n else np.nan,
                    q1=float(np.percentile(x, 25)) if n else np.nan,
                    q3=float(np.percentile(x, 75)) if n else np.nan,
                )

            s_block = _summary_row(seg["duration_weighted_mean_blockade_nA"])
            s_dwell = _summary_row(seg["total_segment_dwell_ms"])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Events with segment data", len(seg))
            m2.metric("Weighted ΔI mean", f'{s_block["mean"]:.3f} nA')
            m3.metric("Weighted ΔI median", f'{s_block["median"]:.3f} nA')
            m4.metric("Median total dwell", f'{s_dwell["median"]:.3f} ms')

            summary_table = pd.DataFrame([
                {"quantity": "Duration-weighted mean ΔI (nA)", **s_block},
                {"quantity": "Total segmented dwell (ms)", **s_dwell},
            ])
            st.dataframe(summary_table.round(4), use_container_width=True)

            st.subheader("A · Event distributions")
            c1, c2 = st.columns(2)

            with c1:
                opts = plot_editor(
                    "seg_weighted_deltaI_hist",
                    "Duration-weighted mean ΔI distribution",
                    "Duration-weighted mean ΔI (nA)",
                    "Event count",
                    default_bins=45,
                    allow_summary_markers=True,
                )
                vals = seg["duration_weighted_mean_blockade_nA"].replace([np.inf, -np.inf], np.nan).dropna()
                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.hist(vals, bins=opts["bins"])
                add_1d_summary_markers(ax, vals, opts)
                apply_plot_editor(ax, opts)
                if opts["show_mean"] or opts["show_median"]:
                    ax.legend(frameon=False)
                render_figure(fig, use_container_width=True)

            with c2:
                opts = plot_editor(
                    "seg_total_dwell_hist",
                    "Total segmented dwell distribution",
                    "Total segmented dwell (ms)",
                    "Event count",
                    default_bins=45,
                    allow_summary_markers=True,
                )
                vals = seg["total_segment_dwell_ms"].replace([np.inf, -np.inf], np.nan).dropna()
                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.hist(vals, bins=opts["bins"])
                add_1d_summary_markers(ax, vals, opts)
                apply_plot_editor(ax, opts)
                if opts["show_mean"] or opts["show_median"]:
                    ax.legend(frameon=False)
                render_figure(fig, use_container_width=True)

            st.subheader("B · Weighted ΔI versus time")
            time_metric_map = {
                "Total segmented dwell time": "total_segment_dwell_ms",
                "Duration-weighted mean segment duration": "duration_weighted_segment_duration_ms",
                "Blockade-weighted mean segment duration": "blockade_weighted_segment_duration_ms",
            }
            time_choice = st.selectbox("Time quantity", list(time_metric_map), key="seg_time_choice")
            time_col = time_metric_map[time_choice]

            opts = plot_editor(
                "seg_weighted_scatter",
                "Weighted blockade versus time",
                time_choice + " (ms)",
                "Duration-weighted mean ΔI (nA)",
                allow_log=True,
                allow_summary_markers=True,
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(seg[time_col], seg["duration_weighted_mean_blockade_nA"], s=18, alpha=0.5)
            add_2d_summary_markers(ax, seg[time_col], seg["duration_weighted_mean_blockade_nA"], opts)
            apply_plot_editor(ax, opts)
            render_figure(fig, use_container_width=True)

            st.subheader("C · Full event table")
            st.dataframe(seg.round(5), use_container_width=True)
            st.download_button(
                "Download segment-weighted event table CSV",
                seg.to_csv(index=False),
                "segment_weighted_event_metrics.csv",
                "text/csv",
            )

# -------------------------------------------------------------------
# 5. EXPORT
# -------------------------------------------------------------------
with tabs[4]:
    st.header("Export grouped files in the same three-file NPZ categories")
    p = st.session_state.loaded

    if not p:
        st.info("Load the files first.")
    else:
        source_names = p["source_names"]

        st.subheader("A. Export K-means clusters")
        r = st.session_state.cluster_result
        if not r:
            st.info("Run K-means first.")
        else:
            assignments = r["assignments"]
            groups = {}
            for c in sorted(assignments.loc[assignments.cluster >= 0, "cluster"].unique()):
                rows = assignments.loc[assignments.cluster == c, "dataset_row"].astype(int).tolist()
                groups[f"cluster_{int(c)}"] = rows

            if st.button("Prepare all cluster files ZIP"):
                blob = export_many_groups_zip(
                    groups,
                    p["dataset_arrays"],
                    p["fitting_arrays"],
                    p["eventdata_arrays"],
                    p["fit_to_dataset"],
                    p["fit_to_raw"],
                    assignments,
                    source_names=source_names,
                )
                st.session_state["cluster_export_blob"] = blob

            if "cluster_export_blob" in st.session_state:
                st.download_button(
                    "Download cluster files",
                    st.session_state["cluster_export_blob"],
                    "kmeans_cluster_files.zip",
                    "application/zip",
                )

        st.divider()
        st.subheader("B. Export ΔI gates")

        gate_groups = st.session_state.get("gate_groups")
        gate_table = st.session_state.get("gate_table")
        if not gate_groups or gate_table is None:
            st.info("Create ΔI gates in Step 3 first.")
        else:
            if st.button("Prepare all gate files ZIP"):
                blob = export_many_groups_zip(
                    gate_groups,
                    p["dataset_arrays"],
                    p["fitting_arrays"],
                    p["eventdata_arrays"],
                    p["fit_to_dataset"],
                    p["fit_to_raw"],
                    gate_table,
                    source_names=source_names,
                )
                st.session_state["gate_export_blob"] = blob

            if "gate_export_blob" in st.session_state:
                st.download_button(
                    "Download current-gated files",
                    st.session_state["gate_export_blob"],
                    "deltaI_gate_files.zip",
                    "application/zip",
                )

        st.info(
            "Exports preserve the original dataset X rows and preserve event-indexed arrays from the uploaded "
            "event-fitting/event-data files wherever they can be matched. Original event IDs are retained rather "
            "than renumbered. A provenance JSON and CSV are included in every group ZIP."
        )

# -------------------------------------------------------------------
# 6. MULTI-SALT COMPARE
# -------------------------------------------------------------------
with tabs[5]:
    st.header("Multiple salts: individual plots first, then combined comparison")
    st.write(
        "Upload the three matching files separately for each salt. "
        "This avoids relying on identical timestamps in the filenames."
    )

    salts = ["LiCl", "NaCl", "KCl", "RbCl", "CsCl"]
    salt_uploads = {}

    for salt in salts:
        with st.expander(f"{salt} files", expanded=False):
            c1, c2, c3 = st.columns(3)
            ds = c1.file_uploader(
                f"{salt} · dataset.npz",
                type=["npz"],
                key=f"multi_{salt}_dataset",
            )
            raw = c2.file_uploader(
                f"{salt} · event_data.npz",
                type=["npz"],
                key=f"multi_{salt}_eventdata",
            )
            fit = c3.file_uploader(
                f"{salt} · event_fitting.npz",
                type=["npz"],
                key=f"multi_{salt}_fitting",
            )
            salt_uploads[salt] = dict(dataset=ds, eventdata=raw, fitting=fit)

    st.subheader("Use the same clustering settings for every salt")
    cmp_clustering_input = st.radio(
        "Clustering input for every salt",
        ["NanoSense 7 scalar features", "event_data whole waveform"],
        horizontal=True,
        key="cmp_clustering_input",
    )
    cmp_waveform_representation = "Native-time midpoint pad/crop"
    cmp_waveform_native_samples = 850
    cmp_waveform_resample_points = 200
    cmp_waveform_scaling = "Physical blockade (nA)"
    if cmp_clustering_input == "event_data whole waveform":
        cw1, cw2, cw3 = st.columns(3)
        cmp_waveform_representation = cw1.selectbox(
            "Waveform representation",
            ["Native-time midpoint pad/crop", "Duration-normalised resample"],
            key="cmp_waveform_representation",
        )
        if cmp_waveform_representation == "Native-time midpoint pad/crop":
            cmp_waveform_native_samples = cw2.slider(
                "Waveform vector length (samples)", 100, 3000, 850, 50, key="cmp_waveform_native_samples"
            )
        else:
            cmp_waveform_resample_points = cw2.slider(
                "Resampled waveform points", 64, 512, 200, 8, key="cmp_waveform_resample_points"
            )
        cmp_waveform_scaling = cw3.selectbox(
            "Waveform scaling",
            ["Physical blockade (nA)", "L2-normalised shape only"],
            key="cmp_waveform_scaling",
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    cmp_mode = c1.radio(
        "Choose k",
        ["Manual", "Silhouette"],
        horizontal=False,
        key="cmp_mode",
    )
    cmp_manual_k = c2.number_input(
        "Manual k",
        min_value=2,
        max_value=12,
        value=4,
        step=1,
        key="cmp_manual_k",
    )
    cmp_kmax = c3.number_input(
        "Maximum k to test",
        min_value=3,
        max_value=12,
        value=8,
        step=1,
        key="cmp_kmax",
    )
    cmp_n_init = c4.number_input(
        "K-means initialisations",
        min_value=10,
        max_value=300,
        value=100,
        step=10,
        key="cmp_n_init",
    )
    cmp_seed = c5.number_input(
        "Random seed",
        min_value=0,
        max_value=1_000_000,
        value=42,
        step=1,
        key="cmp_seed",
    )

    cmp_tol = st.number_input(
        "Timestamp matching tolerance (s)",
        min_value=1e-10,
        max_value=1e-3,
        value=1e-7,
        format="%.1e",
        key="cmp_tol",
    )
    cmp_window = st.slider(
        "Centered waveform window (samples)",
        200,
        3000,
        850,
        50,
        key="cmp_window",
    )
    cmp_x_mode = st.radio(
        "Waveform x-axis",
        ["time", "data_index"],
        horizontal=True,
        key="cmp_x_mode",
    )

    if st.button("Run salt analyses", type="primary"):
        results_by_salt = {}
        errors = []

        for salt in salts:
            files = salt_uploads[salt]
            present = [files["dataset"] is not None, files["eventdata"] is not None, files["fitting"] is not None]

            if not any(present):
                continue
            if not all(present):
                errors.append(f"{salt}: upload all three files or leave all three empty.")
                continue

            try:
                rec = load_recording(
                    read_uploaded_bytes(files["dataset"]),
                    read_uploaded_bytes(files["eventdata"]),
                    read_uploaded_bytes(files["fitting"]),
                    files["dataset"].name,
                    files["eventdata"].name,
                    files["fitting"].name,
                    float(cmp_tol),
                )

                result = run_kmeans_pipeline(
                    rec,
                    cmp_mode,
                    int(cmp_manual_k),
                    int(cmp_kmax),
                    int(cmp_n_init),
                    int(cmp_seed),
                    clustering_input=cmp_clustering_input,
                    waveform_representation=cmp_waveform_representation,
                    waveform_native_samples=int(cmp_waveform_native_samples),
                    waveform_resample_points=int(cmp_waveform_resample_points),
                    waveform_scaling=cmp_waveform_scaling,
                )

                stats = cross_salt_cluster_stats(result["assignments"], salt)
                info, medians = get_waveform_family_data(
                    rec,
                    result["assignments"],
                    window_samples=int(cmp_window),
                )

                results_by_salt[salt] = dict(
                    recording=rec,
                    result=result,
                    stats=stats,
                    profile_info=info,
                    medians=medians,
                )

            except Exception as exc:
                errors.append(f"{salt}: {exc}")

        for err in errors:
            st.error(err)

        if results_by_salt:
            st.session_state.multi_compare = dict(
                results_by_salt=results_by_salt,
                x_mode=cmp_x_mode,
                window=int(cmp_window),
                clustering_input=cmp_clustering_input,
                waveform_representation=cmp_waveform_representation,
            )
            st.success(f"Analysed {len(results_by_salt)} salt recording(s).")
        elif not errors:
            st.warning("Upload at least one complete three-file salt recording.")

    mc = st.session_state.multi_compare

    if mc:
        results_by_salt = mc["results_by_salt"]

        # -------------------------------------------------------
        # Individual salt results
        # -------------------------------------------------------
        st.divider()
        st.header("A · Individual salt results")
        st.caption(
            "Each salt is clustered independently using the same K-means settings. "
            "Every numerical summary reports both mean and median where appropriate."
        )

        for salt, item in results_by_salt.items():
            with st.expander(f"{salt} · individual results", expanded=True):
                result = item["result"]
                assignments = result["assignments"]
                valid = assignments[assignments["cluster"] >= 0].copy()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("k", result["k"])
                m2.metric("Silhouette", f'{result["diagnostics"]["silhouette"]:.3f}')
                m3.metric("Davies–Bouldin", f'{result["diagnostics"]["davies_bouldin"]:.3f}')
                m4.metric("Events", len(valid))

                item["stats"] = cross_salt_cluster_stats(assignments, salt)
                st.dataframe(item["stats"].round(4), use_container_width=True)

                wtab, bdtab, htab, ptab, stab = st.tabs(
                    [
                        "Waveform families",
                        "Blockade–dwell",
                        "Histograms",
                        "Population",
                        "Segment weighted",
                    ]
                )

                with wtab:
                    if item["medians"]:
                        xlab = "Time relative to event midpoint (ms)" if mc["x_mode"] == "time" else "Centered data index"
                        opts = plot_editor(
                            f"multi_{salt}_waveforms",
                            f"{salt} · midpoint-aligned cluster families",
                            xlab,
                            "Blockade (nA)",
                        )
                        family_panels, overlay = plot_waveform_family_panels(
                            item["profile_info"],
                            item["medians"],
                            x_mode=mc["x_mode"],
                        )
                        apply_figure_editor(family_panels, opts, title_as_suptitle=True)
                        render_figure(family_panels, use_container_width=True)
                        st.caption(
                            "Grey traces are real cluster members; the red curve is the pointwise median representative."
                        )
                        overlay_opts = plot_editor(
                            f"multi_{salt}_waveform_overlay",
                            f"{salt} · representative profiles",
                            xlab,
                            "Median blockade (nA)",
                        )
                        apply_figure_editor(overlay, overlay_opts, title_as_suptitle=False)
                        render_figure(overlay, use_container_width=True)
                    else:
                        st.warning("No matched event-fitting traces were available.")

                with bdtab:
                    opts = plot_editor(
                        f"multi_{salt}_bd",
                        f"{salt} · blockade versus dwell",
                        "Event width / dwell-like duration (ms)",
                        "Blockade height (nA)",
                        allow_log=True,
                        allow_summary_markers=True,
                    )
                    fig, ax = plt.subplots(figsize=(8, 5))
                    for c, g in valid.groupby("cluster"):
                        ax.scatter(
                            g["width_ms"],
                            g["height_nA"],
                            s=16,
                            alpha=0.45,
                            label=f"Cluster {c}",
                        )
                        add_2d_summary_markers(ax, g["width_ms"], g["height_nA"], opts)
                    apply_plot_editor(ax, opts)
                    ax.legend()
                    render_figure(fig, use_container_width=True)
                    if summary_marker_caption(opts):
                        st.caption(summary_marker_caption(opts))

                with htab:
                    dwell_opts = plot_editor(
                        f"multi_{salt}_dwell_hist",
                        f"{salt} · dwell-time distributions by cluster",
                        "Event width / dwell-like duration (ms)",
                        "Event count",
                        default_bins=30,
                        allow_summary_markers=True,
                    )
                    fig = plot_histograms_by_cluster(
                        valid,
                        "width_ms",
                        dwell_opts["xlabel"],
                        bins=dwell_opts["bins"],
                        title=dwell_opts["title"],
                        ylabel=dwell_opts["ylabel"],
                        xlim=(dwell_opts["xmin"], dwell_opts["xmax"]) if dwell_opts["xmin"] is not None and dwell_opts["xmax"] is not None else None,
                        ylim=(dwell_opts["ymin"], dwell_opts["ymax"]) if dwell_opts["ymin"] is not None and dwell_opts["ymax"] is not None else None,
                        show_mean=dwell_opts["show_mean"],
                        show_median=dwell_opts["show_median"],
                    )
                    render_figure(fig, use_container_width=True)

                    block_opts = plot_editor(
                        f"multi_{salt}_block_hist",
                        f"{salt} · blockade distributions by cluster",
                        "Blockade height (nA)",
                        "Event count",
                        default_bins=30,
                        allow_summary_markers=True,
                    )
                    fig = plot_histograms_by_cluster(
                        valid,
                        "height_nA",
                        block_opts["xlabel"],
                        bins=block_opts["bins"],
                        title=block_opts["title"],
                        ylabel=block_opts["ylabel"],
                        xlim=(block_opts["xmin"], block_opts["xmax"]) if block_opts["xmin"] is not None and block_opts["xmax"] is not None else None,
                        ylim=(block_opts["ymin"], block_opts["ymax"]) if block_opts["ymin"] is not None and block_opts["ymax"] is not None else None,
                        show_mean=block_opts["show_mean"],
                        show_median=block_opts["show_median"],
                    )
                    render_figure(fig, use_container_width=True)
                    if summary_marker_caption(block_opts):
                        st.caption(summary_marker_caption(block_opts))

                with ptab:
                    pop = valid["cluster"].value_counts().sort_index()
                    pct = 100 * pop / pop.sum()
                    opts = plot_editor(
                        f"multi_{salt}_population",
                        f"{salt} · cluster population",
                        "Cluster",
                        "Population (%)",
                    )
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.bar([f"Cluster {c}" for c in pct.index], pct.values)
                    for i, value in enumerate(pct.values):
                        ax.text(i, value, f"{value:.1f}%", ha="center", va="bottom")
                    apply_plot_editor(ax, opts)
                    render_figure(fig, use_container_width=True)

                with stab:
                    seg = segment_metrics(item["recording"]["fitting_arrays"], item["recording"]["fit_to_dataset"])
                    if not len(seg):
                        st.warning("No segment-level arrays available for this salt.")
                    else:
                        def _mstats(s):
                            x = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
                            sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
                            return dict(
                                n=len(x), mean=float(np.mean(x)), median=float(np.median(x)),
                                sd=sd, sem=float(sd / np.sqrt(len(x))),
                                q1=float(np.percentile(x, 25)), q3=float(np.percentile(x, 75)),
                            )
                        b = _mstats(seg["duration_weighted_mean_blockade_nA"])
                        d = _mstats(seg["total_segment_dwell_ms"])
                        st.dataframe(pd.DataFrame([
                            {"quantity": "Duration-weighted mean ΔI (nA)", **b},
                            {"quantity": "Total segmented dwell (ms)", **d},
                        ]).round(4), use_container_width=True)

                        c1, c2 = st.columns(2)
                        with c1:
                            opts = plot_editor(
                                f"multi_{salt}_seg_di",
                                f"{salt} · weighted ΔI distribution",
                                "Duration-weighted mean ΔI (nA)",
                                "Event count",
                                default_bins=35,
                                allow_summary_markers=True,
                            )
                            vals = seg["duration_weighted_mean_blockade_nA"].dropna()
                            fig, ax = plt.subplots(figsize=(7, 4))
                            ax.hist(vals, bins=opts["bins"])
                            add_1d_summary_markers(ax, vals, opts)
                            apply_plot_editor(ax, opts)
                            if opts["show_mean"] or opts["show_median"]:
                                ax.legend(frameon=False)
                            render_figure(fig, use_container_width=True)
                        with c2:
                            opts = plot_editor(
                                f"multi_{salt}_seg_dt",
                                f"{salt} · total segmented dwell distribution",
                                "Total segmented dwell (ms)",
                                "Event count",
                                default_bins=35,
                                allow_summary_markers=True,
                            )
                            vals = seg["total_segment_dwell_ms"].dropna()
                            fig, ax = plt.subplots(figsize=(7, 4))
                            ax.hist(vals, bins=opts["bins"])
                            add_1d_summary_markers(ax, vals, opts)
                            apply_plot_editor(ax, opts)
                            if opts["show_mean"] or opts["show_median"]:
                                ax.legend(frameon=False)
                            render_figure(fig, use_container_width=True)

        # -------------------------------------------------------
        # Family mapping
        # -------------------------------------------------------
        st.divider()
        st.header("B · Match salt-specific clusters to common families")
        st.info(
            "Cluster IDs are not assumed homologous across salts. "
            "The default suggestion follows increasing median blockade, but you can change any assignment."
        )

        family_options = ["Unmapped"] + [chr(65 + i) for i in range(10)]
        mapping = {}

        for salt, item in results_by_salt.items():
            clusters = sorted(item["stats"]["cluster"].astype(int).tolist())
            st.markdown(f"**{salt}**")
            cols = st.columns(min(5, max(1, len(clusters))))
            mapping[salt] = {}
            for j, cluster in enumerate(clusters):
                default_family = chr(65 + j)
                default_index = family_options.index(default_family)
                family = cols[j % len(cols)].selectbox(
                    f"Cluster {cluster}",
                    family_options,
                    index=default_index,
                    key=f"family_map_{salt}_{cluster}",
                )
                mapping[salt][int(cluster)] = family

        stats = mapped_cross_salt_stats(results_by_salt, mapping)

        st.subheader("Matched family summary · mean + median")
        summary_columns = [
            "salt", "cluster", "family", "n", "population_pct",
            "mean_dwell_ms", "median_dwell_ms", "std_dwell_ms", "sem_dwell_ms", "q1_dwell_ms", "q3_dwell_ms",
            "mean_blockade_nA", "median_blockade_nA", "std_blockade_nA", "sem_blockade_nA", "q1_blockade_nA", "q3_blockade_nA",
        ]
        available_summary_columns = [c for c in summary_columns if c in stats.columns]
        st.dataframe(
            stats[available_summary_columns].round(4),
            use_container_width=True,
        )

        # -------------------------------------------------------
        # Combined plots
        # -------------------------------------------------------
        st.divider()
        st.header("C · Combined cross-salt plots")

        st.subheader("1 · Blockade–dwell distributions for all salts")
        opts = plot_editor(
            "cross_bd",
            "Blockade–dwell distributions with shared axes",
            "Event width / dwell-like duration (ms)",
            "Blockade height (nA)",
            allow_log=True,
            allow_summary_markers=True,
        )
        fig = cross_salt_blockade_dwell_panels(
            results_by_salt,
            show_mean=opts["show_mean"],
            show_median=opts["show_median"],
        )
        apply_figure_editor(fig, opts, title_as_suptitle=True)
        render_figure(fig, use_container_width=True)

        st.subheader("2 · Matched event-family shapes across salts")
        xlab = "Time relative to event midpoint (ms)" if mc["x_mode"] == "time" else "Centered data index"
        opts = plot_editor(
            "cross_family_profiles",
            "Matched median DNA event families across salts",
            xlab,
            "Median blockade (nA)",
        )
        family_fig = cross_salt_profile_figure(
            results_by_salt,
            mapping,
            x_mode=mc["x_mode"],
        )
        if family_fig is not None:
            apply_figure_editor(family_fig, opts, title_as_suptitle=True)
            render_figure(family_fig, use_container_width=True)
        else:
            st.warning("No mapped waveform families are available.")

        st.subheader("3 · Family population fractions")
        opts = plot_editor(
            "cross_population",
            "Event-family population fractions across salts",
            "Electrolyte",
            "Population (%)",
        )
        fig = cross_salt_population_figure(stats)
        apply_plot_editor(fig.axes[0], opts)
        render_figure(fig, use_container_width=True)

        st.subheader("4 · Family-resolved dwell and blockade")
        summary_mode = st.radio(
            "Central value and error bars",
            ["Median + IQR", "Mean ± SD", "Mean ± SEM"],
            horizontal=True,
            key="cross_summary_mode",
        )
        st.caption(
            "Median + IQR is robust for skewed event distributions. Mean ± SD describes spread; mean ± SEM describes uncertainty of the event-level mean, not biological replicate uncertainty."
        )

        c1, c2 = st.columns(2)
        with c1:
            opts = plot_editor(
                "cross_dwell_metric",
                f"Family-resolved dwell time across salts · {summary_mode}",
                "Electrolyte",
                "Dwell time (ms)",
            )
            fig = cross_salt_family_metric_figure(
                stats, quantity="dwell", summary_mode=summary_mode,
                title=opts["title"], ylabel=opts["ylabel"]
            )
            apply_plot_editor(fig.axes[0], opts)
            render_figure(fig, use_container_width=True)

        with c2:
            opts = plot_editor(
                "cross_block_metric",
                f"Family-resolved blockade across salts · {summary_mode}",
                "Electrolyte",
                "Blockade (nA)",
            )
            fig = cross_salt_family_metric_figure(
                stats, quantity="blockade", summary_mode=summary_mode,
                title=opts["title"], ylabel=opts["ylabel"]
            )
            apply_plot_editor(fig.axes[0], opts)
            render_figure(fig, use_container_width=True)

        st.subheader("5 · Cross-salt heatmaps")
        heat_stat = st.radio("Heatmap statistic", ["Median", "Mean"], horizontal=True, key="heat_stat")
        dwell_metric = "median_dwell_ms" if heat_stat == "Median" else "mean_dwell_ms"
        block_metric = "median_blockade_nA" if heat_stat == "Median" else "mean_blockade_nA"
        c1, c2 = st.columns(2)
        with c1:
            opts = plot_editor(
                "cross_dwell_heatmap",
                f"{heat_stat} dwell time",
                "Electrolyte",
                "Event family",
            )
            fig = cross_salt_heatmap(stats, dwell_metric, opts["title"], "ms")
            # Keep categorical tick locations while applying editable titles/typography.
            apply_plot_editor(fig.axes[0], opts)
            render_figure(fig, use_container_width=True)
        with c2:
            opts = plot_editor(
                "cross_block_heatmap",
                f"{heat_stat} blockade",
                "Electrolyte",
                "Event family",
            )
            fig = cross_salt_heatmap(stats, block_metric, opts["title"], "nA")
            apply_plot_editor(fig.axes[0], opts)
            render_figure(fig, use_container_width=True)

        # -------------------------------------------------------
        # Segment-weighted cross-salt analysis
        # -------------------------------------------------------
        st.divider()
        st.header("D · Segment-weighted comparison across salts")
        st.write(
            "This section uses the original undivided recordings. Each event contributes its duration-weighted mean segment blockade and total segmented dwell."
        )

        seg_all, seg_summary = segment_weighted_salt_summary(results_by_salt)
        if not len(seg_all):
            st.warning("No segment-level information was available across the loaded salts.")
        else:
            st.subheader("Mean + median numerical summary")
            st.dataframe(seg_summary.round(4), use_container_width=True)

            norm_choice = st.radio(
                "Combined histogram y-axis",
                ["Probability density", "Event count"],
                horizontal=True,
                key="seg_hist_norm",
            )
            density = norm_choice == "Probability density"
            hist_ylabel = "Probability density" if density else "Event count"

            c1, c2 = st.columns(2)
            with c1:
                opts = plot_editor(
                    "cross_seg_weighted_di_hist",
                    "Weighted ΔI distributions across salts",
                    "Duration-weighted mean ΔI (nA)",
                    hist_ylabel,
                    default_bins=40,
                    allow_summary_markers=True,
                )
                fig = overlay_histogram_by_salt(
                    seg_all,
                    "duration_weighted_mean_blockade_nA",
                    bins=opts["bins"],
                    density=density,
                    title=opts["title"],
                    xlabel=opts["xlabel"],
                    ylabel=opts["ylabel"],
                    show_mean=opts["show_mean"],
                    show_median=opts["show_median"],
                )
                apply_plot_editor(fig.axes[0], opts)
                render_figure(fig, use_container_width=True)

            with c2:
                opts = plot_editor(
                    "cross_seg_dwell_hist",
                    "Segmented dwell distributions across salts",
                    "Total segmented dwell (ms)",
                    hist_ylabel,
                    default_bins=40,
                    allow_summary_markers=True,
                )
                fig = overlay_histogram_by_salt(
                    seg_all,
                    "total_segment_dwell_ms",
                    bins=opts["bins"],
                    density=density,
                    title=opts["title"],
                    xlabel=opts["xlabel"],
                    ylabel=opts["ylabel"],
                    show_mean=opts["show_mean"],
                    show_median=opts["show_median"],
                )
                apply_plot_editor(fig.axes[0], opts)
                render_figure(fig, use_container_width=True)

            seg_error_mode = st.radio(
                "Error-bar summary",
                ["Median + IQR", "Mean ± SD", "Mean ± SEM"],
                horizontal=True,
                key="seg_error_mode",
            )
            seg_plot_style = st.radio(
                "Summary plot style",
                ["Line + error bars", "Bar plot"],
                horizontal=True,
                key="seg_summary_plot_style",
            )
            c1, c2 = st.columns(2)
            with c1:
                opts = plot_editor(
                    "cross_seg_di_error",
                    f"Weighted ΔI across salts · {seg_error_mode}",
                    "Electrolyte",
                    "Duration-weighted mean ΔI (nA)",
                )
                fig = salt_errorbar_summary(
                    seg_summary,
                    "duration_weighted_mean_blockade_nA",
                    seg_error_mode,
                    opts["title"],
                    opts["ylabel"],
                    plot_style=seg_plot_style,
                )
                apply_plot_editor(fig.axes[0], opts)
                render_figure(fig, use_container_width=True)

            with c2:
                opts = plot_editor(
                    "cross_seg_dwell_error",
                    f"Segmented dwell across salts · {seg_error_mode}",
                    "Electrolyte",
                    "Total segmented dwell (ms)",
                )
                fig = salt_errorbar_summary(
                    seg_summary,
                    "total_segment_dwell_ms",
                    seg_error_mode,
                    opts["title"],
                    opts["ylabel"],
                    plot_style=seg_plot_style,
                )
                apply_plot_editor(fig.axes[0], opts)
                render_figure(fig, use_container_width=True)

            opts = plot_editor(
                "cross_seg_scatter",
                "Weighted ΔI versus total segmented dwell",
                "Total segmented dwell (ms)",
                "Duration-weighted mean ΔI (nA)",
                allow_log=True,
                allow_summary_markers=True,
            )
            fig, ax = plt.subplots(figsize=(9, 5))
            for salt, g in seg_all.groupby("salt"):
                ax.scatter(
                    g["total_segment_dwell_ms"],
                    g["duration_weighted_mean_blockade_nA"],
                    s=14,
                    alpha=0.40,
                    label=salt,
                )
                add_2d_summary_markers(
                    ax,
                    g["total_segment_dwell_ms"],
                    g["duration_weighted_mean_blockade_nA"],
                    opts,
                )
            apply_plot_editor(ax, opts)
            ax.legend(frameon=False)
            render_figure(fig, use_container_width=True)

            st.download_button(
                "Download all-salt segment-weighted event table CSV",
                seg_all.to_csv(index=False),
                "cross_salt_segment_weighted_events.csv",
                "text/csv",
            )
            st.download_button(
                "Download all-salt segment-weighted summary CSV",
                seg_summary.to_csv(index=False),
                "cross_salt_segment_weighted_summary.csv",
                "text/csv",
            )

        st.download_button(
            "Download cross-salt matched-family summary CSV",
            stats.to_csv(index=False),
            "cross_salt_family_summary.csv",
            "text/csv",
        )
