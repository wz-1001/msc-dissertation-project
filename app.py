"""app.py: sidebar posts to the fastapi backend, this tab renders whatever comes back."""

import os
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from theme import COLOUR_BEST_HIGHLIGHT, COLOUR_MUTED, PLOTLY_COLOURWAY, apply_theme

BACKEND_URL = os.environ.get("BENCHMARK_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Green AI Benchmarking & Telemetry Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_theme()

if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = []

st.title("Green AI: Accuracy vs Energy Benchmarking Platform")
st.markdown(
    '<p class="app-subtitle">Quantifying computational environmental externalities across Machine Learning model hierarchies.</p>',
    unsafe_allow_html=True
)

DEFAULT_SAMPLE_SIZES = [1000, 5000, 10000, 25000, 50000, 100000]

OVERFIT_PENALTY_SCALE = 0.5   # a gap at or above this hits the maximum penalty
OVERFIT_PENALTY_WEIGHT = 0.3  # maximum fraction subtracted from the score

NON_METRIC_COLUMNS = {
    "Model", "Family", "Note", "Error", "Sample Size",
    "Synthetic (Bootstrapped)", "Best Pick", "Best", "Task Type",
}

LOWER_IS_BETTER_METRICS = {
    "MAE", "RMSE", "Emissions (gCO2)", "Emissions CodeCarbon (gCO2)",
    "Emissions Manual (gCO2)", "Training Time (s)"
}

TASK_TYPE_OPTIONS = {
    "Auto-detect": "auto",
    "Classification": "classification",
    "Regression": "regression"
}

COLUMN_DISPLAY_PRIORITY = [
    "Best", "Sample Size", "Model", "Family",
    "Accuracy", "Train Accuracy", "Val Accuracy",
    "Precision", "Recall", "F1 Score",
    "R2", "Train R2", "Val R2", "MAE", "RMSE",
    "Emissions (gCO2)", "Emissions CodeCarbon (gCO2)", "Emissions Manual (gCO2)",
    "Emissions Variance (%)", "Emissions Diff (gCO2)",
    "Training Time (s)", "Green Score",
    "Synthetic (Bootstrapped)", "Task Type", "Note", "Error",
]


def ordered_columns(df: pd.DataFrame) -> list:
    """sort columns by priority, dump anything unlisted on the end."""
    ordered = [c for c in COLUMN_DISPLAY_PRIORITY if c in df.columns]
    leftover = [c for c in df.columns if c not in ordered]
    return ordered + leftover


def get_plottable_metrics(df: pd.DataFrame) -> list:
    """drop non-metric and non-numeric columns from the plot dropdowns."""
    numeric_cols = [
        c for c in df.columns
        if c not in NON_METRIC_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if "Emissions CodeCarbon (gCO2)" in df.columns and "Emissions (gCO2)" in numeric_cols:
        numeric_cols.remove("Emissions (gCO2)")
    return numeric_cols


def get_performance_column(df: pd.DataFrame) -> str | None:
    """pick whichever primary performance metric is actually present."""
    if "Accuracy" in df.columns:
        return "Accuracy"
    if "R2" in df.columns:
        return "R2"
    return None


def compute_green_scores(
    results_df: pd.DataFrame,
    acc_weight: float,
    performance_col: str,
    efficiency_col: str = "Emissions (gCO2)",
    group_col: str | None = None,
    apply_overfit_penalty: bool = False
) -> pd.DataFrame:
    """normalise performance + efficiency into the Green Score, flag the best row per group."""
    df = results_df.copy()
    df["Green Score"] = pd.NA
    df["Best Pick"] = False

    if performance_col not in df.columns or efficiency_col not in df.columns:
        return df

    is_candidate = ~df["Family"].isin(["Reference", "Error"])
    if "Error" in df.columns:
        is_candidate &= df["Error"].isna()

    group_keys = df[group_col].unique() if (group_col and group_col in df.columns) else [None]

    perf_higher_is_better = performance_col not in LOWER_IS_BETTER_METRICS
    eff_lower_is_better = efficiency_col in LOWER_IS_BETTER_METRICS

    for key in group_keys:
        in_group = (df[group_col] == key) if (group_col and key is not None) else pd.Series(True, index=df.index)
        mask = is_candidate & in_group
        candidates = df[mask]
        if candidates.empty:
            continue

        perf_min, perf_max = candidates[performance_col].min(), candidates[performance_col].max()
        eff_min, eff_max = candidates[efficiency_col].min(), candidates[efficiency_col].max()

        for idx, row in candidates.iterrows():
            raw_perf = (row[performance_col] - perf_min) / (perf_max - perf_min) if perf_max > perf_min else 1.0
            raw_eff = (row[efficiency_col] - eff_min) / (eff_max - eff_min) if eff_max > eff_min else 0.0

            norm_perf = raw_perf if perf_higher_is_better else (1.0 - raw_perf)
            norm_eff = (1.0 - raw_eff) if eff_lower_is_better else raw_eff

            overfit_penalty = 0.0
            if apply_overfit_penalty:
                gap = row.get("Overfit Gap", 0.0)
                gap = 0.0 if pd.isna(gap) else max(gap, 0.0)
                overfit_penalty = min(gap / OVERFIT_PENALTY_SCALE, 1.0) * OVERFIT_PENALTY_WEIGHT

            raw_score = acc_weight * norm_perf + (1 - acc_weight) * norm_eff
            df.at[idx, "Green Score"] = round(max(raw_score - overfit_penalty, 0.0), 4)

        group_scores = df.loc[candidates.index, "Green Score"].astype(float)
        best_idx = group_scores.idxmax()
        df.at[best_idx, "Best Pick"] = True

    return df


def render_emissions_comparison_summary(results_df: pd.DataFrame) -> None:
    """info box: how far apart codecarbon and the manual estimator landed."""
    if "Emissions Variance (%)" not in results_df.columns:
        return

    model_rows = results_df[~results_df["Family"].isin(["Reference", "Error"])]
    if model_rows.empty:
        return

    mean_primary = model_rows["Emissions CodeCarbon (gCO2)"].abs().mean()
    mean_secondary = model_rows["Emissions Manual (gCO2)"].abs().mean()
    pct = ((mean_secondary - mean_primary) / mean_primary * 100) if mean_primary else 0

    st.info(
        f"**Emissions tool agreement:** On average the manual estimator readings varied from CodeCarbon by "
        f"**{pct:+.2f}%**. This describes how closely the two tools "
        f"agree with each other and not which one is more correct as neither has access to a ground truth "
        f"power reading."
    )


def render_column_associations(result_data: dict) -> None:
    """warnings/info badges for dropped or flagged-but-kept columns."""
    associations = result_data.get("column_associations")
    if not associations:
        return

    manual = result_data.get("columns_manually_selected", False)
    threshold_val = result_data.get("threshold_used", 0.95)
    dropped = [a for a in associations if a.get("dropped")]
    kept = [a for a in associations if not a.get("dropped")]

    source_note = "your column selection" if manual else f"the automatic {threshold_val:.2f} cutoff"

    if dropped:
        details = ", ".join(f"`{a['column']}` ({a['association']:.4f})" for a in dropped)
        st.warning(f"**Excluded from training based on {source_note}:** {details}")

    high_kept = [a for a in kept if a.get("suggested_drop")]
    if high_kept:
        details = ", ".join(f"`{a['column']}` ({a['association']:.4f})" for a in high_kept)
        st.info(
            f"**Scored {threshold_val:.2f}+ but kept in training:** {details}. "
            "Either you chose to keep it or dropping every flagged column would have left "
            "nothing to train on."
        )


def render_metrics_glossary(threshold_cutoff: float = 0.95) -> None:
    """expandable dictionary for the column/metric names."""
    with st.expander("ℹ️ What do these columns and metrics mean?"):
        st.markdown(f"""
**Model / Family:** the trained algorithm and which tier it belongs to (Baseline, Ensemble, Deep Learning or Reference for the trivial comparison model below).

**Task Type:** whether this run treated the target as **Classification** (predicting a category) or **Regression** (predicting a continuous number). Auto-detect picks this based on the target column: numeric with many unique values (e.g. price or inflation rate) → Regression; anything else (categories or a small set of numeric values like a 0/1 flag) → Classification.

**Classification metrics**

**Accuracy:** % of test set predictions the model got exactly right. Higher is better.

**Train Accuracy:** same measure but on the training data itself. A big gap between this and Accuracy usually means overfitting.

**Precision:** of everything predicted as a given class what fraction was actually correct.

**Recall:** of everything that actually belonged to a given class what fraction the model correctly found.

**F1 Score:** a single balance of Precision and Recall.

**Majority-Class Reference:** a trivial "model" that always guesses the single most common class. Any real model scoring at or below this isn't learning anything useful.

**Regression metrics**

**R2 (R-squared):** how much of the variation in the target the model explains. 1.0 = perfect, 0.0 = no better than always guessing the average and negative = worse than that.

**Train R2:** same measure on the training data. A big gap vs test R2 usually means overfitting.

**MAE (Mean Absolute Error):** average size of the prediction error in the target's original units (e.g. £ for price or percentage points for inflation).

**RMSE (Root Mean Squared Error):** similar to MAE but penalises large errors more heavily.

**Log-Transformed Target:** `True` when the training target was right-skewed enough (measured, not assumed) that the model was fit against log1p(target) rather than the raw values, with predictions inverse-transformed back before every metric was computed. Only ever applies to regression.

**Mean-Value Reference:** a trivial "model" that always predicts the training set average. Any real model scoring at or below its R2 isn't learning anything useful.

**Column associations**

**Association (Pearson |r| / Cramer's V):** how strongly a feature column statistically determines the target on a 0 to 1 scale.
* **Pearson |r|** is used for continuous Regression targets (e.g. price).
* **Cramer's V** is used for categorical Classification targets.

0 = unrelated and 1 = the column is basically a duplicate of the target under another name. Columns scoring {threshold_cutoff:.2f}+ are flagged as suggested exclusions since high association usually means the column is leaking the answer and not genuinely predicting it. However you choose what actually gets excluded before running.

**Shared across both**

**Emissions (gCO2):** estimated carbon emitted while training and evaluating this model.

**Training Time (s):** wall-clock seconds spent fitting this model, timed separately from the emissions trackers. Under k-fold this includes every fold's fit time plus the final fit on the full training set. Useful alongside Emissions since the two don't always move together. A model can be slow but energy-efficient (e.g. lots of waiting on I/O) or fast but power-hungry depending on hardware. Majority-Class/Mean-Value Reference rows show 0.0 since nothing is actually trained.

**Green Score:** a single 0 to 1 score combining performance and efficiency weighted by the sidebar slider. Only ranks models *within the same comparison group* and not comparable across different runs.

**Best Pick:** whichever model currently has the highest Green Score given the current slider position.

**Sample Size / Synthetic (Bootstrapped):** only present during a sample size sweep. Shows the row count used at that step and whether it was reached by genuine subsampling (`False`) or bootstrap resampling with replacement (`True`) when the dataset was too small to reach that size naturally.

**Emissions CodeCarbon / Emissions Manual / Emissions Variance (%):** only present when "Compare emissions tools" is on. Both tools' readings for the *same* training run side by side and their percentage variance.
        """)


st.sidebar.header("Dataset Management")

task_type_label = st.sidebar.selectbox(
    "Prediction task type",
    options=list(TASK_TYPE_OPTIONS.keys()),
    index=0,
    help="Auto-detect looks at the target column: numeric with many unique values "
         "(e.g. price or inflation rate) → treated as Regression; anything else (categories "
         "or a small set of numeric values) → treated as Classification. Override this if "
         "you want a specific framing regardless of what is auto-detected."
)
task_type_param = TASK_TYPE_OPTIONS[task_type_label]

emissions_library = st.sidebar.selectbox(
    "Emissions tracking library",
    options=["codecarbon", "manual"],
    index=0,
    help="Which tool measures carbon emissions during training. 'codecarbon' samples real "
         "hardware power counters. 'manual' needs no extra install as it estimates emissions "
         "from wall-clock time × an assumed power draw × grid carbon intensity following the "
         "Green Algorithms methodology (Lannelongue et al. 2021). Ignored if the emissions "
         "comparison checkbox below is switched on."
)
st.sidebar.caption(
    "Learn more about the methodologies: "
    "[CodeCarbon](https://codecarbon.io/) | "
    "[Green Algorithms](https://www.green-algorithms.org/)"
)
st.sidebar.markdown("---")
compare_emissions = st.sidebar.checkbox(
    "Compare emissions tools (CodeCarbon vs Manual)",
    value=False,
    help="Runs both CodeCarbon and the manual estimator on every model in the same training "
         "run (not two separate runs which would introduce timing noise) so their readings "
         "can be compared directly. When this is on it overrides the 'Emissions tracking "
         "library' choice above as both tools are always used together."
)

show_manual_settings = (emissions_library == "manual") or compare_emissions

tdp_watts = grid_intensity_input = pue_input = None
if show_manual_settings:
    st.sidebar.markdown("**Manual tracker settings**")
    tdp_watts = st.sidebar.number_input(
        "CPU TDP (Watts)",
        min_value=1.0, max_value=300.0, value=15.0, step=1.0,
        help="Your CPU's Thermal Design Power. Look it up on Intel ARK or AMD's spec pages "
             "(on Windows Task Manager → Performance → CPU shows the exact model name). "
             "This is the 'power' term in the manual estimator's formula. Defaults to a rough "
             "laptop estimate if left unchanged."
    )
    grid_intensity_input = st.sidebar.number_input(
        "Grid carbon intensity (gCO2/kWh)",
        min_value=1.0, max_value=900.0, value=130.0, step=5.0,
        help="How much CO2 is emitted per kWh of electricity on your local grid. UK average "
             "is ≈130 (National Grid ESO). Outside the UK use your own country's figure or a "
             "rough global average fallback of 450 to 480."
    )
    pue_input = st.sidebar.number_input(
        "PUE (datacenter overhead)",
        min_value=1.0, max_value=3.0, value=1.0, step=0.05,
        help="Power Usage Effectiveness which is how much extra power a datacenter draws beyond the "
             "computer itself. Leave at 1.0 for a personal laptop or desktop since there is no "
             "datacenter overhead to account for. This matches the convention used in Green "
             "Algorithms (Lannelongue et al. 2021) for personal machines."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**Green Score Tradeoff Settings**")

custom_perf_metric = st.sidebar.selectbox(
    "Performance metric",
    options=["Auto-detect", "Accuracy", "F1 Score", "Precision", "Recall", "R2", "MAE", "RMSE"],
    index=0,
    help="Select the benchmark metric used for the performance component of the Green Score."
)

custom_eff_metric = st.sidebar.selectbox(
    "Efficiency metric",
    options=["Emissions (gCO2)", "Training Time (s)", "Emissions Manual (gCO2)"],
    index=0,
    help="Select the resource metric used for the environmental/cost component."
)

accuracy_weight = st.sidebar.slider(
    "Tradeoff weighting",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="0.0 = pick solely based on efficiency, 1.0 = pick solely based on performance. "
         "0.5 balances both equally. Drag any time to update instantly with no re-run needed."
)
label_col1, label_col2 = st.sidebar.columns(2)
with label_col1:
    st.caption("Lowest cost / emissions")
with label_col2:
    st.markdown(
        f"<div style='text-align: right;'><span style='font-size: 0.8rem; color: {COLOUR_MUTED};'>Best performance</span></div>",
        unsafe_allow_html=True
    )

penalise_overfitting = st.sidebar.checkbox(
    "Penalise overfitting in Green Score", 
    value=False, 
    help="If enabled, models with a large gap between training and validation performance will have their Green Score heavily penalised."
)

st.sidebar.markdown("---")
run_sweep = st.sidebar.checkbox(
    "Run sample size sweep",
    value=False,
    help="Instead of one run on the full dataset this trains every model at each selected sample "
         "size below and compares how performance and emissions scale with data volume."
)

sweep_sizes = []
if run_sweep:
    sweep_sizes = st.sidebar.multiselect(
        "Sample sizes to test",
        options=DEFAULT_SAMPLE_SIZES,
        default=DEFAULT_SAMPLE_SIZES,
        help="Each size trains every model from scratch. More sizes equals more thorough results but "
             "takes proportionally longer to run."
    )
    st.sidebar.caption(
        "If a dataset has fewer rows than a selected size that size is reached via "
        "bootstrap resampling (sampling with replacement) and flagged as synthetic in results "
        "rather than real additional data."
    )

st.sidebar.markdown("---")
uploaded_files = st.sidebar.file_uploader(
    "Upload Datasets (CSV)", type=["csv"], accept_multiple_files=True,
    help="Upload one or more CSV files. ONS time series exports and HM Land Registry Price "
         "Paid data are auto-detected and parsed specially while anything else falls back to "
         "general purpose CSV parsing."
)

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
        fname = uploaded_file.name

        preview_key = f"preview_cols_{fname}"
        shape_key = f"preview_shape_{fname}"
        ts_key = f"is_time_series_{fname}"

        if preview_key not in st.session_state:
            try:
                files_payload = {"file": (fname, uploaded_file.getvalue(), "text/csv")}
                resp = requests.post(f"{BACKEND_URL}/preview-columns/", files=files_payload, timeout=120)
                if resp.status_code == 200 and resp.json().get("status") == "success":
                    payload = resp.json()
                    st.session_state[preview_key] = payload["columns"]
                    st.session_state[shape_key] = (payload.get("rows"), payload.get("cols"))
                    st.session_state[ts_key] = payload.get("is_time_series", False)
                else:
                    st.session_state[preview_key] = None
                    st.session_state[shape_key] = None
                    st.session_state[ts_key] = False
            except requests.exceptions.Timeout:
                st.sidebar.warning(
                    f"Previewing {fname} timed out. The backend server might be busy training a model in the background."
                )
                st.session_state[preview_key] = None
                st.session_state[shape_key] = None
                st.session_state[ts_key] = False
            except requests.exceptions.ConnectionError:
                st.sidebar.error("Connection Failed! Is your FastAPI backend running?")
                st.session_state[preview_key] = None
                st.session_state[shape_key] = None
                st.session_state[ts_key] = False

        shape = st.session_state.get(shape_key)
        if shape:
            st.sidebar.caption(f"📊 Dataset size: **{shape[0]:,} rows** × **{shape[1]} columns**")

        target_choice = None
        cols = st.session_state.get(preview_key)
        is_time_series_ui = st.session_state.get(ts_key, False)

        if cols:
            target_choice = st.sidebar.selectbox(
                f"Target column: {fname}",
                options=cols,
                index=len(cols) - 1,
                key=f"target_{fname}",
                help="Which column the models should learn to predict. Defaults to the "
                     "last column if left unchanged. Reflects the actual parsed columns "
                     "including non-standard formats like ONS exports."
            )
        else:
            st.sidebar.caption(f"(Couldn't preview columns for {fname} so we will auto-detect on the backend.)")

        assoc_key = f"associations_{fname}"
        excl_key = f"excluded_{fname}"
        thresh_key = f"threshold_{fname}"

        if is_time_series_ui:
            st.sidebar.info(
                "📈 **Time-series detected.** Column analysis is disabled because historical "
                "lag features are naturally expected to correlate with the target."
            )
        else:
            if st.sidebar.button(
                f"Analyse columns: {fname}", key=f"analyse_{fname}",
                help="Scores every feature column's statistical association with the target "
                     "column so you can decide what to exclude before running the full benchmark."
            ):
                with st.spinner(f"Analysing columns in {fname}..."):
                    try:
                        files_payload = {"file": (fname, uploaded_file.getvalue(), "text/csv")}
                        post_data = {"task_type": task_type_param}
                        if target_choice:
                            post_data["target_column"] = target_choice
                        resp = requests.post(
                            f"{BACKEND_URL}/analyse-columns/",
                            files=files_payload,
                            data=post_data,
                            timeout=180
                        )
                        if resp.status_code == 200:
                            payload = resp.json()
                            if payload.get("status") == "success":
                                st.session_state[assoc_key] = payload
                                st.session_state[thresh_key] = payload.get("threshold_used", 0.95)
                                st.session_state[excl_key] = {
                                    a["column"] for a in payload.get("column_associations", [])
                                    if a.get("suggested_drop")
                                }
                            else:
                                st.sidebar.error("Column analysis returned an error. Please check your target column and task type.")
                        else:
                            st.sidebar.error(f"Column analysis failed: {resp.status_code}")
                    except requests.exceptions.Timeout:
                        st.sidebar.error("Column analysis timed out. Your backend is currently overloaded.")
                    except requests.exceptions.ConnectionError:
                        st.sidebar.error("Connection Failed! Is your FastAPI backend running?")

            if assoc_key in st.session_state and st.session_state[assoc_key].get("status") == "success":
                associations_data = st.session_state[assoc_key].get("column_associations", [])
                threshold_setting = st.session_state.get(thresh_key, 0.95)

                metric_name = associations_data[0].get("metric", "Cramer's V") if associations_data else "Cramer's V"

                with st.sidebar.expander(f"Columns to exclude: {fname}", expanded=True):
                    if not associations_data:
                        st.info("No scoreable feature columns found.")
                    else:
                        st.caption(
                            f"Score is **{metric_name}** against the target (0 = unrelated and 1 = duplicate). "
                            f"**{threshold_setting:.2f} and above is the preferred cutoff** "
                            "for excluding a column as a likely near duplicate of the target. These are "
                            "pre-checked below but you decide what actually gets excluded."
                        )

                    st.session_state.setdefault(excl_key, set())
                    for item in associations_data:
                        col_name = item["column"]
                        default_checked = col_name in st.session_state[excl_key]
                        checked = st.checkbox(
                            f"{col_name}: {item['association']:.4f}"
                            + (" (suggested)" if item.get("suggested_drop") else ""),
                            value=default_checked,
                            key=f"exclcb_{fname}_{col_name}",
                        )
                        if checked:
                            st.session_state[excl_key].add(col_name)
                        else:
                            st.session_state[excl_key].discard(col_name)

        if st.sidebar.button(
            f"Run Benchmark: {fname}", key=f"btn_{fname}",
            help="Sends this dataset to the backend and trains every model in the suite on it "
                 "using the settings currently selected in the sidebar."
        ):
            if run_sweep and not sweep_sizes:
                st.sidebar.error("Select at least one sample size or turn off the sweep toggle.")
            else:
                with st.spinner(f"Training models for {fname}... This can take several minutes."):
                    files_payload = {"file": (fname, uploaded_file.getvalue(), "text/csv")}
                    post_data = {"emissions_library": emissions_library, "task_type": task_type_param}
                    if target_choice:
                        post_data["target_column"] = target_choice
                    if run_sweep and sweep_sizes:
                        post_data["sample_sizes"] = ",".join(str(s) for s in sorted(sweep_sizes))
                    if show_manual_settings:
                        post_data["tdp_watts"] = str(tdp_watts)
                        post_data["grid_intensity_g_per_kwh"] = str(grid_intensity_input)
                        post_data["pue"] = str(pue_input)
                    post_data["compare_emissions"] = "true" if compare_emissions else "false"

                    if excl_key in st.session_state:
                        post_data["excluded_columns"] = ",".join(sorted(st.session_state[excl_key]))

                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/run-benchmark/",
                            files=files_payload,
                            data=post_data,
                            timeout=None
                        )

                        if response.status_code == 200:
                            st.sidebar.success(f"Finished {fname}!")
                            st.session_state.benchmark_results.append(response.json())
                        else:
                            st.sidebar.error(f"API Error {response.status_code}: Something went wrong.")
                    except requests.exceptions.Timeout:
                        st.sidebar.error(
                            "Benchmark timed out! The models took longer than expected. "
                            "Try reducing sample sizes or unselecting the largest sweep points."
                        )
                    except requests.exceptions.ConnectionError:
                        st.sidebar.error("Connection Failed! Is your FastAPI backend running?")

tab_about, tab_dashboard = st.tabs(["📖 About this tool", "📊 Dashboard"])

with tab_about:
    st.header("What is this tool for?")
    st.markdown("""
Most machine learning gets judged on one thing: how accurate is it? This tool adds a second question alongside it — **how much carbon did it cost to get that accuracy?** You can see the trade-off between the two directly, instead of treating sustainability as something to think about later, if at all.

---

## Tutorial

### 1. Uploading a dataset

Drop a CSV file into the sidebar uploader.

### 2. Classification vs. regression detection

Once you've picked a target column, the tool works out which kind of problem it's dealing with:

$$\\text{task} = \\begin{cases} \\text{regression} & \\text{if numeric and } > 10 \\text{ unique values} \\\\ \\text{classification} & \\text{otherwise} \\end{cases}$$

This matters because squeezing a continuous target into a classification framing throws away information you'd otherwise keep. When classification is the chosen path — either automatically for a categorical column, or via a median split for a continuous one — a numeric target with lots of unique values gets converted into a binary label:

$$y_{\\text{binary}} = \\begin{cases} 1 & \\text{if } y > \\text{median}(y) \\\\ 0 & \\text{otherwise} \\end{cases}$$

If that split happens to produce only one class — an edge case that can crop up with heavily skewed distributions — the tool falls back to splitting around the mean instead. You can override the automatic choice from the sidebar any time you want a specific framing regardless of what's detected.

### 3. Column association screening (leakage check)

Before any training happens, every feature column gets scored against the target to see how strongly they're statistically linked. The goal is to catch columns that are basically the target in disguise — these tend to produce results that look suspiciously perfect rather than genuinely predictive.

**For a regression target**, that score is the absolute Pearson correlation coefficient:

$$r = \\frac{\\sum (x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum (x_i - \\bar{x})^2 \\sum (y_i - \\bar{y})^2}}, \\quad \\text{score} = |r|$$

**For a classification target**, it's Cramér's V instead, derived from a chi-squared test of independence between feature and target. It's the right tool here because it's built specifically to measure association between two categorical variables:

$$V = \\sqrt{\\frac{\\chi^2 / n}{\\min(r - 1,\\ k - 1)}}$$

where $n$ is the number of observations, and $r$, $k$ are the row and column counts of the feature × target contingency table. Both scores sit on a 0 to 1 scale — 0 means no relationship, 1 means the column is essentially a re-encoded copy of the target.

Anything scoring **0.95 or above** gets pre-checked as a suggested exclusion in the "Analyse columns" panel, but nothing is ever dropped automatically — you decide what actually gets excluded before the benchmark runs. This check is skipped entirely for time-series data, since lag features are meant to correlate strongly with the target — that's intentional signal, not leakage.

### 4. Training the model suite

Every run trains a full tiered hierarchy of models:

*   **Baseline:** Decision Tree, Naive Bayes *(classification only)*, Logistic or Linear Regression, and SVM.
*   **Ensemble:** Random Forest and XGBoost.
*   **Deep Learning:** Multi-Layer Perceptrons (MLP) are always included. Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) models only get added when a time series is detected, since both assume adjacent features have a meaningful order to them.

Each model trains on a 60% training / 20% validation / 20% test split. The test set is kept completely separate and only ever used for the final performance numbers shown in the results table — never for anything during training. Validation does two jobs: spotting overfitting, and controlling early stopping in the neural networks.


### 5. Measuring emissions

Two tracking methods are on offer:

**CodeCarbon** reads physical hardware power counters (via Intel RAPL or NVIDIA NVML) throughout training and integrates the readings over time.

**The manual estimator** follows the method set out by Lannelongue, Grealey and Inouye (2021) in *"Green Algorithms: Quantifying the Carbon Footprint of Computation"*:

$$E \\text{ (kWh)} = t \\times P \\times \\text{PUE}$$
$$C \\text{ (gCO}_2\\text{)} = E \\times CI$$

Here $t$ is training time in hours, $P$ is the processor's Thermal Design Power (TDP) in kilowatts, PUE is the Power Usage Effectiveness, and $CI$ is the local grid's carbon intensity.

### 6. The Green Score

Predictive performance and environmental cost live on completely different scales, so both get min-max normalised to a 0 to 1 range across the models in the run before being combined. For metrics where lower is better (error, emissions, time), the normalisation flips so a smaller value still counts as a bigger contribution:

$$\\text{norm}_{\\text{perf}} = \\frac{\\text{perf} - \\text{perf}_{\\min}}{\\text{perf}_{\\max} - \\text{perf}_{\\min}}, \\qquad \\text{norm}_{\\text{eff}} = 1 - \\frac{\\text{cost} - \\text{cost}_{\\min}}{\\text{cost}_{\\max} - \\text{cost}_{\\min}}$$

From there, the Green Score is just a weighted blend, controlled by the sidebar slider ($w$, from 0 to 1):

$$\\text{GreenScore} = w \\cdot \\text{norm}_{\\text{perf}} + (1 - w) \\cdot \\text{norm}_{\\text{eff}}$$

If the **overfitting penalty** is switched on via the sidebar, models that overfit their training data receive a heavy deduction. The penalty scales based on the performance gap between the training and validation sets. It maxes out at a 0.3 point deduction for a gap of 0.5 or worse:

$$\\text{penalty} = \\min\\left(\\frac{\\text{gap}}{0.5}, 1.0\\right) \\times 0.3$$

$$\\text{Green Score} = \\max(\\text{Raw Score} - \\text{penalty}, 0.0)$$

### 7. Sample-size sweep

Turning this on repeats the entire model suite at each row count you select, so you can see how performance and emissions scale as data volume grows. If a requested size is bigger than the dataset actually has, it's reached via bootstrap resampling (with replacement) and flagged as synthetic.

### 8. Comparing emissions tools

Switching on "Compare emissions tools" runs CodeCarbon and the manual estimator over the exact same training run at the same time, so their readings can be compared directly rather than across two separate (and slightly different) runs:

$$\\text{Variance \\%} = \\frac{C_{\\text{manual}} - C_{\\text{codecarbon}}}{C_{\\text{codecarbon}}} \\times 100$$
    """)

with tab_dashboard:
    if len(st.session_state.benchmark_results) > 0:

        if st.button("Clear All Results", help="Removes every result currently shown below."):
            st.session_state.benchmark_results = []
            st.rerun()

        st.markdown("---")

        for i, result_payload in enumerate(st.session_state.benchmark_results):
            lib_used = result_payload.get("emissions_library", "codecarbon")
            target_used = result_payload.get("target_column_used", "auto-detected")
            task_used = result_payload.get("task_type_used", "classification")
            current_thresh = result_payload.get("threshold_used", 0.95)
            st.subheader(
                f"{result_payload['filename']}  |  tracked with `{lib_used}`  |  "
                f"target: `{target_used}`  |  task: `{task_used}`"
            )

            render_column_associations(result_payload)

            results_dataframe = pd.DataFrame(result_payload["data"])

            if results_dataframe.empty or "Emissions (gCO2)" not in results_dataframe.columns:
                st.warning(f"Could not generate graphs for {result_payload['filename']} because the data was empty, highly imbalanced or invalid.")
                st.dataframe(results_dataframe, use_container_width=True)
                st.markdown("---")
                continue

            results_dataframe = results_dataframe.drop(columns=["Green Score", "Best Pick"], errors="ignore")

            is_sweep = "Sample Size" in results_dataframe.columns and results_dataframe["Sample Size"].nunique() > 1
            has_comparison = "Emissions CodeCarbon (gCO2)" in results_dataframe.columns

            # figure out the performance column to use
            if custom_perf_metric == "Auto-detect" or custom_perf_metric not in results_dataframe.columns:
                active_perf_col = get_performance_column(results_dataframe)
            else:
                active_perf_col = custom_perf_metric

            # figure out the efficiency column (maps Emissions to CodeCarbon during comparisons)
            if custom_eff_metric == "Emissions (gCO2)" and "Emissions (gCO2)" not in results_dataframe.columns:
                active_eff_col = "Emissions CodeCarbon (gCO2)"
            elif custom_eff_metric in results_dataframe.columns:
                active_eff_col = custom_eff_metric
            else:
                active_eff_col = "Emissions (gCO2)"

            render_metrics_glossary(current_thresh)

            if is_sweep:
                results_dataframe = compute_green_scores(
                    results_dataframe,
                    accuracy_weight,
                    performance_col=active_perf_col,
                    efficiency_col=active_eff_col,
                    group_col="Sample Size",
                    apply_overfit_penalty=penalise_overfitting
                )

                render_emissions_comparison_summary(results_dataframe)

                if results_dataframe.get("Synthetic (Bootstrapped)", pd.Series(dtype=bool)).any():
                    st.info(
                        "Some sample sizes exceed this dataset's natural row count and were reached via "
                        "**bootstrap resampling (sampling with replacement)**: marked `Synthetic (Bootstrapped) = True` "
                        "below. These stress test energy scaling at fixed volumes and they are not additional real data."
                    )

                model_rows = results_dataframe[~results_dataframe["Family"].isin(["Reference", "Error"])]
                metric_options = get_plottable_metrics(model_rows)

                col1, col2 = st.columns(2)
                with col1:
                    idx1 = metric_options.index(active_perf_col) if active_perf_col in metric_options else 0
                    metric_1 = st.selectbox(
                        "Metric for left chart", options=metric_options, index=idx1,
                        key=f"sweep_metric1_{i}_{result_payload['filename']}",
                        help="Choose which metric to plot against sample size."
                    )
                    fig_1 = px.line(
                        model_rows.sort_values("Sample Size"),
                        x="Sample Size", y=metric_1, color="Model",
                        markers=True, title=f"{metric_1} vs Sample Size",
                        color_discrete_sequence=PLOTLY_COLOURWAY, template="plotly_white"
                    )
                    st.plotly_chart(fig_1, use_container_width=True, key=f"sweep_chart1_{i}_{result_payload['filename']}")
                with col2:
                    default_idx = metric_options.index(active_eff_col) if active_eff_col in metric_options else min(1, len(metric_options) - 1)
                    metric_2 = st.selectbox(
                        "Metric for right chart", options=metric_options, index=default_idx,
                        key=f"sweep_metric2_{i}_{result_payload['filename']}",
                        help="Choose which metric to plot against sample size."
                    )
                    fig_2 = px.line(
                        model_rows.sort_values("Sample Size"),
                        x="Sample Size", y=metric_2, color="Model",
                        markers=True, title=f"{metric_2} vs Sample Size",
                        color_discrete_sequence=PLOTLY_COLOURWAY, template="plotly_white"
                    )
                    st.plotly_chart(fig_2, use_container_width=True, key=f"sweep_chart2_{i}_{result_payload['filename']}")

                display_df = results_dataframe.copy()
                display_df["Best"] = display_df["Best Pick"].map({True: "Best", False: ""})
                display_df = display_df.drop(columns=["Best Pick"])
                if has_comparison and "Emissions (gCO2)" in display_df.columns:
                    display_df = display_df.drop(columns=["Emissions (gCO2)"])
                display_df = display_df[ordered_columns(display_df)].sort_values("Sample Size")

                def highlight_best(row):
                    is_best = row.get("Best") == "Best"
                    return [f'background-color: {COLOUR_BEST_HIGHLIGHT}' if is_best else '' for _ in row]

                fmt_dict = {}
                for pct_col in ["Accuracy", "Train Accuracy", "Val Accuracy", "Precision", "Recall", "F1 Score"]:
                    if pct_col in display_df.columns:
                        fmt_dict[pct_col] = "{:.2%}"
                for reg_col in ["R2", "Train R2", "Val R2", "MAE", "RMSE"]:
                    if reg_col in display_df.columns:
                        fmt_dict[reg_col] = "{:.4f}"
                if "Emissions (gCO2)" in display_df.columns:
                    fmt_dict["Emissions (gCO2)"] = "{:.4e}"
                for extra_col in ["Emissions CodeCarbon (gCO2)", "Emissions Manual (gCO2)", "Emissions Diff (gCO2)"]:
                    if extra_col in display_df.columns:
                        fmt_dict[extra_col] = "{:.4e}"
                if "Emissions Variance (%)" in display_df.columns:
                    fmt_dict["Emissions Variance (%)"] = "{:+.2f}%"

                st.dataframe(
                    display_df.style.apply(highlight_best, axis=1).format(fmt_dict),
                    use_container_width=True
                )

            else:
                results_dataframe = compute_green_scores(
                    results_dataframe,
                    accuracy_weight,
                    performance_col=active_perf_col,
                    efficiency_col=active_eff_col,
                    apply_overfit_penalty=penalise_overfitting
                )

                render_emissions_comparison_summary(results_dataframe)

                best_rows = results_dataframe[results_dataframe["Best Pick"].astype(bool)]
                if not best_rows.empty and active_perf_col in best_rows.columns:
                    best = best_rows.iloc[0]
                    perf_val = best[active_perf_col]
                    perf_fmt = f"{perf_val:.2%}" if "Accuracy" in active_perf_col or "F1" in active_perf_col else f"{perf_val:.4f}"
                    eff_val = best[active_eff_col]
                    eff_fmt = f"{eff_val:.4e}" if "Emissions" in active_eff_col else f"{eff_val:.2f}s"
                    st.success(
                        f"Best pick based on **{active_perf_col}** vs **{active_eff_col}** "
                        f"(weight: {accuracy_weight:.2f} toward performance): "
                        f"**{best['Model']}** ({best['Family']}) | "
                        f"{active_perf_col}: {perf_fmt}, {active_eff_col}: {eff_fmt}, "
                        f"Green Score: {best['Green Score']}"
                    )

                display_df = results_dataframe.copy()
                display_df["Best"] = display_df["Best Pick"].map({True: "Best", False: ""})
                display_df = display_df.drop(columns=["Best Pick"])
                if has_comparison and "Emissions (gCO2)" in display_df.columns:
                    display_df = display_df.drop(columns=["Emissions (gCO2)"])
                display_df = display_df[ordered_columns(display_df)]

                def highlight_best(row):
                    is_best = row.get("Best") == "Best"
                    return [f'background-color: {COLOUR_BEST_HIGHLIGHT}' if is_best else '' for _ in row]

                fmt_dict = {}
                for pct_col in ["Accuracy", "Train Accuracy", "Val Accuracy", "Precision", "Recall", "F1 Score"]:
                    if pct_col in display_df.columns:
                        fmt_dict[pct_col] = "{:.2%}"
                for reg_col in ["R2", "Train R2", "Val R2", "MAE", "RMSE"]:
                    if reg_col in display_df.columns:
                        fmt_dict[reg_col] = "{:.4f}"
                if "Emissions (gCO2)" in display_df.columns:
                    fmt_dict["Emissions (gCO2)"] = "{:.4e}"
                for extra_col in ["Emissions CodeCarbon (gCO2)", "Emissions Manual (gCO2)", "Emissions Diff (gCO2)"]:
                    if extra_col in display_df.columns:
                        fmt_dict[extra_col] = "{:.4e}"
                if "Emissions Variance (%)" in display_df.columns:
                    fmt_dict["Emissions Variance (%)"] = "{:+.2f}%"

                styled = display_df.style.apply(highlight_best, axis=1)
                if fmt_dict:
                    styled = styled.format(fmt_dict)
                st.dataframe(styled, use_container_width=True)

                metric_options = get_plottable_metrics(results_dataframe)

                col1, col2 = st.columns(2)
                with col1:
                    x_default = metric_options.index(active_eff_col) if active_eff_col in metric_options else 0
                    y_default = metric_options.index(active_perf_col) if active_perf_col in metric_options else min(1, len(metric_options) - 1)
                    x_metric = st.selectbox(
                        "X-axis metric", options=metric_options, index=x_default,
                        key=f"scatter_x_{i}_{result_payload['filename']}",
                        help="Choose what to plot on the horizontal axis."
                    )
                    y_metric = st.selectbox(
                        "Y-axis metric", options=metric_options, index=y_default,
                        key=f"scatter_y_{i}_{result_payload['filename']}",
                        help="Choose what to plot on the vertical axis."
                    )
                    size_metric = "F1 Score" if "F1 Score" in results_dataframe.columns else None
                    fig_scatter = px.scatter(
                        results_dataframe, x=x_metric, y=y_metric, color="Family",
                        hover_name="Model", size=size_metric,
                        title=f"{y_metric} vs {x_metric}",
                        color_discrete_sequence=PLOTLY_COLOURWAY, template="plotly_white"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True, key=f"scatter_{i}_{result_payload['filename']}")
                with col2:
                    bar_default = metric_options.index(active_eff_col) if active_eff_col in metric_options else 0
                    bar_metric = st.selectbox(
                        "Metric to compare across models", options=metric_options, index=bar_default,
                        key=f"bar_metric_{i}_{result_payload['filename']}",
                        help="Choose which metric to compare across models as a bar chart."
                    )
                    fig_bar = px.bar(
                        results_dataframe.sort_values(bar_metric), x="Model", y=bar_metric,
                        color="Family", title=f"{bar_metric} by Model",
                        color_discrete_sequence=PLOTLY_COLOURWAY, template="plotly_white"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{i}_{result_payload['filename']}")

            st.markdown("---")
    else:
        st.info("Upload your datasets in the sidebar and run the benchmarks to populate the dashboard.")