"""
api.py

fastapi shell around benchmark.py. three endpoints: preview a file's
columns, run the leakage/association check on its own, run the full
benchmark. keeps the streamlit frontend from ever having to import
pandas or sklearn directly - it just posts a file and gets json back.
"""

from fastapi import FastAPI, UploadFile, File, Form
from benchmark import run_experiment, prepare_target, should_skip_association_check, looks_like_time_series
from column_analysis import compute_column_associations, DEFAULT_ASSOCIATION_THRESHOLD
from data_utils import smart_read_csv

app = FastAPI(title="Green AI Benchmarking API")

def resolve_target_column(df, requested: str):
    if requested and requested in df.columns:
        return requested
    return df.columns[-1]

@app.get("/")
def read_root():
    return {"message": "Green AI API is running! Ready to receive datasets."}

@app.post("/preview-columns/")
async def api_preview_columns(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = smart_read_csv(contents)
    except Exception as e:
        return {"filename": file.filename, "status": "error", "Error": f"{type(e).__name__}: {e}"}

    return {
        "filename": file.filename,
        "status": "success",
        "columns": df.columns.tolist(),
        "rows": df.shape[0],
        "cols": df.shape[1],
        "is_time_series": looks_like_time_series(df),  # real backend check, not a frontend guess off column names
    }

# keep this in sync with the UK spelling in app.py
@app.post("/analyse-columns/")
async def api_analyse_columns(
    file: UploadFile = File(...),
    target_column: str = Form(None),
    task_type: str = Form("auto"),
):
    contents = await file.read()

    try:
        df = smart_read_csv(contents)
    except Exception as e:
        return {"filename": file.filename, "status": "error", "Error": f"{type(e).__name__}: {e}"}

    target_col = resolve_target_column(df, target_column)

    if task_type not in ("auto", "classification", "regression"):
        task_type = "auto"

    prepped_df, resolved_target_col, error, resolved_task, is_time_series = prepare_target(
        df, target_col, file.filename, task_type
    )
    if error is not None:
        return {"filename": file.filename, "status": "error", "data": [error]}

    # same check benchmark.py runs for real - so the preview can't disagree with what actually happens
    if should_skip_association_check(prepped_df):
        return {
            "filename": file.filename,
            "status": "success",
            "target_column_used": target_col,
            "task_type_used": resolved_task,
            "column_associations": [],
            "threshold_used": DEFAULT_ASSOCIATION_THRESHOLD,
        }

    associations = compute_column_associations(prepped_df, resolved_target_col, task_type=resolved_task)

    return {
        "filename": file.filename,
        "status": "success",
        "target_column_used": target_col,
        "task_type_used": resolved_task,
        "column_associations": associations,
        "threshold_used": DEFAULT_ASSOCIATION_THRESHOLD,
    }

@app.post("/run-benchmark/")
async def api_run_benchmark(
    file: UploadFile = File(...),
    emissions_library: str = Form("codecarbon"),
    target_column: str = Form(None),
    sample_sizes: str = Form(None),
    tdp_watts: float = Form(None),
    grid_intensity_g_per_kwh: float = Form(None),
    pue: float = Form(None),
    compare_emissions: bool = Form(False),
    task_type: str = Form("auto"),
    excluded_columns: str = Form(None),
):
    contents = await file.read()

    try:
        df = smart_read_csv(contents)
    except Exception as e:
        return {
            "filename": file.filename,
            "status": "error",
            "data": [{"Model": "File Read Error", "Family": "Error", "Emissions (gCO2)": 0, "Error": f"{type(e).__name__}: {e}"}],
            "emissions_library": emissions_library,
        }

    target_col = resolve_target_column(df, target_column)

    parsed_sizes = None
    if sample_sizes:
        try:
            parsed_sizes = [int(s.strip()) for s in sample_sizes.split(",") if s.strip()]
            parsed_sizes = [s for s in parsed_sizes if s > 0]
            if not parsed_sizes:
                parsed_sizes = None
        except ValueError:
            return {
                "filename": file.filename,
                "status": "error",
                "data": [{"Model": "Config Error", "Family": "Error", "Emissions (gCO2)": 0, "Error": f"Could not parse sample_sizes '{sample_sizes}' as a comma-separated list of integers."}],
                "emissions_library": emissions_library,
            }

    if task_type not in ("auto", "classification", "regression"):
        task_type = "auto"

    parsed_excluded = None
    if excluded_columns is not None:
        parsed_excluded = [c.strip() for c in excluded_columns.split(",") if c.strip()]


    results, column_associations, task_type_used = run_experiment(
        df,
        target_col=target_col,
        dataset_name=file.filename,
        emissions_library=emissions_library,
        sample_sizes=parsed_sizes,
        tdp_watts=tdp_watts,
        grid_intensity_g_per_kwh=grid_intensity_g_per_kwh,
        pue=pue,
        compare_emissions=compare_emissions,
        task_type=task_type,
        excluded_columns=parsed_excluded,
    )

    return {
        "filename": file.filename,
        "status": "success",
        "data": results,
        "emissions_library": "codecarbon + manual (comparison)" if compare_emissions else emissions_library,
        "target_column_used": target_col,
        "sample_sizes_used": parsed_sizes,
        "compare_emissions": compare_emissions,
        "column_associations": column_associations,
        "task_type_used": task_type_used,
        "columns_manually_selected": parsed_excluded is not None,
        "threshold_used": DEFAULT_ASSOCIATION_THRESHOLD,
    }