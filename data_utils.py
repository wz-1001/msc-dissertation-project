"""
data_utils.py

CSV parsing. smart_read_csv() is the general entry point. Two formats
get dedicated handling before falling back to generic parsing:

- parse_ons_series(): ONS single-series exports (metadata preamble,
  no header row, possibly multiple time frequencies stacked together).
- parse_land_registry_price_paid(): HM Land Registry Price Paid Data -
  16 fixed columns, no header row at all, format documented by HMLR.
"""

import pandas as pd
import io
import re
import csv

_MONTHS = {
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
}

_UUID_RE = re.compile(r"^\{?[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}?$")

# order matters here - matches the column order HMLR actually ships the file in, not alphabetical
LAND_REGISTRY_COLUMNS = [
    "transaction_id", "price", "date_of_transfer", "postcode",
    "property_type", "old_new", "duration", "paon", "saon",
    "street", "locality", "town_city", "district", "county",
    "ppd_category_type", "record_status",
]


def _classify_period(raw: str):
    s = str(raw).strip()

    if re.fullmatch(r"\d{4}", s):
        return "annual", pd.Period(s, freq="Y")

    m = re.fullmatch(r"(\d{4})\s+Q([1-4])", s)
    if m:
        return "quarterly", pd.Period(f"{m.group(1)}Q{m.group(2)}", freq="Q")

    m = re.fullmatch(r"(\d{4})\s+([A-Za-z]{3})", s)
    if m and m.group(2).upper() in _MONTHS:
        return "monthly", pd.Period(f"{m.group(1)}-{m.group(2).upper()}", freq="M")

    return None, None


def parse_ons_series(text: str):
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if len(r) >= 2]

    classified = []
    for row in rows:
        kind, period = _classify_period(row[0])
        if kind is None:
            continue
        value = str(row[1]).strip()
        classified.append((kind, period, value))

    if len(classified) < 10:
        return None

    # ONS files sometimes stack annual + quarterly + monthly for the same
    # series - just keep whichever one actually has the most rows
    kind_counts = {}
    for kind, _, _ in classified:
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    dominant_kind = max(kind_counts, key=kind_counts.get)

    filtered = [(p, v) for k, p, v in classified if k == dominant_kind]

    df = pd.DataFrame(filtered, columns=["period", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["period"] = df["period"].apply(lambda p: p.to_timestamp())
    df = df.dropna(subset=["value"]).sort_values("period").reset_index(drop=True)

    if len(df) < 10:
        return None

    return df


def _sniff_delimiter(text: str) -> str:
    """HMLR files are comma-delimited officially, but handle tab-delimited exports too."""
    sample = text[:5000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        return dialect.delimiter
    except Exception:
        return ","  # sniffer gives up on short/weird samples - comma is the documented default anyway


def parse_land_registry_price_paid(text: str):
    """
    Detects HM Land Registry Price Paid Data: 16 fixed columns, no header
    row. Confirmed by checking the first row's shape - GUID transaction
    id, numeric price, dd/mm/yyyy date - before committing to this format.
    """
    delimiter = _sniff_delimiter(text)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [r for r in reader if len(r) >= 15]  # allow 15 in case the trailing empty field got dropped somewhere

    if not rows:
        return None

    first = rows[0]
    if len(first) < 16:
        return None

    looks_like_id = bool(_UUID_RE.match(first[0].strip()))
    looks_like_price = first[1].strip().isdigit()  # plain integer, no £ sign or commas in the raw export
    looks_like_date = bool(re.match(r"^\d{2}/\d{2}/\d{4}", first[2].strip()))

    if not (looks_like_id and looks_like_price and looks_like_date):
        return None

    trimmed = [r[:16] for r in rows]  # some exports carry a trailing blank 17th field - drop anything past col 16
    df = pd.DataFrame(trimmed, columns=LAND_REGISTRY_COLUMNS)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["date_of_transfer"] = pd.to_datetime(df["date_of_transfer"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["price"]).reset_index(drop=True)

    return df


def smart_read_csv(file_bytes: bytes, max_skip: int = 15) -> pd.DataFrame:
    text = file_bytes.decode("utf-8-sig", errors="replace")

    ons_df = parse_ons_series(text)
    if ons_df is not None:
        return ons_df

    land_registry_df = parse_land_registry_price_paid(text)
    if land_registry_df is not None:
        return land_registry_df

    # neither known format matched - fall back to guessing where the real header row starts
    best_df = None
    for skip in range(0, max_skip):
        try:
            candidate = pd.read_csv(
                io.StringIO(text), skiprows=skip, on_bad_lines="skip", engine="python"
            )
        except Exception:
            continue

        if candidate.shape[1] < 2 or candidate.shape[0] < 3:
            continue

        # numeric column names basically always mean we haven't skipped
        # past the real header row yet
        header_looks_numeric = all(
            str(c).replace(".", "", 1).replace("-", "", 1).isdigit()
            for c in candidate.columns
        )
        unnamed_ratio = sum(
            str(c).startswith("Unnamed") for c in candidate.columns
        ) / candidate.shape[1]

        if header_looks_numeric or unnamed_ratio > 0.5:
            continue

        best_df = candidate
        break

    if best_df is None:
        best_df = pd.read_csv(io.StringIO(text), on_bad_lines="skip", engine="python")

    best_df = best_df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return best_df