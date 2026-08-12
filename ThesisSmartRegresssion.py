import re
from io import BytesIO

import altair as alt
import pandas as pd
import streamlit as st

# E2E-level position-sensitive similarity threshold
MIN_SIMILARITY_E2E = 0.10
# Only E2E pairs with similarity >= this will be used for test-set analysis
HIGH_SIM_E2E_THRESHOLD = 0.80
# Minimum similarity for test set names to be kept
MIN_SIMILARITY_TESTSET = 0.10


def base_prefix_strip(e2e: str) -> str:
    """Strip leading 'E2E...' prefix but keep the rest, including < and >."""
    if e2e is None:
        return ""
    s = str(e2e).strip()
    prefix_regex = re.compile(
        r'^(?:\d{1,3}_)?'
        r'(?:e2e(?:_[A-Za-z0-9]+)*)_',
        flags=re.IGNORECASE,
    )
    m = prefix_regex.match(s)
    if not m:
        return s
    cleaned = s[m.end():]
    cleaned = re.sub(r'^[\s_\-:]+', ' ', cleaned).strip()
    return cleaned


def clean_e2e_name(e2e: str) -> str:
    """Normalize E2E folder names for similarity comparison."""
    if e2e is None:
        return ""
    s = str(e2e).strip()

    if s.count("_") >= 4:
        parts = s.split("_", 4)
        return parts[4].strip()

    return base_prefix_strip(s)


def tokenize_with_signs(s: str):
    """Tokenize while preserving < and > and keeping token order."""
    if not s:
        return []
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9<>]+", " ", s)
    return [w for w in s.split() if w]


def position_sensitive_similarity(a: str, b: str) -> float:
    """Compute positional token similarity between two values."""
    ta = tokenize_with_signs(a)
    tb = tokenize_with_signs(b)

    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0

    min_len = min(len(ta), len(tb))
    matches = sum(1 for i in range(min_len) if ta[i] == tb[i])
    denom = max(len(ta), len(tb))
    return matches / denom


def find_col(df: pd.DataFrame, keyword_list):
    """Find a column whose name contains all keywords (case-insensitive)."""
    for c in df.columns:
        c_low = str(c).lower()
        if all(k in c_low for k in keyword_list):
            return c
    return None


def load_first_non_empty_sheet(uploaded_file):
    """Load the first non-empty sheet from the uploaded Excel file."""
    uploaded_file.seek(0)
    xls = pd.ExcelFile(uploaded_file)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if df.shape[1] > 0:
            return df, sheet_name
    raise ValueError("No valid sheet found in workbook.")


def create_output_workbook(unique_df, df_cross, df_same, df_testset_same, df_testset_cross, df_testcases):
    """Write result DataFrames to an in-memory Excel file."""
    output = BytesIO()

    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
            engine = "openpyxl"
        except Exception:
            engine = None

    if engine is None:
        raise ModuleNotFoundError(
            "Neither xlsxwriter nor openpyxl is installed. Please install one to export Excel."
        )

    def build_testset_name_similarity_sheet(df_testset_same, threshold=0.8):
        columns = [
            "Country",
            "E2E_A",
            "E2E_B",
            "Similarity_E2E",
            "TestSets_A",
            "TestSets_B",
            "MatchingTestSets",
            "MatchCount",
            "TotalTestSets_A",
            "TotalTestSets_B",
            "TestSetName_Similarity",
        ]
        if df_testset_same is None or df_testset_same.empty:
            return pd.DataFrame(columns=columns)

        grouped = df_testset_same.groupby(["Country", "E2E_A", "E2E_B"], dropna=False)
        result_rows = []

        for (country, e2eA, e2eB), group in grouped:
            similarity_e2e = group["Similarity_E2E"].iloc[0] if "Similarity_E2E" in group.columns else ""
            testsetsA = sorted(group["TestSetName_A"].dropna().astype(str).str.strip().unique())
            testsetsB = sorted(group["TestSetName_B"].dropna().astype(str).str.strip().unique())
            normalizedA = {ts.lower() for ts in testsetsA if ts}
            normalizedB = {ts.lower() for ts in testsetsB if ts}
            if not normalizedA or not normalizedB:
                continue
            common_lower = normalizedA & normalizedB
            common = sorted({ts for ts in testsetsA if ts.lower() in common_lower} | {ts for ts in testsetsB if ts.lower() in common_lower})
            similarity = len(common_lower) / max(len(normalizedA), len(normalizedB)) if max(len(normalizedA), len(normalizedB)) > 0 else 0.0
            if similarity < threshold:
                continue

            result_rows.append({
                "Country": country,
                "E2E_A": e2eA,
                "E2E_B": e2eB,
                "Similarity_E2E": similarity_e2e,
                "TestSets_A": "; ".join(testsetsA),
                "TestSets_B": "; ".join(testsetsB),
                "MatchingTestSets": "; ".join(common),
                "MatchCount": len(common_lower),
                "TotalTestSets_A": len(normalizedA),
                "TotalTestSets_B": len(normalizedB),
                "TestSetName_Similarity": round(similarity, 3),
            })

        return pd.DataFrame(result_rows, columns=columns)

    combined_columns = [
        "ComparisonType",
        "Country",
        "Country_A",
        "Country_B",
        "Test set Folder ID A",
        "E2E_A",
        "TestSetName_A",
        "Test set Folder ID B",
        "E2E_B",
        "TestSetName_B",
        "TestSet_Similarity",
        "Similarity_E2E",
    ]

    frames = []
    if df_testset_same is not None and not df_testset_same.empty:
        same_frame = df_testset_same.copy()
        same_frame["ComparisonType"] = "Same country"
        same_frame["Country_A"] = ""
        same_frame["Country_B"] = ""
        frames.append(same_frame.reindex(columns=combined_columns))
    if df_testset_cross is not None and not df_testset_cross.empty:
        cross_frame = df_testset_cross.copy()
        cross_frame["ComparisonType"] = "Cross country"
        if "Country" not in cross_frame.columns:
            cross_frame["Country"] = cross_frame.get("Country_A", "") + " -> " + cross_frame.get("Country_B", "")
        frames.append(cross_frame.reindex(columns=combined_columns))

    if frames:
        df_testset_comparison = pd.concat(frames, ignore_index=True)
    else:
        df_testset_comparison = pd.DataFrame(columns=combined_columns)

    df_testset_name_similarity = build_testset_name_similarity_sheet(df_testset_same, threshold=0.8)

    with pd.ExcelWriter(output, engine=engine) as writer:
        unique_df.to_excel(writer, sheet_name="Unique_Cleaned", index=False)
        df_cross.to_excel(writer, sheet_name="E2E_CROSS_COUNTRY", index=False)
        df_same.to_excel(writer, sheet_name="E2E_SAME_COUNTRY", index=False)
        df_testset_same.to_excel(writer, sheet_name="TEST_SET_SAME_COUNTRY", index=False)
        df_testset_name_similarity.to_excel(writer, sheet_name="TEST_SET_NAME_SIMILARITY", index=False)
        df_testset_comparison.to_excel(writer, sheet_name="TEST_SET_COMPARISON", index=False)
        df_testcases.to_excel(writer, sheet_name="TEST_CASE", index=False)

    output.seek(0)
    return output.getvalue()


def process_dataframe(df_raw):
    """Process input data and return the output DataFrames."""
    def _safe_default_col(df, idx, fallback=0):
        if df is None or len(df.columns) == 0:
            return None
        try:
            return df.columns[idx]
        except Exception:
            # fallback to first column or provided fallback index
            if 0 <= fallback < len(df.columns):
                return df.columns[fallback]
            return df.columns[0]

    col_folder = find_col(df_raw, ["folder"]) or _safe_default_col(df_raw, 0)
    col_country = find_col(df_raw, ["country"]) or _safe_default_col(df_raw, 4)
    # Look for either 'e2e' or 's2e' tokens (not both together)
    col_e2e = find_col(df_raw, ["e2e"]) or find_col(df_raw, ["s2e"]) or _safe_default_col(df_raw, 8)
    col_testset = find_col(df_raw, ["test", "set"]) or _safe_default_col(df_raw, 11)
    col_testcase = find_col(df_raw, ["test", "case"]) or find_col(df_raw, ["testcase"])
    col_config = find_col(df_raw, ["configuration", "config"]) or find_col(df_raw, ["configuration id", "config id"])
    # Prefer explicit TestID-style column names; avoid generic ['test','id'] which matches 'Test Set Folder ID'
    col_testid = find_col(df_raw, ["testid", "test_id"]) or find_col(df_raw, ["test", "identifier"]) 

    cols_to_take = [col_folder, col_country, col_e2e, col_testset]
    rename_map = {
        col_folder: "TestSetFolderID",
        col_country: "Country",
        col_e2e: "E2EFolder",
        col_testset: "TestSetName",
    }
    if col_testcase is not None:
        cols_to_take.append(col_testcase)
        rename_map[col_testcase] = "TestCaseName"
    if col_config is not None:
        cols_to_take.append(col_config)
        rename_map[col_config] = "ConfigurationID"
    if col_testid is not None:
        cols_to_take.append(col_testid)
        rename_map[col_testid] = "TestID"

    # deduplicate columns (preserve order) and remove any None values
    seen = set()
    dedup_cols = []
    for c in cols_to_take:
        if c is None:
            continue
        if c in seen:
            continue
        seen.add(c)
        dedup_cols.append(c)

    base = df_raw[dedup_cols].rename(columns=rename_map).copy()
    if "TestCaseName" not in base.columns:
        base["TestCaseName"] = None
    if "ConfigurationID" not in base.columns:
        base["ConfigurationID"] = None

    base["TestCaseName"] = base["TestCaseName"].astype(str).replace({"nan": ""}).str.strip()
    base["ConfigurationID"] = base["ConfigurationID"].astype(str).replace({"nan": ""}).str.strip()
    if "TestID" in base.columns:
        base["TestID"] = base["TestID"].astype(str).replace({"nan": ""}).str.strip()
    # Ensure E2EFolder exists or find best match
    def _pick_col(df, desired, keywords):
        if desired in df.columns:
            return desired
        for c in df.columns:
            cl = str(c).lower()
            for kw in keywords:
                if kw in cl:
                    return c
        return None

    e2e_col = _pick_col(base, "E2EFolder", ["e2e", "s2e"]) or None
    folder_col = _pick_col(base, "TestSetFolderID", ["folder", "testset", "folder id"]) or None
    country_col = _pick_col(base, "Country", ["country"]) or None

    if e2e_col is not None:
        base["Cleaned_Folder_Name"] = base[e2e_col].apply(clean_e2e_name)
        # ensure standardized column name exists
        if e2e_col != "E2EFolder":
            base = base.rename(columns={e2e_col: "E2EFolder"})
    else:
        base["Cleaned_Folder_Name"] = ""

    # ensure folder and country standardized names if possible
    if folder_col and folder_col != "TestSetFolderID":
        base = base.rename(columns={folder_col: "TestSetFolderID"})
    if country_col and country_col != "Country":
        base = base.rename(columns={country_col: "Country"})

    # Build unique_df from standardized columns if present
    cols_for_unique = [c for c in ["TestSetFolderID", "E2EFolder", "Cleaned_Folder_Name", "Country"] if c in base.columns]
    unique_df = (
        base[cols_for_unique]
        .dropna(subset=[c for c in ["TestSetFolderID"] if c in cols_for_unique])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    # Helper to check exact E2E equality
    def exact_e2e_match(a, b):
        return str(a).strip().lower() == str(b).strip().lower()

    # Build pairwise E2E comparisons (same-country and cross-country)
    cross_rows = []
    same_rows = []
    records = unique_df.to_dict("records")
    n = len(records)
    for i in range(n):
        for j in range(i + 1, n):
            a = records[i]
            b = records[j]
            sim = position_sensitive_similarity(a.get("Cleaned_Folder_Name", ""), b.get("Cleaned_Folder_Name", ""))
            exact = exact_e2e_match(a.get("E2EFolder", ""), b.get("E2EFolder", ""))
            if str(a.get("Country", "")).strip().lower() == str(b.get("Country", "")).strip().lower():
                same_rows.append({
                    "Country": a.get("Country", ""),
                    "Test set Folder ID A": a.get("TestSetFolderID", ""),
                    "E2E_A": a.get("E2EFolder", ""),
                    "Cleaned_A": a.get("Cleaned_Folder_Name", ""),
                    "Test set Folder ID B": b.get("TestSetFolderID", ""),
                    "E2E_B": b.get("E2EFolder", ""),
                    "Cleaned_B": b.get("Cleaned_Folder_Name", ""),
                    "Similarity": round(sim, 3),
                    "Exact_E2E": exact,
                })
            else:
                cross_rows.append({
                    "Country_A": a.get("Country", ""),
                    "Test set Folder ID A": a.get("TestSetFolderID", ""),
                    "E2E_A": a.get("E2EFolder", ""),
                    "Country_B": b.get("Country", ""),
                    "Test set Folder ID B": b.get("TestSetFolderID", ""),
                    "E2E_B": b.get("E2EFolder", ""),
                    "Similarity": round(sim, 3),
                })

    df_cross = pd.DataFrame(cross_rows).sort_values("Similarity", ascending=False).reset_index(drop=True) if cross_rows else pd.DataFrame(columns=[
        "Country_A", "Test set Folder ID A", "E2E_A", "Country_B", "Test set Folder ID B", "E2E_B", "Similarity"
    ])

    df_same = pd.DataFrame(same_rows).sort_values("Similarity", ascending=False).reset_index(drop=True) if same_rows else pd.DataFrame(columns=[
        "Country", "Test set Folder ID A", "E2E_A", "Cleaned_A", "Test set Folder ID B", "E2E_B", "Cleaned_B", "Similarity", "Exact_E2E"
    ])
    # Ensure TestSetName exists (fallback to folder ID or empty string)
    if "TestSetName" not in base.columns:
        if "TestSetFolderID" in base.columns:
            base["TestSetName"] = base["TestSetFolderID"].astype(str)
        else:
            base["TestSetName"] = ""

    e2e_to_testsets = (
        base[["E2EFolder", "TestSetName"]]
        .dropna(subset=["E2EFolder", "TestSetName"])
        .drop_duplicates()
        .groupby("E2EFolder")["TestSetName"]
        .apply(list)
        .to_dict()
    )

    testcases_by_set = {
        (e2e, ts): group
        for (e2e, ts), group in base.dropna(subset=["E2EFolder", "TestSetName"]).groupby(["E2EFolder", "TestSetName"])
    }

    def testset_similarity(a, b):
        return position_sensitive_similarity(str(a), str(b))

    testset_same_rows = []
    testset_cross_rows = []
    testcase_rows = []

    for _, row in df_same[df_same["Similarity"] >= HIGH_SIM_E2E_THRESHOLD].iterrows():
        e2eA = row["E2E_A"]
        e2eB = row["E2E_B"]
        testsetsA = e2e_to_testsets.get(e2eA, [])
        testsetsB = e2e_to_testsets.get(e2eB, [])
        for tsa in testsetsA:
            for tsb in testsetsB:
                score = testset_similarity(tsa, tsb)
                if score < MIN_SIMILARITY_TESTSET:
                    continue
                testset_same_rows.append({
                    "Country": row["Country"],
                    "Test set Folder ID A": row["Test set Folder ID A"],
                    "E2E_A": e2eA,
                    "TestSetName_A": tsa,
                    "Test set Folder ID B": row["Test set Folder ID B"],
                    "E2E_B": e2eB,
                    "TestSetName_B": tsb,
                    "TestSet_Similarity": round(score, 3),
                })

    for _, row in df_cross[df_cross["Similarity"] >= HIGH_SIM_E2E_THRESHOLD].iterrows():
        e2eA = row["E2E_A"]
        e2eB = row["E2E_B"]
        testsetsA = e2e_to_testsets.get(e2eA, [])
        testsetsB = e2e_to_testsets.get(e2eB, [])
        for tsa in testsetsA:
            for tsb in testsetsB:
                score = testset_similarity(tsa, tsb)
                if score < MIN_SIMILARITY_TESTSET:
                    continue
                testset_cross_rows.append({
                    "Country_A": row.get("Country_A", ""),
                    "Country_B": row.get("Country_B", ""),
                    "Test set Folder ID A": row.get("Test set Folder ID A", ""),
                    "E2E_A": e2eA,
                    "TestSetName_A": tsa,
                    "Test set Folder ID B": row.get("Test set Folder ID B", ""),
                    "E2E_B": e2eB,
                    "TestSetName_B": tsb,
                    "TestSet_Similarity": round(score, 3),
                    "Similarity_E2E": row.get("Similarity", ""),
                })

    def build_case_rows(groupA, groupB, country, e2eA, setA, e2eB, setB):
        if groupA is None:
            groupA = base.iloc[0:0]
        if groupB is None:
            groupB = base.iloc[0:0]

        # Prefer TestID grouping if available and populated
        use_testid = False
        a_by_testid = {}
        b_by_testid = {}
        if "TestID" in groupA.columns or "TestID" in groupB.columns:
            import pandas as _pd
            testid_colA = groupA["TestID"] if "TestID" in groupA.columns else _pd.Series([""] * len(groupA), index=groupA.index)
            testid_colB = groupB["TestID"] if "TestID" in groupB.columns else _pd.Series([""] * len(groupB), index=groupB.index)
            a_by_testid = (
                groupA[testid_colA != ""]
                .groupby("TestID")["TestCaseName"]
                .apply(lambda x: "; ".join(sorted(set(v for v in x if v))))
                .to_dict()
            )
            b_by_testid = (
                groupB[testid_colB != ""]
                .groupby("TestID")["TestCaseName"]
                .apply(lambda x: "; ".join(sorted(set(v for v in x if v))))
                .to_dict()
            )
            if a_by_testid or b_by_testid:
                use_testid = True

        if use_testid:
            testids = sorted(set(a_by_testid) | set(b_by_testid))
            for tid in testids:
                testcase_rows.append({
                    "Country": country,
                    "E2E_A": e2eA,
                    "TestSetName_A": setA,
                    "E2E_B": e2eB,
                    "TestSetName_B": setB,
                    "TestID": tid,
                    "TestCaseName_A": a_by_testid.get(tid, ""),
                    "TestCaseName_B": b_by_testid.get(tid, ""),
                    "MatchStatus": "Matched" if tid in a_by_testid and tid in b_by_testid else ("Only in A" if tid in a_by_testid else "Only in B"),
                })
            # done using TestID
        else:
            a_by_config = (
                groupA[groupA["ConfigurationID"] != ""]
                .groupby("ConfigurationID")["TestCaseName"]
                .apply(lambda x: "; ".join(sorted(set(v for v in x if v))))
                .to_dict()
            )
            b_by_config = (
                groupB[groupB["ConfigurationID"] != ""]
                .groupby("ConfigurationID")["TestCaseName"]
                .apply(lambda x: "; ".join(sorted(set(v for v in x if v))))
                .to_dict()
            )
            config_ids = sorted(set(a_by_config) | set(b_by_config))

            if config_ids:
                for config in config_ids:
                    testcase_rows.append({
                        "Country": country,
                        "E2E_A": e2eA,
                        "TestSetName_A": setA,
                        "E2E_B": e2eB,
                        "TestSetName_B": setB,
                        "ConfigurationID": config,
                        "TestCaseName_A": a_by_config.get(config, ""),
                        "TestCaseName_B": b_by_config.get(config, ""),
                        "MatchStatus": "Matched" if config in a_by_config and config in b_by_config else ("Only in A" if config in a_by_config else "Only in B"),
                    })
            else:
                a_names = sorted(set(groupA["TestCaseName"].dropna().astype(str).str.strip()))
                b_names = sorted(set(groupB["TestCaseName"].dropna().astype(str).str.strip()))
                all_names = sorted(set(a_names) | set(b_names))
                for case in all_names:
                    testcase_rows.append({
                        "Country": country,
                        "E2E_A": e2eA,
                        "TestSetName_A": setA,
                        "E2E_B": e2eB,
                        "TestSetName_B": setB,
                        "ConfigurationID": "",
                        "TestCaseName_A": case if case in a_names else "",
                        "TestCaseName_B": case if case in b_names else "",
                        "MatchStatus": "Matched" if case in a_names and case in b_names else ("Only in A" if case in a_names else "Only in B"),
                    })

    for row in testset_same_rows:
        groupA = testcases_by_set.get((row["E2E_A"], row["TestSetName_A"]))
        groupB = testcases_by_set.get((row["E2E_B"], row["TestSetName_B"]))
        build_case_rows(groupA, groupB, row["Country"], row["E2E_A"], row["TestSetName_A"], row["E2E_B"], row["TestSetName_B"])

    df_testset_same = pd.DataFrame(testset_same_rows).sort_values("TestSet_Similarity", ascending=False).reset_index(drop=True) if testset_same_rows else pd.DataFrame(columns=[
        "Country",
        "Test set Folder ID A",
        "E2E_A",
        "TestSetName_A",
        "Test set Folder ID B",
        "E2E_B",
        "TestSetName_B",
        "TestSet_Similarity",
    ])

    df_testset_cross = pd.DataFrame(testset_cross_rows).sort_values("TestSet_Similarity", ascending=False).reset_index(drop=True) if testset_cross_rows else pd.DataFrame(columns=[
        "Country_A",
        "Country_B",
        "Test set Folder ID A",
        "E2E_A",
        "TestSetName_A",
        "Test set Folder ID B",
        "E2E_B",
        "TestSetName_B",
        "TestSet_Similarity",
        "Similarity_E2E",
    ])

    # create DataFrame dynamically (may include TestID or ConfigurationID)
    df_testcases = pd.DataFrame(testcase_rows)

    return unique_df, df_cross, df_same, df_testset_same, df_testset_cross, df_testcases


def main():
    st.set_page_config(
        page_title="SMART REGRESSION",
        page_icon="📊",
        layout="wide",
    )

    st.markdown("""
    <div style='background: linear-gradient(135deg, #2E6BF8 0%, #00C1F7 100%); padding: 24px; border-radius: 16px; color: white;'>
        <h1 style='margin: 0; font-size: 2.3rem;'>📊 SMART REGRESSION</h1>
        <p style='margin: 8px 0 0; font-size: 1rem; color: #F0F8FF;'>Upload an Excel file, create a polished similarity report, and download your result instantly.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    left_col, right_col = st.columns([2, 1])
    with left_col:
        uploaded_file = st.file_uploader("Upload your input Excel file", type=["xlsx", "xls"], help="Select the workbook to analyze.")
        output_name = st.text_input("Output file name", "core_regression_output.xlsx")
        generate_button = st.button("Generate output file", key="generate_button")

    with right_col:
        st.markdown("### Quick tips")
        st.write("- Use .xlsx or .xls files only")
        st.write("- Upload the source workbook first")
        st.write("- Then click **Generate output file**")
        st.info("The generated Excel will be available as a download button once processing is complete.")

    if uploaded_file is None:
        st.warning("Please upload an Excel file to enable output generation.")
        return

    if not generate_button:
        st.info("Click the **Generate output file** button to process the uploaded workbook.")
        return

    try:
        df_raw, sheet_name = load_first_non_empty_sheet(uploaded_file)
        st.success(f"Loaded sheet: {sheet_name} (shape={df_raw.shape})")
        st.write("### Detected input data preview")
        st.dataframe(df_raw.head().astype(str))

        unique_df, df_cross, df_same, df_testset_same, df_testset_cross, df_testcases = process_dataframe(df_raw)
        total_cross = len(df_cross)
        total_same = len(df_same)
        total_testset = len(df_testset_same)
        total_testset_cross = len(df_testset_cross)
        total_testcases = len(df_testcases)

        st.markdown("### Key analytics")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Unique E2E folders", len(unique_df), delta=None)
        m2.metric("Similar same-country E2E pairs", total_same, delta=None)
        m3.metric("Cross-country E2E pairs", total_cross, delta=None)
        m4.metric("Same-country test-set matches", total_testset, delta=None)
        m5.metric("Cross-country test-set matches", total_testset_cross, delta=None)
        m6.metric("Test-case rows", total_testcases, delta=None)

        with st.container():
            analytics_option = st.selectbox("Analytics view", ["E2E Similarity", "Same-Country Test Cases"], index=0)
            chart_col1, chart_col2 = st.columns(2)
            if analytics_option == "E2E Similarity":
                with chart_col1:
                    st.markdown("#### Pair type distribution")
                    pie_data = pd.DataFrame({
                        "Pair type": ["Cross-country", "Same-country"],
                        "Count": [total_cross, total_same],
                    })
                    pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta(field="Count", type="quantitative"),
                        color=alt.Color(field="Pair type", type="nominal", legend=alt.Legend(title="Pair type")),
                        tooltip=[alt.Tooltip(field="Pair type", type="nominal"), alt.Tooltip(field="Count", type="quantitative")],
                    )
                    st.altair_chart(pie_chart, use_container_width=True)

                with chart_col2:
                    st.markdown("#### Same-country pairs by country")
                    if total_same > 0:
                        country_counts = (
                            df_same.groupby("Country").size().reset_index(name="Count").sort_values("Count", ascending=False)
                        )
                        bar_chart = alt.Chart(country_counts.head(8)).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                            x=alt.X("Count:Q", title="Pair count"),
                            y=alt.Y("Country:N", sort="-x", title="Country"),
                            tooltip=[alt.Tooltip("Country:N"), alt.Tooltip("Count:Q")],
                            color=alt.Color("Country:N", legend=None)
                        )
                        st.altair_chart(bar_chart, use_container_width=True)
                    else:
                        st.info("No same-country pairs available to chart.")
            else:
                with chart_col1:
                    st.markdown("#### Same-country matched test-sets by country")
                    if len(df_testset_same) > 0:
                        ts_counts = df_testset_same.groupby("Country").size().reset_index(name="Count").sort_values("Count", ascending=False)
                        bar_chart = alt.Chart(ts_counts.head(8)).mark_bar().encode(
                            x=alt.X("Count:Q", title="Matched test-sets"),
                            y=alt.Y("Country:N", sort="-x", title="Country"),
                            tooltip=[alt.Tooltip("Country:N"), alt.Tooltip("Count:Q")],
                            color=alt.Color("Country:N", legend=None)
                        )
                        st.altair_chart(bar_chart, use_container_width=True)
                    else:
                        st.info("No same-country matched test-sets to display.")

                with chart_col2:
                    st.markdown("#### Same-country matched test-cases by country")
                    if (df_testcases is not None) and ("MatchStatus" in df_testcases.columns):
                        matched = df_testcases[df_testcases["MatchStatus"].astype(str).str.lower() == "matched"]
                        if len(matched) > 0 and "Country" in matched.columns:
                            m_counts = matched.groupby("Country").size().reset_index(name="Count").sort_values("Count", ascending=False)
                            bar_chart2 = alt.Chart(m_counts.head(8)).mark_bar().encode(
                                x=alt.X("Count:Q", title="Matched test-cases"),
                                y=alt.Y("Country:N", sort="-x", title="Country"),
                                tooltip=[alt.Tooltip("Country:N"), alt.Tooltip("Count:Q")],
                                color=alt.Color("Country:N", legend=None)
                            )
                            st.altair_chart(bar_chart2, use_container_width=True)
                        else:
                            st.info("No matched test-case rows to display.")
                    else:
                        st.info("Test-case comparison data not available.")

        st.markdown("### Output preview")
        st.write("Unique cleaned E2E rows")
        st.dataframe(unique_df.head().astype(str))
        st.write("Cross-country E2E pairs")
        st.dataframe(df_cross.head().astype(str))
        st.write("Same-country E2E pairs")
        st.dataframe(df_same.head().astype(str))
        st.write("Same-country matched test-set pairs")
        st.dataframe(df_testset_same.head().astype(str))
        st.write("Test-case comparison")
        st.dataframe(df_testcases.head().astype(str))

        output_bytes = create_output_workbook(
            unique_df, df_cross, df_same, df_testset_same, df_testset_cross, df_testcases
        )

        st.download_button(
            label="Output file",
            data=output_bytes,
            file_name=output_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download the generated Excel workbook.",
        )
    except Exception as exc:
        st.error(f"Error: {exc}")


if __name__ == "__main__":
    main()
    