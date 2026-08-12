import re
import time
from io import BytesIO
from itertools import combinations

import pandas as pd
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================
E2E_THRESHOLD = 0.80  # Test-set analysis is only done for E2E pairs >= 80%

REQUIRED_COLUMNS = {
    "Country": ["country"],
    "E2EName": ["s2e_e2e_name", "e2e_name", "e2e"],
    "TestSetID": ["test set id", "test_set_id"],
    "TestSetName": ["test_set_name", "test set name"],
    "TestID": ["test id", "test_id", "testid"],
    "TestName": ["test name", "test_name"],
}


# ============================================================
# EXCEL / COLUMN HELPERS
# ============================================================
def _norm(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def detect_header_row(raw_preview: pd.DataFrame):
    """Find the row containing the real Excel headers."""
    for idx, row in raw_preview.iterrows():
        vals = {_norm(v) for v in row.tolist() if pd.notna(v)}
        has_country = "country" in vals
        has_e2e = any(v in vals for v in ["s2e e2e name", "e2e name", "e2e"])
        has_testset = any(v in vals for v in ["test set name", "test set id"])
        has_testid = "test id" in vals
        if has_country and has_e2e and has_testset and has_testid:
            return int(idx)
    return None


def load_source_workbook(file_like):
    """Load the first worksheet that contains the expected header row."""
    file_like.seek(0)
    xls = pd.ExcelFile(file_like)

    for sheet in xls.sheet_names:
        preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=20)
        if preview.empty:
            continue
        header_row = detect_header_row(preview)
        if header_row is not None:
            df = pd.read_excel(xls, sheet_name=sheet, header=header_row)
            return df, sheet, header_row

    raise ValueError(
        "Could not find a sheet containing Country, E2E, Test Set and Test ID headers."
    )


def find_column(df, candidates):
    normalized = {_norm(c): c for c in df.columns}
    for candidate in candidates:
        key = _norm(candidate)
        if key in normalized:
            return normalized[key]

    # contains fallback
    for c in df.columns:
        c_norm = _norm(c)
        for candidate in candidates:
            cand_norm = _norm(candidate)
            if cand_norm and cand_norm in c_norm:
                return c
    return None


def standardize_input(df_raw):
    mapping = {}
    missing = []
    for standard_name, candidates in REQUIRED_COLUMNS.items():
        found = find_column(df_raw, candidates)
        if found is None:
            missing.append(standard_name)
        else:
            mapping[found] = standard_name

    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    optional_candidates = {
        "E2EID": ["e2e_id", "e2e id"],
        "TestSetFolderID": ["test set folder id", "test_set_folder_id"],
    }
    for standard_name, candidates in optional_candidates.items():
        found = find_column(df_raw, candidates)
        if found is not None:
            mapping[found] = standard_name

    base = df_raw[list(mapping.keys())].rename(columns=mapping).copy()

    for c in base.columns:
        base[c] = base[c].where(base[c].notna(), "")
        base[c] = base[c].astype(str).str.strip()
        base[c] = base[c].replace({"nan": "", "None": ""})

    # Remove rows with no useful hierarchy information
    base = base[(base["Country"] != "") & (base["E2EName"] != "")].copy()
    return base


# ============================================================
# NAME SIMILARITY
# ============================================================
def clean_e2e_name(name):
    """
    Keep the existing project logic: remove technical E2E prefix and compare
    the meaningful part of the E2E name position-by-position.
    """
    if name is None:
        return ""
    s = str(name).strip()
    if not s:
        return ""

    # Existing naming convention: technical prefix occupies first 4 '_' sections.
    if s.count("_") >= 4:
        return s.split("_", 4)[4].strip()

    prefix_regex = re.compile(
        r"^(?:\d{1,3}_)?(?:e2e(?:_[A-Za-z0-9]+)*)_",
        flags=re.IGNORECASE,
    )
    m = prefix_regex.match(s)
    if m:
        s = s[m.end():]
    return re.sub(r"^[\s_\-:]+", " ", s).strip()


def tokenize_name(value):
    if value is None:
        return []
    s = str(value).lower()
    s = re.sub(r"[^a-z0-9<>]+", " ", s)
    return [t for t in s.split() if t]


def name_similarity(a, b):
    """Position-sensitive token similarity used for E2E and Test Set names."""
    ta = tokenize_name(a)
    tb = tokenize_name(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    matches = sum(1 for x, y in zip(ta, tb) if x == y)
    return matches / max(len(ta), len(tb))


# ============================================================
# 1) E2E COMPARISON
# ============================================================
def build_e2e_comparisons(base):
    e2es = (
        base[["Country", "E2EName"] + (["E2EID"] if "E2EID" in base.columns else [])]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    e2es["CleanedE2EName"] = e2es["E2EName"].apply(clean_e2e_name)

    same_rows = []
    cross_rows = []
    records = e2es.to_dict("records")

    for a, b in combinations(records, 2):
        score = name_similarity(a["CleanedE2EName"], b["CleanedE2EName"])
        row = {
            "Country_A": a["Country"],
            "Country_B": b["Country"],
            "E2E_A": a["E2EName"],
            "E2E_B": b["E2EName"],
            "Cleaned_E2E_A": a["CleanedE2EName"],
            "Cleaned_E2E_B": b["CleanedE2EName"],
            "E2E_Similarity": round(score, 4),
            "E2E_Similarity_%": round(score * 100, 2),
            "Eligible_For_TestSet_Comparison": score >= E2E_THRESHOLD,
        }
        if "E2EID" in e2es.columns:
            row["E2E_ID_A"] = a.get("E2EID", "")
            row["E2E_ID_B"] = b.get("E2EID", "")

        if a["Country"].lower() == b["Country"].lower():
            row["ComparisonType"] = "Same country"
            same_rows.append(row)
        else:
            row["ComparisonType"] = "Cross country"
            cross_rows.append(row)

    same_df = pd.DataFrame(same_rows)
    cross_df = pd.DataFrame(cross_rows)

    if not same_df.empty:
        same_df = same_df.sort_values("E2E_Similarity", ascending=False).reset_index(drop=True)
    if not cross_df.empty:
        cross_df = cross_df.sort_values("E2E_Similarity", ascending=False).reset_index(drop=True)

    return e2es, same_df, cross_df


# ============================================================
# 2) TEST SET NAME COMPARISON
#    Only for E2E pairs with E2E similarity >= 80%
# ============================================================
def build_testset_comparison(base, same_e2e, cross_e2e):
    testset_cols = ["Country", "E2EName", "TestSetID", "TestSetName"]
    testsets = base[testset_cols].drop_duplicates()

    lookup = {
        (country, e2e): group[["TestSetID", "TestSetName"]].drop_duplicates().to_dict("records")
        for (country, e2e), group in testsets.groupby(["Country", "E2EName"])
    }

    eligible_frames = []
    for df in [same_e2e, cross_e2e]:
        if df is not None and not df.empty:
            eligible_frames.append(df[df["E2E_Similarity"] >= E2E_THRESHOLD])

    if not eligible_frames:
        return pd.DataFrame()

    eligible = pd.concat(eligible_frames, ignore_index=True)
    rows = []

    for _, pair in eligible.iterrows():
        key_a = (pair["Country_A"], pair["E2E_A"])
        key_b = (pair["Country_B"], pair["E2E_B"])
        sets_a = lookup.get(key_a, [])
        sets_b = lookup.get(key_b, [])

        for tsa in sets_a:
            for tsb in sets_b:
                score = name_similarity(tsa["TestSetName"], tsb["TestSetName"])
                rows.append({
                    "ComparisonType": pair["ComparisonType"],
                    "Country_A": pair["Country_A"],
                    "Country_B": pair["Country_B"],
                    "E2E_A": pair["E2E_A"],
                    "E2E_B": pair["E2E_B"],
                    "E2E_Similarity": pair["E2E_Similarity"],
                    "E2E_Similarity_%": pair["E2E_Similarity_%"],
                    "TestSetID_A": tsa["TestSetID"],
                    "TestSetName_A": tsa["TestSetName"],
                    "TestSetID_B": tsb["TestSetID"],
                    "TestSetName_B": tsb["TestSetName"],
                    "TestSet_Name_Similarity": round(score, 4),
                    "TestSet_Name_Similarity_%": round(score * 100, 2),
                })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["E2E_Similarity", "TestSet_Name_Similarity"],
            ascending=[False, False],
        ).reset_index(drop=True)
    return result


# ============================================================
# 3) TEST CASE COMPARISON BY TEST ID
# ============================================================
def test_id_similarity(ids_a, ids_b):
    """
    Similarity = common Test IDs / larger Test-ID set.
    This gives 100% only when both test sets contain the same IDs.
    """
    a = {str(x).strip() for x in ids_a if str(x).strip()}
    b = {str(x).strip() for x in ids_b if str(x).strip()}
    if not a and not b:
        return 1.0, set(), a, b
    if not a or not b:
        return 0.0, set(), a, b
    common = a & b
    return len(common) / max(len(a), len(b)), common, a, b


def build_testcase_comparison(base, testset_comparison):
    case_lookup = {
        (country, e2e, tsid, tsname): group[["TestID", "TestName"]].drop_duplicates()
        for (country, e2e, tsid, tsname), group in base.groupby(
            ["Country", "E2EName", "TestSetID", "TestSetName"], dropna=False
        )
    }

    summary_rows = []
    detail_rows = []

    if testset_comparison is None or testset_comparison.empty:
        return pd.DataFrame(), pd.DataFrame()

    for _, row in testset_comparison.iterrows():
        key_a = (row["Country_A"], row["E2E_A"], row["TestSetID_A"], row["TestSetName_A"])
        key_b = (row["Country_B"], row["E2E_B"], row["TestSetID_B"], row["TestSetName_B"])
        ga = case_lookup.get(key_a, pd.DataFrame(columns=["TestID", "TestName"]))
        gb = case_lookup.get(key_b, pd.DataFrame(columns=["TestID", "TestName"]))

        ids_a = ga["TestID"].tolist()
        ids_b = gb["TestID"].tolist()
        score, common, set_a, set_b = test_id_similarity(ids_a, ids_b)

        summary_rows.append({
            "ComparisonType": row["ComparisonType"],
            "Country_A": row["Country_A"],
            "Country_B": row["Country_B"],
            "E2E_A": row["E2E_A"],
            "E2E_B": row["E2E_B"],
            "E2E_Similarity_%": row["E2E_Similarity_%"],
            "TestSetID_A": row["TestSetID_A"],
            "TestSetName_A": row["TestSetName_A"],
            "TestSetID_B": row["TestSetID_B"],
            "TestSetName_B": row["TestSetName_B"],
            "TestSet_Name_Similarity_%": row["TestSet_Name_Similarity_%"],
            "Total_TestIDs_A": len(set_a),
            "Total_TestIDs_B": len(set_b),
            "Matching_TestIDs": len(common),
            "TestID_Similarity": round(score, 4),
            "TestID_Similarity_%": round(score * 100, 2),
        })

        names_a = (
            ga.groupby("TestID")["TestName"]
            .apply(lambda s: "; ".join(sorted({x for x in s if x})))
            .to_dict()
            if not ga.empty else {}
        )
        names_b = (
            gb.groupby("TestID")["TestName"]
            .apply(lambda s: "; ".join(sorted({x for x in s if x})))
            .to_dict()
            if not gb.empty else {}
        )

        for tid in sorted(set_a | set_b):
            if tid in set_a and tid in set_b:
                status = "Matched"
            elif tid in set_a:
                status = "Only in A"
            else:
                status = "Only in B"

            detail_rows.append({
                "ComparisonType": row["ComparisonType"],
                "Country_A": row["Country_A"],
                "Country_B": row["Country_B"],
                "E2E_A": row["E2E_A"],
                "E2E_B": row["E2E_B"],
                "TestSetID_A": row["TestSetID_A"],
                "TestSetName_A": row["TestSetName_A"],
                "TestSetID_B": row["TestSetID_B"],
                "TestSetName_B": row["TestSetName_B"],
                "TestID": tid,
                "TestName_A": names_a.get(tid, ""),
                "TestName_B": names_b.get(tid, ""),
                "MatchStatus": status,
                "TestID_Similarity_%": round(score * 100, 2),
            })

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["E2E_Similarity_%", "TestSet_Name_Similarity_%", "TestID_Similarity_%"],
            ascending=False,
        ).reset_index(drop=True)
    return summary_df, detail_df


# ============================================================
# 4) SUMMARY
# ============================================================
def build_summary(base, same_e2e, cross_e2e, testsets, testcase_summary):
    metrics = [
        ("Input rows", len(base)),
        ("Countries", base["Country"].nunique()),
        ("Unique E2Es", base[["Country", "E2EName"]].drop_duplicates().shape[0]),
        ("Same-country E2E pairs", len(same_e2e)),
        ("Cross-country E2E pairs", len(cross_e2e)),
        ("Same-country E2E pairs >= 80%", int((same_e2e["E2E_Similarity"] >= E2E_THRESHOLD).sum()) if not same_e2e.empty else 0),
        ("Cross-country E2E pairs >= 80%", int((cross_e2e["E2E_Similarity"] >= E2E_THRESHOLD).sum()) if not cross_e2e.empty else 0),
        ("Test-set comparisons for eligible E2Es", len(testsets)),
        ("Test-set name matches >= 80%", int((testsets["TestSet_Name_Similarity"] >= 0.80).sum()) if not testsets.empty else 0),
        ("Test-case pair summaries", len(testcase_summary)),
        ("Test-case pairs with 100% Test-ID overlap", int((testcase_summary["TestID_Similarity"] == 1.0).sum()) if not testcase_summary.empty else 0),
    ]
    overall = pd.DataFrame(metrics, columns=["Metric", "Value"])

    by_country = (
        base.groupby("Country")
        .agg(
            Input_Rows=("E2EName", "size"),
            Unique_E2Es=("E2EName", "nunique"),
            Unique_TestSets=("TestSetID", "nunique"),
            Unique_TestIDs=("TestID", "nunique"),
        )
        .reset_index()
        .sort_values("Country")
    )
    return overall, by_country


# ============================================================
# EXCEL OUTPUT
# ============================================================
def make_excel_output(base, e2e_unique, same_e2e, cross_e2e, testsets, testcase_summary, testcase_detail, summary_overall, summary_country, time_saved_df=None):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        summary_overall.to_excel(writer, sheet_name="SUMMARY", index=False, startrow=0)
        summary_country.to_excel(writer, sheet_name="SUMMARY", index=False, startrow=len(summary_overall) + 3)
        e2e_unique.to_excel(writer, sheet_name="UNIQUE_E2E", index=False)
        same_e2e.to_excel(writer, sheet_name="E2E_SAME_COUNTRY", index=False)
        cross_e2e.to_excel(writer, sheet_name="E2E_CROSS_COUNTRY", index=False)
        testsets.to_excel(writer, sheet_name="TEST_SET_COMPARISON", index=False)
        testcase_summary.to_excel(writer, sheet_name="TEST_CASE_SIMILARITY", index=False)
        testcase_detail.to_excel(writer, sheet_name="TEST_CASE_DETAIL", index=False)
        if time_saved_df is not None and not time_saved_df.empty:
            time_saved_df.to_excel(writer, sheet_name="TIME_SAVED", index=False)

        # Basic readability formatting
        for sheet_name, ws in writer.sheets.items():
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, 0, max(0, ws.dim_colmax))
            ws.set_column(0, max(0, ws.dim_colmax), 18)

    out.seek(0)
    return out.getvalue()


# ============================================================
# FULL PROCESS
# ============================================================
def process_dataframe(df_raw):
    base = standardize_input(df_raw)
    e2e_unique, same_e2e, cross_e2e = build_e2e_comparisons(base)
    testsets = build_testset_comparison(base, same_e2e, cross_e2e)
    testcase_summary, testcase_detail = build_testcase_comparison(base, testsets)
    summary_overall, summary_country = build_summary(
        base, same_e2e, cross_e2e, testsets, testcase_summary
    )
    return {
        "base": base,
        "e2e_unique": e2e_unique,
        "same_e2e": same_e2e,
        "cross_e2e": cross_e2e,
        "testsets": testsets,
        "testcase_summary": testcase_summary,
        "testcase_detail": testcase_detail,
        "summary_overall": summary_overall,
        "summary_country": summary_country,
    }


# ============================================================
# STREAMLIT EXPLORER
# ============================================================

def hierarchy_explorer(base):
    st.markdown("---")
    st.header("E2E → Test Set → Test Case Explorer")

    all_countries = sorted(
        [c for c in base["Country"].dropna().astype(str).unique().tolist() if c]
    )
    country_options = ["All"] + all_countries

    if base.empty:
        st.info("No hierarchy data available.")
        return

    col_a, col_b = st.columns(2)

    def side(label, container, default_index):
        with container:
            st.subheader(label)

            # ------------------------------------------------------------
            # 1. COUNTRY FIRST
            # ------------------------------------------------------------
            country = st.selectbox(
                f"Country for {label}",
                country_options,
                index=0,
                key=f"{label}_country",
            )

            if country == "All":
                country_rows = base.copy()
            else:
                country_rows = base[base["Country"] == country].copy()

            # ------------------------------------------------------------
            # 2. E2E FILTERED BY COUNTRY
            # ------------------------------------------------------------
            e2e_options = sorted(
                [e for e in country_rows["E2EName"].dropna().astype(str).unique().tolist() if e]
            )

            if not e2e_options:
                st.info(f"No E2Es available for {country}.")
                return {
                    "country_filter": country,
                    "e2e": None,
                    "e2e_rows": country_rows.iloc[0:0],
                    "testsets": pd.DataFrame(columns=["Country", "TestSetID", "TestSetName"]),
                    "selected_country": None,
                    "selected_testset_id": None,
                    "selected_testset_name": None,
                }

            e2e = st.selectbox(
                f"Select {label}",
                e2e_options,
                index=min(default_index, len(e2e_options) - 1),
                key=f"{label}_e2e",
            )

            e2e_rows = country_rows[country_rows["E2EName"] == e2e].copy()

            # ------------------------------------------------------------
            # 3. LIST ALL TEST SETS INSIDE THIS E2E
            # ------------------------------------------------------------
            ts_table = (
                e2e_rows[["Country", "TestSetID", "TestSetName"]]
                .drop_duplicates()
                .sort_values(["Country", "TestSetName", "TestSetID"])
                .reset_index(drop=True)
            )

            st.markdown("#### Test Sets")
            st.caption(
                f"{len(ts_table):,} test sets in this E2E"
                + (" across all countries" if country == "All" else f" for {country}")
            )
            st.dataframe(
                ts_table,
                use_container_width=True,
                hide_index=True,
            )

            if ts_table.empty:
                return {
                    "country_filter": country,
                    "e2e": e2e,
                    "e2e_rows": e2e_rows,
                    "testsets": ts_table,
                    "selected_country": None,
                    "selected_testset_id": None,
                    "selected_testset_name": None,
                }

            # Country is included in the option so that "All" remains unambiguous.
            ts_options = [
                f"{r.Country} | {r.TestSetID} | {r.TestSetName}"
                for r in ts_table.itertuples(index=False)
            ]

            selected = st.selectbox(
                f"Select Test Set in {label}",
                ts_options,
                key=f"{label}_testset",
            )

            parts = selected.split(" | ", 2)
            selected_country = parts[0]
            selected_id = parts[1] if len(parts) > 1 else ""
            selected_name = parts[2] if len(parts) > 2 else ""

            # ------------------------------------------------------------
            # 4. TEST CASES FOR THE INDIVIDUALLY SELECTED TEST SET
            # ------------------------------------------------------------
            cases = e2e_rows[
                (e2e_rows["Country"] == selected_country)
                & (e2e_rows["TestSetID"] == selected_id)
                & (e2e_rows["TestSetName"] == selected_name)
            ][["TestID", "TestName"]].drop_duplicates().sort_values("TestID")

            st.markdown("#### Test Cases")
            st.caption(f"{len(cases):,} test cases in the selected Test Set")
            st.dataframe(
                cases,
                use_container_width=True,
                hide_index=True,
            )

            return {
                "country_filter": country,
                "e2e": e2e,
                "e2e_rows": e2e_rows,
                "testsets": ts_table,
                "selected_country": selected_country,
                "selected_testset_id": selected_id,
                "selected_testset_name": selected_name,
            }

    a = side("E2E_A", col_a, 0)
    b = side("E2E_B", col_b, 1)

    # ============================================================
    # ALL TEST-SET SIMILARITIES FOR THE TWO SELECTED E2Es
    # This does NOT depend on the individually selected Test Sets.
    # ============================================================
    if a["e2e"] and b["e2e"]:
        st.markdown("---")
        st.subheader("All Test Set Similarities for the selected E2Es")
        st.caption(
            "This compares every Test Set in E2E_A with every Test Set in E2E_B. "
            "It is independent of the Test Sets selected above for Test Case drill-down."
        )

        all_ts_rows = []

        for tsa in a["testsets"].itertuples(index=False):
            for tsb in b["testsets"].itertuples(index=False):
                score = name_similarity(tsa.TestSetName, tsb.TestSetName)
                all_ts_rows.append({
                    "Country_A": tsa.Country,
                    "E2E_A": a["e2e"],
                    "TestSetID_A": tsa.TestSetID,
                    "TestSetName_A": tsa.TestSetName,
                    "Country_B": tsb.Country,
                    "E2E_B": b["e2e"],
                    "TestSetID_B": tsb.TestSetID,
                    "TestSetName_B": tsb.TestSetName,
                    "TestSet_Similarity": round(score, 4),
                    "TestSet_Similarity_%": round(score * 100, 2),
                })

        all_ts_similarity = pd.DataFrame(all_ts_rows)

        if all_ts_similarity.empty:
            st.info("There are no Test Sets available to compare for these E2Es.")
        else:
            all_ts_similarity = all_ts_similarity.sort_values(
                ["TestSet_Similarity", "TestSetName_A", "TestSetName_B"],
                ascending=[False, True, True],
            ).reset_index(drop=True)

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Test Set pairs compared",
                f"{len(all_ts_similarity):,}",
            )
            m2.metric(
                "Best Test Set similarity",
                f"{all_ts_similarity['TestSet_Similarity'].max():.1%}",
            )
            m3.metric(
                "Pairs ≥ 80%",
                f"{(all_ts_similarity['TestSet_Similarity'] >= 0.80).sum():,}",
            )

            st.dataframe(
                all_ts_similarity[
                    [
                        "Country_A",
                        "TestSetID_A",
                        "TestSetName_A",
                        "Country_B",
                        "TestSetID_B",
                        "TestSetName_B",
                        "TestSet_Similarity_%",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    # ============================================================
    # SELECTED TEST-SET / TEST-CASE DRILL-DOWN COMPARISON
    # ============================================================
    selected_fields_exist = all([
        a.get("e2e"),
        b.get("e2e"),
        a.get("selected_country"),
        b.get("selected_country"),
        a.get("selected_testset_id"),
        b.get("selected_testset_id"),
    ])

    if selected_fields_exist:
        ca = a["selected_country"]
        cb = b["selected_country"]
        ea = a["e2e"]
        eb = b["e2e"]
        tsa_id = a["selected_testset_id"]
        tsb_id = b["selected_testset_id"]
        tsa_name = a["selected_testset_name"]
        tsb_name = b["selected_testset_name"]

        e2e_score = name_similarity(clean_e2e_name(ea), clean_e2e_name(eb))
        ts_score = name_similarity(tsa_name, tsb_name)

        ga = base[
            (base["Country"] == ca)
            & (base["E2EName"] == ea)
            & (base["TestSetID"] == tsa_id)
            & (base["TestSetName"] == tsa_name)
        ]
        gb = base[
            (base["Country"] == cb)
            & (base["E2EName"] == eb)
            & (base["TestSetID"] == tsb_id)
            & (base["TestSetName"] == tsb_name)
        ]

        id_score, common, ids_a, ids_b = test_id_similarity(
            ga["TestID"],
            gb["TestID"],
        )

        st.markdown("---")
        st.subheader("Selected Test Set / Test Case Comparison")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("E2E name similarity", f"{e2e_score:.1%}")
        m2.metric("Selected Test Set similarity", f"{ts_score:.1%}")
        m3.metric("Test ID similarity", f"{id_score:.1%}")
        m4.metric("Matching Test IDs", f"{len(common):,}")

        ids = sorted(ids_a | ids_b)

        na = (
            ga.groupby("TestID")["TestName"]
            .apply(lambda s: "; ".join(sorted(set(s))))
            .to_dict()
        )
        nb = (
            gb.groupby("TestID")["TestName"]
            .apply(lambda s: "; ".join(sorted(set(s))))
            .to_dict()
        )

        detail = pd.DataFrame([
            {
                "TestID": tid,
                "TestName_A": na.get(tid, ""),
                "TestName_B": nb.get(tid, ""),
                "MatchStatus": (
                    "Matched"
                    if tid in ids_a and tid in ids_b
                    else ("Only in A" if tid in ids_a else "Only in B")
                ),
            }
            for tid in ids
        ])

        st.dataframe(
            detail,
            use_container_width=True,
            hide_index=True,
        )



# ============================================================
# VISUAL ANALYTICS + TIME SAVED
# ============================================================
def render_visual_analytics_and_time_saved(result, processing_seconds):
    same = result["same_e2e"].copy()
    cross = result["cross_e2e"].copy()
    testsets = result["testsets"].copy()
    testcase_summary = result["testcase_summary"].copy()
    testcase_detail = result["testcase_detail"].copy()

    st.markdown("---")
    st.header("📊 Visual Analytics")

    # ------------------------------------------------------------
    # Chart 1: E2E comparison volume
    # ------------------------------------------------------------
    pair_counts = pd.DataFrame({
        "Comparison type": ["Same country", "Cross country"],
        "E2E pairs": [len(same), len(cross)],
    }).set_index("Comparison type")
    st.markdown("#### E2E comparison volume")
    st.bar_chart(pair_counts)

    # ------------------------------------------------------------
    # Chart 2: E2E similarity distribution
    # ------------------------------------------------------------
    similarity_frames = []
    for label, df in [("Same country", same), ("Cross country", cross)]:
        if not df.empty and "E2E_Similarity" in df.columns:
            temp = df[["E2E_Similarity"]].copy()
            temp["Comparison type"] = label
            temp["Similarity band"] = pd.cut(
                temp["E2E_Similarity"],
                bins=[-0.001, 0.25, 0.50, 0.80, 1.001],
                labels=["0–25%", "25–50%", "50–80%", "80–100%"],
                right=False,
            )
            similarity_frames.append(temp)

    if similarity_frames:
        sim_df = pd.concat(similarity_frames, ignore_index=True)
        sim_chart = (
            sim_df.groupby(["Similarity band", "Comparison type"], observed=False)
            .size()
            .unstack(fill_value=0)
        )
        st.markdown("#### E2E similarity distribution")
        st.bar_chart(sim_chart)

    # ------------------------------------------------------------
    # Chart 3: Test-set similarity distribution
    # ------------------------------------------------------------
    if not testsets.empty and "TestSet_Similarity" in testsets.columns:
        ts = testsets.copy()
        ts["Similarity band"] = pd.cut(
            ts["TestSet_Similarity"],
            bins=[-0.001, 0.25, 0.50, 0.80, 1.001],
            labels=["0–25%", "25–50%", "50–80%", "80–100%"],
            right=False,
        )
        ts_chart = (
            ts.groupby("Similarity band", observed=False)
            .size()
            .rename("Test Set pairs")
            .to_frame()
        )
        st.markdown("#### Test Set similarity distribution")
        st.bar_chart(ts_chart)

    # ------------------------------------------------------------
    # Chart 4: Test-ID match status
    # ------------------------------------------------------------
    if not testcase_detail.empty and "MatchStatus" in testcase_detail.columns:
        status = (
            testcase_detail["MatchStatus"]
            .value_counts()
            .rename_axis("Status")
            .rename("Test cases")
            .to_frame()
        )
        st.markdown("#### Test Case match status")
        st.bar_chart(status)

    # ------------------------------------------------------------
    # TIME SAVED
    # Baseline supplied by user:
    # 5 test cases / 8-hour working day
    # ------------------------------------------------------------
    st.markdown("---")
    st.header("⏱️ Time Saved")

    c1, c2 = st.columns(2)
    with c1:
        cases_per_day = st.number_input(
            "Manual capacity: test cases per working day",
            min_value=1,
            value=5,
            step=1,
            help="Your baseline is 5 test cases reviewed manually per day.",
        )
    with c2:
        hours_per_day = st.number_input(
            "Working hours per day",
            min_value=1.0,
            value=8.0,
            step=0.5,
        )

    # Count unique comparison workload rows represented by test-case details.
    # Each detail row is one Test ID assessment in a Test Set pair.
    total_test_case_assessments = len(testcase_detail)

    manual_days = (
        total_test_case_assessments / cases_per_day
        if cases_per_day > 0 else 0.0
    )
    manual_hours = manual_days * hours_per_day
    automation_hours = processing_seconds / 3600.0
    hours_saved = max(manual_hours - automation_hours, 0.0)
    days_saved = hours_saved / hours_per_day if hours_per_day > 0 else 0.0
    saving_pct = (
        (hours_saved / manual_hours) * 100
        if manual_hours > 0 else 0.0
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test case assessments", f"{total_test_case_assessments:,}")
    m2.metric("Manual effort", f"{manual_hours:,.1f} h")
    m3.metric("Working days", f"{manual_days:,.1f}")
    m4.metric("Estimated hours saved", f"{hours_saved:,.1f} h")

    st.caption(
        f"Baseline: {cases_per_day} test cases/day × {hours_per_day:g} hours/day. "
        f"Automated processing time for this run: {processing_seconds:.2f} seconds."
    )

    if manual_hours > 0:
        st.success(
            f"Estimated saving: {hours_saved:,.1f} hours "
            f"({days_saved:,.1f} working days), approximately {saving_pct:.1f}%."
        )

    # Small workload vs manual-hours chart.
    time_chart = pd.DataFrame({
        "Measure": ["Manual effort", "Automated processing"],
        "Hours": [manual_hours, automation_hours],
    }).set_index("Measure")
    st.markdown("#### Manual vs automated effort")
    st.bar_chart(time_chart)

    return pd.DataFrame([{
        "TestCaseAssessments": total_test_case_assessments,
        "CasesPerWorkingDay": cases_per_day,
        "HoursPerWorkingDay": hours_per_day,
        "EstimatedManualDays": round(manual_days, 3),
        "EstimatedManualHours": round(manual_hours, 3),
        "AutomatedProcessingSeconds": round(processing_seconds, 3),
        "AutomatedProcessingHours": round(automation_hours, 6),
        "EstimatedHoursSaved": round(hours_saved, 3),
        "EstimatedWorkingDaysSaved": round(days_saved, 3),
        "EstimatedSavingPercent": round(saving_pct, 2),
    }])


# ============================================================
# STREAMLIT APP
# ============================================================
def main():
    st.set_page_config(page_title="SMART REGRESSION", page_icon="📊", layout="wide")

    st.title("📊 SMART REGRESSION")
    st.write(
        "Compare E2E names, compare Test Set names for E2E pairs with at least 80% similarity, "
        "then compare Test Cases by Test ID."
    )

    uploaded = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    output_name = st.text_input("Output file name", "smart_regression_output.xlsx")

    if uploaded is None:
        st.info("Upload the source workbook to begin.")
        return

    file_bytes = uploaded.getvalue()

    try:
        df_raw, sheet_name, header_row = load_source_workbook(BytesIO(file_bytes))
        st.success(
            f"Loaded '{sheet_name}'. Detected header on Excel row {header_row + 1}. "
            f"Rows: {len(df_raw):,}, columns: {len(df_raw.columns)}."
        )
        base = standardize_input(df_raw)
        hierarchy_explorer(base)
    except Exception as exc:
        st.error(f"Could not read workbook: {exc}")
        return

    st.markdown("---")
    if st.button("Process and generate output file", type="primary"):
        try:
            with st.spinner("Processing comparisons..."):
                processing_start = time.perf_counter()

                result = process_dataframe(df_raw)

                processing_seconds = time.perf_counter() - processing_start

            # Charts and time-saved calculation use the completed processing result.
            time_saved_df = render_visual_analytics_and_time_saved(
                result,
                processing_seconds,
            )

            output = make_excel_output(
                result["base"], result["e2e_unique"], result["same_e2e"], result["cross_e2e"],
                result["testsets"], result["testcase_summary"], result["testcase_detail"],
                result["summary_overall"], result["summary_country"],
                time_saved_df=time_saved_df,
            )

            st.success("Analysis complete.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Same-country E2E pairs", f"{len(result['same_e2e']):,}")
            m2.metric("Cross-country E2E pairs", f"{len(result['cross_e2e']):,}")
            m3.metric("Test-set comparisons", f"{len(result['testsets']):,}")
            m4.metric("Test-case summaries", f"{len(result['testcase_summary']):,}")

            st.subheader("Summary")
            st.dataframe(result["summary_overall"], use_container_width=True, hide_index=True)

            st.subheader("Top E2E matches")
            tab1, tab2 = st.tabs(["Same country", "Cross country"])
            with tab1:
                st.dataframe(result["same_e2e"].head(100), use_container_width=True, hide_index=True)
            with tab2:
                st.dataframe(result["cross_e2e"].head(100), use_container_width=True, hide_index=True)

            st.subheader("Top Test Set matches")
            st.dataframe(result["testsets"].head(200), use_container_width=True, hide_index=True)

            st.download_button(
                "Download generated workbook",
                data=output,
                file_name=output_name if output_name.lower().endswith(".xlsx") else output_name + ".xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as exc:
            st.exception(exc)


if __name__ == "__main__":
    main()
