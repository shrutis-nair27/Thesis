import pandas as pd
import streamlit as st
from pathlib import Path
from io import BytesIO

# reuse processing logic from the existing module
from ThesisSmartRegresssion import (
    load_first_non_empty_sheet,
    process_dataframe,
    create_output_workbook,
)


@st.cache_data(show_spinner=False)
def load_uploaded_sheet(file_bytes: bytes):
    uploaded_file = BytesIO(file_bytes)
    return load_first_non_empty_sheet(uploaded_file)


def identify_hierarchy_columns(df_raw):
    def _get_col(df, idx, name_hints=None):
        name_hints = name_hints or []
        if 0 <= idx < len(df.columns):
            return df.columns[idx]
        for hint in name_hints:
            for c in df.columns:
                if hint.lower() in str(c).lower():
                    return c
        return None

    e2e_col = _get_col(df_raw, 8, ['s2e_e2e_name', 'e2e'])
    testset_col = _get_col(df_raw, 11, ['test_set_name', 'test set'])
    testid_col = _get_col(df_raw, 14, ['test id', 'testid'])
    testcase_col = _get_col(df_raw, 16, ['test name', 'testcase'])
    return e2e_col, testset_col, testid_col, testcase_col


@st.cache_data(show_spinner=False)
def build_e2e_hierarchy(file_bytes: bytes):
    df_raw, _ = load_uploaded_sheet(file_bytes)
    e2e_col, testset_col, testid_col, testcase_col = identify_hierarchy_columns(df_raw)
    if not all([e2e_col, testset_col, testid_col, testcase_col]):
        return None

    df_lookup = df_raw[[e2e_col, testset_col, testid_col, testcase_col]].copy()
    df_lookup = df_lookup.dropna(subset=[e2e_col])
    df_lookup[e2e_col] = df_lookup[e2e_col].astype(str)
    df_lookup[testset_col] = df_lookup[testset_col].astype(str)
    df_lookup[testid_col] = df_lookup[testid_col].astype(str)
    df_lookup[testcase_col] = df_lookup[testcase_col].astype(str)

    e2e_values = sorted(df_lookup[e2e_col].dropna().unique())
    sets_map = {
        e2e: sorted(df_lookup[df_lookup[e2e_col] == e2e][testset_col].dropna().unique())
        for e2e in e2e_values
    }
    cases_map = {}
    for e2e in e2e_values:
        subset = df_lookup[df_lookup[e2e_col] == e2e]
        cases_map[(e2e, '-- all --')] = subset[[testid_col, testcase_col]].drop_duplicates()
        for testset in sets_map[e2e]:
            cases_map[(e2e, testset)] = subset[subset[testset_col] == testset][[testid_col, testcase_col]].drop_duplicates()

    return {
        'e2e_col': e2e_col,
        'testset_col': testset_col,
        'testid_col': testid_col,
        'testcase_col': testcase_col,
        'e2e_values': e2e_values,
        'sets_map': sets_map,
        'cases_map': cases_map,
        'df_lookup': df_lookup,
    }


@st.cache_data(show_spinner=False)
def process_cached_dataframe(file_bytes: bytes):
    uploaded_file = BytesIO(file_bytes)
    df_raw, _ = load_first_non_empty_sheet(uploaded_file)
    return process_dataframe(df_raw)


def main():
    st.set_page_config(page_title='Excel Upload UI', layout='wide')
    st.markdown(
        """
    <div style='background: linear-gradient(135deg, #2E6BF8 0%, #00C1F7 100%); padding: 18px; border-radius: 14px; color: white;'>
        <h1 style='margin: 0; font-size: 2rem;'>Excel Upload UI</h1>
        <p style='margin: 8px 0 0; font-size: 1rem; color: #F0F8FF;'>Upload a workbook, preview the sheet, and process it with SMART REGRESSION logic.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 1])
    with left:
        uploaded_file = st.file_uploader('Upload Excel file', type=['xlsx', 'xls'])
        output_name = st.text_input('Output file name', value='core_regression_output.xlsx')
        process_button = st.button('Process and generate workbook')

    with right:
        st.markdown('### Quick tips')
        st.write('- Files must be .xlsx or .xls')
        st.write('- Uploaded files should have the expected columns')

    if uploaded_file is None:
        st.info('Please upload an Excel file to begin.')
        return

    if 'file_bytes' not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
        st.session_state['file_bytes'] = uploaded_file.read()
        st.session_state['uploaded_file_name'] = uploaded_file.name

    file_bytes = st.session_state['file_bytes']

    try:
        df_raw, sheet_name = load_uploaded_sheet(file_bytes)
    except Exception as exc:
        st.error(f'Failed to read Excel file: {exc}')
        return

    selected_columns = []
    for idx in [8, 11, 14, 16]:
        if idx < len(df_raw.columns):
            selected_columns.append(df_raw.columns[idx])

    hierarchy_data = build_e2e_hierarchy(file_bytes)

    st.markdown(f'### Preview: {sheet_name}')
    st.write(f'Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns')
    if df_raw.empty:
        st.warning('The selected sheet is empty.')
    else:
        if selected_columns:
            st.write('Showing input columns I, L, O, Q only:')
            st.dataframe(df_raw[selected_columns].head(20).astype(str), use_container_width=True)
            st.write('**Columns:**')
            st.write(selected_columns)
        else:
            st.warning('The uploaded sheet has fewer than 17 columns, so columns I, L, O, Q cannot be displayed.')

    if process_button:
        if hierarchy_data is None:
            st.warning('The uploaded sheet does not contain valid E2E/TestSet/TestID/TestCase columns in the expected positions.')
        else:
            try:
                unique_df, df_cross, df_same, df_testset_same, df_testset_cross, df_testcases = process_cached_dataframe(file_bytes)

                output_bytes = create_output_workbook(
                    unique_df, df_cross, df_same, df_testset_same, df_testset_cross, df_testcases
                )

                # save to uploads folder and show download
                output_dir = Path(__file__).resolve().parent / "uploads"
                output_dir.mkdir(parents=True, exist_ok=True)
                safe_name = (output_name or "core_regression_output.xlsx").strip()
                if Path(safe_name).suffix.lower() != ".xlsx":
                    safe_name = f"{safe_name}.xlsx"
                saved_path = output_dir / safe_name
                saved_path.write_bytes(output_bytes)
                st.success(f'Workbook generated and saved to: {saved_path}')

                st.download_button(
                    label='Download generated workbook',
                    data=output_bytes,
                    file_name=output_name,
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )

            except Exception as exc:
                st.error(f'Error during processing: {exc}')
    else:
        st.info('Click "Process and generate workbook" to run analysis and create the output Excel.')

    # --- Hierarchical E2E / Test Set / Test Case explorer and comparator ---
    st.markdown('---')
    st.markdown('## E2E explorer and comparison')

    if hierarchy_data is None:
        st.warning('Could not identify the E2E / TestSet / TestID / TestCase columns automatically. The explorer requires columns in positions I, L, O, Q (or similarly named).')
        return

    e2e_col = hierarchy_data['e2e_col']
    testset_col = hierarchy_data['testset_col']
    testid_col = hierarchy_data['testid_col']
    testcase_col = hierarchy_data['testcase_col']
    e2e_values = hierarchy_data['e2e_values']
    sets_map = hierarchy_data['sets_map']
    cases_map = hierarchy_data['cases_map']
    df_lookup = hierarchy_data['df_lookup']

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('### E2E A')
        e2e_a = st.selectbox('Select E2E A', e2e_values, key='e2e_a')
        sets_a = ['-- all --'] + sets_map.get(e2e_a, [])
        sel_set_a = st.selectbox('Test set (A)', sets_a, key='set_a')
        cases_a = cases_map.get((e2e_a, sel_set_a), pd.DataFrame(columns=[testid_col, testcase_col]))
        st.markdown(f'**Test cases (A): {len(cases_a)} rows**')
        st.dataframe(cases_a.rename(columns={testid_col: 'TestID', testcase_col: 'TestCaseName'}).astype(str).head(200), use_container_width=True)

    with col_b:
        st.markdown('### E2E B')
        e2e_b = st.selectbox('Select E2E B', e2e_values, index=1 if len(e2e_values) > 1 else 0, key='e2e_b')
        sets_b = ['-- all --'] + sets_map.get(e2e_b, [])
        sel_set_b = st.selectbox('Test set (B)', sets_b, key='set_b')
        cases_b = cases_map.get((e2e_b, sel_set_b), pd.DataFrame(columns=[testid_col, testcase_col]))
        st.markdown(f'**Test cases (B): {len(cases_b)} rows**')
        st.dataframe(cases_b.rename(columns={testid_col: 'TestID', testcase_col: 'TestCaseName'}).astype(str).head(200), use_container_width=True)

    # Comparison
    if st.button('Compare selected E2Es'):
        try:
            set_names_a = set(sets_a[1:]) if sets_a else set()
            set_names_b = set(sets_b[1:]) if sets_b else set()
            common_sets = sorted(list(set_names_a & set_names_b))

            ids_a = set(cases_a[testid_col].dropna().astype(str).unique())
            ids_b = set(cases_b[testid_col].dropna().astype(str).unique())
            common_ids = sorted(list(ids_a & ids_b))

            st.markdown('### Comparison results')
            st.write(f'- E2E A: {e2e_a} — test sets: {len(sets_a)-1} — test cases: {len(ids_a)}')
            st.write(f'- E2E B: {e2e_b} — test sets: {len(sets_b)-1} — test cases: {len(ids_b)}')
            st.write(f'- Common test sets: {len(common_sets)}')
            st.write(f'- Common test cases (by TestID): {len(common_ids)}')

            if common_sets:
                st.markdown('#### Shared test sets')
                st.write(common_sets)

            if common_ids:
                st.markdown('#### Shared test cases (by TestID)')
                shared_cases = df_lookup[df_lookup[testid_col].astype(str).isin(common_ids)][[e2e_col, testset_col, testid_col, testcase_col]]
                shared_cases = shared_cases.drop_duplicates()
                shared_cases = shared_cases.rename(columns={e2e_col: 'E2E', testset_col: 'TestSet', testid_col: 'TestID', testcase_col: 'TestCaseName'})
                st.dataframe(shared_cases.astype(str).head(500), use_container_width=True)
        except Exception as exc:
            st.error(f'Comparison failed: {exc}')


if __name__ == '__main__':
    main()
