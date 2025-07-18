import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

st.set_page_config(page_title="UPS Australia Matching Tool", layout="wide")

# --- UI Header ---
st.title("🇦🇺 UPS Australia Recipient Matching Tool")
st.markdown("""
Upload shipment and account Excel or CSV files. The tool will try to match recipient names to the correct account.
- If no suggestions are found, it will default to 'Cash'.
- Up to 3 account suggestions per shipment.
- Use filters to view only unmatched or only cash results.
""")

# --- File Uploads ---
shipment_file = st.file_uploader("📦 Upload Shipment File", type=['xlsx', 'csv'])
account_file = st.file_uploader("🏢 Upload Account File", type=['xlsx', 'csv'])
sim_threshold = st.slider("🔍 Similarity Threshold", 0.5, 1.0, 0.75, 0.01)

@st.cache_data(show_spinner=False)
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

@st.cache_resource(show_spinner=False)
def preprocess_text(s):
    if pd.isna(s): return ""
    if isinstance(s, (int, float)): s = str(s)
    s = s.upper()
    s = re.sub(r"[^A-Z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

@st.cache_resource(show_spinner=False)
def is_personal_name(name):
    keywords = ["PTY", "LTD", "INC", "CORP", "CO", "LLC", "PLC"]
    return not any(k in name for k in keywords) and len(name.split()) <= 4

@st.cache_resource(show_spinner=False)
def compute_matches(shipment_df, account_df, threshold):
    shipment_df['Normalized Recipient'] = shipment_df['Recipient Name'].apply(preprocess_text)
    account_df['Normalized Account'] = account_df['Account Name'].apply(preprocess_text)

    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4))
    tfidf_matrix_accounts = tfidf.fit_transform(account_df['Normalized Account'])
    tfidf_matrix_recipients = tfidf.transform(shipment_df['Normalized Recipient'])

    similarities = cosine_similarity(tfidf_matrix_recipients, tfidf_matrix_accounts)
    results = []

    for i, row in shipment_df.iterrows():
        recipient = row['Normalized Recipient']
        original_name = row['Recipient Name']
        sims = similarities[i]

        top_idx = np.argsort(sims)[::-1][:3]
        top_scores = sims[top_idx]
        top_accounts = account_df.iloc[top_idx]

        suggestions = []

        for j, score in enumerate(top_scores):
            if score >= threshold:
                account_name = top_accounts.iloc[j]['Account Name']
                acc_num = top_accounts.iloc[j]['Account Number']
                suggestions.append(f"{account_name} ({acc_num})")

        # Assign matched account based on suggestions
        if suggestions:
            matched_account = top_accounts.iloc[0]['Account Number']
        else:
            matched_account = "Cash"

        results.append({
            **row,
            "Matched Account": matched_account,
            "Top Suggestions": "; ".join(suggestions)
        })

    return pd.DataFrame(results)

# --- Main Logic ---
if shipment_file and account_file:
    try:
        shipment_df = load_data(shipment_file)
        account_df = load_data(account_file)

        if 'Recipient Name' not in shipment_df.columns or 'Account Name' not in account_df.columns:
            st.error("❌ Please ensure 'Recipient Name' in shipment file and 'Account Name' in account file.")
        else:
            with st.spinner("Matching in progress..."):
                df = compute_matches(shipment_df, account_df, sim_threshold)

            st.success(f"✅ Matching complete! {len(df)} rows processed.")

            # Filters
            col1, col2 = st.columns(2)
            with col1:
                show_cash_only = st.checkbox("💵 Show Only 'Cash' Matches")
            with col2:
                show_unmatched_only = st.checkbox("🚫 Show Only Unmatched (Cash + No Suggestions)")

            filtered_df = df.copy()
            if show_cash_only:
                filtered_df = filtered_df[filtered_df['Matched Account'] == 'Cash']
            if show_unmatched_only:
                filtered_df = filtered_df[(filtered_df['Matched Account'] == 'Cash') & (filtered_df['Top Suggestions'] == '')]

            st.dataframe(filtered_df, use_container_width=True)

            # Excel Export
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='All Matches')
                df[df['Matched Account'] == 'Cash'].to_excel(writer, index=False, sheet_name='Only Cash')
                df[(df['Matched Account'] == 'Cash') & (df['Top Suggestions'] == '')].to_excel(writer, index=False, sheet_name='Only Unmatched')

            st.download_button(
                label="📥 Download Results as Excel",
                data=output.getvalue(),
                file_name="ups_matching_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.warning("⬆️ Please upload both shipment and account files to begin.")
