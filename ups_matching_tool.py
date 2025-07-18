import streamlit as st
import pandas as pd
import difflib
import re
from io import BytesIO

st.set_page_config(page_title="UPS Australia Matching Tool", layout="wide")

# Helper: Normalize name
def normalize_name(name):
    if not isinstance(name, str):
        name = str(name)
    name = re.sub(r'[^A-Z0-9 ]+', '', name.upper())
    return name.strip()

# Helper: Detect if personal name (simplified)
def is_personal_name(name):
    keywords = ['PTY', 'LTD', 'INC', 'CORP', 'LLC', 'GROUP', 'CO', 'LIMITED', 'COMPANY']
    return all(kw not in name for kw in keywords)

# Matching logic
def match_recipient_to_account(recipient_name, acc_df, threshold=0.75):
    norm_rec_name = normalize_name(recipient_name)
    acc_df['Normalized Name'] = acc_df['Customer Name'].apply(normalize_name)
    scores = acc_df['Normalized Name'].apply(lambda x: difflib.SequenceMatcher(None, norm_rec_name, x).ratio())
    acc_df['Similarity'] = scores

    top_matches = acc_df.sort_values(by='Similarity', ascending=False).head(3)
    suggestions = top_matches[['Customer Name', 'Account Number', 'Similarity']].to_dict('records')

    best_match = top_matches.iloc[0]
    first_two_words = ' '.join(norm_rec_name.split()[:2])
    filtered = acc_df[acc_df['Normalized Name'].str.startswith(first_two_words)]
    filtered = filtered[filtered['Similarity'] >= threshold]

    if not is_personal_name(norm_rec_name):
        if len(filtered) == 1:
            selected = filtered.iloc[0]
            return selected['Account Number'], selected['Similarity'], suggestions, "Matched by prefix rule"
        elif best_match['Similarity'] >= threshold:
            return best_match['Account Number'], best_match['Similarity'], suggestions, "Top match above threshold"

    return "Cash", 0.0, suggestions, "Defaulted to Cash"

# UI Elements
st.markdown("### 🇦🇺 UPS AU Recipient Name Matching Tool")
uploaded_shipments = st.file_uploader("Upload shipment file (Excel)", type=["xlsx"])
uploaded_accounts = st.file_uploader("Upload account database (Excel)", type=["xlsx"])
threshold = st.slider("Similarity threshold", 0.5, 0.95, 0.75, 0.01)

if uploaded_shipments and uploaded_accounts:
    try:
        ship_df = pd.read_excel(uploaded_shipments)
        acc_df = pd.read_excel(uploaded_accounts)

        result_rows = []
        for _, row in ship_df.iterrows():
            acct, score, suggestions, comment = match_recipient_to_account(
                row['Recipient Company Name'], acc_df, threshold)
            result_rows.append({
                "Tracking Number": row['Tracking Number'],
                "Recipient Company Name": row['Recipient Company Name'],
                "Matched Account": acct,
                "Match Score": score,
                "Comment": comment,
                "Suggested Matches": " | ".join([f"{s['Customer Name']} ({s['Account Number']}, {s['Similarity']:.2f})" for s in suggestions])
            })

        result_df = pd.DataFrame(result_rows)
        st.dataframe(result_df)

        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            result_df.to_excel(writer, index=False, sheet_name="Results")
        st.download_button("Download Result as Excel", data=output.getvalue(),
                           file_name="matched_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")