import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import base64

# ---------- Display Header and Flag ----------
def show_flag(country_code="au"):
    st.image(f"https://flagcdn.com/w320/{country_code.lower()}.png", width=80)

st.set_page_config(page_title="UPS AU Matching Tool", layout="wide")
st.title("🇦🇺 UPS Australia Recipient Matching Tool")
show_flag("au")

# ---------- Upload Section ----------
shipment_file = st.file_uploader("Upload Shipment File (CSV)", type=["csv"])
account_file = st.file_uploader("Upload Account List File (CSV)", type=["csv"])
similarity_threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.85, step=0.01)

# ---------- Normalize Function ----------
def normalize(text):
    if pd.isna(text): return ""
    if isinstance(text, (int, float)): text = str(text)
    text = text.upper()
    text = re.sub(r'[^A-Z0-9 ]+', '', text)
    return text.strip()

# ---------- Check for personal name ----------
def is_likely_personal(name):
    if pd.isna(name): return True
    name = name.upper()
    if "PTY" in name or "LTD" in name or "CORP" in name or "CO" in name or "INC" in name:
        return False
    return len(name.split()) <= 2

# ---------- Match Logic with TF-IDF ----------
def match_shipments(shipments_df, accounts_df, threshold):
    accounts_df["NormName"] = accounts_df["Customer Name"].apply(normalize)
    shipment_names = shipments_df["Recipient Company Name"].fillna("").astype(str).apply(normalize)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
    tfidf_matrix = vectorizer.fit_transform(accounts_df["NormName"])

    matched_accounts = []
    similarity_scores = []
    match_comments = []

    for recipient in shipment_names:
        if is_likely_personal(recipient):
            matched_accounts.append("Cash")
            similarity_scores.append(0)
            match_comments.append("Likely personal name")
            continue

        rec_vec = vectorizer.transform([recipient])
        cosine_sim = cosine_similarity(rec_vec, tfidf_matrix).flatten()
        top_idx = cosine_sim.argsort()[::-1][:3]
        top_scores = cosine_sim[top_idx]

        if top_scores[0] >= threshold:
            matched_account = accounts_df.iloc[top_idx[0]]["Account Number"]
            matched_accounts.append(matched_account)
            similarity_scores.append(top_scores[0])
            comment = f"Matched with score {top_scores[0]:.2f}"
        else:
            matched_accounts.append("Cash")
            similarity_scores.append(top_scores[0])
            comment = "No good match found"

        match_comments.append(comment)

    shipments_df["Matched Account"] = matched_accounts
    shipments_df["Similarity Score"] = similarity_scores
    shipments_df["Match Comment"] = match_comments

    return shipments_df

# ---------- Run Matching ----------
if shipment_file and account_file:
    try:
        shipments = pd.read_csv(shipment_file)
        accounts = pd.read_csv(account_file)

        if "Recipient Company Name" not in shipments.columns or "Customer Name" not in accounts.columns:
            st.error("❌ Missing required columns in your files.")
        else:
            matched_df = match_shipments(shipments.copy(), accounts.copy(), similarity_threshold)

            st.success(f"✅ Matching completed for {len(matched_df)} rows")

            # Filters
            filter_option = st.selectbox("Filter", ["All", "Only Cash", "Only Unmatched"])
            if filter_option == "Only Cash":
                filtered_df = matched_df[matched_df["Matched Account"] == "Cash"]
            elif filter_option == "Only Unmatched":
                filtered_df = matched_df[matched_df["Similarity Score"] < similarity_threshold]
            else:
                filtered_df = matched_df

            st.dataframe(filtered_df)

            # Export to Excel
            def to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Matched Results')
                return output.getvalue()

            excel_data = to_excel(matched_df)

            b64 = base64.b64encode(excel_data).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="matched_results.xlsx">📥 Download Full Results as Excel</a>'
            st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
