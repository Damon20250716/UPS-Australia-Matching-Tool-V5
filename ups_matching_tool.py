
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

def normalize_name(name):
    if pd.isna(name):
        return ""
    return ''.join(e for e in str(name).upper() if e.isalnum())

def match_recipient_to_account(recipient_name, account_df, vectorizer, threshold=0.7):
    recipient_clean = normalize_name(recipient_name)
    if not recipient_clean:
        return "Cash", 0.0, [], "Empty recipient name"

    account_df['Normalized'] = account_df['Customer Name'].apply(normalize_name)
    names = account_df['Normalized'].tolist()
    tfidf_matrix = vectorizer.fit_transform([recipient_clean] + names)
    cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    account_df['Similarity'] = cosine_scores
    sorted_df = account_df.sort_values(by='Similarity', ascending=False)
    top_matches = sorted_df.head(3)
    best_score = top_matches.iloc[0]['Similarity']

    if best_score >= threshold:
        return top_matches.iloc[0]['Account Number'], best_score, top_matches[['Customer Name', 'Account Number', 'Similarity']].values.tolist(), "Matched"
    return "Cash", best_score, top_matches[['Customer Name', 'Account Number', 'Similarity']].values.tolist(), "No match"

st.set_page_config(page_title="UPS AU Matching Tool", layout="wide")
st.markdown("## 🇦🇺 UPS AU Recipient Matching Tool")

shipment_file = st.file_uploader("Upload Shipment File (Excel or CSV)", type=["xlsx", "csv"])
account_file = st.file_uploader("Upload Account List File (Excel or CSV)", type=["xlsx", "csv"])
threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.75, 0.01)
filter_option = st.selectbox("Filter results", ["All", "Only Cash", "Only Unmatched"])

if shipment_file and account_file:
    try:
        ship_df = pd.read_excel(shipment_file) if shipment_file.name.endswith(".xlsx") else pd.read_csv(shipment_file)
        acc_df = pd.read_excel(account_file) if account_file.name.endswith(".xlsx") else pd.read_csv(account_file)
        vectorizer = TfidfVectorizer().fit([])

        results = []
        for _, row in ship_df.iterrows():
            recipient = row.get('Recipient Company Name', '')
            acct, score, suggestions, comment = match_recipient_to_account(recipient, acc_df.copy(), vectorizer, threshold)
            results.append({
                'Tracking Number': row.get('Tracking Number', ''),
                'Recipient Company Name': recipient,
                'Matched Account': acct,
                'Score': score,
                'Comment': comment,
                'Suggestions': "; ".join([f"{s[0]} ({s[1]}) [{s[2]:.2f}]" for s in suggestions])
            })

        result_df = pd.DataFrame(results)
        if filter_option == "Only Cash":
            result_df = result_df[result_df['Matched Account'] == "Cash"]
        elif filter_option == "Only Unmatched":
            result_df = result_df[result_df['Comment'] != "Matched"]

        st.dataframe(result_df)

        if st.button("Download Result as Excel"):
            result_path = "/mnt/data/ups_matching_result.xlsx"
            result_df.to_excel(result_path, index=False, engine="xlsxwriter")
            with open(result_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="ups_matching_result.xlsx">📥 Click here to download result</a>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
