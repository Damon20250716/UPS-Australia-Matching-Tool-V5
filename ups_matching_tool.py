import streamlit as st
import pandas as pd
import numpy as np
import difflib
import base64
import io
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

st.set_page_config(layout="wide")

# ---------- Helper functions ----------
def normalize_name(name):
    if not isinstance(name, str):
        name = str(name)
    return name.upper().strip()

def is_personal_name(name):
    name = normalize_name(name)
    return (
        len(name.split()) <= 3
        and all(word.isalpha() for word in name.split())
        and not any(x in name for x in ["PTY", "LTD", "CORP", "INC", "CO", "LIMITED"])
    )

def match_recipient_to_account_tfidf(recipient_name, acc_df, vectorizer, account_vectors, threshold):
    recipient_name_cleaned = normalize_name(recipient_name)
    if is_personal_name(recipient_name_cleaned):
        return "Cash", 0, [], "Personal name"

    recipient_vector = vectorizer.transform([recipient_name_cleaned])
    similarities = cosine_similarity(recipient_vector, account_vectors).flatten()
    best_indices = similarities.argsort()[::-1][:3]
    suggestions = [(acc_df.iloc[i]['Customer Name'], similarities[i], acc_df.iloc[i]['Account Number']) for i in best_indices]

    best_match_index = best_indices[0]
    best_score = similarities[best_match_index]
    best_name = acc_df.iloc[best_match_index]['Customer Name']
    best_account = acc_df.iloc[best_match_index]['Account Number']

    if best_score >= threshold:
        return best_account, best_score, suggestions, f"Matched with credit account"
    else:
        return "Cash", best_score, suggestions, "Below threshold"

def display_flag():
    flag_path = "australia_flag.png"
    if os.path.exists(flag_path):
        st.image(flag_path, width=60)
    else:
        st.write(":flag-au:")

# ---------- Streamlit UI ----------
st.title("UPS Australia Recipient Matching Tool")
display_flag()

shipment_file = st.file_uploader("Upload Shipment File (Excel or CSV)", type=["xlsx", "xls", "csv"])
account_file = st.file_uploader("Upload Account List (Excel or CSV)", type=["xlsx", "xls", "csv"])
threshold = st.slider("Similarity Threshold", min_value=0.1, max_value=1.0, value=0.75, step=0.01)

filter_option = st.selectbox("Filter Results", ["All", "Only Unmatched (Cash)", "Only Matched (Credit)"])

if shipment_file and account_file:
    try:
        shipment_df = pd.read_excel(shipment_file) if shipment_file.name.endswith("xlsx") else pd.read_csv(shipment_file)
        acc_df = pd.read_excel(account_file) if account_file.name.endswith("xlsx") else pd.read_csv(account_file)

        shipment_df['Recipient Company Name'] = shipment_df['Recipient Company Name'].astype(str).str.strip()
        acc_df['Customer Name'] = acc_df['Customer Name'].astype(str).str.strip()
        acc_df = acc_df[acc_df['Customer Name'].str.len() > 0]

        if acc_df.empty:
            st.error("Account list is empty or invalid after cleaning. Please check your file.")
            st.stop()

        vectorizer = TfidfVectorizer(stop_words='english')
        account_vectors = vectorizer.fit_transform(acc_df['Customer Name'])

        matched_accounts = []
        scores = []
        comments = []
        suggestions_all = []

        for _, row in shipment_df.iterrows():
            recipient = row['Recipient Company Name']
            account, score, suggestions, comment = match_recipient_to_account_tfidf(recipient, acc_df, vectorizer, account_vectors, threshold)
            matched_accounts.append(account)
            scores.append(score)
            comments.append(comment)
            suggestions_all.append("; ".join([f"{x[0]} ({x[1]:.2f}) - {x[2]}" for x in suggestions]))

        shipment_df['Matched Account'] = matched_accounts
        shipment_df['Match Score'] = scores
        shipment_df['Comment'] = comments
        shipment_df['Top 3 Suggestions'] = suggestions_all

        if filter_option == "Only Unmatched (Cash)":
            shipment_df = shipment_df[shipment_df['Matched Account'] == "Cash"]
        elif filter_option == "Only Matched (Credit)":
            shipment_df = shipment_df[shipment_df['Matched Account'] != "Cash"]

        st.dataframe(shipment_df)

        # Excel export
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            shipment_df.to_excel(writer, index=False, sheet_name="Matching Result")
            writer.save()
        st.download_button("Download Result as Excel", data=output.getvalue(), file_name="matching_result.xlsx")

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
