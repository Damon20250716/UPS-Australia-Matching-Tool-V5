import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

st.set_page_config(page_title="UPS AU Name Matcher 🇦🇺", layout="wide")
st.title("📦 UPS Australia Recipient Matching Tool 🇦🇺")

# Upload shipment and account files
shipment_file = st.file_uploader("Upload Shipment File (CSV)", type=["csv"])
account_file = st.file_uploader("Upload Account List File (CSV)", type=["csv"])

threshold = st.slider("Set similarity threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.01)

@st.cache_data

def load_data(file):
    return pd.read_csv(file)

@st.cache_resource

def build_tfidf_matrix(names):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(names)
    return tfidf_matrix, vectorizer


def match_names_tfidf(shipment_df, account_df, threshold=0.7):
    shipment_names = shipment_df['Recipient Company Name'].astype(str).fillna("").str.lower()
    account_names = account_df['Customer Name'].astype(str).fillna("").str.lower()

    account_matrix, vectorizer = build_tfidf_matrix(account_names)
    shipment_matrix = vectorizer.transform(shipment_names)

    similarity_scores = cosine_similarity(shipment_matrix, account_matrix)

    matched_accounts = []
    similarity_values = []
    suggestions = []
    comments = []

    for i, scores in enumerate(similarity_scores):
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        if best_score >= threshold:
            matched_account = account_df.iloc[best_idx]['Account Number']
            matched_name = account_df.iloc[best_idx]['Customer Name']
            comment = "Matched to credit account"
        else:
            matched_account = 'Cash'
            matched_name = ''
            comment = 'Low similarity - defaulted to Cash'

        # Get top 3 suggestions
        top_indices = scores.argsort()[-3:][::-1]
        top_suggestions = [str(account_df.iloc[j]['Customer Name']) for j in top_indices]

        matched_accounts.append(matched_account)
        similarity_values.append(round(best_score, 4))
        suggestions.append(", ".join(top_suggestions))
        comments.append(comment)

    result_df = shipment_df.copy()
    result_df['Matched Account'] = matched_accounts
    result_df['Similarity Score'] = similarity_values
    result_df['Top Suggestions'] = suggestions
    result_df['Comment'] = comments

    return result_df

if shipment_file and account_file:
    try:
        shipment_df = load_data(shipment_file)
        account_df = load_data(account_file)

        with st.spinner("Matching in progress..."):
            result_df = match_names_tfidf(shipment_df, account_df, threshold)

        st.success("✅ Matching complete!")
        st.dataframe(result_df.head(20))

        csv_output = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Result as CSV", data=csv_output, file_name="matched_result.csv", mime='text/csv')

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("Please upload both shipment and account CSV files to start.")
