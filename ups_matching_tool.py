import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

def normalize_name(name):
    if not isinstance(name, str):
        name = str(name)
    name = name.upper()
    # remove punctuation and common company suffixes
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\b(PTY|LTD|LIMITED|PL|P\/L|AUST|AUSTRALIA|CORP|INC|CO|COMPANY|LLC|THE|AND|&)\b', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def match_recipient_to_account(recipient_name, accounts_df, threshold=0.6):
    recipient_norm = normalize_name(recipient_name)
    accounts_norm = accounts_df['Normalized Name'].values
    
    # Vectorize normalized names
    vectorizer = TfidfVectorizer().fit(accounts_norm.tolist() + [recipient_norm])
    recipient_vec = vectorizer.transform([recipient_norm])
    accounts_vec = vectorizer.transform(accounts_norm)
    
    # Calculate cosine similarity
    sims = cosine_similarity(recipient_vec, accounts_vec).flatten()
    
    # Find matches above threshold
    matched_idxs = np.where(sims >= threshold)[0]
    if len(matched_idxs) == 0:
        return 'Cash', 0, [], "No match above threshold"
    
    # Sort matches by similarity descending
    sorted_matches = sorted([(idx, sims[idx]) for idx in matched_idxs], key=lambda x: x[1], reverse=True)
    
    # Pick top 1 as main match
    top_idx, top_score = sorted_matches[0]
    main_account = accounts_df.iloc[top_idx]['Account Number']
    # Suggestions (up to 3)
    suggestions = [(accounts_df.iloc[idx]['Account Number'], sims[idx]) for idx, _ in sorted_matches[:3]]
    
    return main_account, top_score, suggestions, "Matched"

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Matches')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

st.title("UPS Australia Recipient Name Matching Tool")

st.markdown("""
Upload shipment CSV and account CSV files.  
Adjust similarity threshold.  
Filter results and export to Excel.
""")

shipment_file = st.file_uploader("Upload Shipment CSV", type=['csv'])
account_file = st.file_uploader("Upload Account CSV", type=['csv'])

threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.65, step=0.01)

if shipment_file and account_file:
    shipments_df = pd.read_csv(shipment_file)
    accounts_df = pd.read_csv(account_file)
    
    # Normalize account names once
    accounts_df['Normalized Name'] = accounts_df['Customer Name'].apply(normalize_name)
    
    # Match all shipments
    results = []
    for idx, row in shipments_df.iterrows():
        acct, score, suggestions, comment = match_recipient_to_account(row['Recipient Company Name'], accounts_df, threshold)
        results.append({
            'Tracking Number': row.get('Tracking Number', ''),
            'Recipient Company Name': row['Recipient Company Name'],
            'Matched Account': acct,
            'Match Score': score,
            'Match Comment': comment,
            'Suggestions': ", ".join([f"{acc}({sim:.2f})" for acc, sim in suggestions])
        })
    results_df = pd.DataFrame(results)
    
    # Filters
    show_unmatched = st.checkbox("Only Unmatched", value=False)
    show_cash = st.checkbox("Only Cash", value=False)
    
    filtered_df = results_df
    if show_unmatched:
        filtered_df = filtered_df[filtered_df['Match Comment'] != "Matched"]
    if show_cash:
        filtered_df = filtered_df[filtered_df['Matched Account'] == 'Cash']
    
    st.dataframe(filtered_df)
    
    # Export to Excel
    if st.button("Export Filtered Results to Excel"):
        excel_data = to_excel(filtered_df)
        st.download_button(label='Download Excel file', data=excel_data, file_name='matching_results.xlsx')
