
import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import re
from io import BytesIO

# --- Utilities ---
def normalize_name(name):
    if pd.isna(name):
        return ""
    name = str(name).upper()
    name = re.sub(r"[^A-Z0-9 ]", "", name)
    name = re.sub(r"\b(PTY LTD|LTD|CO|INC|CORP|LIMITED)\b", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def is_probably_personal(name):
    if len(name.split()) <= 2 and not any(x in name for x in ["PTY", "LTD", "CORP", "INC"]):
        return True
    return False

def token_similarity(a, b):
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0
    return len(set_a & set_b) / len(set_a | set_b)

def match_recipient_to_account(recipient_name, accounts_df, threshold):
    recipient_norm = normalize_name(recipient_name)

    if is_probably_personal(recipient_norm):
        return "Cash", 0, [], "Likely a personal name"

    accounts_df["Score"] = accounts_df["Normalized Name"].apply(lambda x: fuzz.ratio(recipient_norm, x))
    top_matches = accounts_df.sort_values(by="Score", ascending=False).head(3)

    if top_matches.iloc[0]["Score"] >= threshold:
        return (top_matches.iloc[0]["Account Number"],
                top_matches.iloc[0]["Score"],
                list(zip(top_matches["Customer Name"], top_matches["Score"])),
                "Matched by fuzzy ratio")
    else:
        return "Cash", top_matches.iloc[0]["Score"], [], "No match above threshold"

# --- Streamlit UI ---
st.set_page_config(page_title="UPS AU Matching Tool 🇦🇺")
st.title("🇦🇺 UPS Australia Recipient to Account Matching Tool")

uploaded_shipments = st.file_uploader("Upload Shipment File", type=["xlsx"])
uploaded_accounts = st.file_uploader("Upload Account Master File", type=["xlsx"])

threshold = st.slider("Similarity Threshold", min_value=60, max_value=100, value=85, step=1)

if uploaded_shipments and uploaded_accounts:
    try:
        shipments_df = pd.read_excel(uploaded_shipments)
        accounts_df = pd.read_excel(uploaded_accounts)
        accounts_df["Normalized Name"] = accounts_df["Customer Name"].apply(normalize_name)

        results = []
        for _, row in shipments_df.iterrows():
            recipient = row.get("Recipient Company Name", "")
            acct, score, suggestions, comment = match_recipient_to_account(recipient, accounts_df.copy(), threshold)
            results.append({
                "Tracking Number": row.get("Tracking Number", ""),
                "Recipient Company Name": recipient,
                "Matched Account": acct,
                "Confidence": score,
                "Comment": comment
            })

        result_df = pd.DataFrame(results)
        st.success("✅ Matching Complete")
        st.dataframe(result_df)

        # Download link
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            result_df.to_excel(writer, index=False)
        st.download_button("Download Result", output.getvalue(), file_name="matching_result.xlsx")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
