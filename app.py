import os
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# ------------------ CONFIGURATION ------------------
GEMINI_API_KEY = "AIzaSyDABvwRicVONM2D2MOqW886-UGo9BmxHdc"  # Replace with your key

genai.configure(api_key=GEMINI_API_KEY)
st.set_page_config(page_title="Legal Incident Analyzer", layout="wide", page_icon="📘")

# ------------------ DATA LOADER ------------------
@st.cache_data
def load_data():
    ipc_df = pd.read_csv('ipc_sections.csv')
    ipc_df = ipc_df.dropna(subset=['Section', 'Offence', 'Punishment'])
    ipc_df['context'] = (
        ipc_df['Section'].astype(str) + ' ' +
        ipc_df['Offence'].astype(str).str.lower() + ' ' +
        ipc_df.get('Description', '').astype(str).str.lower()
    )
    return ipc_df

# ------------------ CRIME TYPE ASSIGNMENT ------------------
def assign_crime_type(row):
    txt = (str(row['Offence']) + ' ' + str(row.get('Description',''))).lower()
    if any(word in txt for word in ['cheat', 'fraud', 'impersonat', 'phishing', 'cyber']): return 'cybercrime'
    if any(word in txt for word in ['theft', 'steal', 'pickpocket']): return 'theft'
    if any(word in txt for word in ['robbery', 'dacoity', 'snatch']): return 'robbery'
    if any(word in txt for word in ['forgery', 'counterfeit']): return 'forgery'
    if any(word in txt for word in ['sexual', 'rape', 'molest']): return 'sexual'
    if any(word in txt for word in ['kidnap', 'abduct']): return 'kidnapping'
    if any(word in txt for word in ['murder', 'homicide']): return 'murder'
    return 'other'

# ------------------ EMBEDDINGS ------------------
@st.cache_resource
def load_model_and_embeddings(ipc_df):
    sbert_model = SentenceTransformer('bhavyagiri/InLegal-Sbert')
    ipc_embs = sbert_model.encode(ipc_df['context'].tolist(), convert_to_tensor=True)
    return sbert_model, ipc_embs

# ------------------ GEMINI SUMMARIZER ------------------
def gemini_simplify_incident(incident_text):
    prompt = (
        "You are an Indian legal assistant. Given the following incident, do two things:\n"
        "1. Summarize the core criminal act in one sentence using IPC-like legal language.\n"
        "2. Identify the most likely crime type (e.g., 'theft', 'robbery', 'cybercrime', etc.).\n\n"
        f"Incident: {incident_text}\n\n"
        "Output format:\n"
        "Summary: <summary>\n"
        "CrimeType: <type>"
    )
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# ------------------ ROUTER ------------------
def route_query(ipc_df, crime_type):
    filtered = ipc_df[ipc_df['crime_type'].str.contains(crime_type, case=False, na=False)]
    return filtered if not filtered.empty else ipc_df

# ------------------ TOP K MATCHER ------------------
def retrieve_sections(summary, candidate_df, sbert_model, top_k=7, threshold=0.40):
    candidate_embs = sbert_model.encode(candidate_df['context'].tolist(), convert_to_tensor=True)
    summary_emb = sbert_model.encode(summary, convert_to_tensor=True)
    scores = util.cos_sim(summary_emb, candidate_embs)[0]
    idxs = (scores >= threshold).nonzero().flatten()
    if len(idxs) < 1:
        idxs = scores.argsort(descending=True)[:top_k]
    else:
        idxs = scores[idxs].argsort(descending=True)[:top_k]

    results = []
    for idx in idxs:
        row = candidate_df.iloc[int(idx)]
        results.append({
            'section': row['Section'],
            'offence': row['Offence'],
            'description': row.get('Description', ''),
            'punishment': row['Punishment'],
            'score': float(scores[int(idx)])
        })
    return results

# ------------------ GEMINI RERANKER ------------------
def llm_rerank_sections(summary, candidates, min_relevance=85):
    model = genai.GenerativeModel("gemini-1.5-flash")
    final_results = []
    for m in candidates:
        prompt = (
            "You are a legal expert. Rate how relevant the following IPC section is to the incident summary from 0 to 100.\n\n"
            f"Incident: {summary}\n"
            f"Section: {m['section']}\nOffence: {m['offence']}\nDescription: {m['description']}\nPunishment: {m['punishment']}\n\n"
            "Only output a number."
        )
        response = model.generate_content(prompt)
        try:
            relevance = int(''.join(filter(str.isdigit, response.text.strip())))
        except:
            relevance = 0
        if relevance >= min_relevance:
            m['llm_relevance'] = relevance
            final_results.append(m)
    return sorted(final_results, key=lambda x: -x['llm_relevance'])

# ------------------ STREAMLIT UI ------------------
def main():
    st.markdown("""
        <style>
            .reportview-container .main .block-container {
                padding: 2rem 2rem;
            }
            .stButton > button {
                background-color: #004466;
                color: white;
                font-weight: bold;
                border-radius: 10px;
                padding: 0.5em 1em;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("📘 Legal Incident Analyzer (IPC)")
    st.subheader("🔍 Analyze criminal incidents and retrieve relevant Indian Penal Code sections")

    incident_text = st.text_area("📝 Describe the incident in detail:", height=150, placeholder="e.g., A man entered the house at night and stole valuables...")

    if st.button("Analyze Incident") and incident_text:
        ipc_df = load_data()
        ipc_df['crime_type'] = ipc_df.apply(assign_crime_type, axis=1)
        sbert_model, _ = load_model_and_embeddings(ipc_df)

        with st.spinner("🤖 Analyzing the incident using Gemini AI..."):
            summary_and_type = gemini_simplify_incident(incident_text)

        st.markdown("### 🧾 Gemini Summary Output")
        st.code(summary_and_type)

        summary, crime_type = '', ''
        for line in summary_and_type.split('\n'):
            if line.lower().startswith('summary:'):
                summary = line.split(':',1)[1].strip()
            if line.lower().startswith('crimetype:'):
                crime_type = line.split(':',1)[1].strip().lower()

        st.success(f"**🔎 Detected Crime Type:** {crime_type}")
        st.info(f"**📜 Legal Summary:** {summary}")

        candidate_df = route_query(ipc_df, crime_type)

        with st.spinner("📖 Retrieving relevant IPC/IT Act sections..."):
            initial_matches = retrieve_sections(summary, candidate_df, sbert_model)

        with st.spinner("🎯 Reranking sections based on LLM feedback..."):
            final_matches = llm_rerank_sections(summary, initial_matches, min_relevance=60)

        st.markdown("### 📚 Matched IPC/IT Sections")
        if not final_matches:
            st.error("❗ No relevant sections found.")
        else:
            for m in final_matches:
                with st.expander(f"📘 Section {m['section']} - {m['offence']}"):
                    st.markdown(f"**🔹 Description:** {m['description']}")
                    st.markdown(f"**🔹 Punishment:** {m['punishment']}")
                    st.markdown(f"**🧠 SBERT Similarity:** `{m['score']:.2f}`")
                    st.markdown(f"**📊 LLM Relevance:** `{m['llm_relevance']}`")

if __name__ == "__main__":
    main()