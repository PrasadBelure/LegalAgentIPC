# 1. Install dependencies (uncomment if running in Colab)
# !pip install --quiet google-generativeai sentence-transformers pandas

import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# 2. Set up your Gemini API key
GEMINI_API_KEY = "AIzaSyDABvwRicVONM2D2MOqW886-UGo9BmxHdc"  # <-- Replace with your Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# 3. Load your IPC/IT Act dataset
ipc_df = pd.read_csv('ipc_sections.csv')
ipc_df = ipc_df.dropna(subset=['Section','Offence','Punishment'])
ipc_df['context'] = (
    ipc_df['Section'].astype(str) + ' ' +
    ipc_df['Offence'].astype(str).str.lower() + ' ' +
    ipc_df.get('Description', '').astype(str).str.lower()
)

# 4. Assign crime_type to each section (simple rule-based mapping)
def assign_crime_type(row):
    txt = (str(row['Offence']) + ' ' + str(row.get('Description',''))).lower()
    if any(word in txt for word in ['cheat', 'fraud', 'impersonat', 'phishing', 'cyber']):
        return 'cybercrime'
    if any(word in txt for word in ['theft', 'steal', 'pickpocket']):
        return 'theft'
    if any(word in txt for word in ['robbery', 'dacoity', 'snatch']):
        return 'robbery'
    if any(word in txt for word in ['forgery', 'counterfeit']):
        return 'forgery'
    if any(word in txt for word in ['sexual', 'rape', 'molest']):
        return 'sexual'
    if any(word in txt for word in ['kidnap', 'abduct']):
        return 'kidnapping'
    if any(word in txt for word in ['murder', 'homicide']):
        return 'murder'
    return 'other'
ipc_df['crime_type'] = ipc_df.apply(assign_crime_type, axis=1)

# 5. Load legal-domain SBERT
sbert_model = SentenceTransformer('bhavyagiri/InLegal-Sbert')
ipc_embs = sbert_model.encode(ipc_df['context'].tolist(), convert_to_tensor=True)

# 6. Gemini LLM summarization and crime type extraction
def gemini_simplify_incident(incident_text):
    prompt = (
        "You are an Indian legal assistant. Given the following incident, do two things:\n"
        "1. Summarize the core criminal act in one sentence, using formal legal language similar to Indian Penal Code (IPC) section descriptions.\n"
        "2. Identify the most likely crime type (e.g., 'cheating', 'theft', 'robbery', 'cybercrime', 'forgery', 'sexual offense', 'kidnapping', etc.).\n\n"
        f"Incident: {incident_text}\n\n"
        "Output format:\n"
        "Summary: <your one-sentence legal summary>\n"
        "CrimeType: <the main crime type>\n"
    )
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# 7. Query routing: filter dataset by detected crime type
def route_query(ipc_df, crime_type):
    filtered = ipc_df[ipc_df['crime_type'].str.contains(crime_type, case=False, na=False)]
    if filtered.empty:
        return ipc_df
    return filtered

# 8. Section retrieval function (SBERT)
def retrieve_sections(summary, candidate_df, top_k=7, threshold=0.40):
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

# 9. LLM-based reranking
def llm_rerank_sections(summary, candidates, min_relevance=85):
    model = genai.GenerativeModel("gemini-1.5-flash")
    final_results = []
    for m in candidates:
        prompt = (
            "You are a legal expert. Given the incident summary and the IPC/IT Act section below, "
            "rate how relevant the section is to the incident on a scale from 0 to 100, where 100 means 'perfect match'.\n\n"
            f"Incident summary: {summary}\n"
            f"Section: {m['section']}\n"
            f"Offence: {m['offence']}\n"
            f"Description: {m['description']}\n"
            f"Punishment: {m['punishment']}\n\n"
            "Respond ONLY with an integer (the relevance score)."
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

# 10. Main workflow
if __name__ == "__main__":
    incident_text = input("Describe the incident:\n")
    print("\n🔎 Gemini LLM summary and crime type:")
    summary_and_type = gemini_simplify_incident(incident_text)
    print(summary_and_type)

    # Parse summary and crime type from LLM output
    summary = ''
    crime_type = ''
    for line in summary_and_type.split('\n'):
        if line.lower().startswith('summary:'):
            summary = line.split(':',1)[1].strip()
        if line.lower().startswith('crimetype:'):
            crime_type = line.split(':',1)[1].strip().lower()

    print(f"\n🔖 Detected crime type: {crime_type}")
    print(f"🔎 Legal summary: {summary}")

    # Route query to relevant subset
    candidate_df = route_query(ipc_df, crime_type)

    # Retrieve top sections (SBERT)
    initial_matches = retrieve_sections(summary, candidate_df)

    # Rerank with Gemini LLM
    print("\n🔍 LLM Reranking (only sections with relevance ≥ 85):\n")
    final_matches = llm_rerank_sections(summary, initial_matches, min_relevance=60)
    if not final_matches:
        print("❗ No sections found with relevance")
    else:
        for m in final_matches:
            print(f"📘 Section: {m['section']}")
            print(f"🔸 Offence: {m['offence']}")
            if m['description']:
                print(f"🔹 Description: {m['description']}")
            print(f"🔹 Punishment: {m['punishment']}")
            print(f"⭐ SBERT Similarity: {m['score']:.2f}")
            print(f"⭐ LLM Relevance: {m['llm_relevance']}")
            print('-'*50)