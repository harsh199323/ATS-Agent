from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import re
from ats_orchestrator import (
    input_pdf_text,
    extract_text_from_url,
    extract_keywords,
    compute_match_scores,
    generate_resume,
    generate_cover_letter,
    analyze_match
)

# Initialize session state for tracking hallucinations and match history.
if 'hallucinations' not in st.session_state:
    st.session_state.hallucinations = []
if 'matches' not in st.session_state:
    st.session_state.matches = []

st.title("Precision ATS Aligner")
st.caption("Primary UI. app.py is legacy.")

low_memory_mode = st.toggle("Low Memory Mode", value=True, help="Use lighter models to avoid RAM errors.")

# --- Job Description Input ---
jd_method = st.radio("Job Description Input Method:", ("Text", "URL"), horizontal=True)
jd_text = ""
if jd_method == "Text":
    jd_text = st.text_area("Paste Job Description", height=150)
else:
    jd_url = st.text_input("Enter Job Posting URL")
    if jd_url:
        jd_text = extract_text_from_url(jd_url)

if jd_text:
    with st.expander("View Extracted JD Keywords"):
        keyword_data = extract_keywords(jd_text)
        jd_keywords = keyword_data["keywords"]
        jd_phrases = keyword_data["phrases"]
        st.write("Top Keywords: " + ", ".join(jd_keywords))
        if jd_phrases:
            st.write("Key Phrases: " + ", ".join(jd_phrases))
else:
    jd_keywords = []
    jd_phrases = []

# --- Document Uploads ---
st.header("Candidate Files")
resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
profile_file = st.file_uploader("Upload Profile (Optional, PDF)", type="pdf")

resume_text = ""
profile_text = ""
if resume_file:
    resume_text = input_pdf_text(resume_file)
if profile_file:
    profile_text = input_pdf_text(profile_file)

st.markdown("---")
st.header("Actions")
col1, col2, col3 = st.columns(3)

if col1.button("Generate Precision Resume"):
    if not jd_text:
        st.error("Please provide a Job Description.")
    elif not resume_text:
        st.error("Please upload your resume.")
    else:
        with st.spinner("Generating resume..."):
            keywords_for_prompt = list(dict.fromkeys(jd_keywords + jd_phrases))
            result = generate_resume(jd_text, resume_text, profile_text, keywords_for_prompt, low_memory=low_memory_mode)
        st.subheader("Generated Resume")
        st.text_area("Result", value=result, height=500)

if col2.button("Generate Cover Letter"):
    if not jd_text:
        st.error("Please provide a Job Description.")
    elif not resume_text:
        st.error("Please upload your resume.")
    else:
        with st.spinner("Generating cover letter..."):
            result = generate_cover_letter(jd_text, resume_text, low_memory=low_memory_mode)
        st.subheader("Generated Cover Letter")
        st.text_area("Result", value=result, height=400)
        match_count = result.count("[RESUME EVIDENCE]")
        st.success(f"Verified Evidence Links: {match_count}")

if col3.button("Analyze Resume-JD Match"):
    if not jd_text:
        st.error("Please provide a Job Description.")
    elif not resume_text:
        st.error("Please upload your resume.")
    else:
        with st.spinner("Analyzing match..."):
            score_data = compute_match_scores(jd_text, resume_text)
            result = analyze_match(jd_text, resume_text, low_memory=low_memory_mode)
        st.subheader("Analysis Results")
        st.metric("Hybrid Match Score", f"{score_data['overall_score']:.1f}%")
        with st.expander("Hybrid Match Details"):
            st.write(f"Semantic Similarity: {score_data['semantic_score']:.3f}")
            st.write(f"TF-IDF Similarity: {score_data['tfidf_score']:.3f}")
            st.write(f"Skill Overlap: {score_data['skill_overlap']:.3f}")
            st.write(f"Experience Match: {score_data['experience_match']:.3f}")
            missing_skills = score_data["missing_skills"]
            st.write("Missing Skills: " + (", ".join(missing_skills) if missing_skills else "None"))
            if score_data["top_matches"]:
                st.write("Top Section Matches:")
                for match_row in score_data["top_matches"]:
                    st.write(
                        f"Score {match_row['score']:.3f} | JD: {match_row['jd_section']} | Resume: {match_row['resume_section']}"
                    )
        st.text_area("Result", value=result, height=400)
        match = re.search(r"Match Percentage:\s*(\d+)%", result)
        if match:
            st.session_state.matches.append(int(match.group(1)))
            st.metric("Match Score", f"{match.group(1)}%")

# --- Sidebar Tracking ---
with st.sidebar:
    st.header("Tracking Information")
    if st.session_state.hallucinations:
        st.subheader("Hallucination Log")
        for i, h in enumerate(st.session_state.hallucinations[-5:], 1):
            st.warning(f"{i}. {h}")
    if st.session_state.matches:
        st.subheader("Match History")
        st.line_chart({"Match %": st.session_state.matches})
