from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import PyPDF2 as pdf
import re
import requests
from urllib.parse import urlparse
from typing import Dict, Any
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Initialize session state
if 'hallucinations' not in st.session_state:
    st.session_state.hallucinations = []
if 'matches' not in st.session_state:
    st.session_state.matches = []

# Initialize Ollama
llm = ChatOllama(
    model="mistral:7b-instruct",
    temperature=0.5,
    top_k=40,
    top_p=0.85,
    repeat_penalty=1.15,
    num_ctx=2048,
    num_predict=512
)
output_parser = StrOutputParser()

# System Prompts
RESUME_SYSTEM_PROMPT = """You are an expert ATS analyst. Create a resume that exactly matches the job description using ONLY provided information.

Rules:
1. Extract keywords from job description
2. Verify every claim against the resume
3. Never invent information
4. Structure:
   [Contact Info]
   [Tailored Summary]
   [Skills (prioritized by JD keywords)]
   [Experience (reworded to match JD)]
   [Education]
5. Tag each match: [MATCH] 
6. Reject unverifiable information"""

COVER_LETTER_SYSTEM_PROMPT = """Create cover letters demonstrating exact qualification matches.

Requirements:
1. Open with specific JD reference
2. Map resume items to JD requirements
3. Highlight unique value
4. Closing with JD-specific motivation

Rules:
- Each claim must show [RESUME EVIDENCE]
- Each JD connection must show [JD REQUIREMENT]
- No unverifiable information"""

ANALYSIS_SYSTEM_PROMPT = """Analyze resume-JD match with precision:

1. Calculate match percentage (0-100%)
2. List missing keywords
3. Identify resume strengths
4. Suggest improvements
5. Provide overall assessment

Format:
Match Percentage: X%
Missing Keywords: [list]
Strengths: [list]
Improvements: [list]
Assessment: [text]"""

def track_hallucinations(response: str) -> str:
    """Analyze response for potential hallucinations"""
    hallucinations = []
    
    if "[INFERENCE]" in response:
        hallucinations.append("Made inferences beyond source material")
    if "invent" in response.lower() or "assume" in response.lower():
        hallucinations.append("Used speculative language")
    if "Not specified" in response:
        hallucinations.append("Identified missing information")
    
    if hallucinations:
        st.session_state.hallucinations.extend(hallucinations)
        st.warning(f"Potential hallucinations detected: {', '.join(hallucinations)}")
    return response

def generate_content(prompt_template: str, inputs: Dict[str, Any], system_prompt: str) -> str:
    """Generate content with proper error handling"""
    try:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(prompt_template)
        ])
        
        chain = prompt | llm | output_parser
        response = chain.invoke(inputs)
        return track_hallucinations(response)
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        return ""

def input_pdf_text(uploaded_file) -> str:
    """Extract text from PDF"""
    try:
        reader = pdf.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        st.error(f"PDF error: {str(e)}")
        return ""

def extract_text_from_url(url: str) -> str:
    """Extract text from job posting URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return re.sub('<[^<]+?>', '', response.text)[:5000]
    except Exception as e:
        st.error(f"URL error: {str(e)}")
        return ""

def extract_keywords(text: str, n: int = 15) -> list:
    """Extract top keywords from text"""
    words = re.findall(r'\w{3,}', text.lower())
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:n]

# UI Layout
st.title("Precision ATS Aligner")

jd_text = "" 
# Job Description Input
jd_method = st.radio("Job Description Input Method:", ("Text", "URL"), horizontal=True)

if jd_method == "Text":
    jd_text = st.text_area("Paste Job Description", height=150)
else:
    jd_url = st.text_input("Enter Job Posting URL")
    if jd_url:
        jd_text = extract_text_from_url(jd_url)

if jd_text:
    with st.expander("View Extracted JD Keywords"):
        jd_keywords = extract_keywords(jd_text)
        st.write("Top Keywords: " + ", ".join(jd_keywords))

# Document Uploads
resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
profile_file = st.file_uploader("Upload Profile (Optional)", type="pdf")

if jd_text and resume_file:
    resume_text = input_pdf_text(resume_file)
    profile_text = input_pdf_text(profile_file) if profile_file else ""
    
    # Action Buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Generate Precision Resume"):
            prompt = """
            Create resume optimized for this job:
            Job Description: {jd}
            Current Resume: {resume}
            Additional Profile: {profile}
            Keywords to emphasize: {keywords}
            
            Requirements:
            1. Include all contact information
            2. Tailor professional summary
            3. Reorder skills by relevance
            4. Rewrite experience bullets to match JD
            5. Keep to 1 page
            """
            
            result = generate_content(
                prompt,
                {
                    "jd": jd_text,
                    "resume": resume_text,
                    "profile": profile_text,
                    "keywords": jd_keywords
                },
                RESUME_SYSTEM_PROMPT
            )
            
            if result:
                st.subheader("Generated Resume")
                st.text_area("Result", value=result, height=500)
    
    with col2:
        if st.button("Generate Cover Letter"):
            prompt = """
            Write cover letter for:
            Job Description: {jd}
            Resume: {resume}
            
            Requirements:
            1. Professional opening
            2. Highlight 3 key qualifications
            3. Show evidence from resume
            4. Strong closing
            """
            
            result = generate_content(
                prompt,
                {
                    "jd": jd_text,
                    "resume": resume_text
                },
                COVER_LETTER_SYSTEM_PROMPT
            )
            
            if result:
                st.subheader("Generated Cover Letter")
                st.text_area("Result", value=result, height=400)
                match_count = result.count("[RESUME EVIDENCE]")
                st.success(f"Verified Evidence Links: {match_count}")
    
    with col3:
        if st.button("Analyze Resume-JD Match"):
            prompt = """
            Analyze resume match with job:
            Job Description: {jd}
            Resume: {resume}
            
            Provide:
            1. Match percentage (0-100%)
            2. Missing keywords
            3. Resume strengths
            4. Suggested improvements
            """
            
            result = generate_content(
                prompt,
                {
                    "jd": jd_text,
                    "resume": resume_text
                },
                ANALYSIS_SYSTEM_PROMPT
            )
            
            if result:
                st.subheader("Analysis Results")
                st.text_area("Result", value=result, height=400)
                if "Match Percentage:" in result:
                    match = re.search(r"Match Percentage:\s*(\d+)%", result)
                    if match:
                        st.session_state.matches.append(int(match.group(1)))
                        st.metric("Match Score", f"{match.group(1)}%")

# Tracking Sidebar
with st.sidebar:
    st.header("Tracking Information")
    
    if st.session_state.hallucinations:
        st.subheader("Hallucination Log")
        for i, h in enumerate(st.session_state.hallucinations[-5:], 1):
            st.warning(f"{i}. {h}")
    
    if st.session_state.matches:
        st.subheader("Match History")
        st.line_chart({"Match %": st.session_state.matches})