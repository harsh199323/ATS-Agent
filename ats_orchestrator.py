from dotenv import load_dotenv
load_dotenv()

import os
import re
import requests
import PyPDF2 as pdf
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from bs4 import BeautifulSoup

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from config import AVAILABLE_MODELS, SCORING_WEIGHTS, EMBEDDING_MODEL_NAME, SKILL_LEXICON

MAX_JD_CHARS = 5000
MAX_RESUME_CHARS = 8000
MAX_PROFILE_CHARS = 4000

# Create a helper to instantiate a ChatOllama model with common parameters
def get_llm_instance(model_name: str) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=0.5,
        top_k=40,
        top_p=0.85,
        repeat_penalty=1.15,
        num_ctx=2048,
        num_predict=512
    )

# Instantiate models based on config.
llm_embedding = get_llm_instance(AVAILABLE_MODELS["embedding"])
llm_general_light = get_llm_instance(AVAILABLE_MODELS["general"]["light"])
llm_general_heavy = get_llm_instance(AVAILABLE_MODELS["general"]["heavy"])
# For technical tasks, here we choose "data_science" as an example; you may extend this logic.
llm_technical_light = get_llm_instance(AVAILABLE_MODELS["technical"]["data_science"]["light"])
llm_technical_heavy = get_llm_instance(AVAILABLE_MODELS["technical"]["data_science"]["heavy"])
llm_creative_polish = get_llm_instance(AVAILABLE_MODELS["creative_polish"])

output_parser = StrOutputParser()

_NLP = None
_EMBEDDING_MODEL: Optional[Any] = None

_SIMPLE_STOPWORDS = {
    "the", "and", "for", "with", "you", "your", "our", "will", "from", "that",
    "this", "are", "was", "were", "has", "have", "had", "into", "over", "under",
    "about", "across", "more", "most", "such", "including", "within", "into",
    "using", "used", "use", "per", "via", "all", "any", "each", "other"
}

# Define system prompts for each task.
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
    """Analyze response for potential hallucinations."""
    hallucinations = []
    if "[INFERENCE]" in response:
        hallucinations.append("Made inferences beyond source material")
    if "invent" in response.lower() or "assume" in response.lower():
        hallucinations.append("Used speculative language")
    if "Not specified" in response:
        hallucinations.append("Identified missing information")
    # (In a Streamlit app, you can log these to session state.)
    return response

def generate_content(prompt_template: str, inputs: Dict[str, Any], system_prompt: str, model_instance: ChatOllama) -> str:
    """Generate content using the provided model instance and LangChain prompt chaining."""
    try:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(prompt_template)
        ])
        chain = prompt | model_instance | output_parser
        response = chain.invoke(inputs)
        return track_hallucinations(response)
    except Exception as e:
        return f"Generation error: {str(e)}"

def generate_with_fallback(
    prompt_template: str,
    inputs: Dict[str, Any],
    system_prompt: str,
    primary_model: ChatOllama,
    fallback_model: ChatOllama
) -> str:
    """Retry with a lighter model when the primary model exceeds memory."""
    result = generate_content(prompt_template, inputs, system_prompt, primary_model)
    if "requires more system memory" in result:
        return generate_content(prompt_template, inputs, system_prompt, fallback_model)
    return result

def input_pdf_text(uploaded_file) -> str:
    """Extract text from a PDF file."""
    try:
        reader = pdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        return f"PDF error: {str(e)}"

def extract_text_from_url(url: str) -> str:
    """Extract text from a job posting URL with basic hardening."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (ATS-Aligner/1.0; +https://example.com)"
        }
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "text" not in content_type and "html" not in content_type:
            return "URL error: unsupported content type"
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        clean = soup.get_text(separator=" ", strip=True)
        return clean[:5000]
    except Exception as e:
        return f"URL error: {str(e)}"

def get_nlp() -> Optional[Any]:
    """Load spaCy model with a lightweight fallback."""
    global _NLP
    if _NLP is None:
        try:
            os.environ.setdefault("THINC_NO_PYTORCH", "1")
            import spacy
            try:
                _NLP = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                download("en_core_web_sm")
                _NLP = spacy.load("en_core_web_sm")
        except Exception:
            return None
    return _NLP

def get_embedding_model() -> Optional[Any]:
    """Load sentence-transformer model once, or return None if unavailable."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            return None
        try:
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception:
            return None
    return _EMBEDDING_MODEL

def normalize_text(text: str) -> str:
    """Normalize text for matching and vectorization."""
    cleaned = re.sub(r"[^\w\s\-\+\.]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()

def get_tfidf_vectorizer() -> Tuple[Optional[Any], Optional[Any]]:
    """Return TF-IDF vectorizer and stopwords if sklearn is available."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
        return TfidfVectorizer, ENGLISH_STOP_WORDS
    except Exception:
        return None, None

def simple_tokenize(text: str, stopwords: Optional[set] = None) -> List[str]:
    """Tokenize text with a basic regex and stopword filter."""
    stop = stopwords or _SIMPLE_STOPWORDS
    return [w for w in re.findall(r"\w{3,}", text.lower()) if w not in stop]

def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between two 2D arrays."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.dot(a_norm, b_norm.T)

def extract_phrases(doc: Any) -> List[str]:
    """Extract noun-phrase and entity phrases."""
    phrases: List[str] = []
    for chunk in doc.noun_chunks:
        tokens = [t.lemma_.lower() for t in chunk if not t.is_stop and not t.is_punct]
        if 2 <= len(tokens) <= 5:
            phrases.append(" ".join(tokens))
    for ent in doc.ents:
        tokens = [t.lemma_.lower() for t in ent if not t.is_stop and not t.is_punct]
        if 2 <= len(tokens) <= 5:
            phrases.append(" ".join(tokens))
    return phrases

def extract_keywords(text: str, n: int = 15) -> Dict[str, List[str]]:
    """Extract top keywords and phrases from text."""
    nlp = get_nlp()
    if nlp is None:
        _, stopwords = get_tfidf_vectorizer()
        stop = set(stopwords) if stopwords else _SIMPLE_STOPWORDS
        words = [w for w in re.findall(r"\w{3,}", text.lower()) if w not in stop]
        counts: Dict[str, int] = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        top_keywords = sorted(counts, key=counts.get, reverse=True)[:n]
        return {"keywords": top_keywords, "phrases": []}

    doc = nlp(text)
    token_counts: Dict[str, int] = {}
    phrase_counts: Dict[str, int] = {}

    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        lemma = token.lemma_.lower()
        if len(lemma) < 3:
            continue
        token_counts[lemma] = token_counts.get(lemma, 0) + 1

    for phrase in extract_phrases(doc):
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

    top_keywords = sorted(token_counts, key=token_counts.get, reverse=True)[:n]
    top_phrases = sorted(phrase_counts, key=phrase_counts.get, reverse=True)[:n]
    return {"keywords": top_keywords, "phrases": top_phrases}

def extract_years_experience(text: str) -> int:
    """Extract the maximum years-of-experience requirement if present."""
    matches = re.findall(r"(\d+)\s*\+?\s*(?:years|yrs)", text.lower())
    return max((int(m) for m in matches), default=0)

def extract_skill_entities(text: str) -> Dict[str, Any]:
    """Extract skills/tools and years of experience from text."""
    lowered = text.lower()
    skills = set()
    for skill in SKILL_LEXICON:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, lowered):
            skills.add(skill)
    years = extract_years_experience(lowered)
    return {"skills": skills, "years": years}

def split_sections(text: str, max_lines: int = 3, max_sections: int = 20) -> List[str]:
    """Split text into compact sections for semantic similarity."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sections: List[str] = []
    current: List[str] = []

    def is_heading(line: str) -> bool:
        return (line.isupper() and len(line) <= 60) or (line.endswith(":") and len(line) <= 80)

    for line in lines:
        if is_heading(line) and current:
            sections.append(" ".join(current))
            current = [line]
            continue
        current.append(line)
        if len(current) >= max_lines:
            sections.append(" ".join(current))
            current = []

    if current:
        sections.append(" ".join(current))

    return sections[:max_sections]

def compute_semantic_similarity(jd_text: str, resume_text: str, top_k: int = 5) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute semantic similarity and return top section matches."""
    jd_sections = split_sections(jd_text)
    resume_sections = split_sections(resume_text)
    if not jd_sections or not resume_sections:
        return 0.0, []

    model = get_embedding_model()
    if model is None:
        vectorizer_cls, _ = get_tfidf_vectorizer()
        if vectorizer_cls is not None:
            vectorizer = vectorizer_cls(stop_words="english", ngram_range=(1, 2), max_features=5000)
            tfidf = vectorizer.fit_transform(jd_sections + resume_sections)
            jd_vecs = tfidf[: len(jd_sections)].toarray()
            resume_vecs = tfidf[len(jd_sections) :].toarray()
            sim_matrix = cosine_matrix(jd_vecs, resume_vecs)
        else:
            sim_matrix = np.zeros((len(jd_sections), len(resume_sections)))
            for i, jd_section in enumerate(jd_sections):
                jd_tokens = set(simple_tokenize(jd_section))
                for j, resume_section in enumerate(resume_sections):
                    resume_tokens = set(simple_tokenize(resume_section))
                    union = jd_tokens | resume_tokens
                    sim_matrix[i, j] = (len(jd_tokens & resume_tokens) / len(union)) if union else 0.0
    else:
        jd_emb = model.encode(jd_sections, normalize_embeddings=True)
        resume_emb = model.encode(resume_sections, normalize_embeddings=True)
        sim_matrix = cosine_matrix(np.array(jd_emb), np.array(resume_emb))

    top_pairs: List[Dict[str, Any]] = []
    max_scores = sim_matrix.max(axis=1)
    avg_max = float(max_scores.mean())

    for i, score in enumerate(max_scores):
        j = int(sim_matrix[i].argmax())
        top_pairs.append({
            "jd_section": jd_sections[i],
            "resume_section": resume_sections[j],
            "score": float(score)
        })

    top_pairs = sorted(top_pairs, key=lambda x: x["score"], reverse=True)[:top_k]
    return avg_max, top_pairs

def compute_tfidf_similarity(jd_text: str, resume_text: str) -> float:
    """Compute TF-IDF cosine similarity as a keyword compliance signal."""
    cleaned_jd = normalize_text(jd_text)
    cleaned_resume = normalize_text(resume_text)
    if not cleaned_jd or not cleaned_resume:
        return 0.0
    vectorizer_cls, _ = get_tfidf_vectorizer()
    if vectorizer_cls is None:
        jd_tokens = set(simple_tokenize(cleaned_jd))
        resume_tokens = set(simple_tokenize(cleaned_resume))
        union = jd_tokens | resume_tokens
        return (len(jd_tokens & resume_tokens) / len(union)) if union else 0.0
    vectorizer = vectorizer_cls(stop_words="english", ngram_range=(1, 2), max_features=5000)
    tfidf = vectorizer.fit_transform([cleaned_jd, cleaned_resume]).toarray()
    sim = cosine_matrix(tfidf[:1], tfidf[1:2])
    return float(sim[0][0])

def compute_match_scores(jd_text: str, resume_text: str) -> Dict[str, Any]:
    """Compute hybrid match scores and diagnostics."""
    semantic_score, top_matches = compute_semantic_similarity(jd_text, resume_text)
    tfidf_score = compute_tfidf_similarity(jd_text, resume_text)

    jd_entities = extract_skill_entities(jd_text)
    resume_entities = extract_skill_entities(resume_text)
    jd_skills = jd_entities["skills"]
    resume_skills = resume_entities["skills"]
    overlap = jd_skills.intersection(resume_skills)
    skill_overlap = (len(overlap) / len(jd_skills)) if jd_skills else 0.0

    jd_years = jd_entities["years"]
    resume_years = resume_entities["years"]
    experience_match = 0.0
    if jd_years > 0:
        experience_match = min(resume_years / jd_years, 1.0) if resume_years > 0 else 0.0

    ontology_boost = 1.0 if skill_overlap >= 0.30 else 0.0

    overall = (
        SCORING_WEIGHTS["semantic"] * semantic_score +
        SCORING_WEIGHTS["skill_overlap"] * skill_overlap +
        SCORING_WEIGHTS["tfidf"] * tfidf_score +
        SCORING_WEIGHTS["experience"] * experience_match +
        SCORING_WEIGHTS["ontology"] * ontology_boost
    )
    overall_score = max(0.0, min(100.0, overall * 100))

    missing_skills = sorted(jd_skills - resume_skills)
    return {
        "overall_score": overall_score,
        "semantic_score": semantic_score,
        "tfidf_score": tfidf_score,
        "skill_overlap": skill_overlap,
        "experience_match": experience_match,
        "ontology_boost": ontology_boost,
        "missing_skills": missing_skills,
        "top_matches": top_matches
    }

def clamp_text(text: str, max_chars: int) -> str:
    """Reduce large inputs to keep model calls responsive."""
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:max_chars]

def generate_resume(jd: str, resume: str, profile: str, keywords: list, low_memory: bool = False) -> str:
    """Generate a precision resume using JD, resume, profile, and extracted keywords."""
    jd_entities = extract_skill_entities(jd)
    resume_entities = extract_skill_entities(resume)
    missing_skills = sorted(jd_entities["skills"] - resume_entities["skills"])

    prompt = """
    Create resume optimized for this job:
    Job Description: {jd}
    Current Resume: {resume}
    Additional Profile: {profile}
    Keywords to emphasize: {keywords}
    Missing skills (do not add if absent): {missing_skills}
    
    Requirements:
    1. Include all contact information
    2. Tailor professional summary
    3. Reorder skills by relevance
    4. Rewrite experience bullets to match JD
    5. Keep to 1 page
    """
    inputs = {
        "jd": clamp_text(jd, MAX_JD_CHARS),
        "resume": clamp_text(resume, MAX_RESUME_CHARS),
        "profile": clamp_text(profile, MAX_PROFILE_CHARS),
        "keywords": ", ".join(keywords),
        "missing_skills": ", ".join(missing_skills) if missing_skills else "None"
    }
    if low_memory:
        return generate_content(prompt, inputs, RESUME_SYSTEM_PROMPT, llm_technical_light)
    # Try heavy model first, then fall back to light if memory is insufficient.
    return generate_with_fallback(prompt, inputs, RESUME_SYSTEM_PROMPT, llm_technical_heavy, llm_technical_light)

def generate_cover_letter(jd: str, resume: str, low_memory: bool = False) -> str:
    """Generate a cover letter based on the job description and resume."""
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
    inputs = {
        "jd": clamp_text(jd, MAX_JD_CHARS),
        "resume": clamp_text(resume, MAX_RESUME_CHARS)
    }
    if low_memory:
        return generate_content(prompt, inputs, COVER_LETTER_SYSTEM_PROMPT, llm_general_light)
    # Try heavy model first, then fall back to light if memory is insufficient.
    return generate_with_fallback(prompt, inputs, COVER_LETTER_SYSTEM_PROMPT, llm_general_heavy, llm_general_light)

def analyze_match(jd: str, resume: str, low_memory: bool = False) -> str:
    """Analyze the match between a resume and a job description."""
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
    # For analysis, use light model regardless of mode.
    inputs = {"jd": clamp_text(jd, MAX_JD_CHARS), "resume": clamp_text(resume, MAX_RESUME_CHARS)}
    return generate_content(prompt, inputs, ANALYSIS_SYSTEM_PROMPT, llm_general_light)
